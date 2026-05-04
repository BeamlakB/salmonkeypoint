"""
test.py — Evaluate a trained Lite-HRNet checkpoint on the FIB test set.

Metrics reported
────────────────
  • OKS  (Object Keypoint Similarity) — overall and per keypoint
  • PCK  (Percentage of Correct Keypoints) @ thresholds 0.05 / 0.10 / 0.20
  • MPJPE (Mean Per-Joint Position Error) in pixels
  • AP   (Average Precision) at OKS thresholds 0.50 : 0.05 : 0.95

Outputs saved to  --out_dir  (default: test_results/)
  predictions.json   — COCO-format keypoint predictions
  metrics.json       — all numeric results
  viz/               — overlay images (one per test fish)

Usage
─────
  python test.py --checkpoint checkpoints/best_oks_model.pth
  python test.py --checkpoint checkpoints/best_oks_model.pth \\
                 --test_json dataset/test.json \\
                 --test_img  dataset/images/test \\
                 --out_dir   test_results/
"""

import os
import json
import math
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

# ── import everything from train.py ──────────────────────────────
from ninetrain import (
    cfg,
    LiteHRNet,
    FishKeypointDataset,
    WeightedHeatmapLoss,
    dark_decode,
    compute_oks,
    SIGMA,
)

# ─────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────

KP_NAMES = FishKeypointDataset.KEYPOINT_NAMES   # 9 names
K        = cfg.NUM_KEYPOINTS                    # 9

# Skeleton edges for visualisation (index pairs)
SKELETON = [
    (0, 1),   # mouth  → eye
    (1, 2),   # eye    → pectoral
    (2, 3),   # pectoral → pelvic
    (3, 5),   # pelvic → tailroot
    (1, 4),   # eye    → dorsal
    (4, 5),   # dorsal → tailroot
    (5, 6),   # tailroot → tailtop
    (5, 7),   # tailroot → tailcenter
    (5, 8),   # tailroot → tailbottom
]

# Colour per keypoint  (BGR for OpenCV)
KP_COLORS = [
    (0,   200, 255),   # mouth       — amber
    (0,   255, 180),   # eye         — green
    (255, 128,   0),   # pectoral    — blue
    (255,   0, 128),   # pelvic      — purple
    (0,   128, 255),   # dorsal      — orange
    (255, 255,   0),   # tailroot    — cyan
    (128,   0, 255),   # tailtop     — magenta
    (0,   255, 255),   # tailcenter  — yellow
    (200,   0, 255),   # tailbottom  — violet
]

PCK_THRESHOLDS = [0.05, 0.10, 0.20]   # fraction of bbox diagonal
OKS_THRESHOLDS = np.arange(0.50, 1.00, 0.05)   # for AP calculation


# ─────────────────────────────────────────────────────────────────
#  DATASET  (with extra info for evaluation)
# ─────────────────────────────────────────────────────────────────

class FishTestDataset(FishKeypointDataset):
    """
    Extends FishKeypointDataset to also return raw annotation metadata
    needed for back-projecting predictions into original image space.
    """

    def __getitem__(self, idx):
        img_t, hm_t, vis_t = super().__getitem__(idx)
        s = self.samples[idx]
        meta = {
            "file_name": s["file_name"],
            "bbox":      s["bbox"],          # [x, y, w, h] in original image
            "area":      s["area"],
            "ann_id":    idx,
        }
        return img_t, hm_t, vis_t, meta


def collate_with_meta(batch):
    imgs  = torch.stack([b[0] for b in batch])
    hms   = torch.stack([b[1] for b in batch])
    vis   = torch.stack([b[2] for b in batch])
    metas = [b[3] for b in batch]
    return imgs, hms, vis, metas


# ─────────────────────────────────────────────────────────────────
#  DECODING  — heatmap → original image pixel coords
# ─────────────────────────────────────────────────────────────────

def decode_batch(hm_pred_np, metas):
    """
    For each sample, run DARK decoding and project back to original image space.

    Returns list of dicts:
        file_name, bbox, pred_kps [K,2], gt_kps [K,2], vis [K], area
    """
    results = []
    for b, meta in enumerate(metas):
        hm  = hm_pred_np[b]                      # [K, Hm, Wm]
        # DARK decoding → coords in heatmap pixel space
        coords_hm = dark_decode(hm,
            input_size=(cfg.HEATMAP_H, cfg.HEATMAP_W))  # [K, 2]  (x,y)

        # Scale to crop space [INPUT_H × INPUT_W]
        scale_x = cfg.INPUT_W / cfg.HEATMAP_W
        scale_y = cfg.INPUT_H / cfg.HEATMAP_H
        coords_crop = coords_hm * np.array([[scale_x, scale_y]])

        # Project crop → original image  (undo the 10 % padding crop)
        bx, by, bw, bh = meta["bbox"]
        pad = 0.10
        bx_p = max(0, bx - bw * pad)
        by_p = max(0, by - bh * pad)
        bw_p = bw * (1 + 2 * pad)
        bh_p = bh * (1 + 2 * pad)

        pred_kps = np.zeros((K, 2), dtype=np.float32)
        pred_kps[:, 0] = coords_crop[:, 0] / cfg.INPUT_W  * bw_p + bx_p
        pred_kps[:, 1] = coords_crop[:, 1] / cfg.INPUT_H  * bh_p + by_p

        results.append({
            "file_name": meta["file_name"],
            "bbox":      meta["bbox"],
            "area":      meta["area"],
            "pred_kps":  pred_kps,         # [K, 2]  original image pixels
            "ann_id":    meta["ann_id"],
        })
    return results


# ─────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────

def _bbox_diagonal(bbox):
    _, _, bw, bh = bbox
    return math.sqrt(bw**2 + bh**2)


def compute_all_metrics(all_results, all_gt):
    """
    all_results : list of decode_batch outputs (dicts with pred_kps)
    all_gt      : list of dicts with gt_kps [K,2] and vis [K]

    Returns a metrics dict.
    """
    per_kp_errors  = defaultdict(list)   # kp_name → list of L2 errors (px)
    per_kp_pck     = {thr: defaultdict(list) for thr in PCK_THRESHOLDS}
    oks_list       = []
    per_kp_oks     = defaultdict(list)

    for res, gt in zip(all_results, all_gt):
        pred = res["pred_kps"]             # [K, 2]
        gtk  = gt["gt_kps"]               # [K, 2]
        vis  = gt["vis"]                   # [K]
        diag = _bbox_diagonal(res["bbox"])
        obj_scale = math.sqrt(max(res["area"], 1))

        # OKS
        oks = compute_oks(pred, gtk, vis, obj_scale)
        oks_list.append(oks)

        for k in range(K):
            if vis[k] == 0:
                continue
            name = KP_NAMES[k]
            err  = math.sqrt((pred[k, 0] - gtk[k, 0])**2 +
                             (pred[k, 1] - gtk[k, 1])**2)
            per_kp_errors[name].append(err)

            # Per-kp OKS contribution
            d2 = err**2
            s2 = (obj_scale * SIGMA[k])**2
            per_kp_oks[name].append(math.exp(-d2 / (2 * s2 + 1e-9)))

            # PCK
            for thr in PCK_THRESHOLDS:
                per_kp_pck[thr][name].append(int(err < thr * diag))

    # Aggregate
    mean_oks = float(np.mean(oks_list)) if oks_list else 0.0
    mpjpe    = {n: float(np.mean(v)) for n, v in per_kp_errors.items()}
    overall_mpjpe = float(np.mean([v for vals in per_kp_errors.values()
                                    for v in vals])) if per_kp_errors else 0.0
    mean_per_kp_oks = {n: float(np.mean(v)) for n, v in per_kp_oks.items()}

    pck_results = {}
    for thr in PCK_THRESHOLDS:
        pck_results[f"PCK@{thr}"] = {
            n: float(np.mean(v)) * 100
            for n, v in per_kp_pck[thr].items()
        }
        pck_results[f"PCK@{thr}"]["overall"] = float(np.mean(
            [v for vals in per_kp_pck[thr].values() for v in vals])) * 100

    # AP over OKS thresholds
    ap_list = []
    for thr in OKS_THRESHOLDS:
        hits = [int(o >= thr) for o in oks_list]
        ap_list.append(float(np.mean(hits)) if hits else 0.0)
    ap50  = ap_list[0]                          # OKS ≥ 0.50
    ap75  = ap_list[int((0.75 - 0.50) / 0.05)] # OKS ≥ 0.75
    mAP   = float(np.mean(ap_list))

    return {
        "num_instances":   len(oks_list),
        "mean_OKS":        round(mean_oks, 4),
        "AP@0.50:0.95":    round(mAP, 4),
        "AP@0.50":         round(ap50, 4),
        "AP@0.75":         round(ap75, 4),
        "overall_MPJPE_px": round(overall_mpjpe, 2),
        "per_keypoint_MPJPE_px": {k: round(v, 2) for k, v in mpjpe.items()},
        "per_keypoint_OKS":      {k: round(v, 4) for k, v in mean_per_kp_oks.items()},
        **pck_results,
    }


# ─────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────

def visualise(img_dir, results, gt_lookup, out_dir, max_images=50):
    """
    Draw predicted (filled circle) and GT (hollow circle) keypoints + skeleton.
    Saves one image per fish instance to out_dir/viz/.
    """
    viz_dir = Path(out_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    img_cache = {}

    for i, res in enumerate(results[:max_images]):
        fname    = res["file_name"]
        img_path = Path(img_dir) / fname

        if fname not in img_cache:
            if not img_path.exists():
                continue
            img_cache[fname] = cv2.imread(str(img_path))

        img = img_cache[fname].copy()
        if img is None:
            continue

        pred = res["pred_kps"]                          # [K, 2]
        gt   = gt_lookup.get(res["ann_id"])
        bx, by, bw, bh = [int(v) for v in res["bbox"]]

        # Draw bounding box
        cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (200, 200, 200), 2)

        # Draw skeleton
        for (a, b_) in SKELETON:
            if gt is not None and (gt["vis"][a] > 0 and gt["vis"][b_] > 0):
                pa = tuple(gt["gt_kps"][a].astype(int))
                pb = tuple(gt["gt_kps"][b_].astype(int))
                cv2.line(img, pa, pb, (180, 180, 180), 1, cv2.LINE_AA)
            pa = tuple(pred[a].astype(int))
            pb = tuple(pred[b_].astype(int))
            cv2.line(img, pa, pb, (255, 255, 255), 1, cv2.LINE_AA)

        for k in range(K):
            color = KP_COLORS[k]
            px, py = int(pred[k, 0]), int(pred[k, 1])

            # Predicted — filled
            cv2.circle(img, (px, py), 5, color, -1, cv2.LINE_AA)
            cv2.putText(img, KP_NAMES[k][:3], (px+6, py+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

            # GT — hollow (if available)
            if gt is not None and gt["vis"][k] > 0:
                gx, gy = int(gt["gt_kps"][k, 0]), int(gt["gt_kps"][k, 1])
                cv2.circle(img, (gx, gy), 6, color, 2, cv2.LINE_AA)

        # Crop to a padded region around the bbox for a tighter view
        pad  = 40
        x1   = max(0, bx - pad);  y1 = max(0, by - pad)
        x2   = min(img.shape[1], bx + bw + pad)
        y2   = min(img.shape[0], by + bh + pad)
        crop = img[y1:y2, x1:x2]

        stem = Path(fname).stem
        out_path = viz_dir / f"{stem}_inst{i:04d}.jpg"
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"  Saved {min(len(results), max_images)} visualisation(s) → {viz_dir}/")


# ─────────────────────────────────────────────────────────────────
#  PRINT RESULTS TABLE
# ─────────────────────────────────────────────────────────────────

def print_results(metrics):
    sep = "─" * 58
    print(f"\n{'═'*58}")
    print(f"  FIB TEST SET EVALUATION")
    print(f"{'═'*58}")
    print(f"  Instances evaluated : {metrics['num_instances']}")
    print(sep)
    print(f"  {'Metric':<30}  {'Value':>10}")
    print(sep)
    print(f"  {'mAP  (OKS 0.50:0.95)':<30}  {metrics['AP@0.50:0.95']:>10.4f}")
    print(f"  {'AP @ OKS ≥ 0.50':<30}  {metrics['AP@0.50']:>10.4f}")
    print(f"  {'AP @ OKS ≥ 0.75':<30}  {metrics['AP@0.75']:>10.4f}")
    print(f"  {'Mean OKS':<30}  {metrics['mean_OKS']:>10.4f}")
    print(f"  {'Overall MPJPE (px)':<30}  {metrics['overall_MPJPE_px']:>10.2f}")
    for thr in PCK_THRESHOLDS:
        key = f"PCK@{thr}"
        print(f"  {key+' (overall)':<30}  {metrics[key]['overall']:>9.2f}%")
    print(sep)
    print(f"  Per-keypoint MPJPE (px):")
    for name in KP_NAMES:
        mpjpe = metrics["per_keypoint_MPJPE_px"].get(name, float("nan"))
        oks   = metrics["per_keypoint_OKS"].get(name, float("nan"))
        print(f"    {name:<14}  MPJPE={mpjpe:6.2f} px   OKS={oks:.4f}")
    print(f"{'═'*58}\n")


# ─────────────────────────────────────────────────────────────────
#  MAIN EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────────

def evaluate(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────
    print(f"\nLoading test set from: {args.test_json}")
    test_ds = FishTestDataset(
        args.test_json, args.test_img,
        input_size=(cfg.INPUT_H, cfg.INPUT_W),
        heatmap_size=(cfg.HEATMAP_H, cfg.HEATMAP_W),
        sigma=cfg.HEATMAP_SIGMA,
        augment=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_with_meta,
    )

    # ── Load model ────────────────────────────────────────────────
    print(f"Loading checkpoint : {args.checkpoint}")
    model = LiteHRNet(num_keypoints=cfg.NUM_KEYPOINTS).to(cfg.DEVICE)
    state = torch.load(args.checkpoint, map_location=cfg.DEVICE)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"Model parameters   : {sum(p.numel() for p in model.parameters()):,}")

    # ── Build GT lookup from JSON ─────────────────────────────────
    # We need original-image GT coords to evaluate and visualise.
    with open(args.test_json) as f:
        coco = json.load(f)
    id2img = {img["id"]: img for img in coco["images"]}

    gt_by_ann_idx = {}   # ann_id (sequential) → {gt_kps, vis}
    for idx, ann in enumerate(a for a in coco["annotations"]
                               if len(a.get("keypoints", [])) >= 3 * K):
        img_info = id2img.get(ann["image_id"])
        if img_info is None:
            continue
        kps = ann["keypoints"]
        gt_kps = np.array([[kps[3*k], kps[3*k+1]] for k in range(K)],
                           dtype=np.float32)
        vis    = np.array([kps[3*k+2] for k in range(K)], dtype=np.float32)
        gt_by_ann_idx[idx] = {"gt_kps": gt_kps, "vis": vis}

    # ── Inference loop ────────────────────────────────────────────
    print("\nRunning inference ...")
    criterion  = WeightedHeatmapLoss(cfg.KEYPOINT_WEIGHTS).to(cfg.DEVICE)
    all_results, all_gt = [], []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for imgs, hm_gt, vis, metas in test_loader:
            imgs  = imgs.to(cfg.DEVICE)
            hm_gt = hm_gt.to(cfg.DEVICE)
            vis_t = vis.to(cfg.DEVICE)

            preds = model(imgs)
            loss  = criterion(preds, hm_gt, vis_t)
            total_loss += loss.item()
            n_batches  += 1

            # Sigmoid → numpy
            hm_np = torch.sigmoid(preds).cpu().numpy()   # [B, K, Hm, Wm]

            batch_res = decode_batch(hm_np, metas)
            for res in batch_res:
                ann_id = res["ann_id"]
                gt     = gt_by_ann_idx.get(ann_id)
                if gt is not None:
                    all_results.append(res)
                    all_gt.append(gt)

    avg_loss = total_loss / max(n_batches, 1)
    print(f"  Test heatmap loss : {avg_loss:.4f}")

    # ── Compute metrics ───────────────────────────────────────────
    print("Computing metrics  ...")
    metrics = compute_all_metrics(all_results, all_gt)
    metrics["heatmap_loss"] = round(avg_loss, 4)

    print_results(metrics)

    # ── Save metrics JSON ─────────────────────────────────────────
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")

    # ── Save COCO-format predictions ──────────────────────────────
    coco_preds = []
    for res, gt in zip(all_results, all_gt):
        flat_kps = []
        for k in range(K):
            flat_kps += [float(res["pred_kps"][k, 0]),
                         float(res["pred_kps"][k, 1]),
                         int(gt["vis"][k])]
        coco_preds.append({
            "image_id":       res["ann_id"],
            "category_id":    1,
            "keypoints":      flat_kps,
            "score":          1.0,
        })
    pred_path = out_dir / "predictions.json"
    with open(pred_path, "w") as f:
        json.dump(coco_preds, f, indent=2)
    print(f"  Predictions saved → {pred_path}")

    # ── Visualise ─────────────────────────────────────────────────
    if not args.no_viz:
        print(f"  Saving visualisations (max {args.max_viz}) ...")
        gt_lookup = {res["ann_id"]: gt
                     for res, gt in zip(all_results, all_gt)}
        visualise(args.test_img, all_results, gt_lookup,
                  out_dir, max_images=args.max_viz)

    print("Done.\n")
    return metrics


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained Lite-HRNet on the FIB test set")

    parser.add_argument("--checkpoint",  required=True,
                        help="Path to .pth checkpoint (best_oks_model.pth or similar)")
    parser.add_argument("--test_json",   default=cfg.TEST_JSON,
                        help="COCO JSON for the test split")
    parser.add_argument("--test_img",    default=cfg.TEST_IMG,
                        help="Directory containing test images")
    parser.add_argument("--out_dir",     default="test_results",
                        help="Output directory for metrics, predictions, and viz")
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_viz",     type=int, default=50,
                        help="Maximum visualisation images to save")
    parser.add_argument("--no_viz",      action="store_true",
                        help="Skip saving visualisation images")

    args = parser.parse_args()

    print(f"Device     : {cfg.DEVICE}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Test JSON  : {args.test_json}")
    print(f"Test imgs  : {args.test_img}")
    print(f"Output dir : {args.out_dir}")

    evaluate(args)