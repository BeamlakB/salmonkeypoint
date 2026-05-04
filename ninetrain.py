"""
Fish Keypoint Detection - Lite-HRNet Training Script
Based on: "A detection-regression based framework for fish keypoints detection"
          Dong et al., Intelligent Marine Technology and Systems (2023)

Adapted for FIB Dataset (Fish Instance Benchmark)
Keypoints (9): mouth, eye, pectoral, pelvic, dorsal, tailroot, tailtop, tailcenter, tailbottom
Data format: COCO Keypoints (train.json / val.json / test.json)
"""

import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import argparse
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────

class Config:
    # ── Dataset paths (FIB COCO split) ───────────────────────────
    DATA_ROOT  = "dataset2"
    TRAIN_JSON = "dataset2/train/_annotations.coco.json"
    VAL_JSON   = "dataset2/val/_annotations.coco.json"
    TEST_JSON  = "dataset2/test/_annotations.coco.json"
    TRAIN_IMG  = "dataset2/train"
    VAL_IMG    = "dataset2/val"
    TEST_IMG   = "dataset2/test"

    # ── Model ─────────────────────────────────────────────────────
    # 9 keypoints: mouth, eye, pectoral, pelvic, dorsal,
    #              tailroot, tailtop, tailcenter, tailbottom
    NUM_KEYPOINTS = 9
    INPUT_H       = 384
    INPUT_W       = 288
    HEATMAP_H     = 96    # INPUT_H // 4
    HEATMAP_W     = 72    # INPUT_W // 4
    HEATMAP_SIGMA = 2.0

    # Per-keypoint loss weights.
    # mouth/eye are small and precise → 1.5×
    # tail landmarks are spread out but structurally important → 1.2×
    # body landmarks are larger and easier → 1.0×
    #                   mouth  eye  pect  pelv  dors  tailroot  tailtop  tailctr  tailbot
    KEYPOINT_WEIGHTS = [1.5,   1.5, 1.0,  1.0,  1.0,  1.2,      1.2,     1.2,     1.2]

    # ── Training ──────────────────────────────────────────────────
    BATCH_SIZE   = 16
    NUM_EPOCHS   = 120
    LR           = 1e-4
    LR_STEP      = [80, 105]
    LR_GAMMA     = 0.1
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS  = 8
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Augmentation ──────────────────────────────────────────────
    FLIP_PROB    = 0.5
    ROT_DEGREES  = 30      # ± degrees for random rotation
    SCALE_RANGE  = (0.85, 1.15)   # random scale factor range

    # ── Checkpointing ─────────────────────────────────────────────
    SAVE_DIR     = "checkpoints"
    SAVE_EVERY   = 10


cfg = Config()


# ─────────────────────────────────────────────────────────────────
#  LITE-HRNET BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                             padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.dw(x)))
        x = self.relu(self.bn2(self.pw(x)))
        return x


def channel_shuffle(x, groups):
    B, C, H, W = x.shape
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, H, W)


class ConditionalChannelWeighting(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid = in_channels // 2
        self.dw   = nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False)
        self.bn_dw = nn.BatchNorm2d(mid)
        self.pw1  = nn.Conv2d(mid, mid, 1, bias=False)
        self.bn1  = nn.BatchNorm2d(mid)
        self.pw2  = nn.Conv2d(mid, mid, 1, bias=False)
        self.bn2  = nn.BatchNorm2d(mid)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.relu(self.bn_dw(self.dw(x2)))
        x2 = self.relu(self.bn1(self.pw1(x2)))
        x1 = self.relu(self.bn2(self.pw2(x1)))
        out = torch.cat([x1, x2], dim=1)
        return channel_shuffle(out, 2)


class ShuffleBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        mid = in_ch // 2
        self.branch = nn.Sequential(
            nn.Conv2d(mid, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.Conv2d(mid, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat([x1, self.branch(x2)], dim=1)
        return channel_shuffle(out, 2)


class LiteHRNetStage(nn.Module):
    def __init__(self, channels, num_blocks=4):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ShuffleBlock(channels) for _ in range(num_blocks)]
        )
        self.ccw = ConditionalChannelWeighting(channels)

    def forward(self, x):
        x = self.blocks(x)
        x = self.ccw(x)
        return x


class LiteHRNet(nn.Module):
    """
    Lite-HRNet-30 adapted for 9-keypoint fish detection.

    Parallel multi-resolution branches:
      Branch 1: full resolution  (stride 1)
      Branch 2: half resolution  (stride 2)
      Branch 3: quarter res      (stride 4)
    """

    def __init__(self, num_keypoints=9):
        super().__init__()
        self.num_keypoints = num_keypoints

        # ── Stem ──────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # → [B, 64, H/4, W/4]

        # ── Transition to 2 branches ──────────────────────────────
        self.trans1_b1 = nn.Sequential(
            nn.Conv2d(64, 40, 1, bias=False), nn.BatchNorm2d(40), nn.ReLU(True))
        self.trans1_b2 = nn.Sequential(
            nn.Conv2d(64, 80, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(80), nn.ReLU(True))

        # ── Stage 1 ───────────────────────────────────────────────
        self.stage1_b1 = LiteHRNetStage(40, num_blocks=4)
        self.stage1_b2 = LiteHRNetStage(80, num_blocks=4)

        # ── Transition to 3 branches ──────────────────────────────
        self.trans2_b3 = nn.Sequential(
            nn.Conv2d(80, 160, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(160), nn.ReLU(True))

        # ── Stage 2 ───────────────────────────────────────────────
        self.stage2_b1 = LiteHRNetStage(40, num_blocks=4)
        self.stage2_b2 = LiteHRNetStage(80, num_blocks=4)
        self.stage2_b3 = LiteHRNetStage(160, num_blocks=4)

        # ── Stage 3 (deeper) ──────────────────────────────────────
        self.stage3_b1 = LiteHRNetStage(40, num_blocks=8)
        self.stage3_b2 = LiteHRNetStage(80, num_blocks=8)
        self.stage3_b3 = LiteHRNetStage(160, num_blocks=8)

        # ── Fusion back to b1 resolution ──────────────────────────
        self.fuse_b2   = nn.Sequential(nn.Conv2d(80,  40, 1, bias=False), nn.BatchNorm2d(40))
        self.fuse_b3   = nn.Sequential(nn.Conv2d(160, 40, 1, bias=False), nn.BatchNorm2d(40))
        self.fuse_relu = nn.ReLU(inplace=True)

        # ── Final head ────────────────────────────────────────────
        self.head = nn.Conv2d(40, num_keypoints, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x  = self.stem(x)

        b1 = self.trans1_b1(x)
        b2 = self.trans1_b2(x)

        b1 = self.stage1_b1(b1)
        b2 = self.stage1_b2(b2)

        b3 = self.trans2_b3(b2)

        b1 = self.stage2_b1(b1)
        b2 = self.stage2_b2(b2)
        b3 = self.stage2_b3(b3)

        b1 = self.stage3_b1(b1)
        b2 = self.stage3_b2(b2)
        b3 = self.stage3_b3(b3)

        h, w   = b1.shape[2], b1.shape[3]
        b2_up  = nn.functional.interpolate(self.fuse_b2(b2), size=(h, w),
                                           mode='bilinear', align_corners=False)
        b3_up  = nn.functional.interpolate(self.fuse_b3(b3), size=(h, w),
                                           mode='bilinear', align_corners=False)
        fused  = self.fuse_relu(b1 + b2_up + b3_up)
        return self.head(fused)                 # [B, K, H/4, W/4]


# ─────────────────────────────────────────────────────────────────
#  DATASET  (FIB COCO Keypoints format)
# ─────────────────────────────────────────────────────────────────

# Horizontal-flip swap pairs for the 9 FIB keypoints.
# When an image is flipped left↔right the anatomical meaning of
# some landmarks swaps (e.g. the fish's mouth stays at the head
# but the left/right-sided fins swap sides).
# Index mapping (0-based):
#   0 mouth      ↔ 0  (midline, no swap)
#   1 eye        ↔ 1  (midline, no swap)
#   2 pectoral   ↔ 2  (single midline fin, no swap)
#   3 pelvic     ↔ 3  (single midline fin, no swap)
#   4 dorsal     ↔ 4  (midline, no swap)
#   5 tailroot   ↔ 5  (midline, no swap)
#   6 tailtop    ↔ 8  tailbottom  (top/bottom swap because image flipped)
#   7 tailcenter ↔ 7  (midline, no swap)
#   8 tailbottom ↔ 6  tailtop
FLIP_PAIRS = [(6, 8)]   # (tailtop, tailbottom)


class FishKeypointDataset(Dataset):
    """
    Loads FIB COCO-format keypoint annotations.

    Keypoint order in annotation (flat list, length = 3 × 9 = 27):
      [mouth_x, mouth_y, mouth_v,
       eye_x,   eye_y,   eye_v,
       pect_x,  pect_y,  pect_v,
       pelv_x,  pelv_y,  pelv_v,
       dors_x,  dors_y,  dors_v,
       tailroot_x, tailroot_y, tailroot_v,
       tailtop_x,  tailtop_y,  tailtop_v,
       tailctr_x,  tailctr_y,  tailctr_v,
       tailbot_x,  tailbot_y,  tailbot_v]

    v: 0 = not labeled, 1 = labeled but occluded, 2 = labeled & visible
    """

    KEYPOINT_NAMES = [
        "mouth", "eye", "pectoral", "pelvic", "dorsal",
        "tailroot", "tailtop", "tailcenter", "tailbottom",
    ]

    def __init__(self, json_path, img_dir,
                 input_size=(384, 288),
                 heatmap_size=(96, 72),
                 sigma=2.0,
                 augment=False):
        self.img_dir   = Path(img_dir)
        self.input_h, self.input_w     = input_size
        self.heatmap_h, self.heatmap_w = heatmap_size
        self.sigma   = sigma
        self.augment = augment

        with open(json_path) as f:
            coco = json.load(f)

        id2img = {img["id"]: img for img in coco["images"]}

        self.samples = []
        for ann in coco["annotations"]:
            kps = ann.get("keypoints", [])
            if len(kps) < 3 * cfg.NUM_KEYPOINTS:
                continue
            img_info = id2img.get(ann["image_id"])
            if img_info is None:
                continue
            self.samples.append({
                "file_name": img_info["file_name"],
                "img_w":     img_info["width"],
                "img_h":     img_info["height"],
                "bbox":      ann["bbox"],       # [x, y, w, h]
                "keypoints": ann["keypoints"],  # flat list len=27
                "area":      ann.get("area", 1),
            })

        print(f"  → Loaded {len(self.samples)} annotated fish from {json_path}")

    def __len__(self):
        return len(self.samples)

    # ── Gaussian heatmap generator ────────────────────────────────
    def _make_heatmap(self, kp_x, kp_y, visible):
        K  = cfg.NUM_KEYPOINTS
        Hm, Wm = self.heatmap_h, self.heatmap_w
        heatmaps = np.zeros((K, Hm, Wm), dtype=np.float32)

        # Build coordinate grids once
        gy, gx = np.mgrid[0:Hm, 0:Wm]

        for k in range(K):
            if visible[k] == 0:
                continue
            cx = kp_x[k] * Wm
            cy = kp_y[k] * Hm
            heatmaps[k] = np.exp(
                -((gx - cx)**2 + (gy - cy)**2) / (2 * self.sigma**2)
            )
        return heatmaps

    # ── Augmentation helpers ──────────────────────────────────────
    @staticmethod
    def _flip_keypoints(kp_x, kp_y, vis):
        kp_x = 1.0 - kp_x.copy()
        for i, j in FLIP_PAIRS:
            kp_x[[i, j]] = kp_x[[j, i]]
            kp_y[[i, j]] = kp_y[[j, i]]
            vis[[i, j]]  = vis[[j, i]]
        return kp_x, kp_y, vis

    @staticmethod
    def _rotate_keypoints(kp_x, kp_y, vis, angle_deg, crop_w, crop_h):
        """Rotate keypoints (in pixel crop coords) around crop center."""
        cx, cy = crop_w / 2, crop_h / 2
        rad = math.radians(-angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        new_x = kp_x.copy()
        new_y = kp_y.copy()
        for k in range(len(kp_x)):
            if vis[k] == 0:
                continue
            px = kp_x[k] * crop_w - cx
            py = kp_y[k] * crop_h - cy
            rx = cos_a * px - sin_a * py + cx
            ry = sin_a * px + cos_a * py + cy
            # Mark invisible if rotated out of bounds
            if rx < 0 or ry < 0 or rx >= crop_w or ry >= crop_h:
                vis[k] = 0
            new_x[k] = rx / crop_w
            new_y[k] = ry / crop_h
        return new_x, new_y, vis

    def __getitem__(self, idx):
        s        = self.samples[idx]
        img_path = self.img_dir / s["file_name"]
        img      = Image.open(img_path).convert("RGB")
        W0, H0   = img.size

        # ── Bounding-box crop with 10 % padding ───────────────────
        bx, by, bw, bh = s["bbox"]
        pad = 0.10
        bx  = max(0,      bx - bw * pad)
        by  = max(0,      by - bh * pad)
        bw  = min(W0 - bx, bw * (1 + 2 * pad))
        bh  = min(H0 - by, bh * (1 + 2 * pad))
        img_crop = img.crop((bx, by, bx + bw, by + bh))
        cw, ch   = img_crop.size   # crop width, height

        # ── Map keypoints → crop-relative [0,1] coords ───────────
        kps = s["keypoints"]
        K   = cfg.NUM_KEYPOINTS
        kp_x = np.array([(kps[3*k]   - bx) / bw for k in range(K)], dtype=np.float32)
        kp_y = np.array([(kps[3*k+1] - by) / bh for k in range(K)], dtype=np.float32)
        vis  = np.array([ kps[3*k+2]             for k in range(K)], dtype=np.float32)

        # ── Augmentation ──────────────────────────────────────────
        if self.augment:
            # 1. Random horizontal flip
            if torch.rand(1).item() < cfg.FLIP_PROB:
                img_crop = TF.hflip(img_crop)
                kp_x, kp_y, vis = self._flip_keypoints(kp_x, kp_y, vis)

            # 2. Random rotation
            if cfg.ROT_DEGREES > 0:
                angle = (torch.rand(1).item() * 2 - 1) * cfg.ROT_DEGREES
                img_crop = TF.rotate(img_crop, angle,
                                     interpolation=TF.InterpolationMode.BILINEAR,
                                     expand=False)
                kp_x, kp_y, vis = self._rotate_keypoints(
                    kp_x, kp_y, vis, angle, cw, ch)

            # 3. Random scale (resize crop slightly)
            scale = cfg.SCALE_RANGE[0] + torch.rand(1).item() * (
                cfg.SCALE_RANGE[1] - cfg.SCALE_RANGE[0])
            new_cw = max(1, int(cw * scale))
            new_ch = max(1, int(ch * scale))
            img_crop = TF.resize(img_crop, (new_ch, new_cw))
            # keypoint coords are relative [0,1], scale doesn't change them

            # 4. Color jitter
            img_crop = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            )(img_crop)

        # ── Resize to model input ─────────────────────────────────
        img_tensor = TF.resize(img_crop, (self.input_h, self.input_w))
        img_tensor = TF.to_tensor(img_tensor)
        img_tensor = TF.normalize(img_tensor,
                                  mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

        heatmaps = self._make_heatmap(kp_x, kp_y, vis)

        return img_tensor, torch.from_numpy(heatmaps), torch.from_numpy(vis)


# ─────────────────────────────────────────────────────────────────
#  LOSS  (weighted MSE on heatmaps)
# ─────────────────────────────────────────────────────────────────

class WeightedHeatmapLoss(nn.Module):
    """
    Per-keypoint weighted MSE loss.
    Ignores keypoints marked as not labeled (visibility == 0).
    """
    def __init__(self, keypoint_weights):
        super().__init__()
        w = torch.tensor(keypoint_weights, dtype=torch.float32)
        self.register_buffer("weights", w)

    def forward(self, pred, target, visibility):
        """
        pred:       [B, K, Hm, Wm]
        target:     [B, K, Hm, Wm]
        visibility: [B, K]
        """
        K = pred.shape[1]
        loss      = torch.zeros(1, device=pred.device)
        num_valid = 0

        for k in range(K):
            vis_mask = (visibility[:, k] > 0).float()
            if vis_mask.sum() == 0:
                continue
            diff = ((pred[:, k] - target[:, k]) ** 2).mean(dim=[1, 2])
            diff = (diff * vis_mask).sum() / vis_mask.sum()
            loss      = loss + self.weights[k] * diff
            num_valid += 1

        return loss / max(num_valid, 1)


# ─────────────────────────────────────────────────────────────────
#  DARK DECODING  (Distribution-Aware Coordinate Representation)
# ─────────────────────────────────────────────────────────────────

def dark_decode(heatmaps, input_size=(384, 288)):
    """
    Refine argmax predictions using the Hessian of the heatmap.

    Args:
        heatmaps:   np.ndarray [K, Hm, Wm]
        input_size: (H, W) of the model input image

    Returns:
        coords: np.ndarray [K, 2]  (x, y) in input_size pixel space
    """
    K, Hm, Wm = heatmaps.shape
    coords = np.zeros((K, 2), dtype=np.float32)

    for k in range(K):
        hm  = cv2.GaussianBlur(heatmaps[k].copy(), (3, 3), 0)
        idx = np.argmax(hm)
        my, mx = np.unravel_index(idx, hm.shape)
        mx = int(np.clip(mx, 1, Wm - 2))
        my = int(np.clip(my, 1, Hm - 2))

        dx  = 0.5 * (hm[my, mx+1] - hm[my, mx-1])
        dy  = 0.5 * (hm[my+1, mx] - hm[my-1, mx])
        dxx = hm[my, mx+1] + hm[my, mx-1] - 2*hm[my, mx]
        dyy = hm[my+1, mx] + hm[my-1, mx] - 2*hm[my, mx]
        dxy = 0.25 * (hm[my+1, mx+1] - hm[my+1, mx-1]
                      - hm[my-1, mx+1] + hm[my-1, mx-1])

        det = dxx * dyy - dxy**2
        if abs(det) < 1e-6:
            coords[k] = [mx * input_size[1] / Wm, my * input_size[0] / Hm]
            continue

        H_inv  = np.array([[dyy, -dxy], [-dxy, dxx]]) / det
        offset = -H_inv @ np.array([dx, dy])

        coords[k, 0] = (mx + offset[0]) * input_size[1] / Wm
        coords[k, 1] = (my + offset[1]) * input_size[0] / Hm

    return coords


# ─────────────────────────────────────────────────────────────────
#  OKS METRIC  (Object Keypoint Similarity)
# ─────────────────────────────────────────────────────────────────

# Normalisation factors σ per keypoint (adapted from COCO; tuned for fish).
# Smaller σ → stricter tolerance (precise landmarks like mouth/eye).
# Larger σ → looser tolerance (spread landmarks like tail edges).
#              mouth  eye   pect  pelv  dors  tailrt tailtop tailctr tailbot
SIGMA = np.array([0.060, 0.060, 0.087, 0.087, 0.087,
                   0.072, 0.072, 0.072, 0.072], dtype=np.float64)


def compute_oks(pred_kps, gt_kps, visibility, obj_scale):
    """
    pred_kps:   [K, 2]  predicted (x, y) in heatmap pixel space
    gt_kps:     [K, 2]  ground-truth
    visibility: [K]     0 / 1 / 2
    obj_scale:  float   sqrt(bbox_area) for normalisation
    """
    oks_sum, n_vis = 0.0, 0
    for k in range(len(pred_kps)):
        if visibility[k] == 0:
            continue
        d2 = ((pred_kps[k, 0] - gt_kps[k, 0])**2 +
              (pred_kps[k, 1] - gt_kps[k, 1])**2)
        s2 = (obj_scale * SIGMA[k])**2
        oks_sum += np.exp(-d2 / (2 * s2 + 1e-9))
        n_vis   += 1
    return oks_sum / max(n_vis, 1)


# ─────────────────────────────────────────────────────────────────
#  VALIDATION: OKS-based evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate_oks(model, loader, device):
    """
    Compute mean OKS over the entire loader using DARK decoding.
    Returns mean OKS score and per-keypoint mean distances.
    """
    model.eval()
    all_oks = []
    per_kp_dists = [[] for _ in range(cfg.NUM_KEYPOINTS)]

    with torch.no_grad():
        for imgs, heatmaps_gt, vis in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            preds = torch.sigmoid(preds)

            B = imgs.shape[0]
            for b in range(B):
                hm_pred = preds[b].cpu().numpy()     # [K, Hm, Wm]
                hm_gt   = heatmaps_gt[b].numpy()     # [K, Hm, Wm]
                v       = vis[b].numpy()             # [K]

                pred_coords = dark_decode(hm_pred,
                    input_size=(cfg.HEATMAP_H, cfg.HEATMAP_W))

                # Decode GT coords from heatmap argmax
                K, Hm, Wm = hm_gt.shape
                gt_coords = np.zeros((K, 2), dtype=np.float32)
                for k in range(K):
                    if v[k] == 0:
                        continue
                    idx = np.argmax(hm_gt[k])
                    gy, gx = np.unravel_index(idx, (Hm, Wm))
                    gt_coords[k] = [gx, gy]

                # Approximate obj_scale from heatmap size
                obj_scale = math.sqrt(Hm * Wm) * 0.25

                oks = compute_oks(pred_coords, gt_coords, v, obj_scale)
                all_oks.append(oks)

                # Per-keypoint L2 in heatmap pixels
                for k in range(K):
                    if v[k] > 0:
                        d = math.sqrt(
                            (pred_coords[k, 0] - gt_coords[k, 0])**2 +
                            (pred_coords[k, 1] - gt_coords[k, 1])**2)
                        per_kp_dists[k].append(d)

    mean_oks = float(np.mean(all_oks)) if all_oks else 0.0
    mean_dists = [float(np.mean(d)) if d else float("nan")
                  for d in per_kp_dists]
    return mean_oks, mean_dists


# ─────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────

def train():
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    train_ds = FishKeypointDataset(
        cfg.TRAIN_JSON, cfg.TRAIN_IMG,
        input_size=(cfg.INPUT_H, cfg.INPUT_W),
        heatmap_size=(cfg.HEATMAP_H, cfg.HEATMAP_W),
        sigma=cfg.HEATMAP_SIGMA, augment=True)

    val_ds = FishKeypointDataset(
        cfg.VAL_JSON, cfg.VAL_IMG,
        input_size=(cfg.INPUT_H, cfg.INPUT_W),
        heatmap_size=(cfg.HEATMAP_H, cfg.HEATMAP_W),
        sigma=cfg.HEATMAP_SIGMA, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,  persistent_workers=True)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True)

    model     = LiteHRNet(num_keypoints=cfg.NUM_KEYPOINTS).to(cfg.DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    criterion = WeightedHeatmapLoss(cfg.KEYPOINT_WEIGHTS).to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.LR_STEP, gamma=cfg.LR_GAMMA)

    best_val_loss = float("inf")
    best_oks      = 0.0

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for imgs, heatmaps, vis in train_loader:
            imgs     = imgs.to(cfg.DEVICE)
            heatmaps = heatmaps.to(cfg.DEVICE)
            vis      = vis.to(cfg.DEVICE)

            preds = model(imgs)
            loss  = criterion(preds, heatmaps, vis)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate (loss) ───────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, heatmaps, vis in val_loader:
                imgs     = imgs.to(cfg.DEVICE)
                heatmaps = heatmaps.to(cfg.DEVICE)
                vis      = vis.to(cfg.DEVICE)
                preds    = model(imgs)
                val_loss += criterion(preds, heatmaps, vis).item()
        val_loss /= len(val_loader)

        scheduler.step()

        # ── OKS every 5 epochs (expensive) ───────────────────────
        oks_str = ""
        if epoch % 5 == 0:
            mean_oks, per_kp = evaluate_oks(model, val_loader, cfg.DEVICE)
            oks_str = f"  val_OKS: {mean_oks:.4f}"
            if mean_oks > best_oks:
                best_oks = mean_oks
                torch.save(model.state_dict(),
                           f"{cfg.SAVE_DIR}/best_oks_model.pth")
                oks_str += "  ✔ best OKS"
            per_kp_fmt = "  ".join(
                f"{FishKeypointDataset.KEYPOINT_NAMES[k]}:{per_kp[k]:.2f}px"
                for k in range(cfg.NUM_KEYPOINTS))
            print(f"   KP dists → {per_kp_fmt}")

        print(f"Epoch [{epoch:3d}/{cfg.NUM_EPOCHS}]  "
              f"train: {train_loss:.4f}  val: {val_loss:.4f}  "
              f"lr: {scheduler.get_last_lr()[0]:.2e}{oks_str}")

        # ── Checkpoint ────────────────────────────────────────────
        if epoch % cfg.SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, f"{cfg.SAVE_DIR}/ckpt_epoch{epoch:03d}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       f"{cfg.SAVE_DIR}/best_loss_model.pth")
            print(f"  ✔ Saved best-loss model (val_loss={best_val_loss:.4f})")

    print("\nTraining complete.")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Best val OKS  : {best_oks:.4f}")


# ─────────────────────────────────────────────────────────────────
#  TEST EVALUATION
# ─────────────────────────────────────────────────────────────────

def test(checkpoint_path):
    """Run evaluation on the held-out test set."""
    test_ds = FishKeypointDataset(
        cfg.TEST_JSON, cfg.TEST_IMG,
        input_size=(cfg.INPUT_H, cfg.INPUT_W),
        heatmap_size=(cfg.HEATMAP_H, cfg.HEATMAP_W),
        sigma=cfg.HEATMAP_SIGMA, augment=False)

    test_loader = DataLoader(
        test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True)

    model = LiteHRNet(num_keypoints=cfg.NUM_KEYPOINTS).to(cfg.DEVICE)
    state = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    # Support both raw state-dict and full checkpoint dict
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    print(f"\nLoaded weights from: {checkpoint_path}")

    mean_oks, per_kp = evaluate_oks(model, test_loader, cfg.DEVICE)

    print(f"\n{'='*55}")
    print(f"  Test OKS: {mean_oks:.4f}")
    print(f"{'='*55}")
    print("  Per-keypoint mean distances (heatmap pixels):")
    for k, name in enumerate(FishKeypointDataset.KEYPOINT_NAMES):
        print(f"    {name:>12s}:  {per_kp[k]:.2f} px")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────
#  INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────

def predict_keypoints(model, image_crop_pil, use_dark=True):
    """
    Run inference on a single cropped-fish PIL image.

    Returns:
        coords:   np.ndarray [K, 2]  (x, y) in input_size pixel space
        heatmaps: np.ndarray [K, Hm, Wm]
    """
    model.eval()
    img = TF.resize(image_crop_pil, (cfg.INPUT_H, cfg.INPUT_W))
    img = TF.to_tensor(img)
    img = TF.normalize(img,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
    img = img.unsqueeze(0).to(cfg.DEVICE)

    with torch.no_grad():
        hm = model(img)
    hm    = torch.sigmoid(hm)
    hm_np = hm[0].cpu().numpy()

    if use_dark:
        coords = dark_decode(hm_np, input_size=(cfg.INPUT_H, cfg.INPUT_W))
    else:
        K, Hm, Wm = hm_np.shape
        coords = np.zeros((K, 2), dtype=np.float32)
        for k in range(K):
            flat = np.argmax(hm_np[k])
            ry, rx = np.unravel_index(flat, (Hm, Wm))
            coords[k] = [rx * cfg.INPUT_W / Wm, ry * cfg.INPUT_H / Hm]

    return coords, hm_np


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lite-HRNet training for FIB 9-keypoint fish detection")

    parser.add_argument("--mode",       choices=["train", "test"], default="train")
    parser.add_argument("--train_json", default=cfg.TRAIN_JSON)
    parser.add_argument("--val_json",   default=cfg.VAL_JSON)
    parser.add_argument("--test_json",  default=cfg.TEST_JSON)
    parser.add_argument("--train_img",  default=cfg.TRAIN_IMG)
    parser.add_argument("--val_img",    default=cfg.VAL_IMG)
    parser.add_argument("--test_img",   default=cfg.TEST_IMG)
    parser.add_argument("--epochs",     type=int,   default=cfg.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=cfg.BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=cfg.LR)
    parser.add_argument("--save_dir",   default=cfg.SAVE_DIR)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint for --mode test")

    args = parser.parse_args()

    cfg.TRAIN_JSON  = args.train_json
    cfg.VAL_JSON    = args.val_json
    cfg.TEST_JSON   = args.test_json
    cfg.TRAIN_IMG   = args.train_img
    cfg.VAL_IMG     = args.val_img
    cfg.TEST_IMG    = args.test_img
    cfg.NUM_EPOCHS  = args.epochs
    cfg.BATCH_SIZE  = args.batch_size
    cfg.LR          = args.lr
    cfg.SAVE_DIR    = args.save_dir

    print(f"Device     : {cfg.DEVICE}")
    print(f"Keypoints  : {FishKeypointDataset.KEYPOINT_NAMES}")
    print(f"Mode       : {args.mode}")

    if args.mode == "train":
        train()
    else:
        ckpt = args.checkpoint or f"{cfg.SAVE_DIR}/best_oks_model.pth"
        test(ckpt)