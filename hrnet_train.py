"""
Fish Keypoint Detection — HRNet-W32 (ImageNet pretrained) + FIB Dataset
========================================================================
Backbone : HRNet-W32 loaded from timm with ImageNet weights.
Head     : Lightweight 1×1 conv → 9 keypoint heatmaps.
Dataset  : FIB COCO keypoint format (train / val / test splits).
Training : Two-stage
  Stage 1 (epochs 1–FREEZE_EPOCHS)  : backbone frozen, head only trained.
  Stage 2 (epochs FREEZE_EPOCHS+1–) : full network fine-tuned, backbone LR
                                       10× smaller than head LR.

"""

import os
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

try:
    import timm
except ImportError:
    raise ImportError(
        "timm is required.  Install with:  pip install timm"
    )




class Config:
    #       Dataset                                                                                                                              
    TRAIN_JSON = "dataset/train.json"
    VAL_JSON   = "dataset/val.json"
    TEST_JSON  = "dataset/test.json"
    TRAIN_IMG  = "dataset/images/train"
    VAL_IMG    = "dataset/images/val"
    TEST_IMG   = "dataset/images/test"

    TRAIN_JSON = "resized_data/train/train.json"  # Use pre-scaled annotations if images were resized
    VAL_JSON   = "resized_data/val/val.json"
    TEST_JSON  = "resized_data/test/test.json"
    TRAIN_IMG  = "resized_data/train"
    VAL_IMG    = "resized_data/val"
    TEST_IMG   = "resized_data/test"
    #       Pre-processing                                                                                                               
    PREPROCESS_MAX_SIZE = 1024   # resize 4K images once; None to skip

    #       Model                                                                                                                                   ─
    BACKBONE      = "hrnet_w32"   # timm model name
    PRETRAINED    = True          # load ImageNet weights
    NUM_KEYPOINTS = 9
    # HRNet-W32 final feature map is stride-4 of the input,
    # matching the heatmap resolution below.
    INPUT_H    = 384
    INPUT_W    = 288
    HEATMAP_H  = 96    # INPUT_H // 4
    HEATMAP_W  = 72    # INPUT_W // 4
    HEATMAP_SIGMA = 3.0   # larger → smoother target → better for small datasets

    # Per-keypoint loss weights
    #                   mouth  eye  pect  pelv  dors  tailroot tailtop tailctr tailbot
    KEYPOINT_WEIGHTS = [1.5,   1.5, 1.0,  1.0,  1.0,  1.2,     1.2,    1.2,    1.2]

    #       Two-stage training                                                                                                     
    # Stage 1: freeze backbone, train head only (fast convergence start)
    # Stage 2: unfreeze everything, fine-tune with lower backbone LR
    FREEZE_EPOCHS  = 20      # how many epochs to keep backbone frozen
    HEAD_LR        = 1e-3    # learning rate for the keypoint head
    BACKBONE_LR    = 1e-4    # backbone LR during fine-tuning (Stage 2)
    LR_STEP        = [180, 240]   # epochs to drop LR (out of 300)
    LR_GAMMA       = 0.2     # gentler drop than 0.1
    WEIGHT_DECAY   = 1e-4

    #       Training                                                                                                                              
    NUM_EPOCHS  = 300
    BATCH_SIZE  = 8      # smaller → more gradient steps per epoch
    NUM_WORKERS = 8
    USE_AMP     = True
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

    #       Augmentation (heavier than Lite-HRNet version)                               
    FLIP_PROB    = 0.5
    ROT_DEGREES  = 45          # was 30
    SCALE_RANGE  = (0.60, 1.40)   # was (0.85, 1.15)
    TRANSLATE    = 0.10        # random shift ±10 % of crop

    #       Checkpointing                                                                                                               ─
    SAVE_DIR   = "checkpoints"
    SAVE_EVERY = 10


cfg = Config()


# IMAGE PRE-PROCESSING

def preprocess_images(img_dirs, max_size=1024):
    """Resize 4K images to max_size on the long edge, in-place. Idempotent."""
    if max_size is None:
        return
    exts = {".jpg", ".jpeg", ".png"}
    for img_dir in img_dirs:
        p = Path(img_dir)
        if not p.exists():
            continue
        files = [f for f in p.iterdir() if f.suffix.lower() in exts]
        resized = 0
        for f in files:
            try:
                img = Image.open(f)
                W, H = img.size
                if max(W, H) <= max_size:
                    continue
                scale = max_size / max(W, H)
                img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
                img.save(f, quality=92)
                resized += 1
            except Exception as e:
                print(f"  Warning: could not resize {f}: {e}")
        print(f"  Preprocessed {resized}/{len(files)} images in {img_dir}")


#  BACKBONE + HEAD

class HRNetKeypointModel(nn.Module):
    """
    HRNet-W32 backbone (ImageNet pretrained via timm) with a lightweight
    keypoint head producing [B, K, H/4, W/4] heatmaps.

    timm's HRNet returns a list of feature maps at different resolutions.
    We use only the highest-resolution branch (stride 4) which is already
    spatially rich thanks to HRNet's multi-scale fusions.

    Architecture
                                  
    backbone → [B, 32, H/4, W/4]   (highest-res branch of HRNet-W32)
    head     → BN → ReLU → Conv1×1(32→K)
    """

    def __init__(self, num_keypoints: int = 9, pretrained: bool = True):
        super().__init__()

        #       Load HRNet-W32 with ImageNet weights                                              
        # features_only=True returns intermediate feature maps instead
        # of a classification logit — perfect for dense prediction.
        self.backbone = timm.create_model(
            "hrnet_w32",
            pretrained=pretrained,
            features_only=True,   # returns list of feature maps
        )

        # HRNet-W32 features_only output channels per stage:
        # [32, 64, 128, 256]  at strides [4, 8, 16, 32]
        # We use feat[1] (stride 4, 128 channels) — highest resolution.
        backbone_out_ch = 128

        #       Keypoint head                                                                                                     ─
        # Small but effective: BN stabilises the pretrained features,
        # ReLU adds non-linearity, 1×1 conv maps to K heatmaps.
        self.head = nn.Sequential(
            nn.BatchNorm2d(backbone_out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(backbone_out_ch, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_keypoints, 1),
        )

        # Initialise only the head — backbone keeps ImageNet weights
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        """Freeze all backbone parameters (Stage 1)."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("  Backbone frozen (Stage 1 — head-only training)")

    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning (Stage 2)."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("  Backbone unfrozen (Stage 2 — full fine-tuning)")

    def forward(self, x):
        # features is a list [stride4, stride8, stride16, stride32]
        features = self.backbone(x)
        # feat[1] is stride-4 (96×72) — matches HEATMAP_H × HEATMAP_W exactly
        f = features[1]           # [B, 128, H/4, W/4]
        return self.head(f)       # [B, K, H/4, W/4]




#  DATASET


# Horizontal flip swap pairs: tailtop (6) ↔ tailbottom (8)
FLIP_PAIRS = [(6, 8)]


class FishKeypointDataset(Dataset):
    """
    FIB COCO-format keypoint dataset with all speed fixes applied.

    Key points
                             
    • Vectorised heatmap generation (numpy mgrid, ~50× faster).
    • In-memory image cache (no repeated disk reads across epochs).
    • Scale-aware bbox/keypoint mapping (works after preprocess_images()).
    • Heavier augmentation: flip, rotation ±45°, scale 0.6–1.4, translate.
    """

    KEYPOINT_NAMES = [
        "mouth", "eye", "pectoral", "pelvic", "dorsal",
        "tailroot", "tailtop", "tailcenter", "tailbottom",
    ]

    def __init__(self, json_path, img_dir,
                 input_size=(384, 288),
                 heatmap_size=(96, 72),
                 sigma=3.0,
                 augment=False):
        self.img_dir = Path(img_dir)
        self.input_h, self.input_w     = input_size
        self.heatmap_h, self.heatmap_w = heatmap_size
        self.sigma   = sigma
        self.augment = augment
        self._img_cache: dict = {}

        # Pre-computed coordinate grids for vectorised heatmap generation
        gy, gx = np.mgrid[0:self.heatmap_h, 0:self.heatmap_w]
        self._gx = gx.astype(np.float32)
        self._gy = gy.astype(np.float32)

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
                "bbox":      ann["bbox"],
                "keypoints": ann["keypoints"],
                "area":      ann.get("area", 1),
            })

        print(f"  → {len(self.samples)} annotated fish from {json_path}")

    def __len__(self):
        return len(self.samples)

    #       Vectorised Gaussian heatmap                                                                            ─
    def _make_heatmap(self, kp_x, kp_y, vis):
        K  = cfg.NUM_KEYPOINTS
        hm = np.zeros((K, self.heatmap_h, self.heatmap_w), dtype=np.float32)
        inv2s2 = 1.0 / (2.0 * self.sigma ** 2)
        for k in range(K):
            if vis[k] == 0:
                continue
            cx = kp_x[k] * self.heatmap_w
            cy = kp_y[k] * self.heatmap_h
            hm[k] = np.exp(-((self._gx - cx) ** 2 +
                             (self._gy - cy) ** 2) * inv2s2)
        return hm

    #       In-memory image loader                                                                                           
    def _load(self, fname):
        if fname not in self._img_cache:
            self._img_cache[fname] = (
                Image.open(self.img_dir / fname).convert("RGB"))
        return self._img_cache[fname]

    #       Augmentation helpers                                                                                                
    @staticmethod
    def _flip_kps(kp_x, kp_y, vis):
        kp_x = 1.0 - kp_x.copy()
        for i, j in FLIP_PAIRS:
            kp_x[[i, j]] = kp_x[[j, i]]
            kp_y[[i, j]] = kp_y[[j, i]]
            vis[[i, j]]  = vis[[j, i]]
        return kp_x, kp_y, vis

    @staticmethod
    def _rotate_kps(kp_x, kp_y, vis, angle_deg, cw, ch):
        cx, cy   = cw / 2, ch / 2
        rad      = math.radians(-angle_deg)
        ca, sa   = math.cos(rad), math.sin(rad)
        nx, ny   = kp_x.copy(), kp_y.copy()
        for k in range(len(kp_x)):
            if vis[k] == 0:
                continue
            px = kp_x[k] * cw - cx
            py = kp_y[k] * ch - cy
            rx = ca * px - sa * py + cx
            ry = sa * px + ca * py + cy
            if rx < 0 or ry < 0 or rx >= cw or ry >= ch:
                vis[k] = 0
            nx[k] = rx / cw
            ny[k] = ry / ch
        return nx, ny, vis

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = self._load(s["file_name"])
        W0, H0 = img.size

        #       Scale JSON coords → actual (possibly resized) image      
        sx = W0 / s["img_w"]
        sy = H0 / s["img_h"]

        bx_r, by_r, bw_r, bh_r = s["bbox"]
        bx_r *= sx;  by_r *= sy
        bw_r *= sx;  bh_r *= sy

        #       Crop with 10 % padding, clamped to image bounds               
        pad = 0.10
        bx  = max(0.0,       bx_r - bw_r * pad)
        by  = max(0.0,       by_r - bh_r * pad)
        x2  = min(float(W0), bx_r + bw_r * (1.0 + pad))
        y2  = min(float(H0), by_r + bh_r * (1.0 + pad))
        bw  = max(1.0, x2 - bx)
        bh  = max(1.0, y2 - by)

        img_crop = img.crop((bx, by, bx + bw, by + bh))
        cw, ch   = img_crop.size

        #       Keypoints → crop-relative [0, 1]                                                   
        kps  = s["keypoints"]
        K    = cfg.NUM_KEYPOINTS
        kp_x = np.array([(kps[3*k]   * sx - bx) / bw for k in range(K)],
                         dtype=np.float32)
        kp_y = np.array([(kps[3*k+1] * sy - by) / bh for k in range(K)],
                         dtype=np.float32)
        vis  = np.array([ kps[3*k+2]               for k in range(K)],
                         dtype=np.float32)

        #       Augmentation                                                                                                          
        if self.augment:
            # 1. Horizontal flip
            if torch.rand(1).item() < cfg.FLIP_PROB:
                img_crop = TF.hflip(img_crop)
                kp_x, kp_y, vis = self._flip_kps(kp_x, kp_y, vis)

            # 2. Random rotation ± ROT_DEGREES
            if cfg.ROT_DEGREES > 0:
                angle    = (torch.rand(1).item() * 2 - 1) * cfg.ROT_DEGREES
                img_crop = TF.rotate(img_crop, angle,
                                     interpolation=TF.InterpolationMode.BILINEAR)
                kp_x, kp_y, vis = self._rotate_kps(
                    kp_x, kp_y, vis, angle, cw, ch)

            # 3. Random scale
            scale    = cfg.SCALE_RANGE[0] + torch.rand(1).item() * (
                       cfg.SCALE_RANGE[1] - cfg.SCALE_RANGE[0])
            new_cw   = max(1, int(cw * scale))
            new_ch   = max(1, int(ch * scale))
            img_crop = TF.resize(img_crop, (new_ch, new_cw))

            # 4. Random translate (shift crop window)
            if cfg.TRANSLATE > 0:
                tx = (torch.rand(1).item() * 2 - 1) * cfg.TRANSLATE
                ty = (torch.rand(1).item() * 2 - 1) * cfg.TRANSLATE
                img_crop = TF.affine(img_crop, angle=0,
                                     translate=(int(tx * new_cw), int(ty * new_ch)),
                                     scale=1.0, shear=0)
                kp_x = np.clip(kp_x - tx, 0, 1)
                kp_y = np.clip(kp_y - ty, 0, 1)

            # 5. Color jitter
            img_crop = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08
            )(img_crop)

        #       Resize → model input                                                                                      
        img_t = TF.resize(img_crop, (self.input_h, self.input_w))
        img_t = TF.to_tensor(img_t)
        img_t = TF.normalize(img_t,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

        heatmaps = self._make_heatmap(kp_x, kp_y, vis)
        return img_t, torch.from_numpy(heatmaps), torch.from_numpy(vis)



#  LOSS


class WeightedHeatmapLoss(nn.Module):
    """Per-keypoint weighted MSE. Skips keypoints with visibility == 0."""

    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights",
                             torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, target, vis):
        K, loss, n = pred.shape[1], torch.zeros(1, device=pred.device,
                                                 dtype=pred.dtype), 0
        for k in range(K):
            mask = (vis[:, k] > 0).float()
            if mask.sum() == 0:
                continue
            diff = ((pred[:, k] - target[:, k]) ** 2).mean(dim=[1, 2])
            loss = loss + self.weights[k] * (diff * mask).sum() / mask.sum()
            n   += 1
        return loss / max(n, 1)


#  DARK DECODING


def dark_decode(heatmaps, input_size=(384, 288)):
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
        dxy = 0.25*(hm[my+1,mx+1] - hm[my+1,mx-1]
                    - hm[my-1,mx+1] + hm[my-1,mx-1])
        det = dxx*dyy - dxy**2
        if abs(det) < 1e-6:
            coords[k] = [mx*input_size[1]/Wm, my*input_size[0]/Hm]
            continue
        H_inv  = np.array([[dyy, -dxy], [-dxy, dxx]]) / det
        offset = -H_inv @ np.array([dx, dy])
        coords[k, 0] = (mx + offset[0]) * input_size[1] / Wm
        coords[k, 1] = (my + offset[1]) * input_size[0] / Hm
    return coords


#  OKS
# Object Keypoint Similarity (OKS) is a common metric for keypoint detection

#              mouth  eye   pect  pelv  dors  tailrt tailtop tailctr tailbot
SIGMA = np.array([0.060, 0.060, 0.087, 0.087, 0.087,
                   0.089, 0.107, 0.089, 0.107], dtype=np.float64)


def compute_oks(pred, gt, vis, obj_scale):
    s, n = 0.0, 0
    for k in range(len(pred)):
        if vis[k] == 0:
            continue
        d2 = (pred[k,0]-gt[k,0])**2 + (pred[k,1]-gt[k,1])**2
        s2 = (obj_scale * SIGMA[k])**2
        s += math.exp(-d2 / (2*s2 + 1e-9))
        n += 1
    return s / max(n, 1)


#  EVALUATION

def evaluate_oks(model, loader, device):
    model.eval()
    use_amp     = cfg.USE_AMP and device == "cuda"
    all_oks     = []
    per_kp_dist = [[] for _ in range(cfg.NUM_KEYPOINTS)]

    with torch.no_grad():
        for imgs, hm_gt, vis in loader:
            imgs = imgs.to(device)
            with autocast("cuda", enabled=use_amp):
                preds = model(imgs)
            preds = torch.sigmoid(preds).cpu().float()

            for b in range(imgs.shape[0]):
                hm_p = preds[b].numpy()
                hm_g = hm_gt[b].numpy()
                v    = vis[b].numpy()
                pc   = dark_decode(hm_p, (cfg.HEATMAP_H, cfg.HEATMAP_W))

                K, Hm, Wm = hm_g.shape
                gc = np.zeros((K, 2), dtype=np.float32)
                for k in range(K):
                    if v[k] == 0:
                        continue
                    idx = np.argmax(hm_g[k])
                    gy, gx = np.unravel_index(idx, (Hm, Wm))
                    gc[k] = [gx, gy]

                oks = compute_oks(pc, gc, v, math.sqrt(Hm*Wm)*0.25)
                all_oks.append(oks)
                for k in range(K):
                    if v[k] > 0:
                        per_kp_dist[k].append(math.sqrt(
                            (pc[k,0]-gc[k,0])**2 + (pc[k,1]-gc[k,1])**2))

    mean_oks = float(np.mean(all_oks)) if all_oks else 0.0
    dists    = [float(np.mean(d)) if d else float("nan") for d in per_kp_dist]
    return mean_oks, dists




def make_loader(ds, shuffle, drop_last=False):
    return DataLoader(
        ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(cfg.DEVICE == "cuda"),
        drop_last=drop_last,
        persistent_workers=(cfg.NUM_WORKERS > 0),
        prefetch_factor=2 if cfg.NUM_WORKERS > 0 else None,
    )




def build_optimizer(model, stage: int):
    """
    Stage 1 — backbone frozen: only head params, lr = HEAD_LR.
    Stage 2 — full model: backbone lr = BACKBONE_LR, head lr = HEAD_LR.
    """
    if stage == 1:
        params = [p for p in model.head.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=cfg.HEAD_LR,
                           weight_decay=cfg.WEIGHT_DECAY)
    else:
        return optim.AdamW([
            {"params": model.backbone.parameters(), "lr": cfg.BACKBONE_LR},
            {"params": model.head.parameters(),     "lr": cfg.HEAD_LR},
        ], weight_decay=cfg.WEIGHT_DECAY)


#  TRAINING

def train(resume_ckpt=None):
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    #       Pre-process images once                                                                                      ─
    if cfg.PREPROCESS_MAX_SIZE:
        print(f"\nPre-processing images (max {cfg.PREPROCESS_MAX_SIZE} px) ...")
        preprocess_images([cfg.TRAIN_IMG, cfg.VAL_IMG],
                          max_size=cfg.PREPROCESS_MAX_SIZE)

    #       Datasets & loaders                                                                                                     
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

    train_loader = make_loader(train_ds, shuffle=True,  drop_last=True)
    val_loader   = make_loader(val_ds,   shuffle=False, drop_last=False)

    #       Model                                                                                                                                   
    print(f"\nLoading {cfg.BACKBONE} "
          f"({'pretrained' if cfg.PRETRAINED else 'random'}) ...")
    model     = HRNetKeypointModel(cfg.NUM_KEYPOINTS, cfg.PRETRAINED).to(cfg.DEVICE)
    criterion = WeightedHeatmapLoss(cfg.KEYPOINT_WEIGHTS).to(cfg.DEVICE)
    use_amp   = cfg.USE_AMP and cfg.DEVICE == "cuda"
    scaler    = GradScaler("cuda", enabled=use_amp)

    total_params    = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params     = sum(p.numel() for p in model.head.parameters())
    print(f"Parameters: total={total_params:,}  "
          f"backbone={backbone_params:,}  head={head_params:,}")

    #       Resume                                                                                                                                   
    start_epoch   = 1
    best_val_loss = float("inf")
    best_oks      = 0.0
    current_stage = 1

    if resume_ckpt and Path(resume_ckpt).exists():
        ckpt = torch.load(resume_ckpt, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_oks      = ckpt.get("best_oks", 0.0)
        current_stage = 2 if start_epoch > cfg.FREEZE_EPOCHS else 1
        print(f"  Resumed from epoch {start_epoch-1} (stage {current_stage})")

    #       Stage 1 setup                                                                                                               ─
    if current_stage == 1 and start_epoch <= cfg.FREEZE_EPOCHS:
        model.freeze_backbone()
        optimizer = build_optimizer(model, stage=1)
    else:
        current_stage = 2
        model.unfreeze_backbone()
        optimizer = build_optimizer(model, stage=2)

    if resume_ckpt and Path(resume_ckpt).exists() and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            pass  

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, m - start_epoch + 1) for m in cfg.LR_STEP],
        gamma=cfg.LR_GAMMA,
    )

    #       Training loop                                                                                                               ─
    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        #       Stage transition: unfreeze backbone after FREEZE_EPOCHS ─
        if current_stage == 1 and epoch > cfg.FREEZE_EPOCHS:
            current_stage = 2
            model.unfreeze_backbone()
            optimizer = build_optimizer(model, stage=2)
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[max(1, m - epoch + 1) for m in cfg.LR_STEP],
                gamma=cfg.LR_GAMMA,
            )
            print(f"  [Epoch {epoch}] Switched to Stage 2 (full fine-tuning)")

        #       Train                                                                                                                         ─
        model.train()
        train_loss = 0.0
        for imgs, hms, vis in train_loader:
            imgs = imgs.to(cfg.DEVICE, non_blocking=True)
            hms  = hms.to(cfg.DEVICE,  non_blocking=True)
            vis  = vis.to(cfg.DEVICE,  non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                preds = model(imgs)
                loss  = criterion(preds, hms, vis)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        #       Validate                                                                                                                    
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, hms, vis in val_loader:
                imgs = imgs.to(cfg.DEVICE, non_blocking=True)
                hms  = hms.to(cfg.DEVICE,  non_blocking=True)
                vis  = vis.to(cfg.DEVICE,  non_blocking=True)
                with autocast("cuda", enabled=use_amp):
                    preds = model(imgs)
                val_loss += criterion(preds, hms, vis).item()
        val_loss /= len(val_loader)

        scheduler.step()
        elapsed = time.time() - t0

        #       OKS every 5 epochs                                                                                           
        oks_str = ""
        if epoch % 5 == 0:
            mean_oks, per_kp = evaluate_oks(model, val_loader, cfg.DEVICE)
            oks_str = f"  OKS: {mean_oks:.4f}"
            if mean_oks > best_oks:
                best_oks = mean_oks
                torch.save(model.state_dict(),
                           f"{cfg.SAVE_DIR}/best_oks_model.pth")
                oks_str += " ✔"
            kp_fmt = "  ".join(
                f"{FishKeypointDataset.KEYPOINT_NAMES[k]}:{per_kp[k]:.1f}px"
                for k in range(cfg.NUM_KEYPOINTS))
            print(f"   KP dists → {kp_fmt}")

        stage_tag = f"[S{current_stage}]"
        print(f"Epoch [{epoch:3d}/{cfg.NUM_EPOCHS}] {stage_tag}  "
              f"train: {train_loss:.4f}  val: {val_loss:.4f}  "
              f"lr_head: {optimizer.param_groups[-1]['lr']:.2e}  "
              f"time: {elapsed:.1f}s{oks_str}")

        #       Checkpoint                                                                                                               
        if epoch % cfg.SAVE_EVERY == 0:
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict":   scaler.state_dict(),
                "val_loss":            val_loss,
                "best_val_loss":       best_val_loss,
                "best_oks":            best_oks,
            }, f"{cfg.SAVE_DIR}/ckpt_epoch{epoch:03d}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       f"{cfg.SAVE_DIR}/best_loss_model.pth")
            print(f"  Best loss model saved (val={best_val_loss:.4f})")

    print(f"\nTraining complete.  Best val loss: {best_val_loss:.4f}  "
          f"Best OKS: {best_oks:.4f}")




def test(checkpoint_path):
    test_ds = FishKeypointDataset(
        cfg.TEST_JSON, cfg.TEST_IMG,
        input_size=(cfg.INPUT_H, cfg.INPUT_W),
        heatmap_size=(cfg.HEATMAP_H, cfg.HEATMAP_W),
        sigma=cfg.HEATMAP_SIGMA, augment=False)
    loader = make_loader(test_ds, shuffle=False)

    model = HRNetKeypointModel(cfg.NUM_KEYPOINTS, pretrained=False).to(cfg.DEVICE)
    state = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    print(f"\nLoaded: {checkpoint_path}")

    mean_oks, per_kp = evaluate_oks(model, loader, cfg.DEVICE)
    print(f"\n{'='*55}")
    print(f"  Test OKS : {mean_oks:.4f}")
    print(f"{'='*55}")
    for k, name in enumerate(FishKeypointDataset.KEYPOINT_NAMES):
        print(f"    {name:>14s}:  {per_kp[k]:.2f} px")
    print(f"{'='*55}\n")



def predict_keypoints(model, crop_pil, use_dark=True):
    """Single-image inference. Returns coords [K,2] and heatmaps [K,Hm,Wm]."""
    model.eval()
    use_amp = cfg.USE_AMP and cfg.DEVICE == "cuda"
    img = TF.normalize(TF.to_tensor(
        TF.resize(crop_pil, (cfg.INPUT_H, cfg.INPUT_W))),
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = img.unsqueeze(0).to(cfg.DEVICE)
    with torch.no_grad():
        with autocast("cuda", enabled=use_amp):
            hm = model(img)
    hm_np = torch.sigmoid(hm)[0].cpu().float().numpy()
    coords = (dark_decode(hm_np, (cfg.INPUT_H, cfg.INPUT_W))
              if use_dark else _argmax_decode(hm_np))
    return coords, hm_np


def _argmax_decode(hm_np):
    K, Hm, Wm = hm_np.shape
    coords = np.zeros((K, 2), dtype=np.float32)
    for k in range(K):
        flat   = np.argmax(hm_np[k])
        ry, rx = np.unravel_index(flat, (Hm, Wm))
        coords[k] = [rx * cfg.INPUT_W / Wm, ry * cfg.INPUT_H / Hm]
    return coords



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HRNet-W32 pretrained fish keypoint detection — FIB dataset")

    parser.add_argument("--mode",          choices=["train", "test"], default="train")
    parser.add_argument("--train_json",    default=cfg.TRAIN_JSON)
    parser.add_argument("--val_json",      default=cfg.VAL_JSON)
    parser.add_argument("--test_json",     default=cfg.TEST_JSON)
    parser.add_argument("--train_img",     default=cfg.TRAIN_IMG)
    parser.add_argument("--val_img",       default=cfg.VAL_IMG)
    parser.add_argument("--test_img",      default=cfg.TEST_IMG)
    parser.add_argument("--epochs",        type=int,   default=cfg.NUM_EPOCHS)
    parser.add_argument("--batch_size",    type=int,   default=cfg.BATCH_SIZE)
    parser.add_argument("--head_lr",       type=float, default=cfg.HEAD_LR)
    parser.add_argument("--backbone_lr",   type=float, default=cfg.BACKBONE_LR)
    parser.add_argument("--freeze_epochs", type=int,   default=cfg.FREEZE_EPOCHS)
    parser.add_argument("--num_workers",   type=int,   default=cfg.NUM_WORKERS)
    parser.add_argument("--save_dir",      default=cfg.SAVE_DIR)
    parser.add_argument("--checkpoint",    default=None,
                        help="Resume training from / evaluate this checkpoint")
    parser.add_argument("--no_amp",        action="store_true")
    parser.add_argument("--no_preprocess", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Train from scratch (disables ImageNet weights)")

    args = parser.parse_args()

    cfg.TRAIN_JSON    = args.train_json
    cfg.VAL_JSON      = args.val_json
    cfg.TEST_JSON     = args.test_json
    cfg.TRAIN_IMG     = args.train_img
    cfg.VAL_IMG       = args.val_img
    cfg.TEST_IMG      = args.test_img
    cfg.NUM_EPOCHS    = args.epochs
    cfg.BATCH_SIZE    = args.batch_size
    cfg.HEAD_LR       = args.head_lr
    cfg.BACKBONE_LR   = args.backbone_lr
    cfg.FREEZE_EPOCHS = args.freeze_epochs
    cfg.NUM_WORKERS   = args.num_workers
    cfg.SAVE_DIR      = args.save_dir
    cfg.USE_AMP       = not args.no_amp
    cfg.PRETRAINED    = not args.no_pretrained
    if args.no_preprocess:
        cfg.PREPROCESS_MAX_SIZE = None

    print(f"Device        : {cfg.DEVICE}")
    print(f"Backbone      : {cfg.BACKBONE} "
          f"({'pretrained' if cfg.PRETRAINED else 'random'})")
    print(f"AMP           : {cfg.USE_AMP}")
    print(f"Freeze epochs : {cfg.FREEZE_EPOCHS}")
    print(f"Epochs        : {cfg.NUM_EPOCHS}")
    print(f"Batch size    : {cfg.BATCH_SIZE}")
    print(f"Sigma         : {cfg.HEATMAP_SIGMA}")
    print(f"Mode          : {args.mode}")

    if args.mode == "train":
        train(resume_ckpt=args.checkpoint)
    else:
        ckpt = args.checkpoint or f"{cfg.SAVE_DIR}/best_oks_model.pth"
        test(ckpt)