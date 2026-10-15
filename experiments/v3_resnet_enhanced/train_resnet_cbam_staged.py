"""
train_resnet_cbam_staged.py
Progressive Staged Fine-Tuning of ResNet50-CBAM on UteroScan Clean Cohort

Staged Fine-Tuning Schedule:
- Stage 1: Frozen backbone, train FC head (lr=1e-4) for 6 epochs.
- Stage 2: Unfreeze layer4 + FC (layer4 lr=1e-5, FC lr=1e-4) for 8 epochs.
- Stage 3: Unfreeze layer3 + layer4 + FC (layer3 lr=3e-6, layer4 lr=1e-5, FC lr=1e-4) with Cosine Decay for 12 epochs.

Input Resolution: 256x256
Batch size: 16
Preprocessing: Ultrasound Viewport Cropper + Caliper Invariance Augmentation (p=0.4)
"""

import sys
import os
import time
import json
import random
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Trapezoid integration for AUC
try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

BASE_DIR = Path("d:/Project/UteroScan")
EXP_DIR = BASE_DIR / "experiments" / "v3_resnet_enhanced"
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(BASE_DIR))

from models.resnet50_cbam import ResNet50_CBAM
from preprocessing.enhanced_dataloader import EnhancedUltrasoundDataset, custom_collate

ANNOTATIONS_DIR = BASE_DIR / "experiments" / "v2_contamination_cleaned" / "annotations"
CHECKPOINTS_DIR = EXP_DIR / "checkpoints"
REPORTS_DIR = EXP_DIR / "reports"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_roc_auc(y_true, y_scores):
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    desc_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_indices]
    y_scores_sorted = y_scores[desc_indices]

    distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_true_sorted == 0)[threshold_idxs]

    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]

    auc = float(trapezoid(tpr, fpr))
    return max(0.0, min(1.0, auc))

def compute_metrics(y_true, y_pred, y_probs):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)

    # Class 0: UF, Class 1: NUF
    tp_uf = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn_uf = int(np.sum((y_true == 0) & (y_pred == 1)))
    fp_uf = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn_uf = int(np.sum((y_true == 1) & (y_pred == 1)))

    total = len(y_true)
    accuracy = float((tp_uf + tn_uf) / total) if total > 0 else 0.0
    sensitivity = float(tp_uf / (tp_uf + fn_uf)) if (tp_uf + fn_uf) > 0 else 0.0
    specificity = float(tn_uf / (tn_uf + fp_uf)) if (tn_uf + fp_uf) > 0 else 0.0
    precision = float(tp_uf / (tp_uf + fp_uf)) if (tp_uf + fp_uf) > 0 else 0.0
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0

    auc = compute_roc_auc(y_true, y_probs)

    return {
        "accuracy": accuracy,
        "auc": auc,
        "uf_sensitivity": sensitivity,
        "nuf_specificity": specificity,
        "uf_precision": precision,
        "uf_f1": f1,
        "confusion_matrix": [[tp_uf, fn_uf], [fp_uf, tn_uf]]
    }

def set_stage1_parameters(model):
    """Stage 1: Only FC layer trainable."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

def set_stage2_parameters(model):
    """Stage 2: layer4 + FC trainable."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

def set_stage3_parameters(model):
    """Stage 3: layer3 + layer4 + FC trainable."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer3.parameters():
        p.requires_grad = True
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

def train_one_epoch(model, loader, criterion, optimizer, device, desc=""):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == targets).item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate_loader(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    all_targets = []
    all_preds = []
    all_probs_nuf = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs_nuf.extend(probs[:, 1].cpu().numpy().tolist())

    metrics = compute_metrics(all_targets, all_preds, all_probs_nuf)
    metrics["val_loss"] = float(running_loss / total) if total > 0 else 0.0
    return metrics, np.array(all_probs_nuf), np.array(all_targets)

def train_staged_resnet_cbam(
    device,
    stage1_epochs=6,
    stage2_epochs=8,
    stage3_epochs=12,
    batch_size=16,
    res=(256, 256),
    save_prefix="resnet50_cbam_single"
):
    set_seed(SEED)
    print("=" * 80)
    print(f"STARTING RESNET50-CBAM PROGRESSIVE STAGED TRAINING ({res[0]}x{res[1]})")
    print("=" * 80)

    train_file = ANNOTATIONS_DIR / "cls_train_split.txt"
    val_file = ANNOTATIONS_DIR / "cls_val_split.txt"

    with open(train_file, "r") as f:
        train_lines = [l.strip() for l in f if l.strip()]
    with open(val_file, "r") as f:
        val_lines = [l.strip() for l in f if l.strip()]

    print(f"Loaded: Train={len(train_lines)} scans, Val={len(val_lines)} scans")

    train_ds = EnhancedUltrasoundDataset(train_lines, input_shape=res, is_train=True, use_viewport=True, caliper_aug_p=0.4)
    val_ds = EnhancedUltrasoundDataset(val_lines, input_shape=res, is_train=False, use_viewport=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=0, pin_memory=True)

    model = ResNet50_CBAM(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_auc = 0.0
    best_ckpt_path = CHECKPOINTS_DIR / f"{save_prefix}_best.pth"

    # -------------------------------------------------------------
    # STAGE 1: Train FC Head Only
    # -------------------------------------------------------------
    print("\n--- STAGE 1: FC Head Training (lr=1e-4) ---")
    set_stage1_parameters(model)
    opt_s1 = optim.Adam(model.fc.parameters(), lr=1e-4, weight_decay=1e-4)

    for ep in range(1, stage1_epochs + 1):
        t0 = time.time()
        loss, acc = train_one_epoch(model, train_loader, criterion, opt_s1, device, desc=f"S1 Ep {ep}")
        val_m, _, _ = evaluate_loader(model, val_loader, criterion, device)
        dur = time.time() - t0
        print(f"Stage 1 [{ep:02d}/{stage1_epochs:02d}] ({dur:4.1f}s) | Train Loss: {loss:.4f} Acc: {acc*100:5.1f}% | "
              f"Val Loss: {val_m['val_loss']:.4f} Acc: {val_m['accuracy']*100:5.1f}% AUC: {val_m['auc']:.4f} Sens: {val_m['uf_sensitivity']*100:5.1f}%", flush=True)

        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            torch.save(model.state_dict(), best_ckpt_path)

    # -------------------------------------------------------------
    # STAGE 2: Unfreeze layer4 + FC
    # -------------------------------------------------------------
    print("\n--- STAGE 2: layer4 + FC Training (layer4 lr=1e-5, FC lr=1e-4) ---", flush=True)
    set_stage2_parameters(model)
    param_groups_s2 = [
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(), "lr": 1e-4}
    ]
    opt_s2 = optim.Adam(param_groups_s2, weight_decay=1e-4)

    for ep in range(1, stage2_epochs + 1):
        t0 = time.time()
        loss, acc = train_one_epoch(model, train_loader, criterion, opt_s2, device, desc=f"S2 Ep {ep}")
        val_m, _, _ = evaluate_loader(model, val_loader, criterion, device)
        dur = time.time() - t0
        print(f"Stage 2 [{ep:02d}/{stage2_epochs:02d}] ({dur:4.1f}s) | Train Loss: {loss:.4f} Acc: {acc*100:5.1f}% | "
              f"Val Loss: {val_m['val_loss']:.4f} Acc: {val_m['accuracy']*100:5.1f}% AUC: {val_m['auc']:.4f} Sens: {val_m['uf_sensitivity']*100:5.1f}%", flush=True)

        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            torch.save(model.state_dict(), best_ckpt_path)

    # -------------------------------------------------------------
    # STAGE 3: Unfreeze layer3 + layer4 + FC with Cosine Annealing
    # -------------------------------------------------------------
    print("\n--- STAGE 3: layer3 + layer4 + FC Training (layer3=3e-6, layer4=1e-5, FC=1e-4) ---", flush=True)
    set_stage3_parameters(model)
    param_groups_s3 = [
        {"params": model.layer3.parameters(), "lr": 3e-6},
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(), "lr": 1e-4}
    ]
    opt_s3 = optim.Adam(param_groups_s3, weight_decay=1e-4)
    sched_s3 = optim.lr_scheduler.CosineAnnealingLR(opt_s3, T_max=stage3_epochs, eta_min=1e-7)

    for ep in range(1, stage3_epochs + 1):
        t0 = time.time()
        loss, acc = train_one_epoch(model, train_loader, criterion, opt_s3, device, desc=f"S3 Ep {ep}")
        sched_s3.step()
        val_m, _, _ = evaluate_loader(model, val_loader, criterion, device)
        dur = time.time() - t0
        print(f"Stage 3 [{ep:02d}/{stage3_epochs:02d}] ({dur:4.1f}s) | Train Loss: {loss:.4f} Acc: {acc*100:5.1f}% | "
              f"Val Loss: {val_m['val_loss']:.4f} Acc: {val_m['accuracy']*100:5.1f}% AUC: {val_m['auc']:.4f} Sens: {val_m['uf_sensitivity']*100:5.1f}%", flush=True)

        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            torch.save(model.state_dict(), best_ckpt_path)

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE. Best Val AUC: {best_val_auc:.4f}")
    print(f"Best model weights saved to: {best_ckpt_path}")
    print("=" * 80, flush=True)
    return best_ckpt_path

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_staged_resnet_cbam(device, stage1_epochs=5, stage2_epochs=8, stage3_epochs=15, batch_size=16)
