"""
train_v2_clean.py
UteroScan Phase 3 (V2 Contamination-Cleaned Baseline)
Retrains the two-stage ResNet50 pipeline on the clean 1,545-image dataset:
- Train Split: 980 scans (526 UF, 454 NUF)
- Val Split  : 244 scans (131 UF, 113 NUF)
- Held-Out   : 321 scans (163 UF, 158 NUF)

Checkpoints saved EXCLUSIVELY to:
  experiments/v2_contamination_cleaned/checkpoints/
Zero modification to the original checkpoints/ directory.
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

# Trapezoid integration for AUC (NumPy 2.x compatible)
try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

BASE_DIR = Path("d:/Project/UteroScan")
COMPANION_DIR = BASE_DIR / "DCGAN&MobileNetV2" / "DCGAN&MobileNetV2"
sys.path.insert(0, str(COMPANION_DIR))

from nets import get_model_from_name
from utils.dataloader import DataGenerator, detection_collate

EXP_DIR = BASE_DIR / "experiments" / "v2_contamination_cleaned"
ANNOTATIONS_DIR = EXP_DIR / "annotations"
CHECKPOINTS_DIR = EXP_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

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

def build_model(device):
    print("\n" + "=" * 75)
    print("INITIALIZING RESNET50 ARCHITECTURE (V2 CLEAN EXPERIMENT)")
    print("=" * 75)
    
    arch_func = get_model_from_name.get("resnet50")
    assert arch_func is not None, "resnet50 not found in nets registry!"
    
    model = arch_func(num_classes=2, pretrained=False)
    arch_name = model.__class__.__name__
    print(f"Loaded Backbone Architecture: {arch_name} (from nets/resnet.py)")
    assert arch_name == "ResNet", f"CRITICAL: Expected ResNet, got {arch_name}!"
    assert not hasattr(model, "last_channel"), "CRITICAL: Detected MobileNetV2 attributes!"
    
    print("Loading ImageNet pretrained weights into backbone...")
    tv_res = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    bb_state = {k: v for k, v in tv_res.state_dict().items() if not k.startswith("fc.")}
    model.load_state_dict(bb_state, strict=False)
    print("ImageNet pretrained weights successfully loaded into backbone.")

    model = model.to(device)
    return model

def configure_stage1_freezing(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Stage 1 Frozen State: Only fc is trainable.")
    print(f"  Trainable Parameters: {trainable_count:,} / {total_count:,} ({trainable_count/total_count*100:.2f}%)")
    assert trainable_count == 2048 * 2 + 2, f"Unexpected Stage 1 parameter count: {trainable_count}"

def configure_stage2_unfreezing(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Stage 2 Unfrozen State: layer4 + fc are trainable.")
    print(f"  Trainable Parameters: {trainable_count:,} / {total_count:,} ({trainable_count/total_count*100:.2f}%)")
    assert trainable_count > 10_000_000, f"Stage 2 parameter count too low: {trainable_count}"

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch_desc=""):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        images, targets = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == targets.data).item()
        total += images.size(0)

        if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == num_batches:
            batch_loss = running_loss / total
            batch_acc = correct / total
            print(f"\r  {epoch_desc} Batch {batch_idx+1:02d}/{num_batches:02d} | "
                  f"Loss: {batch_loss:.4f} | Acc: {batch_acc*100:5.1f}%", end="", flush=True)

    print()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate_loader(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    all_targets = []
    all_preds = []
    all_probs_nuf = []

    with torch.no_grad():
        for batch in dataloader:
            images, targets = batch
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

    val_loss = running_loss / total if total > 0 else 0.0
    metrics = compute_metrics(all_targets, all_preds, all_probs_nuf)
    metrics["val_loss"] = float(val_loss)
    return metrics

def run_v2_training(device, stage1_epochs=15, stage2_epochs=10, batch_size=16):
    print("\n" + "=" * 75)
    print("STARTING V2 CLEAN RESNET50 TWO-STAGE TRAINING")
    print("=" * 75)
    print(f"Target Checkpoints Directory: {CHECKPOINTS_DIR}")

    train_split_file = ANNOTATIONS_DIR / "cls_train_split.txt"
    val_split_file = ANNOTATIONS_DIR / "cls_val_split.txt"

    with open(train_split_file, "r", encoding="utf-8") as f:
        train_lines = [l.strip() for l in f if l.strip()]
    with open(val_split_file, "r", encoding="utf-8") as f:
        val_lines = [l.strip() for l in f if l.strip()]

    print(f"\nLoaded Datasets (V2 Clean 3-Way Split):")
    print(f"  Internal Train: {len(train_lines)} scans (526 UF, 454 NUF)")
    print(f"  Internal Val  : {len(val_lines)} scans (131 UF, 113 NUF)")
    print(f"  Held-out Test : 321 scans (Untouched until Phase 4 Evaluation)", flush=True)

    train_ds = DataGenerator(train_lines, input_shape=(224, 224), random=True)
    val_ds = DataGenerator(val_lines, input_shape=(224, 224), random=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=detection_collate, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=detection_collate, num_workers=0, pin_memory=True)

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()

    history = {
        "stage1": [],
        "stage2": [],
        "best_stage1_metric": None,
        "best_stage2_metric": None
    }

    # =========================================================================
    # STAGE 1: Frozen Backbone Head Training
    # =========================================================================
    print("\n" + "=" * 75)
    print("STAGE 1: TRAINING HEAD WITH FROZEN BACKBONE (15 EPOCHS)")
    print("=" * 75, flush=True)

    configure_stage1_freezing(model)
    optimizer_stage1 = optim.Adam(model.fc.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)

    best_s1_auc = -1.0
    best_s1_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(1, stage1_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage1, device,
                                                epoch_desc=f"[Stage 1 {epoch:02d}/{stage1_epochs:02d}]")
        val_metrics = evaluate_loader(model, val_loader, criterion, device)
        dur = time.time() - t0

        is_best = val_metrics["auc"] > best_s1_auc or (val_metrics["auc"] == best_s1_auc and val_metrics["val_loss"] < best_s1_loss)
        
        # Early stopping tracker
        if val_metrics["val_loss"] < best_s1_loss:
            patience_counter = 0
        else:
            patience_counter += 1

        if is_best:
            best_s1_auc = val_metrics["auc"]
            best_s1_loss = val_metrics["val_loss"]
            ckpt_path = CHECKPOINTS_DIR / "resnet50_stage1_best.pth"
            torch.save({
                "epoch": epoch,
                "stage": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_stage1.state_dict(),
                "metrics": val_metrics,
                "train_loss": train_loss,
                "train_acc": train_acc
            }, str(ckpt_path))
            star = " [*BEST*]"
        else:
            star = ""

        print(f"  [Stage 1 Epoch {epoch:02d}/{stage1_epochs:02d}] ({dur:5.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.1f}% | "
              f"Val Loss: {val_metrics['val_loss']:.4f} Acc: {val_metrics['accuracy']*100:5.1f}% | "
              f"AUC: {val_metrics['auc']:.4f} Sens: {val_metrics['uf_sensitivity']*100:5.1f}% Spec: {val_metrics['nuf_specificity']*100:5.1f}%{star}", flush=True)

        history["stage1"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "duration": dur,
            **val_metrics
        })

        if patience_counter >= patience:
            print(f"\n[Stage 1 Early Stopping] Val loss did not improve for {patience} consecutive epochs. Proceeding to Stage 2.", flush=True)
            break

    history["best_stage1_metric"] = {"best_val_auc": best_s1_auc, "best_val_loss": best_s1_loss}
    print(f"\nStage 1 Complete. Best Val AUC = {best_s1_auc:.4f} (Saved to {CHECKPOINTS_DIR / 'resnet50_stage1_best.pth'})", flush=True)

    # Load best stage 1 weights before starting stage 2
    best_s1_ckpt = torch.load(str(CHECKPOINTS_DIR / "resnet50_stage1_best.pth"), map_location=device)
    model.load_state_dict(best_s1_ckpt["model_state_dict"])

    # =========================================================================
    # STAGE 2: Fine-Tuning layer4 + Head at lr=1e-5
    # =========================================================================
    print("\n" + "=" * 75)
    print("STAGE 2: FINE-TUNING (layer4 + Head Unfrozen at lr=1e-5, 10 EPOCHS)")
    print("=" * 75, flush=True)

    configure_stage2_unfreezing(model)
    stage2_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_stage2 = optim.Adam(stage2_params, lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-4)

    best_s2_auc = best_s1_auc
    best_s2_loss = best_s1_loss

    for epoch in range(1, stage2_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage2, device,
                                                epoch_desc=f"[Stage 2 {epoch:02d}/{stage2_epochs:02d}]")
        val_metrics = evaluate_loader(model, val_loader, criterion, device)
        dur = time.time() - t0

        is_best = val_metrics["auc"] > best_s2_auc or (val_metrics["auc"] == best_s2_auc and val_metrics["val_loss"] < best_s2_loss)
        if is_best:
            best_s2_auc = val_metrics["auc"]
            best_s2_loss = val_metrics["val_loss"]
            ckpt_path = CHECKPOINTS_DIR / "resnet50_stage2_best.pth"
            torch.save({
                "epoch": epoch,
                "stage": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_stage2.state_dict(),
                "metrics": val_metrics,
                "train_loss": train_loss,
                "train_acc": train_acc
            }, str(ckpt_path))
            star = " [*BEST*]"
        else:
            star = ""

        print(f"  [Stage 2 Epoch {epoch:02d}/{stage2_epochs:02d}] ({dur:5.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.1f}% | "
              f"Val Loss: {val_metrics['val_loss']:.4f} Acc: {val_metrics['accuracy']*100:5.1f}% | "
              f"AUC: {val_metrics['auc']:.4f} Sens: {val_metrics['uf_sensitivity']*100:5.1f}% Spec: {val_metrics['nuf_specificity']*100:5.1f}%{star}", flush=True)

        history["stage2"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "duration": dur,
            **val_metrics
        })

    history["best_stage2_metric"] = {"best_val_auc": best_s2_auc, "best_val_loss": best_s2_loss}

    # Save final model
    final_path = CHECKPOINTS_DIR / "resnet50_final.pth"
    torch.save({
        "stage": 2,
        "model_state_dict": model.state_dict(),
        "final_metrics": val_metrics
    }, str(final_path))
    print(f"\nSaved final model weights to {final_path}")

    # Save complete history JSON
    history_path = CHECKPOINTS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved complete training history to {history_path}", flush=True)

    return history

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")
    
    run_v2_training(device, stage1_epochs=15, stage2_epochs=10, batch_size=16)

if __name__ == "__main__":
    main()
