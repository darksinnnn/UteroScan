"""
train_uteroscan_resnet50.py
UteroScan Phase 3 — Two-Stage ResNet50 Training Pipeline

Locked Methodology per PRD.md Section 7 & Implementation_Plan.md Phase 3:
- Architecture: ResNet50 (nets/resnet.py), ImageNet pretrained.
- Input Resolution: 224x224.
- Class Mapping: 0 = UF (Uterine Fibroid), 1 = NUF (Non-Uterine Fibroid).
- Class Weighting: Skipped as moot due to near-perfect 50/50 balance (695 UF / 694 NUF).
- Preprocessing: Phase 1 clinical transform (Grayscale -> 224x224 INTER_AREA -> 3x3 Median -> CLAHE 1.2 -> RGB replicate -> ImageNet norm).
- Augmentation: Phase 2 training transform (Rotation +-15 deg, H-Flip p=0.5, Zoom 0.9-1.1, Shear +-10 deg; NO vertical flip).
- Evaluation: Strict determinism (no augmentation on val/test).
- Stage 1: Freeze backbone, train classification head only (15 epochs, Adam lr=1e-4).
- Stage 2: Unfreeze layer4 (~30 layers) + head, fine-tune at lower lr (10 epochs, Adam lr=1e-5).
- Checkpoints: Saved to checkpoints/ (stage1_best, stage2_best, final, and training_history.json).
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

# Paths
BASE_DIR = Path("d:/Project/UteroScan")
COMPANION_DIR = BASE_DIR / "DCGAN&MobileNetV2" / "DCGAN&MobileNetV2"
sys.path.insert(0, str(COMPANION_DIR))

from nets import get_model_from_name
from utils.dataloader import DataGenerator, detection_collate

CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_roc_auc(y_true, y_scores):
    """
    Computes exact Area Under the ROC Curve (AUC) via trapezoidal integration.
    y_true: binary labels (0 = UF, 1 = NUF)
    y_scores: float predicted probabilities for class 1
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort descending by predicted score
    desc_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_indices]
    y_scores_sorted = y_scores[desc_indices]

    # Find unique score thresholds
    distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    # Cumulative counts of True Positives and False Positives
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
    tp_uf = int(np.sum((y_true == 0) & (y_pred == 0)))  # UF correctly predicted
    fn_uf = int(np.sum((y_true == 0) & (y_pred == 1)))  # UF missed (predicted NUF)
    fp_uf = int(np.sum((y_true == 1) & (y_pred == 0)))  # NUF incorrectly called UF
    tn_uf = int(np.sum((y_true == 1) & (y_pred == 1)))  # NUF correctly predicted

    total = len(y_true)
    accuracy = float((tp_uf + tn_uf) / total) if total > 0 else 0.0
    sensitivity = float(tp_uf / (tp_uf + fn_uf)) if (tp_uf + fn_uf) > 0 else 0.0  # UF Recall
    specificity = float(tn_uf / (tn_uf + fp_uf)) if (tn_uf + fp_uf) > 0 else 0.0  # NUF Recall
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
    print("INITIALIZING MODEL ARCHITECTURE")
    print("=" * 75)
    
    arch_func = get_model_from_name.get("resnet50")
    assert arch_func is not None, "resnet50 not found in nets registry!"
    
    # Instantiate ResNet50 from companion framework nets/resnet.py
    model = arch_func(num_classes=2, pretrained=False)
    
    arch_name = model.__class__.__name__
    print(f"Loaded Backbone Architecture: {arch_name} (from nets/resnet.py)")
    assert arch_name == "ResNet", f"CRITICAL: Expected ResNet, got {arch_name}!"
    assert not hasattr(model, "last_channel"), "CRITICAL: Detected MobileNetV2 attributes!"
    
    # Load ImageNet pretrained backbone weights
    print("Loading ImageNet pretrained weights into backbone...")
    tv_res = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    tv_state = tv_res.state_dict()
    pretrained_backbone = {k: v for k, v in tv_state.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(pretrained_backbone, strict=False)
    print(f"  Pretrained backbone weights successfully transferred. New head keys: {missing}")
    del tv_res, tv_state, pretrained_backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model = model.to(device)
    return model

def configure_stage1_freezing(model):
    model.freeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[Stage 1 Configuration] Backbone Frozen. Trainable Parameters: {trainable:,} / {total:,} (Head Only)")
    assert trainable == 4098, f"Expected 4098 trainable parameters in head, got {trainable}"

def configure_stage2_unfreezing(model):
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[Stage 2 Configuration] layer4 + Head Unfrozen. Trainable Parameters: {trainable:,} / {total:,}")
    return trainable

def evaluate_loader(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_total = 0

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * imgs.size(0)
            val_total += imgs.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    val_loss_avg = float(val_loss / val_total)
    metrics = compute_metrics(all_targets, all_preds, all_probs)
    metrics["val_loss"] = val_loss_avg
    return metrics

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch_desc=""):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    total_batches = len(train_loader)

    for batch_idx, (imgs, targets) in enumerate(train_loader, 1):
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        train_correct += (preds == targets).sum().item()
        train_total += imgs.size(0)

        # Real-time intra-epoch status (every 25 batches)
        if batch_idx % 25 == 0 or batch_idx == total_batches:
            cur_loss = train_loss / train_total
            cur_acc = train_correct / train_total
            print(f"    {epoch_desc} [Batch {batch_idx:02d}/{total_batches:02d}] Train Loss: {cur_loss:.4f} | Acc: {cur_acc*100:5.1f}%", flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_loss = float(train_loss / train_total)
    avg_acc = float(train_correct / train_total)
    return avg_loss, avg_acc


def run_smoke_test(device):
    print("\n" + "=" * 75)
    print("PHASE 3 SMOKE TEST (2 Epochs, ~50 img/class Train, 25 img/class Val)")
    print("=" * 75, flush=True)

    train_split_file = COMPANION_DIR / "cls_train_split.txt"
    val_split_file = COMPANION_DIR / "cls_val_split.txt"

    with open(train_split_file, "r", encoding="utf-8") as f:
        train_lines = [l.strip() for l in f if l.strip()]
    with open(val_split_file, "r", encoding="utf-8") as f:
        val_lines = [l.strip() for l in f if l.strip()]

    set_seed(SEED)
    uf_train = [l for l in train_lines if l.startswith("0;")]
    nuf_train = [l for l in train_lines if l.startswith("1;")]
    uf_val = [l for l in val_lines if l.startswith("0;")]
    nuf_val = [l for l in val_lines if l.startswith("1;")]

    random.shuffle(uf_train)
    random.shuffle(nuf_train)
    random.shuffle(uf_val)
    random.shuffle(nuf_val)

    smoke_train_lines = uf_train[:50] + nuf_train[:50]
    smoke_val_lines = uf_val[:25] + nuf_val[:25]
    random.shuffle(smoke_train_lines)

    print(f"Smoke Test Dataset Subsets:")
    print(f"  Train: {len(smoke_train_lines)} images (50 UF, 50 NUF)")
    print(f"  Val  : {len(smoke_val_lines)} images (25 UF, 25 NUF)")

    train_ds = DataGenerator(smoke_train_lines, input_shape=(224, 224), random=True)
    val_ds = DataGenerator(smoke_val_lines, input_shape=(224, 224), random=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=detection_collate)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=detection_collate)

    model = build_model(device)
    configure_stage1_freezing(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    history = []
    print("\nStarting Smoke Test Epochs (live per-epoch output):", flush=True)
    for epoch in range(1, 3):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_loader(model, val_loader, criterion, device)
        dur = time.time() - t0

        print(f"  [Smoke Test Epoch {epoch}/2] ({dur:.2f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.1f}% | "
              f"Val Loss: {val_metrics['val_loss']:.4f} Acc: {val_metrics['accuracy']*100:5.1f}% | "
              f"Val AUC: {val_metrics['auc']:.4f} Sens: {val_metrics['uf_sensitivity']*100:5.1f}% Spec: {val_metrics['nuf_specificity']*100:5.1f}%", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "duration": dur,
            **val_metrics
        })

    assert history[1]["train_loss"] < history[0]["train_loss"], "Smoke test loss failed to decrease!"
    assert not np.isnan(history[1]["train_loss"]), "Smoke test encountered NaN loss!"
    assert not np.isnan(history[1]["val_loss"]), "Smoke test encountered NaN val loss!"

    print("\n>>> SMOKE TEST PASSED: Pipeline runs end-to-end without crashes, loss decreases, zero NaNs. <<<", flush=True)
    return history

def run_full_training(device, stage1_epochs=15, stage2_epochs=10, batch_size=16):
    print("\n" + "=" * 75)
    print("STARTING FULL TWO-STAGE TRAINING (RESNET50)")
    print("=" * 75)
    print(f"Configuration:")
    print(f"  Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Batch Size    : {batch_size}")
    print(f"  Stage 1 Epochs: {stage1_epochs} (Frozen Backbone, Adam lr=1e-4)")
    print(f"  Stage 2 Epochs: {stage2_epochs} (layer4 + Head Fine-Tuning, Adam lr=1e-5)")
    print(f"  Class Balance : 695 UF vs 694 NUF (Class weighting skipped as moot)")

    train_split_file = COMPANION_DIR / "cls_train_split.txt"
    val_split_file = COMPANION_DIR / "cls_val_split.txt"

    with open(train_split_file, "r", encoding="utf-8") as f:
        train_lines = [l.strip() for l in f if l.strip()]
    with open(val_split_file, "r", encoding="utf-8") as f:
        val_lines = [l.strip() for l in f if l.strip()]

    print(f"\nLoaded Datasets from Annotation Files (Genuine 3-Way Split):")
    print(f"  Internal Train: {len(train_lines)} scans (556 UF, 555 NUF)")
    print(f"  Internal Val  : {len(val_lines)} scans (139 UF, 139 NUF)")
    print(f"  Held-out Test : 358 scans (Untouched until Phase 4 Evaluation)", flush=True)

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
    print("STAGE 1: TRAINING HEAD WITH FROZEN BACKBONE")
    print("=" * 75, flush=True)

    configure_stage1_freezing(model)
    optimizer_stage1 = optim.Adam(model.fc.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)

    best_s1_auc = 0.0
    best_s1_loss = float("inf")
    best_loss_for_early_stop = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(1, stage1_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage1, device,
                                                epoch_desc=f"[Stage 1 {epoch:02d}/{stage1_epochs:02d}]")
        val_metrics = evaluate_loader(model, val_loader, criterion, device)
        dur = time.time() - t0

        # Correct early stopping check: evaluated BEFORE updating best_loss_for_early_stop
        if val_metrics["val_loss"] < best_loss_for_early_stop:
            best_loss_for_early_stop = val_metrics["val_loss"]
            patience_counter = 0
        else:
            patience_counter += 1

        is_best = val_metrics["auc"] > best_s1_auc or (val_metrics["auc"] == best_s1_auc and val_metrics["val_loss"] < best_s1_loss)
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
    print(f"\nStage 1 Complete. Best Val AUC = {best_s1_auc:.4f} (Saved to checkpoints/resnet50_stage1_best.pth)", flush=True)

    # Load best stage 1 weights before starting stage 2
    best_s1_ckpt = torch.load(str(CHECKPOINTS_DIR / "resnet50_stage1_best.pth"), map_location=device)
    model.load_state_dict(best_s1_ckpt["model_state_dict"])

    # =========================================================================
    # STAGE 2: Fine-Tuning layer4 + Head at lr=1e-5
    # =========================================================================
    print("\n" + "=" * 75)
    print("STAGE 2: FINE-TUNING (layer4 + Head Unfrozen at lr=1e-5)")
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
    parser = argparse.ArgumentParser(description="UteroScan ResNet50 Two-Stage Training")
    parser.add_argument("--mode", choices=["smoke_test", "full_train"], default="smoke_test",
                        help="Execution mode: 'smoke_test' (2 epochs on subset) or 'full_train' (Stages 1 and 2)")
    parser.add_argument("--stage1_epochs", type=int, default=15)
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"UteroScan ResNet50 Training Engine")
    print(f"Target Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")

    if args.mode == "smoke_test":
        run_smoke_test(device)
    elif args.mode == "full_train":
        run_full_training(device, stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
