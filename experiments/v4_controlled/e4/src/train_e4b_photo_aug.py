#!/usr/bin/env python3
"""
E4B: Sequential Training with Mild Photometric Augmentation Only
Location: experiments/v4_controlled/e4/src/train_e4b_photo_aug.py

Photometric Augmentations (Train Only):
  - Gamma adjustment: [0.85, 1.15], p=0.5
  - Contrast scaling: [0.90, 1.10], p=0.5
  - Multiplicative speckle noise: sigma <= 0.02, p=0.3

Identical to E0:
  - Architecture: ResNet50Baseline
  - Single uniform learning rate: 1e-4
  - Adam optimizer (weight decay 1e-4)
  - 5-Fold Nested Grouped CV (folds_clean_v5.csv)
  - Full-frame 224x224 input
  - Seed 42, Batch size 16, Max epochs 8 (patience 4)
  - Decision threshold 0.50
"""

import os
import sys
import time
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("d:/Project/UteroScan")
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "v4_controlled" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "v4_controlled" / "e4" / "src"))

try:
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
except Exception:
    pass

from models import ResNet50Baseline
from e4_dataset import E4UltrasoundDataset
from metrics import compute_roc_auc, compute_pr_auc, evaluate_metrics_at_threshold

FOLDS_V5_CSV = PROJECT_ROOT / "experiments" / "v4_controlled" / "data" / "clean_v5" / "folds_clean_v5.csv"
CHECKPOINTS_DIR = PROJECT_ROOT / "experiments" / "v4_controlled" / "e4" / "checkpoints" / "e4b_photometric_aug"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "v4_controlled" / "e4" / "reports"
LIVE_PROGRESS_FILE = REPORTS_DIR / "LIVE_PROGRESS_E4B.md"

def write_live_status(outer_fold, epoch, max_epochs, history_rows, status="Running", extra_notes=""):
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LIVE_PROGRESS_FILE, "w", encoding="utf-8") as f:
            f.write(f"# 📊 UteroScan V4 E4B (Mild Photometric Augmentation): Live Progress\n\n")
            f.write(f"**Experiment:** `E4B_PHOTOMETRIC_AUG`  \n")
            f.write(f"**Status:** **{status}**  \n")
            f.write(f"**Last Updated:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`  \n")
            f.write(f"**Current State:** Outer Fold {outer_fold}/4 | Epoch {epoch}/{max_epochs}  \n\n")
            
            if extra_notes:
                f.write(f"> {extra_notes}\n\n")
                
            f.write("### Fold Execution Log\n\n")
            f.write("| Outer Fold | Epoch | Time | Train Loss | Inner Val Loss | Inner Val Acc | Outer Test Acc (Rule 1) | Outer Test Acc (Rule 2) | Note |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for r in history_rows:
                f.write(f"| Fold {r['outer_fold']} | Ep {r['epoch']:02d} | {r['dur']:.1f}s | {r['train_loss']:.4f} | {r['inner_val_loss']:.4f} | {r['inner_val_acc']*100:.1f}% | {r.get('outer_test_acc_r1', '-') } | {r.get('outer_test_acc_r2', '-') } | {r['note']} |\n")
            f.flush()
    except Exception:
        pass

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_dataloader(model, loader, device, cls_criterion):
    model.eval()
    val_loss = 0.0
    total_val = 0
    val_preds_uf = []
    val_targets_list = []
    val_stems_list = []

    with torch.no_grad():
        for imgs, targets, stems in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            logits = model(imgs)
            loss = cls_criterion(logits, targets)
            probs = torch.softmax(logits, dim=1)

            val_loss += loss.item() * imgs.size(0)
            total_val += imgs.size(0)

            # Class 0 = UF, Class 1 = NUF
            val_preds_uf.extend(probs[:, 0].cpu().numpy().tolist())
            val_targets_list.extend(targets.cpu().numpy().tolist())
            val_stems_list.extend(stems)

    val_loss /= total_val
    y_val_uf = (np.array(val_targets_list) == 0).astype(int)
    p_val_uf = np.array(val_preds_uf)
    val_auc = compute_roc_auc(y_val_uf, p_val_uf)
    val_pr_auc = compute_pr_auc(y_val_uf, p_val_uf)
    m_t50 = evaluate_metrics_at_threshold(y_val_uf, p_val_uf, threshold_uf=0.50)

    return {
        "loss": val_loss,
        "auc": val_auc,
        "pr_auc": val_pr_auc,
        "probs_uf": p_val_uf,
        "targets_uf": y_val_uf,
        "stems": val_stems_list,
        "metrics_050": m_t50
    }

def run_e4b(max_epochs=8, batch_size=16, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print("STARTING E4B: MILD PHOTOMETRIC AUGMENTATION EXPERIMENT")
    print("=" * 80)
    print(f"Device              : {device}")
    print(f"Architecture        : ResNet50Baseline")
    print(f"Augmentations       : Gamma [0.85, 1.15], Contrast +-10%, Speckle sigma<=0.02")
    print(f"Outer Folds         : 5 (Nested Grouped Cross-Validation)")
    print(f"Batch Size          : {batch_size}")
    print(f"Max Epochs          : {max_epochs} (patience=4)")
    print(f"Checkpoint Dir      : {CHECKPOINTS_DIR}")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df_folds = pd.read_csv(FOLDS_V5_CSV)
    set_seed(42)

    history_rows = []
    epoch_metric_rows = []
    cls_criterion = nn.CrossEntropyLoss()

    outer_results_best_loss = []
    outer_results_best_acc = []

    for outer_fold in range(5):
        inner_val_fold = (outer_fold + 1) % 5
        inner_train_folds = [f for f in range(5) if f not in (outer_fold, inner_val_fold)]

        print("\n" + "-" * 70)
        print(f"[E4B_PHOTO_AUG] OUTER FOLD {outer_fold}/4 | Inner Val: Fold {inner_val_fold} | Inner Train: Folds {inner_train_folds}")
        print("-" * 70)

        outer_test_df = df_folds[df_folds["fold"] == outer_fold].copy()
        inner_val_df = df_folds[df_folds["fold"] == inner_val_fold].copy()
        inner_train_df = df_folds[df_folds["fold"].isin(inner_train_folds)].copy()

        # Training set uses mild photometric augmentation
        train_ds = E4UltrasoundDataset(
            inner_train_df, input_size=(224, 224), is_train=True, use_photometric_aug=True, project_root=str(PROJECT_ROOT)
        )
        # Validation and test sets are never augmented
        inner_val_ds = E4UltrasoundDataset(
            inner_val_df, input_size=(224, 224), is_train=False, use_photometric_aug=False, project_root=str(PROJECT_ROOT)
        )
        outer_test_ds = E4UltrasoundDataset(
            outer_test_df, input_size=(224, 224), is_train=False, use_photometric_aug=False, project_root=str(PROJECT_ROOT)
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        inner_val_loader = DataLoader(inner_val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        outer_test_loader = DataLoader(outer_test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        model = ResNet50Baseline(num_classes=2, pretrained=True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

        best_val_loss = float("inf")
        best_loss_epoch = -1
        best_loss_state = None

        best_val_acc = -1.0
        best_acc_loss_tiebreaker = float("inf")
        best_acc_epoch = -1
        best_acc_state = None

        patience = 4
        patience_counter = 0

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            model.train()
            train_loss = 0.0
            total_train = 0

            for imgs, targets, stems in train_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                logits = model(imgs)
                loss = cls_criterion(logits, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * imgs.size(0)
                total_train += imgs.size(0)

            train_loss /= total_train

            # Evaluate on Inner Validation Fold
            val_res = evaluate_dataloader(model, inner_val_loader, device, cls_criterion)
            dur = time.time() - t0

            val_loss = val_res["loss"]
            val_acc = val_res["metrics_050"]["accuracy"]
            val_sens = val_res["metrics_050"]["uf_sensitivity"]
            val_spec = val_res["metrics_050"]["nuf_specificity"]
            val_auc = val_res["auc"]

            is_best_loss = val_loss < best_val_loss
            is_best_acc = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_acc_loss_tiebreaker)

            notes = []
            if is_best_loss:
                notes.append("[*] Min Loss")
            if is_best_acc:
                notes.append("[^] Max Acc")
            note_str = " | ".join(notes)

            print(f"  Epoch {epoch:02d}/{max_epochs:02d} ({dur:.1f}s) | Train Loss: {train_loss:.4f} | Inner Val Loss: {val_loss:.4f} | Inner Val Acc: {val_acc*100:.1f}% | Sens: {val_sens*100:.1f}% | Spec: {val_spec*100:.1f}% | AUC: {val_auc:.4f} | {note_str}", flush=True)

            history_rows.append({
                "outer_fold": outer_fold,
                "epoch": epoch,
                "dur": dur,
                "train_loss": train_loss,
                "inner_val_loss": val_loss,
                "inner_val_acc": val_acc,
                "note": note_str
            })

            epoch_metric_rows.append({
                "experiment": "E4B_PHOTOMETRIC_AUG",
                "outer_fold": outer_fold,
                "epoch": epoch,
                "duration_sec": dur,
                "train_loss": train_loss,
                "inner_val_loss": val_loss,
                "inner_val_acc_050": val_acc,
                "inner_val_sens_050": val_sens,
                "inner_val_spec_050": val_spec,
                "inner_val_roc_auc": val_auc,
                "is_best_loss": is_best_loss,
                "is_best_acc": is_best_acc
            })

            write_live_status(outer_fold, epoch, max_epochs, history_rows, status=f"Training Outer Fold {outer_fold}/4 (Epoch {epoch}/{max_epochs})")

            if is_best_loss:
                best_val_loss = val_loss
                best_loss_epoch = epoch
                best_loss_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if is_best_acc:
                best_val_acc = val_acc
                best_acc_loss_tiebreaker = val_loss
                best_acc_epoch = epoch
                best_acc_state = {k: v.cpu() for k, v in model.state_dict().items()}

            if patience_counter >= patience:
                print(f"  Early stopping triggered on inner validation at epoch {epoch}.")
                break

        # Save Both Checkpoints
        ckpt_loss_path = CHECKPOINTS_DIR / f"e4b_photo_aug_fold_{outer_fold}_best_loss.pth"
        ckpt_acc_path = CHECKPOINTS_DIR / f"e4b_photo_aug_fold_{outer_fold}_best_acc.pth"
        torch.save(best_loss_state, ckpt_loss_path)
        torch.save(best_acc_state, ckpt_acc_path)
        print(f"  [SAVED] Min Loss Ckpt -> {ckpt_loss_path.name} (Epoch {best_loss_epoch}, Inner Loss: {best_val_loss:.4f})")
        print(f"  [SAVED] Max Acc Ckpt  -> {ckpt_acc_path.name} (Epoch {best_acc_epoch}, Inner Acc: {best_val_acc*100:.1f}%)")

        # Evaluate on Completely Untouched Outer Evaluation Fold
        model.load_state_dict({k: v.to(device) for k, v in best_loss_state.items()})
        eval_loss_res = evaluate_dataloader(model, outer_test_loader, device, cls_criterion)

        model.load_state_dict({k: v.to(device) for k, v in best_acc_state.items()})
        eval_acc_res = evaluate_dataloader(model, outer_test_loader, device, cls_criterion)

        outer_results_best_loss.append({
            "outer_fold": outer_fold,
            "selected_epoch": best_loss_epoch,
            "outer_loss": eval_loss_res["loss"],
            "outer_auc": eval_loss_res["auc"],
            "outer_pr_auc": eval_loss_res["pr_auc"],
            "probs_uf": eval_loss_res["probs_uf"],
            "targets_uf": eval_loss_res["targets_uf"],
            "stems": eval_loss_res["stems"],
            **eval_loss_res["metrics_050"]
        })

        outer_results_best_acc.append({
            "outer_fold": outer_fold,
            "selected_epoch": best_acc_epoch,
            "outer_loss": eval_acc_res["loss"],
            "outer_auc": eval_acc_res["auc"],
            "outer_pr_auc": eval_acc_res["pr_auc"],
            "probs_uf": eval_acc_res["probs_uf"],
            "targets_uf": eval_acc_res["targets_uf"],
            "stems": eval_acc_res["stems"],
            **eval_acc_res["metrics_050"]
        })

        print(f"  [OUTER EVAL FOLD {outer_fold}] (Rule 1 Min Loss, Ep {best_loss_epoch}): Acc = {eval_loss_res['metrics_050']['accuracy']*100:.2f}%, Sens = {eval_loss_res['metrics_050']['uf_sensitivity']*100:.2f}%, Spec = {eval_loss_res['metrics_050']['nuf_specificity']*100:.2f}%, AUC = {eval_loss_res['auc']:.4f}")
        print(f"  [OUTER EVAL FOLD {outer_fold}] (Rule 2 Max Acc,  Ep {best_acc_epoch}):  Acc = {eval_acc_res['metrics_050']['accuracy']*100:.2f}%, Sens = {eval_acc_res['metrics_050']['uf_sensitivity']*100:.2f}%, Spec = {eval_acc_res['metrics_050']['nuf_specificity']*100:.2f}%, AUC = {eval_acc_res['auc']:.4f}")

        history_rows[-1]["outer_test_acc_r1"] = f"{eval_loss_res['metrics_050']['accuracy']*100:.1f}%"
        history_rows[-1]["outer_test_acc_r2"] = f"{eval_acc_res['metrics_050']['accuracy']*100:.1f}%"
        write_live_status(outer_fold, max_epochs, max_epochs, history_rows, status=f"Completed Outer Fold {outer_fold}/4")

        # Save per-fold predictions CSV
        preds_df = pd.DataFrame({
            "source_stem": eval_loss_res["stems"],
            "true_uf": eval_loss_res["targets_uf"],
            "pred_prob_uf_rule1": eval_loss_res["probs_uf"],
            "pred_prob_uf_rule2": eval_acc_res["probs_uf"]
        })
        preds_df.to_csv(REPORTS_DIR / f"preds_e4b_outer_fold_{outer_fold}.csv", index=False)

    # Save epoch metrics CSV
    pd.DataFrame(epoch_metric_rows).to_csv(REPORTS_DIR / "e4b_epoch_metrics.csv", index=False)

    mean_acc_r1 = np.mean([r["accuracy"] for r in outer_results_best_loss])
    std_acc_r1 = np.std([r["accuracy"] for r in outer_results_best_loss])
    mean_acc_r2 = np.mean([r["accuracy"] for r in outer_results_best_acc])
    std_acc_r2 = np.std([r["accuracy"] for r in outer_results_best_acc])

    print("\n" + "=" * 80)
    print("E4B PHOTOMETRIC AUGMENTATION TRAINING COMPLETE")
    print(f"Rule 1 (Min Loss Checkpoint) Mean Outer Acc: {mean_acc_r1*100:.2f}% ± {std_acc_r1*100:.2f}%")
    print(f"Rule 2 (Max Acc Checkpoint)  Mean Outer Acc: {mean_acc_r2*100:.2f}% ± {std_acc_r2*100:.2f}%")
    print("=" * 80)

    write_live_status(4, max_epochs, max_epochs, history_rows, status="COMPLETED", extra_notes=f"E4B Complete. Rule 1: {mean_acc_r1*100:.2f}% ± {std_acc_r1*100:.2f}% | Rule 2: {mean_acc_r2*100:.2f}% ± {std_acc_r2*100:.2f}%")

    return {
        "outer_results_best_loss": outer_results_best_loss,
        "outer_results_best_acc": outer_results_best_acc
    }

if __name__ == "__main__":
    run_e4b()
