#!/usr/bin/env python3
"""
Phase 3: Sequential Training of E0 Baseline on Clean V5 Dataset
Location: experiments/v4_controlled/src/train_clean_v5_e0.py

Architecture: ResNet50 Baseline (Pretrained ImageNet)
Full frame (224x224), no viewport cropping, no caliper augmentation, no aux localization.
Saves BOTH:
  - e0_clean_v5_fold_{fold}_best_loss.pth
  - e0_clean_v5_fold_{fold}_best_acc.pth
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

from dataset import ControlledUltrasoundDataset
from models import ResNet50Baseline
from metrics import compute_roc_auc, compute_pr_auc, evaluate_metrics_at_threshold, optimize_threshold_on_val

FOLDS_V5_CSV = PROJECT_ROOT / "experiments" / "v4_controlled" / "data" / "clean_v5" / "folds_clean_v5.csv"
TEST_LINES_FILE = PROJECT_ROOT / "DCGAN&MobileNetV2" / "DCGAN&MobileNetV2" / "cls_test.txt"
CHECKPOINTS_DIR = PROJECT_ROOT / "experiments" / "v4_controlled" / "checkpoints" / "clean_v5_e0"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "v4_controlled" / "reports"
LIVE_PROGRESS_FILE = REPORTS_DIR / "LIVE_PROGRESS_CLEAN_V5.md"

def write_live_status(fold, epoch, max_epochs, history_rows, status="Running", extra_notes=""):
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LIVE_PROGRESS_FILE, "w", encoding="utf-8") as f:
            f.write(f"# 📊 UteroScan V4 Clean V5 E0 Baseline Training: Live Progress\n\n")
            f.write(f"**Experiment:** `E0_CLEAN_V5_BASELINE`  \n")
            f.write(f"**Status:** **{status}**  \n")
            f.write(f"**Last Updated:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`  \n")
            f.write(f"**Current State:** Fold {fold}/4 | Epoch {epoch}/{max_epochs}  \n\n")
            
            if extra_notes:
                f.write(f"> {extra_notes}\n\n")
                
            f.write("### Fold Execution Log\n\n")
            f.write("| Fold | Epoch | Time | Train Loss | Val Loss | Val Acc | UF Sens | NUF Spec | ROC-AUC | Note |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for r in history_rows:
                f.write(f"| Fold {r['fold']} | Ep {r['epoch']:02d} | {r['dur']:.1f}s | {r['train_loss']:.4f} | {r['val_loss']:.4f} | {r['val_acc']*100:.1f}% | {r['val_sens']*100:.1f}% | {r['val_spec']*100:.1f}% | {r['val_auc']:.4f} | {r['note']} |\n")
            f.flush()
    except Exception:
        pass

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_clean_v5_e0(max_epochs=8, batch_size=16, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print("\n" + "=" * 80)
    print("STARTING E0 BASELINE RETRAINING ON CLEAN V5 DATASET")
    print("=" * 80)
    print(f"Device              : {device}")
    print(f"Architecture        : ResNet50 (Pretrained ImageNet)")
    print(f"Viewport Cropping   : False (Full Frame)")
    print(f"Caliper Augmentation: False (Disabled)")
    print(f"Aux Localization    : False (Disabled)")
    print(f"Input Resolution    : (224, 224)")
    print(f"Max Epochs per Fold : {max_epochs} (patience=4)")
    print(f"Batch Size          : {batch_size}")
    print(f"Checkpoint Dir      : {CHECKPOINTS_DIR}")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df_folds = pd.read_csv(FOLDS_V5_CSV)
    set_seed(42)

    history_rows = []
    epoch_metric_rows = []
    
    # Store per-fold best results
    fold_results_best_loss = []
    fold_results_best_acc = []
    
    oof_predictions_loss = []
    oof_predictions_acc = []

    cls_criterion = nn.CrossEntropyLoss()
    write_live_status(0, 0, max_epochs, history_rows, status="Initializing Clean V5 E0 Training...")

    for fold in range(5):
        print("\n" + "-" * 70)
        print(f"[E0_CLEAN_V5] TRAINING FOLD {fold}/4")
        print("-" * 70)

        train_df = df_folds[df_folds["fold"] != fold].copy()
        val_df = df_folds[df_folds["fold"] == fold].copy()

        train_ds = ControlledUltrasoundDataset(
            train_df,
            input_size=(224, 224),
            is_train=True,
            use_viewport=False,
            use_caliper_aug=False,
            project_root=str(PROJECT_ROOT)
        )
        val_ds = ControlledUltrasoundDataset(
            val_df,
            input_size=(224, 224),
            is_train=False,
            use_viewport=False,
            use_caliper_aug=False,
            project_root=str(PROJECT_ROOT)
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        model = ResNet50Baseline(num_classes=2, pretrained=True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

        best_val_loss = float("inf")
        best_loss_epoch = -1
        best_loss_state = None
        best_loss_val_probs = None
        best_loss_val_targets = None
        best_loss_val_stems = None
        best_loss_metrics = None

        best_val_acc = -1.0
        best_acc_loss_tiebreaker = float("inf")
        best_acc_epoch = -1
        best_acc_state = None
        best_acc_val_probs = None
        best_acc_val_targets = None
        best_acc_val_stems = None
        best_acc_metrics = None

        patience = 4
        patience_counter = 0

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            model.train()
            train_loss = 0.0
            total_train = 0

            for imgs, targets, _, _, stems in train_loader:
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

            # Validation Loop
            model.eval()
            val_loss = 0.0
            total_val = 0
            val_preds_uf = []
            val_targets_list = []
            val_stems_list = []

            with torch.no_grad():
                for imgs, targets, _, _, stems in val_loader:
                    imgs = imgs.to(device)
                    targets = targets.to(device)

                    logits = model(imgs)
                    loss = cls_criterion(logits, targets)
                    probs = torch.softmax(logits, dim=1)

                    val_loss += loss.item() * imgs.size(0)
                    total_val += imgs.size(0)

                    val_preds_uf.extend(probs[:, 0].cpu().numpy().tolist())
                    val_targets_list.extend(targets.cpu().numpy().tolist())
                    val_stems_list.extend(stems)

            val_loss /= total_val
            y_val_uf = (np.array(val_targets_list) == 0).astype(int)
            p_val_uf = np.array(val_preds_uf)
            val_auc = compute_roc_auc(y_val_uf, p_val_uf)
            val_pr_auc = compute_pr_auc(y_val_uf, p_val_uf)
            m_t50 = evaluate_metrics_at_threshold(y_val_uf, p_val_uf, threshold_uf=0.50)

            dur = time.time() - t0
            print(f"  Epoch {epoch:02d}/{max_epochs:02d} ({dur:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {m_t50['accuracy']*100:.1f}% | Val Sens: {m_t50['uf_sensitivity']*100:.1f}% | Val Spec: {m_t50['nuf_specificity']*100:.1f}% | AUC: {val_auc:.4f}", flush=True)

            is_best_loss = val_loss < best_val_loss
            is_best_acc = (m_t50['accuracy'] > best_val_acc) or (m_t50['accuracy'] == best_val_acc and val_loss < best_acc_loss_tiebreaker)

            notes = []
            if is_best_loss:
                notes.append("★ Min Loss")
            if is_best_acc:
                notes.append("▲ Max Acc")
            note_str = " | ".join(notes)

            history_rows.append({
                "fold": fold,
                "epoch": epoch,
                "dur": dur,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": m_t50['accuracy'],
                "val_sens": m_t50['uf_sensitivity'],
                "val_spec": m_t50['nuf_specificity'],
                "val_auc": val_auc,
                "note": note_str
            })

            epoch_metric_rows.append({
                "experiment": "E0_CLEAN_V5",
                "fold": fold,
                "epoch": epoch,
                "duration_sec": dur,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc_050": m_t50['accuracy'],
                "val_sens_050": m_t50['uf_sensitivity'],
                "val_spec_050": m_t50['nuf_specificity'],
                "val_f1_050": m_t50['f1'],
                "val_precision_050": m_t50['uf_precision'],
                "roc_auc": val_auc,
                "pr_auc": val_pr_auc,
                "is_best_loss": is_best_loss,
                "is_best_acc": is_best_acc
            })

            write_live_status(fold, epoch, max_epochs, history_rows, status=f"Training Fold {fold}/4 (Epoch {epoch}/{max_epochs})")

            # Update Rule 1: Min Loss
            if is_best_loss:
                best_val_loss = val_loss
                best_loss_epoch = epoch
                best_loss_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_loss_val_probs = p_val_uf
                best_loss_val_targets = y_val_uf
                best_loss_val_stems = val_stems_list
                best_loss_metrics = {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "roc_auc": val_auc,
                    "pr_auc": val_pr_auc,
                    **m_t50
                }
                patience_counter = 0
            else:
                patience_counter += 1

            # Update Rule 2: Max Acc (val loss tiebreaker)
            if is_best_acc:
                best_val_acc = m_t50['accuracy']
                best_acc_loss_tiebreaker = val_loss
                best_acc_epoch = epoch
                best_acc_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_acc_val_probs = p_val_uf
                best_acc_val_targets = y_val_uf
                best_acc_val_stems = val_stems_list
                best_acc_metrics = {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "roc_auc": val_auc,
                    "pr_auc": val_pr_auc,
                    **m_t50
                }

            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch} (patience={patience}).")
                break

        # Save BOTH checkpoints
        ckpt_loss_path = CHECKPOINTS_DIR / f"e0_clean_v5_fold_{fold}_best_loss.pth"
        ckpt_acc_path = CHECKPOINTS_DIR / f"e0_clean_v5_fold_{fold}_best_acc.pth"

        torch.save(best_loss_state, ckpt_loss_path)
        torch.save(best_acc_state, ckpt_acc_path)
        print(f"  [SAVED] Min Loss Ckpt -> {ckpt_loss_path.name} (Epoch {best_loss_epoch}, Loss: {best_val_loss:.4f}, Acc: {best_loss_metrics['accuracy']*100:.1f}%)")
        print(f"  [SAVED] Max Acc Ckpt  -> {ckpt_acc_path.name} (Epoch {best_acc_epoch}, Acc: {best_val_acc*100:.1f}%, Loss: {best_acc_loss_tiebreaker:.4f})")

        fold_results_best_loss.append({"fold": fold, **best_loss_metrics})
        fold_results_best_acc.append({"fold": fold, **best_acc_metrics})

        for s, t, p in zip(best_loss_val_stems, best_loss_val_targets, best_loss_val_probs):
            oof_predictions_loss.append({"stem": s, "true": t, "prob_uf": p, "fold": fold})
        for s, t, p in zip(best_acc_val_stems, best_acc_val_targets, best_acc_val_probs):
            oof_predictions_acc.append({"stem": s, "true": t, "prob_uf": p, "fold": fold})

    # Save epoch metrics CSV
    epoch_metrics_df = pd.DataFrame(epoch_metric_rows)
    epoch_metrics_csv = REPORTS_DIR / "clean_v5_e0_epoch_metrics.csv"
    epoch_metrics_df.to_csv(epoch_metrics_csv, index=False)
    print(f"\nSaved epoch-by-epoch metrics to {epoch_metrics_csv}")

    return fold_results_best_loss, fold_results_best_acc, oof_predictions_loss, oof_predictions_acc

if __name__ == "__main__":
    run_clean_v5_e0()
