#!/usr/bin/env python3
"""
Phase 8: Retrain Original Full-Frame ResNet50 Baseline (E0) on Recovered Dataset
Location: experiments/final_data_audit/train_e0_recovered.py

Strict Specifications:
  - Input: Full-frame 224x224 (Grayscale -> Median Blur -> CLAHE -> 3ch RGB -> ImageNet Norm)
  - Architecture: ResNet50 (ImageNet pretrained)
  - Optimizer: Adam (lr=1e-4, weight_decay=1e-4)
  - Training augmentations: Horizontal Flip (p=0.5), Rotation (+-15 deg)
  - Seed: 42, Batch size: 16, Max epochs: 8 (patience 4 on inner validation loss)
  - Nested Grouped CV:
      Inner loop selects checkpoints (Rule 1: Min Loss, Rule 2: Max Acc)
      Outer fold evaluated strictly once on untouched outer validation data
  - Checkpoints saved under experiments/final_data_audit/e0_retrained/
  - Preserves all historical V5/V6 checkpoints and manifests
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
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import cv2

PROJECT_ROOT = Path("d:/Project/UteroScan")
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "v4_controlled" / "src"))

from metrics import compute_roc_auc, compute_pr_auc, evaluate_metrics_at_threshold

AUDIT_DIR = PROJECT_ROOT / "experiments" / "final_data_audit"
RECOVERED_FOLDS_CSV = AUDIT_DIR / "recovered_folds.csv"
OUTPUT_CKPT_DIR = AUDIT_DIR / "e0_retrained" / "checkpoints"
OUTPUT_REPORTS_DIR = AUDIT_DIR / "e0_retrained" / "reports"
OUTPUT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 16
MAX_EPOCHS = 8
PATIENCE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class RecoveredUltrasoundDataset(Dataset):
    def __init__(self, records_df, is_train=False):
        self.records = records_df.reset_index(drop=True)
        self.is_train = is_train
        self.clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.cache = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if idx in self.cache:
            img_bgr, target_idx, stem = self.cache[idx]
            img_bgr = img_bgr.copy()
        else:
            row = self.records.iloc[idx]
            rel_p = row["rel_path"]
            full_p = PROJECT_ROOT / rel_p
            img_bgr = cv2.imread(str(full_p))
            if img_bgr is None:
                raise FileNotFoundError(f"Image not found at {full_p}")
            target_idx = int(row["target_idx"]) # 0=UF, 1=NUF
            stem = str(row["canonical_image_id"])
            self.cache[idx] = (img_bgr.copy(), target_idx, stem)

        h, w = img_bgr.shape[:2]

        # Training-only geometric augmentations
        if self.is_train:
            if random.random() < 0.5:
                img_bgr = cv2.flip(img_bgr, 1)
            angle = random.uniform(-15.0, 15.0)
            if abs(angle) > 1.0:
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                img_bgr = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Standard Clinical Preprocessing
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
        blurred = cv2.medianBlur(resized, 3)
        clahe_out = self.clahe.apply(blurred)

        # 3-channel RGB replicate
        rgb = np.stack([clahe_out, clahe_out, clahe_out], axis=-1)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        tensor = self.normalize(tensor)

        return tensor, torch.tensor(target_idx, dtype=torch.long), stem

def build_resnet50_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model

def evaluate_loader(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 0].cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    mean_loss = total_loss / len(loader.dataset)
    y_true_uf = (np.array(all_targets) == 0).astype(int)
    p_uf = np.array(all_probs)

    m = evaluate_metrics_at_threshold(y_true_uf, p_uf, threshold_uf=0.50)
    auc = compute_roc_auc(y_true_uf, p_uf)
    pr_auc = compute_pr_auc(y_true_uf, p_uf)

    return {
        "loss": mean_loss,
        "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
        "sensitivity": m["uf_sensitivity"],
        "specificity": m["nuf_specificity"],
        "precision": m["uf_precision"],
        "f1": m["uf_f1"],
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "probs": p_uf,
        "targets": y_true_uf,
        "tp": m["tp"], "fn": m["fn"], "fp": m["fp"], "tn": m["tn"]
    }

def run_retraining():
    print("=" * 80)
    print("PHASE 8: RETRAINING E0 (FULL-FRAME RESNET50) ON RECOVERED CLEAN DATASET")
    print("=" * 80)
    seed_everything(SEED)

    df_folds = pd.read_csv(RECOVERED_FOLDS_CSV)
    print(f"Total recovered development scans: {len(df_folds)}")
    print(f"UF: {len(df_folds[df_folds['verified_class'] == 'UF'])}, NUF: {len(df_folds[df_folds['verified_class'] == 'NUF'])}")

    criterion = nn.CrossEntropyLoss()
    fold_summaries = []
    all_outer_preds = []

    for outer_fold in range(5):
        print("\n" + "-" * 70)
        inner_val_fold = (outer_fold + 1) % 5
        inner_train_folds = [f for f in range(5) if f != outer_fold and f != inner_val_fold]
        print(f"[RECOVERED E0] OUTER FOLD {outer_fold}/4 | Inner Val: Fold {inner_val_fold} | Inner Train: Folds {inner_train_folds}")
        print("-" * 70)

        df_train = df_folds[df_folds["fold"].isin(inner_train_folds)].copy()
        df_val = df_folds[df_folds["fold"] == inner_val_fold].copy()
        df_outer = df_folds[df_folds["fold"] == outer_fold].copy()

        train_ds = RecoveredUltrasoundDataset(df_train, is_train=True)
        val_ds = RecoveredUltrasoundDataset(df_val, is_train=False)
        outer_ds = RecoveredUltrasoundDataset(df_outer, is_train=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        outer_loader = DataLoader(outer_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = build_resnet50_model().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best_val_loss = float("inf")
        best_val_acc = -1.0
        best_loss_state = None
        best_acc_state = None
        best_loss_ep = -1
        best_acc_ep = -1
        patience_counter = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            t_ep = time.time()
            model.train()
            train_loss = 0.0

            for inputs, targets, _ in train_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)
            val_m = evaluate_loader(model, val_loader, criterion)
            ep_dur = time.time() - t_ep

            flag = ""
            if val_m["loss"] < best_val_loss:
                best_val_loss = val_m["loss"]
                best_loss_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_loss_ep = epoch
                patience_counter = 0
                flag += "[*] Min Loss "
            else:
                patience_counter += 1

            if val_m["accuracy"] > best_val_acc:
                best_val_acc = val_m["accuracy"]
                best_acc_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_acc_ep = epoch
                flag += "[^] Max Acc"

            print(f"  Epoch {epoch:02d}/{MAX_EPOCHS:02d} ({ep_dur:.1f}s) | Train Loss: {train_loss:.4f} | Inner Val Loss: {val_m['loss']:.4f} | Acc: {val_m['accuracy']*100:.1f}% | Sens: {val_m['sensitivity']*100:.1f}% | Spec: {val_m['specificity']*100:.1f}% | AUC: {val_m['roc_auc']:.4f} | {flag}")

            if patience_counter >= PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

        # Save checkpoints
        ckpt_loss_path = OUTPUT_CKPT_DIR / f"recovered_e0_fold_{outer_fold}_best_loss.pth"
        torch.save({"model_state": best_loss_state, "epoch": best_loss_ep, "inner_val_loss": best_val_loss}, ckpt_loss_path)

        ckpt_acc_path = OUTPUT_CKPT_DIR / f"recovered_e0_fold_{outer_fold}_best_acc.pth"
        torch.save({"model_state": best_acc_state, "epoch": best_acc_ep, "inner_val_acc": best_val_acc}, ckpt_acc_path)

        # Outer evaluation for Rule 1 (Min Loss)
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_loss_state.items()})
        outer_m_r1 = evaluate_loader(model, outer_loader, criterion)

        # Outer evaluation for Rule 2 (Max Acc)
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_acc_state.items()})
        outer_m_r2 = evaluate_loader(model, outer_loader, criterion)

        print(f"  [OUTER EVAL FOLD {outer_fold}] (Rule 1 Min Loss, Ep {best_loss_ep}): Acc = {outer_m_r1['accuracy']*100:.2f}%, Sens = {outer_m_r1['sensitivity']*100:.2f}%, Spec = {outer_m_r1['specificity']*100:.2f}%, AUC = {outer_m_r1['roc_auc']:.4f}")
        print(f"  [OUTER EVAL FOLD {outer_fold}] (Rule 2 Max Acc,  Ep {best_acc_ep}):  Acc = {outer_m_r2['accuracy']*100:.2f}%, Sens = {outer_m_r2['sensitivity']*100:.2f}%, Spec = {outer_m_r2['specificity']*100:.2f}%, AUC = {outer_m_r2['roc_auc']:.4f}")

        # Save per-image predictions
        df_outer_preds = df_outer.copy()
        df_outer_preds["outer_fold"] = outer_fold
        df_outer_preds["true_uf"] = outer_m_r1["targets"]
        df_outer_preds["pred_prob_uf_rule1"] = outer_m_r1["probs"]
        df_outer_preds["pred_prob_uf_rule2"] = outer_m_r2["probs"]
        df_outer_preds.to_csv(OUTPUT_REPORTS_DIR / f"preds_recovered_e0_outer_fold_{outer_fold}.csv", index=False)
        all_outer_preds.append(df_outer_preds)

        fold_summaries.append({
            "outer_fold": outer_fold,
            "rule_1_min_loss": {
                "epoch": best_loss_ep,
                "accuracy": outer_m_r1["accuracy"],
                "balanced_accuracy": outer_m_r1["balanced_accuracy"],
                "sensitivity": outer_m_r1["sensitivity"],
                "specificity": outer_m_r1["specificity"],
                "precision": outer_m_r1["precision"],
                "f1": outer_m_r1["f1"],
                "roc_auc": outer_m_r1["roc_auc"],
                "pr_auc": outer_m_r1["pr_auc"],
                "tp": outer_m_r1["tp"], "fn": outer_m_r1["fn"], "fp": outer_m_r1["fp"], "tn": outer_m_r1["tn"]
            },
            "rule_2_max_acc": {
                "epoch": best_acc_ep,
                "accuracy": outer_m_r2["accuracy"],
                "balanced_accuracy": outer_m_r2["balanced_accuracy"],
                "sensitivity": outer_m_r2["sensitivity"],
                "specificity": outer_m_r2["specificity"],
                "precision": outer_m_r2["precision"],
                "f1": outer_m_r2["f1"],
                "roc_auc": outer_m_r2["roc_auc"],
                "pr_auc": outer_m_r2["pr_auc"],
                "tp": outer_m_r2["tp"], "fn": outer_m_r2["fn"], "fp": outer_m_r2["fp"], "tn": outer_m_r2["tn"]
            }
        })

    # Pooled analysis
    df_all_outer = pd.concat(all_outer_preds, ignore_index=True)
    
    def compute_summary(rule_key, col_prob):
        accs = [f[rule_key]["accuracy"] for f in fold_summaries]
        bal_accs = [f[rule_key]["balanced_accuracy"] for f in fold_summaries]
        senss = [f[rule_key]["sensitivity"] for f in fold_summaries]
        specs = [f[rule_key]["specificity"] for f in fold_summaries]
        precs = [f[rule_key]["precision"] for f in fold_summaries]
        f1s = [f[rule_key]["f1"] for f in fold_summaries]
        aucs = [f[rule_key]["roc_auc"] for f in fold_summaries]
        pr_aucs = [f[rule_key]["pr_auc"] for f in fold_summaries]

        y_all = df_all_outer["true_uf"].values
        p_all = df_all_outer[col_prob].values
        pool_m = evaluate_metrics_at_threshold(y_all, p_all, threshold_uf=0.50)
        pool_auc = compute_roc_auc(y_all, p_all)
        pool_pr = compute_pr_auc(y_all, p_all)

        return {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_balanced_accuracy": float(np.mean(bal_accs)),
            "std_balanced_accuracy": float(np.std(bal_accs)),
            "mean_sensitivity": float(np.mean(senss)),
            "std_sensitivity": float(np.std(senss)),
            "mean_specificity": float(np.mean(specs)),
            "std_specificity": float(np.std(specs)),
            "mean_precision": float(np.mean(precs)),
            "std_precision": float(np.std(precs)),
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
            "mean_roc_auc": float(np.mean(aucs)),
            "std_roc_auc": float(np.std(aucs)),
            "mean_pr_auc": float(np.mean(pr_aucs)),
            "std_pr_auc": float(np.std(pr_aucs)),
            "pooled": {
                "accuracy": pool_m["accuracy"],
                "balanced_accuracy": pool_m["balanced_accuracy"],
                "sensitivity": pool_m["uf_sensitivity"],
                "specificity": pool_m["nuf_specificity"],
                "precision": pool_m["uf_precision"],
                "f1": pool_m["uf_f1"],
                "roc_auc": pool_auc,
                "pr_auc": pool_pr,
                "tp": pool_m["tp"], "fn": pool_m["fn"], "fp": pool_m["fp"], "tn": pool_m["tn"]
            }
        }

    summary_r1 = compute_summary("rule_1_min_loss", "pred_prob_uf_rule1")
    summary_r2 = compute_summary("rule_2_max_acc", "pred_prob_uf_rule2")

    full_report = {
        "benchmark": "RECOVERED_E0_RESNET50_BASELINE",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "5-Fold Nested Grouped Cross-Validation on Recovered Dataset",
        "dataset_summary": {
            "total_development_scans": len(df_folds),
            "uf_count": int(len(df_folds[df_folds["verified_class"] == "UF"])),
            "nuf_count": int(len(df_folds[df_folds["verified_class"] == "NUF"])),
            "study_groups": int(df_folds["study_proxy_group"].nunique())
        },
        "rule_1_min_loss": summary_r1,
        "rule_2_max_acc": summary_r2,
        "per_fold_details": fold_summaries
    }

    report_json_path = OUTPUT_REPORTS_DIR / "recovered_e0_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 80)
    print("RECOVERED E0 RETRAINING COMPLETE")
    print("=" * 80)
    print(f"Rule 1 (Min Loss) Mean Outer Acc: {summary_r1['mean_accuracy']*100:.2f}% ± {summary_r1['std_accuracy']*100:.2f}% | Pooled: {summary_r1['pooled']['accuracy']*100:.2f}% | AUC: {summary_r1['mean_roc_auc']:.4f}")
    print(f"Rule 2 (Max Acc)  Mean Outer Acc: {summary_r2['mean_accuracy']*100:.2f}% ± {summary_r2['std_accuracy']*100:.2f}% | Pooled: {summary_r2['pooled']['accuracy']*100:.2f}% | AUC: {summary_r2['mean_roc_auc']:.4f}")
    print(f"Saved full report to: {report_json_path.name}")

    return full_report

if __name__ == "__main__":
    run_retraining()
