#!/usr/bin/env python3
"""
UteroScan — Experiment E5: Differential Learning Rates on Recovered Dataset
Location: experiments/final_data_audit/e5_differential_lr/verify_and_train_e5.py

Differential Learning Rates:
  - conv1, bn1, layer1, layer2: 1e-5
  - layer3, layer4: 5e-5
  - fc: 1e-4
  - Optimizer: Adam, weight_decay: 1e-4
  - Batch size: 16, Seed: 42, Max epochs: 8, Patience: 4
  - Evaluation: 5-Fold Nested Grouped Cross-Validation on recovered_folds.csv
  - Checkpoint rules: Rule 1 (Min Val Loss), Rule 2 (Max Val Acc with Loss Tie-Breaker)
  - Paired comparison against Recovered E0 baseline
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
MASTER_MANIFEST_CSV = AUDIT_DIR / "master_review_manifest.csv"
RECOVERED_FOLDS_CSV = AUDIT_DIR / "recovered_folds.csv"
SPLIT_VALIDATION_REPORT_JSON = AUDIT_DIR / "split_validation_report.json"
E0_REPORTS_DIR = AUDIT_DIR / "e0_retrained" / "reports"
E0_REPORT_JSON = E0_REPORTS_DIR / "recovered_e0_report.json"

E5_DIR = AUDIT_DIR / "e5_differential_lr"
OUTPUT_CKPT_DIR = E5_DIR / "checkpoints"
OUTPUT_REPORTS_DIR = E5_DIR / "reports"
OUTPUT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 16
MAX_EPOCHS = 8
PATIENCE = 4
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

def verify_inputs():
    print("=" * 80)
    print("STEP 1: VERIFYING INPUTS AND DATA INTEGRITY")
    print("=" * 80)
    
    if not MASTER_MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Missing master manifest: {MASTER_MANIFEST_CSV}")
    if not RECOVERED_FOLDS_CSV.exists():
        raise FileNotFoundError(f"Missing recovered folds: {RECOVERED_FOLDS_CSV}")
    if not SPLIT_VALIDATION_REPORT_JSON.exists():
        raise FileNotFoundError(f"Missing split validation report: {SPLIT_VALIDATION_REPORT_JSON}")
    if not E0_REPORT_JSON.exists():
        raise FileNotFoundError(f"Missing E0 report: {E0_REPORT_JSON}")

    df_master = pd.read_csv(MASTER_MANIFEST_CSV)
    df_folds = pd.read_csv(RECOVERED_FOLDS_CSV)

    assert len(df_folds) == 1474, f"Expected 1,474 scans, got {len(df_folds)}"
    uf_count = len(df_folds[df_folds["verified_class"] == "UF"])
    nuf_count = len(df_folds[df_folds["verified_class"] == "NUF"])
    assert uf_count == 685, f"Expected 685 UF, got {uf_count}"
    assert nuf_count == 789, f"Expected 789 NUF, got {nuf_count}"

    # Verify fold distribution
    fold_counts = df_folds["fold"].value_counts().sort_index().to_dict()
    print(f"Verified 1,474 development scans ({uf_count} UF, {nuf_count} NUF).")
    print(f"Per-fold scan counts: {fold_counts}")

    # Check that quarantined fetal scans (EXCLUDE_FETUS_VISIBLE) are 100% absent
    fetal_scans = set(df_master[df_master["new_disposition"] == "EXCLUDE_FETUS_VISIBLE"]["canonical_image_id"])
    assert len(fetal_scans) == 64, f"Expected 64 fetal scans, found {len(fetal_scans)}"
    fetal_overlap = fetal_scans.intersection(set(df_folds["canonical_image_id"]))
    assert len(fetal_overlap) == 0, f"Error: {len(fetal_overlap)} fetal scans leaked into development folds!"

    # Check that held-out evaluation scans are 100% absent
    held_out_scans = set(df_master[df_master["new_disposition"] == "HELD_OUT_EVALUATION_ONLY"]["canonical_image_id"])
    assert len(held_out_scans) == 456, f"Expected 456 held out scans, found {len(held_out_scans)}"
    held_out_overlap = held_out_scans.intersection(set(df_folds["canonical_image_id"]))
    assert len(held_out_overlap) == 0, f"Error: {len(held_out_overlap)} held-out scans leaked into development folds!"

    # Check zero study group and duplicate family leakage across folds
    for i in range(5):
        gi = set(df_folds[df_folds["fold"] == i]["study_proxy_group"])
        fi = set(df_folds[df_folds["fold"] == i]["family_id"])
        for j in range(i + 1, 5):
            gj = set(df_folds[df_folds["fold"] == j]["study_proxy_group"])
            fj = set(df_folds[df_folds["fold"] == j]["family_id"])
            assert len(gi.intersection(gj)) == 0, f"Study group leakage between Fold {i} and Fold {j}"
            assert len(fi.intersection(fj)) == 0, f"Duplicate family leakage between Fold {i} and Fold {j}"

    config_and_validation = {
        "status": "INPUTS_VERIFIED",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_development_scans": len(df_folds),
        "uf_count": uf_count,
        "nuf_count": nuf_count,
        "fold_counts": fold_counts,
        "distinct_study_groups": int(df_folds["study_proxy_group"].nunique()),
        "distinct_duplicate_families": int(df_folds["family_id"].nunique()),
        "quarantined_fetal_scans_leakage": 0,
        "held_out_evaluation_leakage": 0,
        "cross_fold_study_group_leakage": 0,
        "cross_fold_duplicate_family_leakage": 0,
        "differential_lrs": {
            "conv1_bn1_layer1_layer2": 1e-5,
            "layer3_layer4": 5e-5,
            "fc": 1e-4
        },
        "optimizer": "Adam",
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "seed": SEED,
        "device": str(DEVICE)
    }

    config_json_path = OUTPUT_REPORTS_DIR / "e5_config_and_split_validation.json"
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(config_and_validation, f, indent=2)
    print(f"Saved configuration and validation report to: {config_json_path.name}")
    print("ALL INPUT VERIFICATION CHECKS PASSED.")
    return df_folds

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

        # Standard Clinical Preprocessing (identical to E0)
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

def build_differential_optimizer(model):
    """
    Param groups:
      - conv1, bn1, layer1, layer2: 1e-5
      - layer3, layer4: 5e-5
      - fc classifier: 1e-4
    """
    group1 = [
        *model.conv1.parameters(),
        *model.bn1.parameters(),
        *model.layer1.parameters(),
        *model.layer2.parameters(),
    ]
    group2 = [
        *model.layer3.parameters(),
        *model.layer4.parameters(),
    ]
    group3 = list(model.fc.parameters())

    optimizer = torch.optim.Adam([
        {"params": group1, "lr": 1e-5},
        {"params": group2, "lr": 5e-5},
        {"params": group3, "lr": 1e-4}
    ], weight_decay=WEIGHT_DECAY)

    return optimizer

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

def run_experiment_e5(df_folds):
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING EXPERIMENT E5 (DIFFERENTIAL LEARNING RATES)")
    print("=" * 80)
    seed_everything(SEED)

    criterion = nn.CrossEntropyLoss()
    fold_summaries = []
    all_outer_preds = []
    epoch_metrics_log = []

    for outer_fold in range(5):
        inner_val_fold = (outer_fold + 1) % 5
        inner_train_folds = [f for f in range(5) if f != outer_fold and f != inner_val_fold]
        print("\n" + "-" * 75)
        print(f"[E5 DIFF-LR] OUTER FOLD {outer_fold}/4 | Inner Val: Fold {inner_val_fold} | Inner Train: Folds {inner_train_folds}")
        print("-" * 75)

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
        optimizer = build_differential_optimizer(model)

        best_val_loss = float("inf")
        best_loss_state = None
        best_loss_ep = -1

        best_val_acc = -1.0
        best_acc_loss_tiebreaker = float("inf")
        best_acc_state = None
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
            # Rule 1: Min validation loss
            if val_m["loss"] < best_val_loss:
                best_val_loss = val_m["loss"]
                best_loss_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_loss_ep = epoch
                patience_counter = 0
                flag += "[*] Min Loss "
            else:
                patience_counter += 1

            # Rule 2: Max validation accuracy with loss tie-breaker
            is_best_acc = False
            if val_m["accuracy"] > best_val_acc:
                is_best_acc = True
            elif abs(val_m["accuracy"] - best_val_acc) < 1e-7 and val_m["loss"] < best_acc_loss_tiebreaker:
                is_best_acc = True

            if is_best_acc:
                best_val_acc = val_m["accuracy"]
                best_acc_loss_tiebreaker = val_m["loss"]
                best_acc_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_acc_ep = epoch
                flag += "[^] Max Acc"

            print(f"  Epoch {epoch:02d}/{MAX_EPOCHS:02d} ({ep_dur:.1f}s) | Train Loss: {train_loss:.4f} | Inner Val Loss: {val_m['loss']:.4f} | Acc: {val_m['accuracy']*100:.1f}% | Sens: {val_m['sensitivity']*100:.1f}% | Spec: {val_m['specificity']*100:.1f}% | AUC: {val_m['roc_auc']:.4f} | {flag}")

            epoch_metrics_log.append({
                "outer_fold": outer_fold,
                "epoch": epoch,
                "duration_s": round(ep_dur, 2),
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_m["loss"], 4),
                "val_accuracy": round(val_m["accuracy"], 4),
                "val_balanced_accuracy": round(val_m["balanced_accuracy"], 4),
                "val_sensitivity": round(val_m["sensitivity"], 4),
                "val_specificity": round(val_m["specificity"], 4),
                "val_roc_auc": round(val_m["roc_auc"], 4),
                "val_pr_auc": round(val_m["pr_auc"], 4),
                "is_min_loss": (epoch == best_loss_ep),
                "is_max_acc": (epoch == best_acc_ep)
            })

            if patience_counter >= PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

        # Save E5 checkpoints
        ckpt_loss_path = OUTPUT_CKPT_DIR / f"e5_fold_{outer_fold}_best_loss.pth"
        torch.save({"model_state": best_loss_state, "epoch": best_loss_ep, "inner_val_loss": best_val_loss}, ckpt_loss_path)

        ckpt_acc_path = OUTPUT_CKPT_DIR / f"e5_fold_{outer_fold}_best_acc.pth"
        torch.save({"model_state": best_acc_state, "epoch": best_acc_ep, "inner_val_acc": best_val_acc, "tiebreaker_loss": best_acc_loss_tiebreaker}, ckpt_acc_path)

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
        
        pred_csv_path = OUTPUT_REPORTS_DIR / f"preds_e5_outer_fold_{outer_fold}.csv"
        df_outer_preds.to_csv(pred_csv_path, index=False)
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

    # Save epoch metrics CSV
    pd.DataFrame(epoch_metrics_log).to_csv(OUTPUT_REPORTS_DIR / "epoch_metrics_e5.csv", index=False)

    # Pooled analysis across all 1,474 outer evaluations
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

    # Load E0 baseline report for paired comparison
    with open(E0_REPORT_JSON, "r", encoding="utf-8") as f:
        e0_data = json.load(f)

    # Paired accuracy differences across folds
    paired_diffs_r1 = []
    paired_diffs_r2 = []
    for f in range(5):
        e0_acc_r1 = e0_data["per_fold_details"][f]["rule_1_min_loss"]["accuracy"]
        e5_acc_r1 = fold_summaries[f]["rule_1_min_loss"]["accuracy"]
        paired_diffs_r1.append(e5_acc_r1 - e0_acc_r1)

        e0_acc_r2 = e0_data["per_fold_details"][f]["rule_2_max_acc"]["accuracy"]
        e5_acc_r2 = fold_summaries[f]["rule_2_max_acc"]["accuracy"]
        paired_diffs_r2.append(e5_acc_r2 - e0_acc_r2)

    paired_comparison = {
        "rule_1_min_loss": {
            "per_fold_diff": paired_diffs_r1,
            "mean_diff": float(np.mean(paired_diffs_r1)),
            "std_diff": float(np.std(paired_diffs_r1)),
            "e0_mean_accuracy": e0_data["rule_1_min_loss"]["mean_accuracy"],
            "e5_mean_accuracy": summary_r1["mean_accuracy"],
            "e0_pooled_accuracy": e0_data["rule_1_min_loss"]["pooled"]["accuracy"],
            "e5_pooled_accuracy": summary_r1["pooled"]["accuracy"]
        },
        "rule_2_max_acc": {
            "per_fold_diff": paired_diffs_r2,
            "mean_diff": float(np.mean(paired_diffs_r2)),
            "std_diff": float(np.std(paired_diffs_r2)),
            "e0_mean_accuracy": e0_data["rule_2_max_acc"]["mean_accuracy"],
            "e5_mean_accuracy": summary_r2["mean_accuracy"],
            "e0_pooled_accuracy": e0_data["rule_2_max_acc"]["pooled"]["accuracy"],
            "e5_pooled_accuracy": summary_r2["pooled"]["accuracy"]
        }
    }

    full_e5_report = {
        "benchmark": "EXPERIMENT_E5_DIFFERENTIAL_LEARNING_RATES",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "5-Fold Nested Grouped Cross-Validation on Recovered Dataset",
        "dataset_summary": {
            "total_development_scans": len(df_folds),
            "uf_count": int(len(df_folds[df_folds["verified_class"] == "UF"])),
            "nuf_count": int(len(df_folds[df_folds["verified_class"] == "NUF"])),
            "study_groups": int(df_folds["study_proxy_group"].nunique())
        },
        "hyperparameters": {
            "differential_lrs": {
                "conv1_bn1_layer1_layer2": 1e-5,
                "layer3_layer4": 5e-5,
                "fc": 1e-4
            },
            "optimizer": "Adam",
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "seed": SEED
        },
        "rule_1_min_loss": summary_r1,
        "rule_2_max_acc": summary_r2,
        "per_fold_details": fold_summaries,
        "paired_comparison_vs_e0": paired_comparison
    }

    report_json_path = OUTPUT_REPORTS_DIR / "e5_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_e5_report, f, indent=2)
    print(f"\nSaved full E5 JSON report to: {report_json_path.name}")

    # Generate Markdown Report
    generate_markdown_report(full_e5_report, e0_data)

    print("\n" + "=" * 80)
    print("EXPERIMENT E5 COMPLETE")
    print("=" * 80)
    print(f"Rule 1 (Min Loss) Mean Outer Acc: {summary_r1['mean_accuracy']*100:.2f}% ± {summary_r1['std_accuracy']*100:.2f}% (E0: {e0_data['rule_1_min_loss']['mean_accuracy']*100:.2f}%, Diff: {paired_comparison['rule_1_min_loss']['mean_diff']*100:+.2f}%)")
    print(f"Rule 2 (Max Acc)  Mean Outer Acc: {summary_r2['mean_accuracy']*100:.2f}% ± {summary_r2['std_accuracy']*100:.2f}% (E0: {e0_data['rule_2_max_acc']['mean_accuracy']*100:.2f}%, Diff: {paired_comparison['rule_2_max_acc']['mean_diff']*100:+.2f}%)")

def generate_markdown_report(e5, e0):
    report_md_path = E5_DIR / "e5_report.md"
    
    r1_e5 = e5["rule_1_min_loss"]
    r2_e5 = e5["rule_2_max_acc"]
    r1_e0 = e0["rule_1_min_loss"]
    r2_e0 = e0["rule_2_max_acc"]
    cmp_r1 = e5["paired_comparison_vs_e0"]["rule_1_min_loss"]
    cmp_r2 = e5["paired_comparison_vs_e0"]["rule_2_max_acc"]

    md = f"""# UteroScan — Experiment E5: Differential Learning Rates on Recovered Dataset Report

**Date:** {e5["timestamp"]}  
**Protocol:** 5-Fold Nested Grouped Cross-Validation on Recovered Dataset (1,474 Clinical Development Scans)  
**Output Directory:** `experiments/final_data_audit/e5_differential_lr/`  

---

## 1. Executive Summary & Objective

The objective of **Experiment E5** is to evaluate whether using **differential layer-wise learning rates** improves upon the audited **Recovered E0 ResNet50 baseline** on the 1,474-image clinical development dataset.

### Differential Learning Rate Architecture
- **Early Feature Layers (`conv1`, `bn1`, `layer1`, `layer2`):** Learning rate **`1e-5`** (preserves low-level edge, texture, and acoustic attenuation filters learned from ImageNet).
- **Mid/High Feature Layers (`layer3`, `layer4`):** Learning rate **`5e-5`** (allows moderate domain adaptation for complex uterine tissue patterns and fibroid capsule boundaries).
- **Classifier Head (`fc`):** Learning rate **`1e-4`** (enables rapid task-specific optimization for binary UF vs. NUF classification).
- **Controlled Baseline Discipline:** Input resolution (224×224 full-frame), preprocessing (CLAHE + median blur), optimizer (Adam), weight decay (`1e-4`), batch size (16), seed (42), max epochs (8), and early stopping patience (4) were kept **100% identical to E0**. No auxiliary heads, CBAM, photometric jitter, synthetic calipers, or cropping were added.

---

## 2. Key Findings & E5 vs. E0 Comparison

### Head-to-Head Comparison: Recovered E0 Baseline vs. E5 Differential LR

| Metric | Recovered E0 (Rule 1 Min Loss) | E5 Diff-LR (Rule 1 Min Loss) | Rule 1 Diff | Recovered E0 (Rule 2 Max Acc) | E5 Diff-LR (Rule 2 Max Acc) | Rule 2 Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Outer Accuracy** | {r1_e0["mean_accuracy"]*100:.2f}% ± {r1_e0["std_accuracy"]*100:.2f}% | **{r1_e5["mean_accuracy"]*100:.2f}% ± {r1_e5["std_accuracy"]*100:.2f}%** | **{cmp_r1["mean_diff"]*100:+.2f}%** | {r2_e0["mean_accuracy"]*100:.2f}% ± {r2_e0["std_accuracy"]*100:.2f}% | **{r2_e5["mean_accuracy"]*100:.2f}% ± {r2_e5["std_accuracy"]*100:.2f}%** | **{cmp_r2["mean_diff"]*100:+.2f}%** |
| **Pooled Accuracy (1,474)** | {r1_e0["pooled"]["accuracy"]*100:.2f}% | **{r1_e5["pooled"]["accuracy"]*100:.2f}%** | **{(r1_e5["pooled"]["accuracy"] - r1_e0["pooled"]["accuracy"])*100:+.2f}%** | {r2_e0["pooled"]["accuracy"]*100:.2f}% | **{r2_e5["pooled"]["accuracy"]*100:.2f}%** | **{(r2_e5["pooled"]["accuracy"] - r2_e0["pooled"]["accuracy"])*100:+.2f}%** |
| **Mean Balanced Accuracy** | {r1_e0["mean_balanced_accuracy"]*100:.2f}% | **{r1_e5["mean_balanced_accuracy"]*100:.2f}%** | {(r1_e5["mean_balanced_accuracy"] - r1_e0["mean_balanced_accuracy"])*100:+.2f}% | {r2_e0["mean_balanced_accuracy"]*100:.2f}% | **{r2_e5["mean_balanced_accuracy"]*100:.2f}%** | {(r2_e5["mean_balanced_accuracy"] - r2_e0["mean_balanced_accuracy"])*100:+.2f}% |
| **Mean UF Sensitivity** | {r1_e0["mean_sensitivity"]*100:.2f}% | **{r1_e5["mean_sensitivity"]*100:.2f}%** | {(r1_e5["mean_sensitivity"] - r1_e0["mean_sensitivity"])*100:+.2f}% | {r2_e0["mean_sensitivity"]*100:.2f}% | **{r2_e5["mean_sensitivity"]*100:.2f}%** | {(r2_e5["mean_sensitivity"] - r2_e0["mean_sensitivity"])*100:+.2f}% |
| **Mean NUF Specificity** | {r1_e0["mean_specificity"]*100:.2f}% | **{r1_e5["mean_specificity"]*100:.2f}%** | {(r1_e5["mean_specificity"] - r1_e0["mean_specificity"])*100:+.2f}% | {r2_e0["mean_specificity"]*100:.2f}% | **{r2_e5["mean_specificity"]*100:.2f}%** | {(r2_e5["mean_specificity"] - r2_e0["mean_specificity"])*100:+.2f}% |
| **Mean UF Precision** | {r1_e0["mean_precision"]*100:.2f}% | **{r1_e5["mean_precision"]*100:.2f}%** | {(r1_e5["mean_precision"] - r1_e0["mean_precision"])*100:+.2f}% | {r2_e0["mean_precision"]*100:.2f}% | **{r2_e5["mean_precision"]*100:.2f}%** | {(r2_e5["mean_precision"] - r2_e0["mean_precision"])*100:+.2f}% |
| **Mean UF F1 Score** | {r1_e0["mean_f1"]:.4f} | **{r1_e5["mean_f1"]:.4f}** | {r1_e5["mean_f1"] - r1_e0["mean_f1"]:+.4f} | {r2_e0["mean_f1"]:.4f} | **{r2_e5["mean_f1"]:.4f}** | {r2_e5["mean_f1"] - r2_e0["mean_f1"]:+.4f} |
| **Mean ROC-AUC** | {r1_e0["mean_roc_auc"]:.4f} | **{r1_e5["mean_roc_auc"]:.4f}** | {r1_e5["mean_roc_auc"] - r1_e0["mean_roc_auc"]:+.4f} | {r2_e0["mean_roc_auc"]:.4f} | **{r2_e5["mean_roc_auc"]:.4f}** | {r2_e5["mean_roc_auc"] - r2_e0["mean_roc_auc"]:+.4f} |
| **Mean PR-AUC** | {r1_e0["mean_pr_auc"]:.4f} | **{r1_e5["mean_pr_auc"]:.4f}** | {r1_e5["mean_pr_auc"] - r1_e0["mean_pr_auc"]:+.4f} | {r2_e0["mean_pr_auc"]:.4f} | **{r2_e5["mean_pr_auc"]:.4f}** | {r2_e5["mean_pr_auc"] - r2_e0["mean_pr_auc"]:+.4f} |

---

## 3. Outer Fold-by-Fold Detailed Breakdown

### Rule 1: Checkpoint Selected by Minimum Validation Loss

| Outer Fold | Selected Epoch | Outer Accuracy | Balanced Acc | UF Sensitivity | NUF Specificity | Precision | F1 Score | ROC-AUC | PR-AUC | E0 Outer Acc | Acc Difference |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for f in range(5):
        d_e5 = e5["per_fold_details"][f]["rule_1_min_loss"]
        d_e0 = e0["per_fold_details"][f]["rule_1_min_loss"]
        diff = d_e5["accuracy"] - d_e0["accuracy"]
        md += f"| **Fold {f}** | Ep {d_e5['epoch']} | {d_e5['accuracy']*100:.2f}% | {d_e5['balanced_accuracy']*100:.2f}% | {d_e5['sensitivity']*100:.2f}% | {d_e5['specificity']*100:.2f}% | {d_e5['precision']*100:.2f}% | {d_e5['f1']:.4f} | {d_e5['roc_auc']:.4f} | {d_e5['pr_auc']:.4f} | {d_e0['accuracy']*100:.2f}% | **{diff*100:+.2f}%** |\n"

    md += f"""| **Mean ± Std** | — | **{r1_e5['mean_accuracy']*100:.2f}% ± {r1_e5['std_accuracy']*100:.2f}%** | **{r1_e5['mean_balanced_accuracy']*100:.2f}%** | **{r1_e5['mean_sensitivity']*100:.2f}%** | **{r1_e5['mean_specificity']*100:.2f}%** | **{r1_e5['mean_precision']*100:.2f}%** | **{r1_e5['mean_f1']:.4f}** | **{r1_e5['mean_roc_auc']:.4f}** | **{r1_e5['mean_pr_auc']:.4f}** | {r1_e0['mean_accuracy']*100:.2f}% | **{cmp_r1['mean_diff']*100:+.2f}%** |
| **Pooled (1,474)** | — | **{r1_e5['pooled']['accuracy']*100:.2f}%** | **{r1_e5['pooled']['balanced_accuracy']*100:.2f}%** | **{r1_e5['pooled']['sensitivity']*100:.2f}%** | **{r1_e5['pooled']['specificity']*100:.2f}%** | **{r1_e5['pooled']['precision']*100:.2f}%** | **{r1_e5['pooled']['f1']:.4f}** | **{r1_e5['pooled']['roc_auc']:.4f}** | **{r1_e5['pooled']['pr_auc']:.4f}** | {r1_e0['pooled']['accuracy']*100:.2f}% | **{(r1_e5['pooled']['accuracy'] - r1_e0['pooled']['accuracy'])*100:+.2f}%** |

**Rule 1 Pooled Confusion Matrix:**  
- **True Positives (UF):** {r1_e5['pooled']['tp']}  
- **False Negatives (UF missed):** {r1_e5['pooled']['fn']}  
- **False Positives (NUF called UF):** {r1_e5['pooled']['fp']}  
- **True Negatives (NUF):** {r1_e5['pooled']['tn']}  

---

### Rule 2: Checkpoint Selected by Maximum Validation Accuracy (with Loss Tie-Breaker)

| Outer Fold | Selected Epoch | Outer Accuracy | Balanced Acc | UF Sensitivity | NUF Specificity | Precision | F1 Score | ROC-AUC | PR-AUC | E0 Outer Acc | Acc Difference |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for f in range(5):
        d_e5 = e5["per_fold_details"][f]["rule_2_max_acc"]
        d_e0 = e0["per_fold_details"][f]["rule_2_max_acc"]
        diff = d_e5["accuracy"] - d_e0["accuracy"]
        md += f"| **Fold {f}** | Ep {d_e5['epoch']} | {d_e5['accuracy']*100:.2f}% | {d_e5['balanced_accuracy']*100:.2f}% | {d_e5['sensitivity']*100:.2f}% | {d_e5['specificity']*100:.2f}% | {d_e5['precision']*100:.2f}% | {d_e5['f1']:.4f} | {d_e5['roc_auc']:.4f} | {d_e5['pr_auc']:.4f} | {d_e0['accuracy']*100:.2f}% | **{diff*100:+.2f}%** |\n"

    md += f"""| **Mean ± Std** | — | **{r2_e5['mean_accuracy']*100:.2f}% ± {r2_e5['std_accuracy']*100:.2f}%** | **{r2_e5['mean_balanced_accuracy']*100:.2f}%** | **{r2_e5['mean_sensitivity']*100:.2f}%** | **{r2_e5['mean_specificity']*100:.2f}%** | **{r2_e5['mean_precision']*100:.2f}%** | **{r2_e5['mean_f1']:.4f}** | **{r2_e5['mean_roc_auc']:.4f}** | **{r2_e5['mean_pr_auc']:.4f}** | {r2_e0['mean_accuracy']*100:.2f}% | **{cmp_r2['mean_diff']*100:+.2f}%** |
| **Pooled (1,474)** | — | **{r2_e5['pooled']['accuracy']*100:.2f}%** | **{r2_e5['pooled']['balanced_accuracy']*100:.2f}%** | **{r2_e5['pooled']['sensitivity']*100:.2f}%** | **{r2_e5['pooled']['specificity']*100:.2f}%** | **{r2_e5['pooled']['precision']*100:.2f}%** | **{r2_e5['pooled']['f1']:.4f}** | **{r2_e5['pooled']['roc_auc']:.4f}** | **{r2_e5['pooled']['pr_auc']:.4f}** | {r2_e0['pooled']['accuracy']*100:.2f}% | **{(r2_e5['pooled']['accuracy'] - r2_e0['pooled']['accuracy'])*100:+.2f}%** |

**Rule 2 Pooled Confusion Matrix:**  
- **True Positives (UF):** {r2_e5['pooled']['tp']}  
- **False Negatives (UF missed):** {r2_e5['pooled']['fn']}  
- **False Positives (NUF called UF):** {r2_e5['pooled']['fp']}  
- **True Negatives (NUF):** {r2_e5['pooled']['tn']}  

---

## 4. Scientific Recommendation & Next Actions

1. **Performance Verdict:** Compare the mean cross-validated accuracy and paired differences across all 5 outer folds.
2. **Clinical Trade-offs:** Check whether sensitivity or specificity experienced any degradation.
3. **Model Selection Decision:** Explicitly state whether E5 should replace E0 as the official development baseline or remain an informative ablation.
4. **Preservation:** All E0 baseline checkpoints, reports, and manifests remain intact.
"""

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved full Markdown report to: {report_md_path.name}")

if __name__ == "__main__":
    df_folds = verify_inputs()
    run_experiment_e5(df_folds)
