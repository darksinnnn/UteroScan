#!/usr/bin/env python3
"""
Phase 4: Evaluation and Reporting for Clean V5 E0 Baseline
Location: experiments/v4_controlled/src/evaluate_and_report_clean_v5.py

Generates:
1. clean_v5_e0_report.json
2. clean_v5_e0_report.md
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("d:/Project/UteroScan")
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "v4_controlled" / "src"))

from dataset import ControlledUltrasoundDataset
from models import ResNet50Baseline
from metrics import (
    compute_roc_auc,
    compute_pr_auc,
    evaluate_metrics_at_threshold,
    optimize_threshold_on_val
)

FOLDS_V5_CSV = PROJECT_ROOT / "experiments/v4_controlled/data/clean_v5/folds_clean_v5.csv"
CLEAN_MANIFEST_CSV = PROJECT_ROOT / "experiments/v4_controlled/data/clean_v5/clean_manifest.csv"
EXCLUSION_SUMMARY_JSON = PROJECT_ROOT / "experiments/v4_controlled/data/clean_v5/exclusion_summary.json"
CHECKPOINTS_DIR = PROJECT_ROOT / "experiments/v4_controlled/checkpoints/clean_v5_e0"
REPORTS_DIR = PROJECT_ROOT / "experiments/v4_controlled/reports"

TEST_LINES_FILE = PROJECT_ROOT / "DCGAN&MobileNetV2/DCGAN&MobileNetV2/cls_test.txt"

def evaluate_fold_checkpoint(ckpt_path, val_df, device):
    val_ds = ControlledUltrasoundDataset(
        val_df,
        input_size=(224, 224),
        is_train=False,
        use_viewport=False,
        use_caliper_aug=False,
        project_root=str(PROJECT_ROOT)
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    
    model = ResNet50Baseline(num_classes=2, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    val_preds_uf = []
    val_targets = []
    val_stems = []
    
    with torch.no_grad():
        for imgs, targets, _, _, stems in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            val_preds_uf.extend(probs[:, 0].cpu().numpy().tolist())
            val_targets.extend(targets.cpu().numpy().tolist())
            val_stems.extend(stems)
            
    y_true = (np.array(val_targets) == 0).astype(int) # 1=UF, 0=NUF
    p_uf = np.array(val_preds_uf)
    return y_true, p_uf, val_stems

def generate_report():
    print("=" * 80)
    print("PHASE 4: EVALUATION & REPORTING FOR CLEAN V5 E0 BASELINE")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_folds = pd.read_csv(FOLDS_V5_CSV)
    
    with open(EXCLUSION_SUMMARY_JSON, "r", encoding="utf-8") as f:
        excl_summary = json.load(f)
        
    # Evaluate Rule 1: Best Loss
    print("\n--- Evaluating Best-Loss Checkpoints ---")
    loss_fold_metrics = []
    oof_loss_true = []
    oof_loss_prob = []
    
    for fold in range(5):
        val_df = df_folds[df_folds["fold"] == fold].copy()
        ckpt = CHECKPOINTS_DIR / f"e0_clean_v5_fold_{fold}_best_loss.pth"
        y_true, p_uf, stems = evaluate_fold_checkpoint(ckpt, val_df, device)
        
        auc = compute_roc_auc(y_true, p_uf)
        pr_auc = compute_pr_auc(y_true, p_uf)
        m = evaluate_metrics_at_threshold(y_true, p_uf, threshold_uf=0.50)
        
        loss_fold_metrics.append({
            "fold": fold,
            "roc_auc": auc,
            "pr_auc": pr_auc,
            **m
        })
        oof_loss_true.extend(y_true.tolist())
        oof_loss_prob.extend(p_uf.tolist())
        print(f"  Fold {fold} (Loss Ckpt): Acc={m['accuracy']*100:.2f}%, Sens={m['uf_sensitivity']*100:.2f}%, Spec={m['nuf_specificity']*100:.2f}%, AUC={auc:.4f}")

    # Evaluate Rule 2: Best Acc
    print("\n--- Evaluating Best-Accuracy Checkpoints ---")
    acc_fold_metrics = []
    oof_acc_true = []
    oof_acc_prob = []
    
    for fold in range(5):
        val_df = df_folds[df_folds["fold"] == fold].copy()
        ckpt = CHECKPOINTS_DIR / f"e0_clean_v5_fold_{fold}_best_acc.pth"
        y_true, p_uf, stems = evaluate_fold_checkpoint(ckpt, val_df, device)
        
        auc = compute_roc_auc(y_true, p_uf)
        pr_auc = compute_pr_auc(y_true, p_uf)
        m = evaluate_metrics_at_threshold(y_true, p_uf, threshold_uf=0.50)
        
        acc_fold_metrics.append({
            "fold": fold,
            "roc_auc": auc,
            "pr_auc": pr_auc,
            **m
        })
        oof_acc_true.extend(y_true.tolist())
        oof_acc_prob.extend(p_uf.tolist())
        print(f"  Fold {fold} (Acc Ckpt): Acc={m['accuracy']*100:.2f}%, Sens={m['uf_sensitivity']*100:.2f}%, Spec={m['nuf_specificity']*100:.2f}%, AUC={auc:.4f}")

    # Compute Pooled Out-of-Fold Metrics
    oof_loss_true = np.array(oof_loss_true)
    oof_loss_prob = np.array(oof_loss_prob)
    oof_loss_auc = compute_roc_auc(oof_loss_true, oof_loss_prob)
    oof_loss_pr_auc = compute_pr_auc(oof_loss_true, oof_loss_prob)
    oof_loss_m50 = evaluate_metrics_at_threshold(oof_loss_true, oof_loss_prob, threshold_uf=0.50)
    tau_loss_star, oof_loss_m_star = optimize_threshold_on_val(oof_loss_true, oof_loss_prob)

    oof_acc_true = np.array(oof_acc_true)
    oof_acc_prob = np.array(oof_acc_prob)
    oof_acc_auc = compute_roc_auc(oof_acc_true, oof_acc_prob)
    oof_acc_pr_auc = compute_pr_auc(oof_acc_true, oof_acc_prob)
    oof_acc_m50 = evaluate_metrics_at_threshold(oof_acc_true, oof_acc_prob, threshold_uf=0.50)
    tau_acc_star, oof_acc_m_star = optimize_threshold_on_val(oof_acc_true, oof_acc_prob)

    # Compute Summary Statistics
    def stats(f_list, key):
        vals = [f[key] for f in f_list]
        return float(np.mean(vals)), float(np.std(vals))

    mean_loss_acc, std_loss_acc = stats(loss_fold_metrics, "accuracy")
    mean_loss_sens, std_loss_sens = stats(loss_fold_metrics, "uf_sensitivity")
    mean_loss_spec, std_loss_spec = stats(loss_fold_metrics, "nuf_specificity")
    mean_loss_auc, std_loss_auc = stats(loss_fold_metrics, "roc_auc")
    mean_loss_f1, std_loss_f1 = stats(loss_fold_metrics, "f1")
    mean_loss_prec, std_loss_prec = stats(loss_fold_metrics, "uf_precision")

    mean_acc_acc, std_acc_acc = stats(acc_fold_metrics, "accuracy")
    mean_acc_sens, std_acc_sens = stats(acc_fold_metrics, "uf_sensitivity")
    mean_acc_spec, std_acc_spec = stats(acc_fold_metrics, "nuf_specificity")
    mean_acc_auc, std_acc_auc = stats(acc_fold_metrics, "roc_auc")
    mean_acc_f1, std_acc_f1 = stats(acc_fold_metrics, "f1")
    mean_acc_prec, std_acc_prec = stats(acc_fold_metrics, "uf_precision")

    # Save JSON Report
    report_json_data = {
        "experiment": "E0_CLEAN_V5_BASELINE",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_integrity": excl_summary,
        "selection_rule_1_min_loss": {
            "mean_cv_accuracy": mean_loss_acc,
            "std_cv_accuracy": std_loss_acc,
            "mean_cv_sensitivity": mean_loss_sens,
            "std_cv_sensitivity": std_loss_sens,
            "mean_cv_specificity": mean_loss_spec,
            "std_cv_specificity": std_loss_spec,
            "mean_cv_roc_auc": mean_loss_auc,
            "std_cv_roc_auc": std_loss_auc,
            "mean_cv_f1": mean_loss_f1,
            "std_cv_f1": std_loss_f1,
            "mean_cv_precision": mean_loss_prec,
            "std_cv_precision": std_loss_prec,
            "pooled_oof_metrics_t50": {
                "roc_auc": oof_loss_auc,
                "pr_auc": oof_loss_pr_auc,
                **oof_loss_m50
            },
            "validation_optimized_threshold": {
                "tau_star": tau_loss_star,
                **oof_loss_m_star
            },
            "per_fold_metrics": loss_fold_metrics
        },
        "selection_rule_2_max_acc": {
            "mean_cv_accuracy": mean_acc_acc,
            "std_cv_accuracy": std_acc_acc,
            "mean_cv_sensitivity": mean_acc_sens,
            "std_cv_sensitivity": std_acc_sens,
            "mean_cv_specificity": mean_acc_spec,
            "std_cv_specificity": std_acc_spec,
            "mean_cv_roc_auc": mean_acc_auc,
            "std_cv_roc_auc": std_acc_auc,
            "mean_cv_f1": mean_acc_f1,
            "std_cv_f1": std_acc_f1,
            "mean_cv_precision": mean_acc_prec,
            "std_cv_precision": std_acc_prec,
            "pooled_oof_metrics_t50": {
                "roc_auc": oof_acc_auc,
                "pr_auc": oof_acc_pr_auc,
                **oof_acc_m50
            },
            "validation_optimized_threshold": {
                "tau_star": tau_acc_star,
                **oof_acc_m_star
            },
            "per_fold_metrics": acc_fold_metrics
        }
    }

    out_json = REPORTS_DIR / "clean_v5_e0_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_json_data, f, indent=2)
    print(f"\nSaved clean V5 E0 JSON report to {out_json}")

    # Generate Markdown Report
    out_md = REPORTS_DIR / "clean_v5_e0_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# UteroScan V4 — Clean V5 E0 Baseline Retraining Report\n\n")
        f.write(f"**Date:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`  \n")
        f.write(f"**Model:** ResNet50 Full-Frame Baseline (Pretrained ImageNet)  \n")
        f.write(f"**Dataset:** Clean V5 Sanitized Development Cohort (Zero Doppler & Zero Obstetric Scans)  \n\n")

        f.write("## 1. Executive Summary\n\n")
        f.write(f"- **Clean Training Cohort Size:** {excl_summary['overall_clean_manifest_distribution']['clean_train_total']} scans ")
        f.write(f"({excl_summary['overall_clean_manifest_distribution']['clean_train_uf']} UF, {excl_summary['overall_clean_manifest_distribution']['clean_train_nuf']} NUF)  \n")
        f.write(f"- **Total New Exclusions Quarantined:** {excl_summary['confirmed_out_of_scope_count']} scans ")
        f.write(f"({excl_summary['new_exclusions_by_class']['UF']} UF, {excl_summary['new_exclusions_by_class']['NUF']} NUF)  \n")
        f.write(f"- **Held-Out Test Cohort (Preserved Read-Only):** {excl_summary['overall_clean_manifest_distribution']['held_out_test_total']} scans  \n\n")

        f.write("### Headline Performance Comparison\n\n")
        f.write("| Model & Selection Rule | Mean Fold Accuracy | UF Sensitivity | NUF Specificity | Mean ROC-AUC |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Clean V5 E0 (Rule 1: Min Loss)** | **{mean_loss_acc*100:.2f}% $\\pm$ {std_loss_acc*100:.2f}%** | **{mean_loss_sens*100:.2f}%** | **{mean_loss_spec*100:.2f}%** | **{mean_loss_auc:.4f}** |\n")
        f.write(f"| **Clean V5 E0 (Rule 2: Max Acc)** | **{mean_acc_acc*100:.2f}% $\\pm$ {std_acc_acc*100:.2f}%** | **{mean_acc_sens*100:.2f}%** | **{mean_acc_spec*100:.2f}%** | **{mean_acc_auc:.4f}** |\n")
        f.write("| Historical E0 (Unsanitized, 1,224 scans) | 95.24% $\\pm$ 1.06% | 97.57% | 92.51% | 0.9875 |\n")
        f.write("| Historical E2 (Caliper, 1,224 scans) | 93.97% $\\pm$ 2.02% | 96.19% | 91.18% | 0.9859 |\n\n")

        f.write("## 2. Per-Fold Confusion Matrices (Clean V5 E0)\n\n")
        f.write("### Rule 1: Minimum Validation Loss Checkpoints\n\n")
        f.write("| Fold | Total | Accuracy | Sensitivity | Specificity | TP (UF) | FN (UF) | TN (NUF) | FP (NUF) | ROC-AUC |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for m in loss_fold_metrics:
            tot = m['tp'] + m['fn'] + m['tn'] + m['fp']
            f.write(f"| Fold {m['fold']} | {tot} | {m['accuracy']*100:.2f}% | {m['uf_sensitivity']*100:.2f}% | {m['nuf_specificity']*100:.2f}% | {m['tp']} | {m['fn']} | {m['tn']} | {m['fp']} | {m['roc_auc']:.4f} |\n")

        f.write("\n### Rule 2: Maximum Validation Accuracy Checkpoints\n\n")
        f.write("| Fold | Total | Accuracy | Sensitivity | Specificity | TP (UF) | FN (UF) | TN (NUF) | FP (NUF) | ROC-AUC |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for m in acc_fold_metrics:
            tot = m['tp'] + m['fn'] + m['tn'] + m['fp']
            f.write(f"| Fold {m['fold']} | {tot} | {m['accuracy']*100:.2f}% | {m['uf_sensitivity']*100:.2f}% | {m['nuf_specificity']*100:.2f}% | {m['tp']} | {m['fn']} | {m['tn']} | {m['fp']} | {m['roc_auc']:.4f} |\n")

        f.write("\n## 3. Pooled Out-of-Fold Analysis\n\n")
        f.write(f"- **Fixed Threshold ($\\tau = 0.50$):** Acc = {oof_loss_m50['accuracy']*100:.2f}%, Sens = {oof_loss_m50['uf_sensitivity']*100:.2f}%, Spec = {oof_loss_m50['nuf_specificity']*100:.2f}%, AUC = {oof_loss_auc:.4f}, PR-AUC = {oof_loss_pr_auc:.4f}\n")
        f.write(f"- **Validation-Optimized Threshold ($\\tau^* = {tau_loss_star:.2f}$):** Acc = {oof_loss_m_star['accuracy']*100:.2f}%, Sens = {oof_loss_m_star['uf_sensitivity']*100:.2f}%, Spec = {oof_loss_m_star['nuf_specificity']*100:.2f}%, Youden J = {oof_loss_m_star['youden_j']:.4f}\n\n")

        f.write("## 4. Checkpoint Persistence Verification\n\n")
        f.write("All fold model weights are preserved in `experiments/v4_controlled/checkpoints/clean_v5_e0/`:\n")
        for fld in range(5):
            f.write(f"- Fold {fld} Min Loss Ckpt: `e0_clean_v5_fold_{fld}_best_loss.pth`\n")
            f.write(f"- Fold {fld} Max Acc Ckpt : `e0_clean_v5_fold_{fld}_best_acc.pth`\n")

    print(f"Saved clean V5 E0 Markdown report to {out_md}")
    return report_json_data

if __name__ == "__main__":
    generate_report()
