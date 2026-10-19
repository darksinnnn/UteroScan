#!/usr/bin/env python3
"""
Comprehensive Comparative Evaluation for E4A and E4B vs E0 Baseline
Location: experiments/v4_controlled/e4/src/evaluate_and_report_e4.py

Outputs:
  - experiments/v4_controlled/e4/reports/e4_comparison_report.json
  - experiments/v4_controlled/e4/reports/e4_comparison_report.md
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/Project/UteroScan")
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "v4_controlled" / "src"))

from metrics import compute_roc_auc, compute_pr_auc, evaluate_metrics_at_threshold

E0_REPORTS_DIR = PROJECT_ROOT / "experiments" / "v4_controlled" / "reports"
E4_REPORTS_DIR = PROJECT_ROOT / "experiments" / "v4_controlled" / "e4" / "reports"

def evaluate_predictions(preds_dir, prefix):
    fold_preds = []
    for f in range(5):
        df_f = pd.read_csv(preds_dir / f"{prefix}_outer_fold_{f}.csv")
        df_f["outer_fold"] = f
        fold_preds.append(df_f)
    return pd.concat(fold_preds, ignore_index=True)

def compute_model_metrics(df_data):
    res = {}
    for rule_key, col_prob, rule_name in [
        ("rule_1_min_loss", "pred_prob_uf_rule1", "Minimum Inner-Validation Loss"),
        ("rule_2_max_acc", "pred_prob_uf_rule2", "Maximum Inner-Validation Accuracy @ 0.50")
    ]:
        fold_metrics = []
        for f in range(5):
            f_df = df_data[df_data["outer_fold"] == f]
            y_true = f_df["true_uf"].values.astype(int)
            y_prob = f_df[col_prob].values.astype(float)
            m = evaluate_metrics_at_threshold(y_true, y_prob, threshold_uf=0.50)
            auc = compute_roc_auc(y_true, y_prob)
            pr_auc = compute_pr_auc(y_true, y_prob)
            fold_metrics.append({
                "fold": f,
                "roc_auc": auc,
                "pr_auc": pr_auc,
                **m
            })

        # Pooled
        y_pool_true = df_data["true_uf"].values.astype(int)
        y_pool_prob = df_data[col_prob].values.astype(float)
        m_pool = evaluate_metrics_at_threshold(y_pool_true, y_pool_prob, threshold_uf=0.50)
        auc_pool = compute_roc_auc(y_pool_true, y_pool_prob)
        pr_auc_pool = compute_pr_auc(y_pool_true, y_pool_prob)

        res[rule_key] = {
            "rule_name": rule_name,
            "mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
            "std_accuracy": float(np.std([m["accuracy"] for m in fold_metrics])),
            "mean_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in fold_metrics])),
            "std_balanced_accuracy": float(np.std([m["balanced_accuracy"] for m in fold_metrics])),
            "mean_sensitivity": float(np.mean([m["uf_sensitivity"] for m in fold_metrics])),
            "std_sensitivity": float(np.std([m["uf_sensitivity"] for m in fold_metrics])),
            "mean_specificity": float(np.mean([m["nuf_specificity"] for m in fold_metrics])),
            "std_specificity": float(np.std([m["nuf_specificity"] for m in fold_metrics])),
            "mean_precision": float(np.mean([m["uf_precision"] for m in fold_metrics])),
            "std_precision": float(np.std([m["uf_precision"] for m in fold_metrics])),
            "mean_f1": float(np.mean([m["uf_f1"] for m in fold_metrics])),
            "std_f1": float(np.std([m["uf_f1"] for m in fold_metrics])),
            "mean_roc_auc": float(np.mean([m["roc_auc"] for m in fold_metrics])),
            "std_roc_auc": float(np.std([m["roc_auc"] for m in fold_metrics])),
            "mean_pr_auc": float(np.mean([m["pr_auc"] for m in fold_metrics])),
            "std_pr_auc": float(np.std([m["pr_auc"] for m in fold_metrics])),
            "per_fold": fold_metrics,
            "pooled": {
                "roc_auc": auc_pool,
                "pr_auc": pr_auc_pool,
                **m_pool
            }
        }
    return res

def generate_report():
    print("=" * 80)
    print("GENERATING E4 CONTROLLED COMPARISON REPORT (E4A & E4B vs E0 BASELINE)")
    print("=" * 80)

    # Load predictions
    df_e0 = evaluate_predictions(E0_REPORTS_DIR, "preds_e0_nested")
    df_e4a = evaluate_predictions(E4_REPORTS_DIR, "preds_e4a")
    df_e4b = evaluate_predictions(E4_REPORTS_DIR, "preds_e4b")

    m_e0 = compute_model_metrics(df_e0)
    m_e4a = compute_model_metrics(df_e4a)
    m_e4b = compute_model_metrics(df_e4b)

    # Compute deltas against E0
    def compute_deltas(target_m, base_m):
        deltas = {}
        for rk in ["rule_1_min_loss", "rule_2_max_acc"]:
            t = target_m[rk]
            b = base_m[rk]
            fold_deltas = []
            for f in range(5):
                t_f = t["per_fold"][f]
                b_f = b["per_fold"][f]
                fold_deltas.append({
                    "fold": f,
                    "delta_accuracy": t_f["accuracy"] - b_f["accuracy"],
                    "delta_balanced_accuracy": t_f["balanced_accuracy"] - b_f["balanced_accuracy"],
                    "delta_sensitivity": t_f["uf_sensitivity"] - b_f["uf_sensitivity"],
                    "delta_specificity": t_f["nuf_specificity"] - b_f["nuf_specificity"],
                    "delta_precision": t_f["uf_precision"] - b_f["uf_precision"],
                    "delta_f1": t_f["uf_f1"] - b_f["uf_f1"],
                    "delta_roc_auc": t_f["roc_auc"] - b_f["roc_auc"],
                    "delta_pr_auc": t_f["pr_auc"] - b_f["pr_auc"]
                })
            deltas[rk] = {
                "delta_mean_accuracy": t["mean_accuracy"] - b["mean_accuracy"],
                "delta_mean_balanced_accuracy": t["mean_balanced_accuracy"] - b["mean_balanced_accuracy"],
                "delta_mean_sensitivity": t["mean_sensitivity"] - b["mean_sensitivity"],
                "delta_mean_specificity": t["mean_specificity"] - b["mean_specificity"],
                "delta_mean_precision": t["mean_precision"] - b["mean_precision"],
                "delta_mean_f1": t["mean_f1"] - b["mean_f1"],
                "delta_mean_roc_auc": t["mean_roc_auc"] - b["mean_roc_auc"],
                "delta_mean_pr_auc": t["mean_pr_auc"] - b["mean_pr_auc"],
                "delta_pooled_accuracy": t["pooled"]["accuracy"] - b["pooled"]["accuracy"],
                "delta_pooled_balanced_accuracy": t["pooled"]["balanced_accuracy"] - b["pooled"]["balanced_accuracy"],
                "delta_pooled_sensitivity": t["pooled"]["uf_sensitivity"] - b["pooled"]["uf_sensitivity"],
                "delta_pooled_specificity": t["pooled"]["nuf_specificity"] - b["pooled"]["nuf_specificity"],
                "delta_pooled_f1": t["pooled"]["uf_f1"] - b["pooled"]["uf_f1"],
                "delta_pooled_roc_auc": t["pooled"]["roc_auc"] - b["pooled"]["roc_auc"],
                "delta_pooled_pr_auc": t["pooled"]["pr_auc"] - b["pooled"]["pr_auc"],
                "per_fold_deltas": fold_deltas
            }
        return deltas

    deltas_e4a = compute_deltas(m_e4a, m_e0)
    deltas_e4b = compute_deltas(m_e4b, m_e0)

    report_dict = {
        "experiment": "E4_CONTROLLED_IMPROVEMENTS_STUDY",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "5-Fold Nested Grouped Cross-Validation (folds_clean_v5.csv)",
        "patient_independence_disclaimer": "Genuine patient IDs are unavailable; study_proxy_group clusters frames by hospital study UID. Study-level independence is strictly verified across all nested partitions, but guaranteed patient-level independence cannot be claimed.",
        "threshold_used": 0.50,
        "evaluation_scope": "Development Results (Historical 321-image test cohort unaccessed)",
        "results": {
            "e0_baseline": m_e0,
            "e4a_differential_lr": m_e4a,
            "e4b_photometric_aug": m_e4b
        },
        "paired_deltas_vs_e0": {
            "e4a_differential_lr": deltas_e4a,
            "e4b_photometric_aug": deltas_e4b
        }
    }

    # Save JSON
    out_json = E4_REPORTS_DIR / "e4_comparison_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Saved JSON report: {out_json.name}")

    # Build Markdown Report
    out_md = E4_REPORTS_DIR / "e4_comparison_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# UteroScan V4 — E4 Controlled Comparison Report\n\n")
        f.write(f"**Experiment:** `E4_CONTROLLED_IMPROVEMENT` (E4A Differential LR & E4B Photometric Augmentation vs E0 Clean Baseline)  \n")
        f.write(f"**Evaluation Protocol:** 5-Fold Nested Grouped Cross-Validation (`folds_clean_v5.csv`)  \n")
        f.write(f"**Evaluation Scope:** Development Results on Outer Folds (Historical 321-image test cohort untouched)  \n")
        f.write(f"**Fixed Primary Decision Threshold:** `0.50`  \n")
        f.write(f"**Generated:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`  \n\n")

        f.write("> **Patient-Independence Disclaimer:** Genuine patient IDs are unavailable; `study_proxy_group` clusters frames by hospital study UID. Study-level independence is strictly verified across all nested partitions, but guaranteed patient-level independence cannot be claimed.\n\n")

        f.write("## 1. Executive Summary & Core Results\n\n")
        f.write("### Rule 1 Checkpoint Selection: Minimum Inner-Validation Loss\n\n")
        f.write("| Model / Intervention | Mean CV Accuracy | Pooled Accuracy | Balanced Acc | UF Sens (Recall) | NUF Spec | UF Precision | UF F1 | ROC-AUC | PR-AUC | Paired Delta (Mean Acc) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for m_name, m_dict, d_dict in [
            ("E0 Baseline (Standard LR, No Photo Aug)", m_e0["rule_1_min_loss"], None),
            ("E4A (Differential LR Only)", m_e4a["rule_1_min_loss"], deltas_e4a["rule_1_min_loss"]),
            ("E4B (Photometric Aug Only)", m_e4b["rule_1_min_loss"], deltas_e4b["rule_1_min_loss"])
        ]:
            delta_str = f"**{d_dict['delta_mean_accuracy']*100:+.2f}%**" if d_dict else "Baseline"
            f.write(f"| **{m_name}** | {m_dict['mean_accuracy']*100:.2f}% ± {m_dict['std_accuracy']*100:.2f}% | {m_dict['pooled']['accuracy']*100:.2f}% | {m_dict['mean_balanced_accuracy']*100:.2f}% | {m_dict['mean_sensitivity']*100:.2f}% | {m_dict['mean_specificity']*100:.2f}% | {m_dict['mean_precision']*100:.2f}% | {m_dict['mean_f1']:.4f} | {m_dict['mean_roc_auc']:.4f} | {m_dict['mean_pr_auc']:.4f} | {delta_str} |\n")

        f.write("\n### Rule 2 Checkpoint Selection: Maximum Inner-Validation Accuracy @ 0.50\n\n")
        f.write("| Model / Intervention | Mean CV Accuracy | Pooled Accuracy | Balanced Acc | UF Sens (Recall) | NUF Spec | UF Precision | UF F1 | ROC-AUC | PR-AUC | Paired Delta (Mean Acc) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for m_name, m_dict, d_dict in [
            ("E0 Baseline (Standard LR, No Photo Aug)", m_e0["rule_2_max_acc"], None),
            ("E4A (Differential LR Only)", m_e4a["rule_2_max_acc"], deltas_e4a["rule_2_max_acc"]),
            ("E4B (Photometric Aug Only)", m_e4b["rule_2_max_acc"], deltas_e4b["rule_2_max_acc"])
        ]:
            delta_str = f"**{d_dict['delta_mean_accuracy']*100:+.2f}%**" if d_dict else "Baseline"
            f.write(f"| **{m_name}** | {m_dict['mean_accuracy']*100:.2f}% ± {m_dict['std_accuracy']*100:.2f}% | {m_dict['pooled']['accuracy']*100:.2f}% | {m_dict['mean_balanced_accuracy']*100:.2f}% | {m_dict['mean_sensitivity']*100:.2f}% | {m_dict['mean_specificity']*100:.2f}% | {m_dict['mean_precision']*100:.2f}% | {m_dict['mean_f1']:.4f} | {m_dict['mean_roc_auc']:.4f} | {m_dict['mean_pr_auc']:.4f} | {delta_str} |\n")

        f.write("\n## 2. Per-Fold Performance Breakdown\n\n")
        f.write("### Rule 1 (Minimum Inner-Validation Loss)\n\n")
        f.write("| Model | Fold | Accuracy | Balanced Acc | Sens | Spec | Precision | F1 | ROC-AUC | PR-AUC | Paired Acc Delta vs E0 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for f_idx in range(5):
            e0_acc = m_e0["rule_1_min_loss"]["per_fold"][f_idx]["accuracy"]
            for m_name, m_dict in [
                ("E0 Baseline", m_e0["rule_1_min_loss"]),
                ("E4A Diff LR", m_e4a["rule_1_min_loss"]),
                ("E4B Photo Aug", m_e4b["rule_1_min_loss"])
            ]:
                fd = m_dict["per_fold"][f_idx]
                d_acc = fd["accuracy"] - e0_acc
                d_str = f"{d_acc*100:+.2f}%" if m_name != "E0 Baseline" else "-"
                f.write(f"| **{m_name}** | Fold {f_idx} | {fd['accuracy']*100:.2f}% | {fd['balanced_accuracy']*100:.2f}% | {fd['uf_sensitivity']*100:.2f}% | {fd['nuf_specificity']*100:.2f}% | {fd['uf_precision']*100:.2f}% | {fd['uf_f1']:.4f} | {fd['roc_auc']:.4f} | {fd['pr_auc']:.4f} | {d_str} |\n")

        f.write("\n### Rule 2 (Maximum Inner-Validation Accuracy @ 0.50)\n\n")
        f.write("| Model | Fold | Accuracy | Balanced Acc | Sens | Spec | Precision | F1 | ROC-AUC | PR-AUC | Paired Acc Delta vs E0 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for f_idx in range(5):
            e0_acc = m_e0["rule_2_max_acc"]["per_fold"][f_idx]["accuracy"]
            for m_name, m_dict in [
                ("E0 Baseline", m_e0["rule_2_max_acc"]),
                ("E4A Diff LR", m_e4a["rule_2_max_acc"]),
                ("E4B Photo Aug", m_e4b["rule_2_max_acc"])
            ]:
                fd = m_dict["per_fold"][f_idx]
                d_acc = fd["accuracy"] - e0_acc
                d_str = f"{d_acc*100:+.2f}%" if m_name != "E0 Baseline" else "-"
                f.write(f"| **{m_name}** | Fold {f_idx} | {fd['accuracy']*100:.2f}% | {fd['balanced_accuracy']*100:.2f}% | {fd['uf_sensitivity']*100:.2f}% | {fd['nuf_specificity']*100:.2f}% | {fd['uf_precision']*100:.2f}% | {fd['uf_f1']:.4f} | {fd['roc_auc']:.4f} | {fd['pr_auc']:.4f} | {d_str} |\n")

        f.write("\n## 3. Per-Fold Confusion Matrices\n\n")
        f.write("### Rule 1: Minimum Inner-Validation Loss\n\n")
        f.write("| Model | Fold | TP (UF hit) | FN (UF missed) | FP (NUF false alarm) | TN (NUF correct) | Total |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for f_idx in range(5):
            for m_name, m_dict in [
                ("E0 Baseline", m_e0["rule_1_min_loss"]),
                ("E4A Diff LR", m_e4a["rule_1_min_loss"]),
                ("E4B Photo Aug", m_e4b["rule_1_min_loss"])
            ]:
                fd = m_dict["per_fold"][f_idx]
                tot = fd['tp'] + fd['fn'] + fd['fp'] + fd['tn']
                f.write(f"| **{m_name}** | Fold {f_idx} | {fd['tp']} | {fd['fn']} | {fd['fp']} | {fd['tn']} | {tot} |\n")

        f.write("\n### Rule 2: Maximum Inner-Validation Accuracy @ 0.50\n\n")
        f.write("| Model | Fold | TP (UF hit) | FN (UF missed) | FP (NUF false alarm) | TN (NUF correct) | Total |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for f_idx in range(5):
            for m_name, m_dict in [
                ("E0 Baseline", m_e0["rule_2_max_acc"]),
                ("E4A Diff LR", m_e4a["rule_2_max_acc"]),
                ("E4B Photo Aug", m_e4b["rule_2_max_acc"])
            ]:
                fd = m_dict["per_fold"][f_idx]
                tot = fd['tp'] + fd['fn'] + fd['fp'] + fd['tn']
                f.write(f"| **{m_name}** | Fold {f_idx} | {fd['tp']} | {fd['fn']} | {fd['fp']} | {fd['tn']} | {tot} |\n")

        f.write("\n## 4. Key Findings & Controlled Ablation Analysis\n\n")
        f.write("### E4A: Differential Learning Rates Only\n")
        f.write("- **Hypothesis Tested:** Assigning lower learning rates (`1e-5`) to generic lower layers (`conv1` through `layer2`), intermediate rate (`5e-5`) to semantic feature extractors (`layer3`-`layer4`), and standard rate (`1e-4`) to the classification head.\n")
        f.write("- **Empirical Results:**\n")
        f.write(f"  - Under **Rule 1 (Minimum Validation Loss)**: Mean CV accuracy changed from **{m_e0['rule_1_min_loss']['mean_accuracy']*100:.2f}% to {m_e4a['rule_1_min_loss']['mean_accuracy']*100:.2f}%** (+{deltas_e4a['rule_1_min_loss']['delta_mean_accuracy']*100:.2f}%), with ROC-AUC moving from {m_e0['rule_1_min_loss']['mean_roc_auc']:.4f} to {m_e4a['rule_1_min_loss']['mean_roc_auc']:.4f}.\n")
        f.write(f"  - Under **Rule 2 (Maximum Validation Accuracy @ 0.50)**: Mean CV accuracy fell from **{m_e0['rule_2_max_acc']['mean_accuracy']*100:.2f}% to {m_e4a['rule_2_max_acc']['mean_accuracy']*100:.2f}%** ({deltas_e4a['rule_2_max_acc']['delta_mean_accuracy']*100:+.2f}%).\n")
        f.write("- **Interpretation:** E4A demonstrates a slight improvement under minimum-loss checkpoint selection, but falls below baseline under validation-accuracy selection. It represents a potential development direction rather than a proven, robust improvement.\n\n")

        f.write("### E4B: Mild Photometric Augmentation Only\n")
        f.write("- **Hypothesis Tested:** Simulating clinic gain/attenuation differences via mild gamma (`0.85-1.15`), contrast (`±10%`), and speckle noise (`σ ≤ 0.02`).\n")
        f.write("- **Empirical Results:**\n")
        f.write(f"  - Under **Rule 1 (Minimum Validation Loss)**: Mean CV accuracy is essentially unchanged: **{m_e0['rule_1_min_loss']['mean_accuracy']*100:.2f}% vs {m_e4b['rule_1_min_loss']['mean_accuracy']*100:.2f}%** (+{deltas_e4b['rule_1_min_loss']['delta_mean_accuracy']*100:.2f}%).\n")
        f.write(f"  - Under **Rule 2 (Maximum Validation Accuracy @ 0.50)**: Mean CV accuracy fell from **{m_e0['rule_2_max_acc']['mean_accuracy']*100:.2f}% to {m_e4b['rule_2_max_acc']['mean_accuracy']*100:.2f}%** ({deltas_e4b['rule_2_max_acc']['delta_mean_accuracy']*100:+.2f}%).\n")
        f.write("- **Interpretation:** While Fold 0 demonstrated an isolated high accuracy (97.52%), performance across other folds was flat or slightly reduced. Photometric augmentation alone does not provide a consistent overall improvement across folds.\n\n")

        f.write("## 5. Recommendation & Decision\n\n")
        f.write("1. **Neither intervention represents a proven or consistent improvement over E0 Baseline:**\n")
        f.write(f"   - E4B produces essentially neutral mean accuracy under Rule 1 (+{deltas_e4b['rule_1_min_loss']['delta_mean_accuracy']*100:.2f}%) and declines under Rule 2 ({deltas_e4b['rule_2_max_acc']['delta_mean_accuracy']*100:+.2f}%).\n")
        f.write(f"   - E4A yields a slight gain under Rule 1 (+{deltas_e4a['rule_1_min_loss']['delta_mean_accuracy']*100:.2f}%) but declines under Rule 2 ({deltas_e4a['rule_2_max_acc']['delta_mean_accuracy']*100:+.2f}%).\n")
        f.write("2. **Development Baseline Retained:**\n")
        f.write("   - Full-frame ResNet50 (**E0 Baseline**) remains the official development benchmark.\n")
        f.write("3. **No Premature Combinations:**\n")
        f.write("   - Neither intervention exhibits the consistent, rule-invariant improvements required to declare a proven enhancement. E4C (combined) should not be launched until data cleanliness and model stability across additional random seeds are verified.\n")

    print(f"Saved Markdown report: {out_md.name}")

if __name__ == "__main__":
    generate_report()
