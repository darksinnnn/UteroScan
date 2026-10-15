"""
evaluate_head_to_head.py
Head-to-Head Evaluation: Original V1 Model vs. Clean V2 Baseline
Evaluates both models on the IDENTICAL held-out clean test cohort (321 scans: 163 UF / 158 NUF).

Eliminates confounding:
- Test dataset held strictly constant (321 clean images).
- Only training dataset varies (V1: 1,747 scans with contamination vs. V2: 1,545 clean scans).
- Performs McNemar's paired test to assess statistical significance of discordant predictions.
- Sweeps decision thresholds for both models to enable apples-to-apples comparisons.
- Saves results to:
    experiments/v2_contamination_cleaned/reports/v1_model_on_clean_testset.json
    experiments/v2_contamination_cleaned/reports/head_to_head_clean_test.json
    experiments/v2_contamination_cleaned/reports/head_to_head_roc_curve.png
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import binomtest, chi2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
REPORTS_DIR = EXP_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

V1_CKPT_PATH = BASE_DIR / "checkpoints" / "resnet50_stage2_best.pth"
V2_CKPT_PATH = EXP_DIR / "checkpoints" / "resnet50_stage2_best.pth"
CLEAN_TEST_FILE = ANNOTATIONS_DIR / "cls_test.txt"
V1_ORIGINAL_METRICS_PATH = BASE_DIR / "reports" / "metrics.json"

def compute_roc_curve(y_true_binary, y_scores):
    """
    Computes full ROC curve.
    y_true_binary: 1 for positive class (UF), 0 for negative class (NUF)
    y_scores: predicted probability of positive class P(UF)
    """
    desc_indices = np.argsort(-y_scores)
    y_true_sorted = y_true_binary[desc_indices]
    y_scores_sorted = y_scores[desc_indices]

    distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_true_sorted == 0)[threshold_idxs]

    n_pos = int(np.sum(y_true_binary == 1))
    n_neg = int(np.sum(y_true_binary == 0))

    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thresholds = np.r_[y_scores_sorted[0] + 1e-5, y_scores_sorted[threshold_idxs]]

    auc = float(trapezoid(tpr, fpr))
    return fpr, tpr, thresholds, auc

def compute_pr_curve(y_true_binary, y_scores):
    """
    Computes Precision-Recall curve for positive class (UF).
    """
    desc_indices = np.argsort(-y_scores)
    y_true_sorted = y_true_binary[desc_indices]
    y_scores_sorted = y_scores[desc_indices]

    distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_true_sorted == 0)[threshold_idxs]

    n_pos = int(np.sum(y_true_binary == 1))

    recall = tps / n_pos
    precision = tps / (tps + fps)

    # Prepend point (Recall=0, Precision=1.0)
    recall = np.r_[0.0, recall]
    precision = np.r_[1.0, precision]

    pr_auc = float(trapezoid(precision, recall))
    return recall, precision, pr_auc

def evaluate_predictions(y_true, y_probs_uf, threshold=0.50):
    """
    y_true: 0 for UF, 1 for NUF
    y_probs_uf: probability of UF
    threshold: cutoff for P(UF) >= threshold -> predict UF (0)
    """
    y_pred = np.where(y_probs_uf >= threshold, 0, 1)

    tp = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 0) & (y_pred == 1)))
    fp = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 1) & (y_pred == 1)))

    total = len(y_true)
    accuracy = float((tp + tn) / total)
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    youden_j = float(sensitivity + specificity - 1.0)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "youden_j": youden_j,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "y_pred": y_pred
    }

def run_model_inference(model_path, data_loader, device):
    arch_func = get_model_from_name.get("resnet50")
    model = arch_func(num_classes=2, pretrained=False)
    ckpt = torch.load(str(model_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_targets = []
    all_probs_uf = []
    all_probs_nuf = []

    with torch.no_grad():
        for batch in data_loader:
            images, targets = batch
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs_uf.extend(probs[:, 0].cpu().numpy().tolist())
            all_probs_nuf.extend(probs[:, 1].cpu().numpy().tolist())

    return np.array(all_targets, dtype=int), np.array(all_probs_uf, dtype=float), np.array(all_probs_nuf, dtype=float)

def run_mcnemar_test(y_true, y_pred1, y_pred2, model1_name="V1", model2_name="V2"):
    """
    Computes McNemar's 2x2 contingency table and statistical tests.
    y_true: true binary labels
    y_pred1, y_pred2: predictions from model 1 and model 2
    """
    corr1 = (y_pred1 == y_true)
    corr2 = (y_pred2 == y_true)

    # Contingency table:
    # a: both correct
    # b: model 1 correct, model 2 incorrect
    # c: model 1 incorrect, model 2 correct
    # d: both incorrect
    a = int(np.sum(corr1 & corr2))
    b = int(np.sum(corr1 & (~corr2)))
    c = int(np.sum((~corr1) & corr2))
    d = int(np.sum((~corr1) & (~corr2)))

    discordant = b + c

    # McNemar chi-square with Edwards continuity correction
    if discordant > 0:
        chi2_corrected = float((abs(b - c) - 1.0) ** 2 / discordant)
        p_val_chi2_corrected = float(1.0 - chi2.cdf(chi2_corrected, df=1))
        chi2_uncorrected = float((b - c) ** 2 / discordant)
        p_val_chi2_uncorrected = float(1.0 - chi2.cdf(chi2_uncorrected, df=1))
    else:
        chi2_corrected = 0.0
        p_val_chi2_corrected = 1.0
        chi2_uncorrected = 0.0
        p_val_chi2_uncorrected = 1.0

    # Exact binomial test (recommended for discrete 2x2 paired data)
    if discordant > 0:
        btest = binomtest(b, discordant, 0.5, alternative="two-sided")
        exact_p_value = float(btest.pvalue)
    else:
        exact_p_value = 1.0

    return {
        "contingency_table": {
            f"both_correct": a,
            f"{model1_name}_correct_{model2_name}_wrong": b,
            f"{model1_name}_wrong_{model2_name}_correct": c,
            f"both_wrong": d
        },
        "b_discordant_m1_better": b,
        "c_discordant_m2_better": c,
        "total_discordant": discordant,
        "mcnemar_chi2_corrected": chi2_corrected,
        "p_value_chi2_corrected": p_val_chi2_corrected,
        "mcnemar_chi2_uncorrected": chi2_uncorrected,
        "p_value_chi2_uncorrected": p_val_chi2_uncorrected,
        "exact_binomial_p_value": exact_p_value,
        "statistically_significant_005": exact_p_value < 0.05
    }

def main():
    print("=" * 80)
    print("CONTROLLED HEAD-TO-HEAD EVALUATION ON IDENTICAL CLEAN TEST SET (321 SCANS)")
    print("V1 (Trained on Contaminated Data) vs. V2 (Trained on Clean Data)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}")

    # Load clean test file lines
    with open(CLEAN_TEST_FILE, "r", encoding="utf-8") as f:
        test_lines = [l.strip() for l in f if l.strip()]

    print(f"Loaded Clean Test Dataset: {len(test_lines)} scans")
    test_ds = DataGenerator(test_lines, input_shape=(224, 224), random=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=detection_collate, num_workers=0)

    # 1. Run V1 Model
    print(f"\n[1/2] Running Inference for V1 Model ({V1_CKPT_PATH.name})...")
    y_true_v1, y_probs_uf_v1, y_probs_nuf_v1 = run_model_inference(V1_CKPT_PATH, test_loader, device)

    # 2. Run V2 Model
    print(f"[2/2] Running Inference for V2 Clean Model ({V2_CKPT_PATH.name})...")
    y_true_v2, y_probs_uf_v2, y_probs_nuf_v2 = run_model_inference(V2_CKPT_PATH, test_loader, device)

    # Verification
    assert np.array_equal(y_true_v1, y_true_v2), "Ground truth mismatch between runs!"
    y_true = y_true_v1
    total_samples = len(y_true)
    n_uf = int(np.sum(y_true == 0))
    n_nuf = int(np.sum(y_true == 1))
    print(f"Dataset Verification: Total = {total_samples}, UF = {n_uf}, NUF = {n_nuf}")

    # Threshold-independent curves
    y_true_uf_binary = (y_true == 0).astype(int)
    fpr_v1, tpr_v1, _, auc_v1 = compute_roc_curve(y_true_uf_binary, y_probs_uf_v1)
    rec_v1, prec_v1, pr_auc_v1 = compute_pr_curve(y_true_uf_binary, y_probs_uf_v1)

    fpr_v2, tpr_v2, _, auc_v2 = compute_roc_curve(y_true_uf_binary, y_probs_uf_v2)
    rec_v2, prec_v2, pr_auc_v2 = compute_pr_curve(y_true_uf_binary, y_probs_uf_v2)

    # Default threshold (0.50) evaluation
    v1_def = evaluate_predictions(y_true, y_probs_uf_v1, threshold=0.50)
    v2_def = evaluate_predictions(y_true, y_probs_uf_v2, threshold=0.50)

    # Threshold sweeps (0.30 to 0.80)
    thresholds = [round(t, 2) for t in np.arange(0.30, 0.81, 0.05)]
    v1_sweep = []
    v2_sweep = []

    for t in thresholds:
        res_v1 = evaluate_predictions(y_true, y_probs_uf_v1, threshold=t)
        res_v2 = evaluate_predictions(y_true, y_probs_uf_v2, threshold=t)
        r1_copy = {k: v for k, v in res_v1.items() if k != "y_pred"}
        r2_copy = {k: v for k, v in res_v2.items() if k != "y_pred"}
        v1_sweep.append(r1_copy)
        v2_sweep.append(r2_copy)

    # Identify optimal thresholds for both models
    v1_opt_acc = max(v1_sweep, key=lambda x: (x["accuracy"], x["f1"]))
    v2_opt_acc = max(v2_sweep, key=lambda x: (x["accuracy"], x["f1"]))

    v1_opt_f1 = max(v1_sweep, key=lambda x: (x["f1"], x["accuracy"]))
    v2_opt_f1 = max(v2_sweep, key=lambda x: (x["f1"], x["accuracy"]))

    # McNemar's Test at Default Threshold (tau = 0.50)
    mcnemar_def = run_mcnemar_test(y_true, v1_def["y_pred"], v2_def["y_pred"], "V1", "V2")

    # McNemar's Test at respective optimal accuracy thresholds
    v1_opt_pred = evaluate_predictions(y_true, y_probs_uf_v1, threshold=v1_opt_acc["threshold"])["y_pred"]
    v2_opt_pred = evaluate_predictions(y_true, y_probs_uf_v2, threshold=v2_opt_acc["threshold"])["y_pred"]
    mcnemar_opt = run_mcnemar_test(y_true, v1_opt_pred, v2_opt_pred, "V1_opt", "V2_opt")

    # Original V1 on contaminated 358-image test set
    v1_original_metrics = {}
    if V1_ORIGINAL_METRICS_PATH.is_file():
        with open(V1_ORIGINAL_METRICS_PATH, "r") as f:
            v1_orig_raw = json.load(f)
            v1_original_metrics = {
                "dataset_size": v1_orig_raw.get("total_test_images", 358),
                "accuracy": v1_orig_raw.get("accuracy"),
                "uf_sensitivity": v1_orig_raw.get("uf_sensitivity_recall"),
                "nuf_specificity": v1_orig_raw.get("nuf_specificity_recall"),
                "uf_precision": v1_orig_raw.get("uf_precision"),
                "uf_f1": v1_orig_raw.get("uf_f1_score"),
                "roc_auc": v1_orig_raw.get("roc_auc"),
                "confusion_matrix": v1_orig_raw.get("confusion_matrix")
            }

    # Print Full Comparative Results
    print("\n" + "=" * 90)
    print("HEAD-TO-HEAD COMPARISON TABLE: SAME TEST SET (321 SCANS)")
    print("=" * 90)
    print(f"{'Metric':<25} | {'V1 on Contam Test (358)':<22} | {'V1 on Clean Test (321)':<22} | {'V2 on Clean Test (321)':<22}")
    print("-" * 90)
    print(f"{'Test Cohort Size':<25} | {'358 (171 UF/187 NUF)':<22} | {'321 (163 UF/158 NUF)':<22} | {'321 (163 UF/158 NUF)':<22}")
    print(f"{'ROC-AUC (Threshold-Free)':<25} | {v1_original_metrics.get('roc_auc', 0.0):<22.4f} | {auc_v1:<22.4f} | {auc_v2:<22.4f}")
    print(f"{'PR-AUC (UF Class)':<25} | {'0.9467':<22} | {pr_auc_v1:<22.4f} | {pr_auc_v2:<22.4f}")
    print("-" * 90)
    print(f"{'At Default tau = 0.50:':<25} | {'':<22} | {'':<22} | {'':<22}")
    print(f"{'  Accuracy':<25} | {v1_original_metrics.get('accuracy', 0.0)*100:<21.2f}% | {v1_def['accuracy']*100:<21.2f}% | {v2_def['accuracy']*100:<21.2f}%")
    print(f"{'  UF Sensitivity':<25} | {v1_original_metrics.get('uf_sensitivity', 0.0)*100:<21.2f}% | {v1_def['sensitivity']*100:<21.2f}% | {v2_def['sensitivity']*100:<21.2f}%")
    print(f"{'  NUF Specificity':<25} | {v1_original_metrics.get('nuf_specificity', 0.0)*100:<21.2f}% | {v1_def['specificity']*100:<21.2f}% | {v2_def['specificity']*100:<21.2f}%")
    print(f"{'  UF Precision':<25} | {v1_original_metrics.get('uf_precision', 0.0)*100:<21.2f}% | {v1_def['precision']*100:<21.2f}% | {v2_def['precision']*100:<21.2f}%")
    print(f"{'  UF F1-Score':<25} | {v1_original_metrics.get('uf_f1', 0.0)*100:<21.2f}% | {v1_def['f1']*100:<21.2f}% | {v2_def['f1']*100:<21.2f}%")
    print(f"{'  False Positives (FP)':<25} | {'41':<22} | {v1_def['fp']:<22d} | {v2_def['fp']:<22d}")
    print(f"{'  Missed Fibroids (FN)':<25} | {'9':<22} | {v1_def['fn']:<22d} | {v2_def['fn']:<22d}")
    v1_opt_tau_str = f"tau = {v1_opt_acc['threshold']:.2f}"
    v2_opt_tau_str = f"tau = {v2_opt_acc['threshold']:.2f}"
    print(f"{'At Optimal Accuracy tau:':<25} | {'tau = 0.65':<22} | {v1_opt_tau_str:<22} | {v2_opt_tau_str:<22}")
    print(f"{'  Accuracy':<25} | {'89.66%':<22} | {v1_opt_acc['accuracy']*100:<21.2f}% | {v2_opt_acc['accuracy']*100:<21.2f}%")
    print(f"{'  UF Sensitivity':<25} | {'88.89%':<22} | {v1_opt_acc['sensitivity']*100:<21.2f}% | {v2_opt_acc['sensitivity']*100:<21.2f}%")
    print(f"{'  NUF Specificity':<25} | {'90.37%':<22} | {v1_opt_acc['specificity']*100:<21.2f}% | {v2_opt_acc['specificity']*100:<21.2f}%")
    print(f"{'  UF Precision':<25} | {'89.41%':<22} | {v1_opt_acc['precision']*100:<21.2f}% | {v2_opt_acc['precision']*100:<21.2f}%")
    print(f"{'  UF F1-Score':<25} | {'89.15%':<22} | {v1_opt_acc['f1']*100:<21.2f}% | {v2_opt_acc['f1']*100:<21.2f}%")
    print(f"{'  False Positives (FP)':<25} | {'18':<22} | {v1_opt_acc['fp']:<22d} | {v2_opt_acc['fp']:<22d}")
    print(f"{'  Missed Fibroids (FN)':<25} | {'19':<22} | {v1_opt_acc['fn']:<22d} | {v2_opt_acc['fn']:<22d}")
    print("=" * 90)

    print("\n" + "=" * 80)
    print("PAIRED MCNEMAR'S TEST: V1 vs. V2 ON IDENTICAL 321 CLEAN IMAGES")
    print("=" * 80)
    print("[1] At Default Cutoff (tau = 0.50):")
    ct = mcnemar_def["contingency_table"]
    print(f"  Contingency Table (2x2 Paired Outcomes):")
    print(f"    Both Models Correct               : {ct['both_correct']}")
    print(f"    V1 Correct, V2 Wrong (b)          : {ct['V1_correct_V2_wrong']}")
    print(f"    V1 Wrong, V2 Correct (c)          : {ct['V1_wrong_V2_correct']}")
    print(f"    Both Models Wrong                 : {ct['both_wrong']}")
    print(f"  Total Discordant Predictions (b + c): {mcnemar_def['total_discordant']}")
    print(f"  McNemar Chi2 (Edwards Corrected)    : {mcnemar_def['mcnemar_chi2_corrected']:.4f} (p = {mcnemar_def['p_value_chi2_corrected']:.4f})")
    print(f"  Exact Binomial Two-Sided p-value    : {mcnemar_def['exact_binomial_p_value']:.4f}")
    print(f"  Statistically Significant (p < 0.05): {mcnemar_def['statistically_significant_005']}")

    print("\n[2] At Model-Specific Optimal Accuracy Thresholds:")
    print(f"  V1 at tau={v1_opt_acc['threshold']:.2f} vs. V2 at tau={v2_opt_acc['threshold']:.2f}")
    ct_opt = mcnemar_opt["contingency_table"]
    print(f"    Both Models Correct               : {ct_opt['both_correct']}")
    print(f"    V1_opt Correct, V2_opt Wrong (b)  : {ct_opt['V1_opt_correct_V2_opt_wrong']}")
    print(f"    V1_opt Wrong, V2_opt Correct (c)  : {ct_opt['V1_opt_wrong_V2_opt_correct']}")
    print(f"    Both Models Wrong                 : {ct_opt['both_wrong']}")
    print(f"  Total Discordant Predictions (b + c): {mcnemar_opt['total_discordant']}")
    print(f"  Exact Binomial Two-Sided p-value    : {mcnemar_opt['exact_binomial_p_value']:.4f}")
    print(f"  Statistically Significant (p < 0.05): {mcnemar_opt['statistically_significant_005']}")

    # 1. Save v1_model_on_clean_testset.json
    v1_clean_report = {
        "model": "V1 (Original resnet50_stage2_best.pth trained on 1,747 scans)",
        "test_dataset": "Clean Held-Out Test Set (Quarantined from contamination)",
        "total_test_samples": total_samples,
        "uf_count": n_uf,
        "nuf_count": n_nuf,
        "roc_auc": float(auc_v1),
        "pr_auc": float(pr_auc_v1),
        "default_threshold_0_50": {
            "accuracy": v1_def["accuracy"],
            "uf_sensitivity": v1_def["sensitivity"],
            "nuf_specificity": v1_def["specificity"],
            "uf_precision": v1_def["precision"],
            "uf_f1": v1_def["f1"],
            "confusion_matrix": {
                "tp": v1_def["tp"],
                "fn": v1_def["fn"],
                "fp": v1_def["fp"],
                "tn": v1_def["tn"]
            }
        },
        "optimal_accuracy_threshold": v1_opt_acc,
        "optimal_f1_threshold": v1_opt_f1,
        "threshold_sweep": v1_sweep
    }
    v1_report_path = REPORTS_DIR / "v1_model_on_clean_testset.json"
    with open(v1_report_path, "w", encoding="utf-8") as f:
        json.dump(v1_clean_report, f, indent=2)
    print(f"\n[Saved] V1 on clean test report: {v1_report_path}")

    # 2. Save head_to_head_clean_test.json
    h2h_report = {
        "evaluation_scope": "Head-to-head comparison on identical 321 clean test scans",
        "total_test_samples": total_samples,
        "uf_count": n_uf,
        "nuf_count": n_nuf,
        "v1_model": {
            "checkpoint": str(V1_CKPT_PATH),
            "training_data": "1,747 scans (contaminated with obstetric/spectral Doppler)",
            "roc_auc": float(auc_v1),
            "pr_auc": float(pr_auc_v1),
            "default_0_50": {k: v for k, v in v1_def.items() if k != "y_pred"},
            "optimal_accuracy": v1_opt_acc
        },
        "v2_model": {
            "checkpoint": str(V2_CKPT_PATH),
            "training_data": "1,545 scans (100% clean pelvic B-mode scans)",
            "roc_auc": float(auc_v2),
            "pr_auc": float(pr_auc_v2),
            "default_0_50": {k: v for k, v in v2_def.items() if k != "y_pred"},
            "optimal_accuracy": v2_opt_acc
        },
        "mcnemar_test_default_0_50": mcnemar_def,
        "mcnemar_test_optimal_thresholds": mcnemar_opt,
        "threshold_sweep_v1": v1_sweep,
        "threshold_sweep_v2": v2_sweep
    }
    h2h_report_path = REPORTS_DIR / "head_to_head_clean_test.json"
    with open(h2h_report_path, "w", encoding="utf-8") as f:
        json.dump(h2h_report, f, indent=2)
    print(f"[Saved] Head-to-head comprehensive report: {h2h_report_path}")

    # 3. Plot Comparative ROC & PR Curves
    plt.figure(figsize=(12, 5))

    # ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr_v1, tpr_v1, color="#1f77b4", lw=2, label=f"V1 Model (AUC = {auc_v1:.4f})")
    plt.plot(fpr_v2, tpr_v2, color="#ff7f0e", lw=2, linestyle="--", label=f"V2 Clean (AUC = {auc_v2:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("ROC Curve Head-to-Head (n=321 Clean Scans)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    # PR
    plt.subplot(1, 2, 2)
    plt.plot(rec_v1, prec_v1, color="#1f77b4", lw=2, label=f"V1 Model (PR-AUC = {pr_auc_v1:.4f})")
    plt.plot(rec_v2, prec_v2, color="#ff7f0e", lw=2, linestyle="--", label=f"V2 Clean (PR-AUC = {pr_auc_v2:.4f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curve Head-to-Head", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    roc_plot_path = REPORTS_DIR / "head_to_head_roc_curve.png"
    plt.savefig(str(roc_plot_path), dpi=300)
    plt.close()
    print(f"[Saved] Head-to-head ROC curve plot: {roc_plot_path}")

    # 4. Side-by-side Confusion Matrix plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    cm_v1 = np.array([[v1_def["tp"], v1_def["fn"]], [v1_def["fp"], v1_def["tn"]]])
    cm_v2 = np.array([[v2_def["tp"], v2_def["fn"]], [v2_def["fp"], v2_def["tn"]]])

    for ax, cm, title in zip(axes, [cm_v1, cm_v2], ["V1 Model (Contaminated Train)", "V2 Model (Clean Train)"]):
        cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / np.sum(cm[i, :]) * 100
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if val > np.max(cm)/2 else "black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred UF", "Pred NUF"], fontsize=10)
        ax.set_yticklabels(["True UF", "True NUF"], fontsize=10)
        ax.set_title(f"{title}\n(tau = 0.50, n=321)", fontsize=11, pad=12, fontweight="bold")

    plt.tight_layout()
    cm_plot_path = REPORTS_DIR / "head_to_head_confusion_matrices.png"
    plt.savefig(str(cm_plot_path), dpi=300)
    plt.close()
    print(f"[Saved] Head-to-head confusion matrix plot: {cm_plot_path}")

if __name__ == "__main__":
    main()
