"""
evaluate_v2_clean.py
Phase 4 Evaluation of V2 Clean ResNet50 on Held-Out Test Set (321 Scans: 163 UF / 158 NUF).
All outputs written EXCLUSIVELY to:
  experiments/v2_contamination_cleaned/reports/
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

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
CHECKPOINTS_DIR = EXP_DIR / "checkpoints"
REPORTS_DIR = EXP_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_METRICS_PATH = BASE_DIR / "reports" / "metrics.json"

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

def main():
    print("=" * 80)
    print("PHASE 4 EVALUATION: V2 CLEAN RESNET50 ON HELD-OUT TEST SET (321 SCANS)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}")

    # Load test set
    test_file = ANNOTATIONS_DIR / "cls_test.txt"
    with open(test_file, "r", encoding="utf-8") as f:
        test_lines = [l.strip() for l in f if l.strip()]

    print(f"Loaded Held-Out Test Set: {len(test_lines)} scans (163 UF, 158 NUF)")

    test_ds = DataGenerator(test_lines, input_shape=(224, 224), random=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=detection_collate, num_workers=0)

    # Load Best Stage 2 Model
    best_ckpt_path = CHECKPOINTS_DIR / "resnet50_stage2_best.pth"
    assert best_ckpt_path.is_file(), f"Model checkpoint not found: {best_ckpt_path}"

    arch_func = get_model_from_name.get("resnet50")
    model = arch_func(num_classes=2, pretrained=False)
    ckpt = torch.load(str(best_ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Successfully loaded checkpoint: {best_ckpt_path.name} (from Stage {ckpt.get('stage')}, Epoch {ckpt.get('epoch')})")

    # Run Inference
    all_targets = []
    all_probs_uf = []
    all_probs_nuf = []

    with torch.no_grad():
        for batch in test_loader:
            images, targets = batch
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs_uf.extend(probs[:, 0].cpu().numpy().tolist())
            all_probs_nuf.extend(probs[:, 1].cpu().numpy().tolist())

    y_true = np.array(all_targets, dtype=int)
    y_probs_uf = np.array(all_probs_uf, dtype=float)
    y_probs_nuf = np.array(all_probs_nuf, dtype=float)

    # Standard default cutoff (0.50)
    y_pred = (y_probs_nuf >= 0.50).astype(int)  # 0 if p_uf >= 0.5, else 1

    tp_uf = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn_uf = int(np.sum((y_true == 0) & (y_pred == 1)))
    fp_uf = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn_uf = int(np.sum((y_true == 1) & (y_pred == 1)))

    total = len(y_true)
    accuracy = float((tp_uf + tn_uf) / total)
    sensitivity = float(tp_uf / (tp_uf + fn_uf))
    specificity = float(tn_uf / (tn_uf + fp_uf))
    precision = float(tp_uf / (tp_uf + fp_uf)) if (tp_uf + fp_uf) > 0 else 0.0
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0

    # Binary UF indicator: 1 = UF, 0 = NUF
    y_true_uf_binary = (y_true == 0).astype(int)
    fpr, tpr, roc_thresh, roc_auc = compute_roc_curve(y_true_uf_binary, y_probs_uf)
    rec_pr, prec_pr, pr_auc = compute_pr_curve(y_true_uf_binary, y_probs_uf)

    print("\n" + "=" * 60)
    print("V2 CLEAN HELD-OUT TEST PERFORMANCE METRICS (Threshold 0.50)")
    print("=" * 60)
    print(f"  Overall Accuracy : {accuracy*100:.2f}% ({tp_uf + tn_uf}/{total})")
    print(f"  UF Sensitivity   : {sensitivity*100:.2f}% ({tp_uf}/{tp_uf + fn_uf}) [Recall]")
    print(f"  NUF Specificity  : {specificity*100:.2f}% ({tn_uf}/{tn_uf + fp_uf})")
    print(f"  UF Precision     : {precision*100:.2f}% ({tp_uf}/{tp_uf + fp_uf})")
    print(f"  UF F1-Score      : {f1*100:.2f}%")
    print(f"  ROC AUC          : {roc_auc:.4f}")
    print(f"  PR AUC (UF)      : {pr_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"               Pred UF    Pred NUF")
    print(f"  Actual UF    {tp_uf:6d}     {fn_uf:6d}   (Total: {tp_uf+fn_uf})")
    print(f"  Actual NUF   {fp_uf:6d}     {tn_uf:6d}   (Total: {fp_uf+tn_uf})")

    # 1. Save metrics.json
    metrics_data = {
        "dataset": "v2_contamination_cleaned",
        "total_test_samples": total,
        "uf_count": int(np.sum(y_true == 0)),
        "nuf_count": int(np.sum(y_true == 1)),
        "default_threshold": 0.50,
        "accuracy": accuracy,
        "uf_sensitivity": sensitivity,
        "nuf_specificity": specificity,
        "uf_precision": precision,
        "uf_f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "tp": tp_uf,
            "fn": fn_uf,
            "fp": fp_uf,
            "tn": tn_uf
        }
    }
    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")

    # 2. Confusion Matrix Plot
    cm = np.array([[tp_uf, fn_uf], [fp_uf, tn_uf]])
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / np.sum(cm[i, :]) * 100
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if val > np.max(cm)/2 else "black")
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred UF (Fibroid)", "Pred NUF (Normal)"], fontsize=10)
    ax.set_yticklabels(["True UF (Fibroid)", "True NUF (Normal)"], fontsize=10)
    ax.set_title("V2 Clean ResNet50 Test Confusion Matrix (n=321)", fontsize=11, pad=15, fontweight="bold")
    plt.tight_layout()
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # 3. ROC & PR Curve Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"V2 Clean ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Chance (AUC = 0.5000)")
    axes[0].plot(1.0 - specificity, sensitivity, marker="o", markersize=9, color="#d62728",
                 label=f"Default Thresh 0.50 (Sens: {sensitivity*100:.1f}%, Spec: {specificity*100:.1f}%)")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    axes[0].set_title("V2 Clean ROC Curve (Held-Out Test Set, n=321)", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right", fontsize=10)

    prevalence = float(np.sum(y_true == 0) / total)
    axes[1].plot(rec_pr, prec_pr, color="#2ca02c", lw=2.5, label=f"V2 Clean PR (PR-AUC = {pr_auc:.4f})")
    axes[1].axhline(y=prevalence, color="gray", linestyle="--", lw=1.5, label=f"Prevalence Baseline ({prevalence*100:.1f}%)")
    axes[1].plot(sensitivity, precision, marker="o", markersize=9, color="#d62728",
                 label=f"Default Thresh 0.50 (Prec: {precision*100:.1f}%, Rec: {sensitivity*100:.1f}%)")
    axes[1].set_xlabel("UF Sensitivity (Recall)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("UF Precision (PPV)", fontsize=11, fontweight="bold")
    axes[1].set_title("V2 Clean Precision-Recall Curve (UF Class)", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    roc_path = REPORTS_DIR / "roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"Saved ROC/PR curves to: {roc_path}")

    # 4. Threshold Sweep
    thresholds_to_test = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    sweep_results = []
    print("\nThreshold Sweep (V2 Clean Dataset):")
    print(f"{'Thresh':<7} | {'Accuracy':<9} | {'UF Sens':<9} | {'NUF Spec':<9} | {'UF Prec':<9} | {'UF F1':<8} | {'FN':<4} | {'FP'}")
    print("-" * 75)

    for th in thresholds_to_test:
        pred_th = (y_probs_uf < th).astype(int)  # 0 if p_uf >= th else 1
        tp_th = int(np.sum((y_true == 0) & (pred_th == 0)))
        fn_th = int(np.sum((y_true == 0) & (pred_th == 1)))
        fp_th = int(np.sum((y_true == 1) & (pred_th == 0)))
        tn_th = int(np.sum((y_true == 1) & (pred_th == 1)))

        acc_th = (tp_th + tn_th) / total
        sens_th = tp_th / (tp_th + fn_th) if (tp_th + fn_th) > 0 else 0.0
        spec_th = tn_th / (tn_th + fp_th) if (tn_th + fp_th) > 0 else 0.0
        prec_th = tp_th / (tp_th + fp_th) if (tp_th + fp_th) > 0 else 0.0
        f1_th = (2 * prec_th * sens_th) / (prec_th + sens_th) if (prec_th + sens_th) > 0 else 0.0

        sweep_results.append({
            "threshold": th,
            "accuracy": round(acc_th, 4),
            "sensitivity": round(sens_th, 4),
            "specificity": round(spec_th, 4),
            "precision": round(prec_th, 4),
            "f1": round(f1_th, 4),
            "tp": tp_th,
            "fn": fn_th,
            "fp": fp_th,
            "tn": tn_th
        })
        print(f"{th:<7.2f} | {acc_th*100:<8.2f}% | {sens_th*100:<8.2f}% | {spec_th*100:<8.2f}% | {prec_th*100:<8.2f}% | {f1_th*100:<7.2f}% | {fn_th:<4d} | {fp_th}")

    thresh_path = REPORTS_DIR / "threshold_analysis.json"
    with open(thresh_path, "w", encoding="utf-8") as f:
        json.dump({"roc_auc": roc_auc, "pr_auc": pr_auc, "sweep": sweep_results}, f, indent=2)
    print(f"\nSaved threshold analysis to: {thresh_path}")

    # 5. Side-by-side comparison with original Phase 4 metrics
    if ORIGINAL_METRICS_PATH.is_file():
        with open(ORIGINAL_METRICS_PATH, "r", encoding="utf-8") as f:
            orig = json.load(f)

        comp_data = {
            "original_v1": {
                "dataset_size": 1747,
                "test_size": 358,
                "accuracy": orig.get("test_accuracy", orig.get("accuracy")),
                "uf_sensitivity": orig.get("test_uf_sensitivity", orig.get("uf_sensitivity")),
                "nuf_specificity": orig.get("test_nuf_specificity", orig.get("nuf_specificity")),
                "uf_precision": orig.get("test_uf_precision", orig.get("uf_precision")),
                "uf_f1": orig.get("test_uf_f1", orig.get("uf_f1")),
                "roc_auc": orig.get("test_auc", orig.get("roc_auc")),
                "confusion_matrix": orig.get("test_confusion_matrix", orig.get("confusion_matrix"))
            },
            "clean_v2": {
                "dataset_size": 1545,
                "test_size": 321,
                "accuracy": accuracy,
                "uf_sensitivity": sensitivity,
                "nuf_specificity": specificity,
                "uf_precision": precision,
                "uf_f1": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "confusion_matrix": {"tp": tp_uf, "fn": fn_uf, "fp": fp_uf, "tn": tn_uf}
            }
        }
        comp_path = REPORTS_DIR / "comparison_v1_vs_v2.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comp_data, f, indent=2)
        print(f"Saved V1 vs V2 comparison to: {comp_path}")

if __name__ == "__main__":
    main()
