"""
evaluate_uteroscan.py
UteroScan Phase 4 — Final Held-Out Test Set Evaluation

Methodology per PRD.md Section 9 & Implementation_Plan.md Phase 4:
- Evaluates strictly on the held-out test set (DCGAN&MobileNetV2/cls_test.txt, 358 scans).
- Model: ResNet50 (nets/resnet.py) loaded with checkpoints/resnet50_stage2_best.pth.
- Input Preprocessing: Deterministic clinical pipeline (random=False).
- Metrics: Accuracy, Sensitivity (UF Recall), Specificity (NUF Recall), Precision, F1-score, AUC, Confusion Matrix.
- Outputs saved: reports/metrics.json and reports/confusion_matrix.png.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_STAGE2_CHECKPOINT = CHECKPOINTS_DIR / "resnet50_stage2_best.pth"
TEST_ANNOTATION_FILE = COMPANION_DIR / "cls_test.txt"

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

def plot_confusion_matrix(cm, output_path):
    tp_uf, fn_uf = cm[0]
    fp_uf, tn_uf = cm[1]
    
    matrix = np.array([[tp_uf, fn_uf], [fp_uf, tn_uf]])
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cax = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax)

    classes = ["UF (Fibroid)", "NUF (Normal)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=11, fontweight="bold")

    total = np.sum(matrix)
    thresh = matrix.max() / 2.0
    for i in range(2):
        for j in range(2):
            count = matrix[i, j]
            pct = count / np.sum(matrix[i, :]) * 100
            color = "white" if count > thresh else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", color=color,
                    fontsize=13, fontweight="bold")

    ax.set_ylabel("True Clinical Label", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Clinical Label", fontsize=12, fontweight="bold")
    ax.set_title("UteroScan ResNet50 Held-Out Test Confusion Matrix\n(358 Pristine Scans: 171 UF / 187 NUF)", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {output_path}")

def run_evaluation():
    print("=" * 80)
    print("UTEROSCAN PHASE 4: HELD-OUT TEST EVALUATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    assert BEST_STAGE2_CHECKPOINT.is_file(), f"Checkpoint not found: {BEST_STAGE2_CHECKPOINT}"
    assert TEST_ANNOTATION_FILE.is_file(), f"Test annotation file not found: {TEST_ANNOTATION_FILE}"

    with open(TEST_ANNOTATION_FILE, "r", encoding="utf-8") as f:
        test_lines = [l.strip() for l in f if l.strip()]

    uf_count = sum(1 for l in test_lines if l.startswith("0;"))
    nuf_count = sum(1 for l in test_lines if l.startswith("1;"))
    print(f"\nLoaded Held-Out Test Set: {len(test_lines)} images")
    print(f"  UF  (Uterine Fibroid)     : {uf_count} scans")
    print(f"  NUF (Non-Uterine Fibroid) : {nuf_count} scans")

    # Build model and load best checkpoint
    print(f"\nInstantiating ResNet50 from nets/resnet.py...")
    model_fn = get_model_from_name["resnet50"]
    model = model_fn(num_classes=2, pretrained=False)

    print(f"Loading Stage 2 Best weights from: {BEST_STAGE2_CHECKPOINT}")
    checkpoint = torch.load(str(BEST_STAGE2_CHECKPOINT), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_ds = DataGenerator(test_lines, input_shape=(224, 224), random=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=detection_collate, num_workers=0)

    criterion = nn.CrossEntropyLoss()

    all_targets = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    total_samples = 0

    t0 = time.time()
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, targets)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    eval_duration = time.time() - t0
    avg_loss = float(total_loss / total_samples)

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    y_probs = np.array(all_probs, dtype=float)

    # Class 0: UF, Class 1: NUF
    tp_uf = int(np.sum((y_true == 0) & (y_pred == 0)))  # UF correctly predicted
    fn_uf = int(np.sum((y_true == 0) & (y_pred == 1)))  # UF missed
    fp_uf = int(np.sum((y_true == 1) & (y_pred == 0)))  # NUF called UF
    tn_uf = int(np.sum((y_true == 1) & (y_pred == 1)))  # NUF correctly predicted

    total = len(y_true)
    accuracy = float((tp_uf + tn_uf) / total)
    sensitivity = float(tp_uf / (tp_uf + fn_uf)) if (tp_uf + fn_uf) > 0 else 0.0
    specificity = float(tn_uf / (tn_uf + fp_uf)) if (tn_uf + fp_uf) > 0 else 0.0
    precision = float(tp_uf / (tp_uf + fp_uf)) if (tp_uf + fp_uf) > 0 else 0.0
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    auc = compute_roc_auc(y_true, y_probs)

    cm = [[tp_uf, fn_uf], [fp_uf, tn_uf]]

    print("\n" + "=" * 80)
    print("FINAL HELD-OUT TEST RESULTS (358 UNTOUCHED IMAGES)")
    print("=" * 80)
    print(f"Evaluation Time : {eval_duration:.2f}s ({eval_duration/total*1000:.1f} ms/image)")
    print(f"Test Loss       : {avg_loss:.4f}")
    print(f"Test Accuracy   : {accuracy*100:6.2f}% ({tp_uf + tn_uf}/{total})")
    print(f"UF Sensitivity  : {sensitivity*100:6.2f}% ({tp_uf}/{tp_uf + fn_uf}) [CLINICAL RECALL - MISSED: {fn_uf}]")
    print(f"NUF Specificity : {specificity*100:6.2f}% ({tn_uf}/{tn_uf + fp_uf})")
    print(f"UF Precision    : {precision*100:6.2f}% ({tp_uf}/{tp_uf + fp_uf})")
    print(f"UF F1-Score     : {f1*100:6.2f}%")
    print(f"ROC-AUC         : {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TP (UF correctly identified) : {tp_uf:3d}  |  FN (UF missed)               : {fn_uf:3d}")
    print(f"  FP (NUF called UF)           : {fp_uf:3d}  |  TN (NUF correctly identified): {tn_uf:3d}")

    # Save metrics JSON
    metrics_payload = {
        "dataset": "Mendeley Uterine Fibroid (Cleaned, 1,747 scans total)",
        "split": "Held-Out Test Set (Quarantined from training/validation)",
        "total_test_images": total,
        "class_breakdown": {
            "UF": uf_count,
            "NUF": nuf_count
        },
        "model_architecture": "ResNet50 (nets/resnet.py)",
        "checkpoint_evaluated": str(BEST_STAGE2_CHECKPOINT.name),
        "evaluation_duration_seconds": eval_duration,
        "inference_latency_ms_per_image": float(eval_duration / total * 1000),
        "test_loss": avg_loss,
        "accuracy": accuracy,
        "uf_sensitivity_recall": sensitivity,
        "nuf_specificity_recall": specificity,
        "uf_precision": precision,
        "uf_f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": {
            "tp_uf": tp_uf,
            "fn_uf": fn_uf,
            "fp_uf": fp_uf,
            "tn_uf": tn_uf
        }
    }

    metrics_json_path = REPORTS_DIR / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\nSaved metrics payload to {metrics_json_path}")

    # Plot and save confusion matrix
    cm_plot_path = REPORTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, str(cm_plot_path))

    return metrics_payload

if __name__ == "__main__":
    run_evaluation()
