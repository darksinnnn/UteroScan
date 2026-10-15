"""
evaluate_resnet_cbam.py
Rigorous Evaluation of ResNet50-CBAM on the 321-Image Clean Held-Out Cohort

Methodological Protocol:
1. Validation-Tuned Threshold: Sweep thresholds strictly on the validation set (244 images) to find tau*.
2. Frozen Evaluation: Apply tau* to the 321-image test set.
3. Test-Time Augmentation (TTA): Evaluate single-view vs horizontal-flip TTA.
4. Detailed Error Audit: Measure exact True Positives, False Positives, False Negatives, True Negatives.
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

BASE_DIR = Path("d:/Project/UteroScan")
EXP_DIR = BASE_DIR / "experiments" / "v3_resnet_enhanced"
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(BASE_DIR))

from models.resnet50_cbam import ResNet50_CBAM
from preprocessing.enhanced_dataloader import EnhancedUltrasoundDataset, custom_collate

ANNOTATIONS_DIR = BASE_DIR / "experiments" / "v2_contamination_cleaned" / "annotations"
CHECKPOINTS_DIR = EXP_DIR / "checkpoints"
REPORTS_DIR = EXP_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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

def compute_metrics_at_threshold(y_true, y_probs_nuf, tau=0.50):
    """
    y_true: 0 for UF, 1 for NUF
    y_probs_nuf: P(NUF).
    If P(NUF) >= tau -> predicted class 1 (NUF).
    Else -> predicted class 0 (UF).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs_nuf = np.asarray(y_probs_nuf, dtype=float)
    y_pred = (y_probs_nuf >= tau).astype(int)

    tp_uf = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn_uf = int(np.sum((y_true == 0) & (y_pred == 1)))
    fp_uf = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn_uf = int(np.sum((y_true == 1) & (y_pred == 1)))

    total = len(y_true)
    accuracy = float((tp_uf + tn_uf) / total) if total > 0 else 0.0
    sensitivity = float(tp_uf / (tp_uf + fn_uf)) if (tp_uf + fn_uf) > 0 else 0.0
    specificity = float(tn_uf / (tn_uf + fp_uf)) if (tn_uf + fp_uf) > 0 else 0.0
    precision = float(tp_uf / (tp_uf + fp_uf)) if (tp_uf + fp_uf) > 0 else 0.0
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    errors = total - (tp_uf + tn_uf)

    auc = compute_roc_auc(y_true, y_probs_nuf)

    return {
        "threshold": float(tau),
        "accuracy": accuracy,
        "auc": auc,
        "uf_sensitivity": sensitivity,
        "nuf_specificity": specificity,
        "uf_precision": precision,
        "uf_f1": f1,
        "total_errors": int(errors),
        "tp_uf": tp_uf,
        "fn_uf": fn_uf,
        "fp_uf": fp_uf,
        "tn_uf": tn_uf,
        "confusion_matrix": [[tp_uf, fn_uf], [fp_uf, tn_uf]]
    }

def get_probabilities(model, loader, device, tta=False):
    model.eval()
    all_targets = []
    all_probs_nuf = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            if tta:
                # Horizontal flip TTA
                images_flip = torch.flip(images, dims=[3])
                outputs_flip = model(images_flip)
                probs_flip = torch.softmax(outputs_flip, dim=1)
                probs = (probs + probs_flip) / 2.0

            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs_nuf.extend(probs[:, 1].cpu().numpy().tolist())

    return np.array(all_targets), np.array(all_probs_nuf)

def evaluate_checkpoint(ckpt_path, device, res=(256, 256), batch_size=16):
    print("=" * 80)
    print(f"EVALUATING MODEL: {Path(ckpt_path).name}")
    print("=" * 80)

    val_file = ANNOTATIONS_DIR / "cls_val_split.txt"
    test_file = ANNOTATIONS_DIR / "cls_test.txt"

    val_lines = [l.strip() for l in open(val_file) if l.strip()]
    test_lines = [l.strip() for l in open(test_file) if l.strip()]

    val_ds = EnhancedUltrasoundDataset(val_lines, input_shape=res, is_train=False, use_viewport=True)
    test_ds = EnhancedUltrasoundDataset(test_lines, input_shape=res, is_train=False, use_viewport=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=0)

    model = ResNet50_CBAM(num_classes=2, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    print("Loaded weights into ResNet50-CBAM.")

    # 1. Evaluate on Validation Set to pick optimal threshold
    print("\n--- Phase 1: Validation Set Threshold Optimization ---")
    val_targets, val_probs = get_probabilities(model, val_loader, device, tta=False)
    
    best_tau = 0.50
    best_val_acc = 0.0
    for tau in np.arange(0.30, 0.71, 0.01):
        m = compute_metrics_at_threshold(val_targets, val_probs, tau=tau)
        if m["accuracy"] > best_val_acc:
            best_val_acc = m["accuracy"]
            best_tau = float(tau)

    print(f"Selected Validation Threshold: tau* = {best_tau:.2f} (Val Acc: {best_val_acc*100:.2f}%)")

    # 2. Frozen Evaluation on 321-Image Test Set (Standard Inference)
    print("\n--- Phase 2: Held-Out Test Evaluation (Standard Inference) ---")
    test_targets, test_probs = get_probabilities(model, test_loader, device, tta=False)

    test_std_def = compute_metrics_at_threshold(test_targets, test_probs, tau=0.50)
    test_std_opt = compute_metrics_at_threshold(test_targets, test_probs, tau=best_tau)

    print(f"Standard (tau=0.50): Acc={test_std_def['accuracy']*100:.2f}% | Sens={test_std_def['uf_sensitivity']*100:.2f}% | "
          f"Spec={test_std_def['nuf_specificity']*100:.2f}% | AUC={test_std_def['auc']:.4f} | Errors={test_std_def['total_errors']}/321")
    print(f"Standard (tau*={best_tau:.2f}): Acc={test_std_opt['accuracy']*100:.2f}% | Sens={test_std_opt['uf_sensitivity']*100:.2f}% | "
          f"Spec={test_std_opt['nuf_specificity']*100:.2f}% | AUC={test_std_opt['auc']:.4f} | Errors={test_std_opt['total_errors']}/321")

    # 3. Held-Out Test Evaluation (Horizontal-Flip TTA)
    print("\n--- Phase 3: Held-Out Test Evaluation (Horizontal-Flip TTA) ---")
    _, test_probs_tta = get_probabilities(model, test_loader, device, tta=True)

    test_tta_def = compute_metrics_at_threshold(test_targets, test_probs_tta, tau=0.50)
    test_tta_opt = compute_metrics_at_threshold(test_targets, test_probs_tta, tau=best_tau)

    print(f"TTA (tau=0.50): Acc={test_tta_def['accuracy']*100:.2f}% | Sens={test_tta_def['uf_sensitivity']*100:.2f}% | "
          f"Spec={test_tta_def['nuf_specificity']*100:.2f}% | AUC={test_tta_def['auc']:.4f} | Errors={test_tta_def['total_errors']}/321")
    print(f"TTA (tau*={best_tau:.2f}): Acc={test_tta_opt['accuracy']*100:.2f}% | Sens={test_tta_opt['uf_sensitivity']*100:.2f}% | "
          f"Spec={test_tta_opt['nuf_specificity']*100:.2f}% | AUC={test_tta_opt['auc']:.4f} | Errors={test_tta_opt['total_errors']}/321")

    results = {
        "checkpoint": str(ckpt_path),
        "validation_selected_tau": best_tau,
        "standard_tau_050": test_std_def,
        "standard_tau_opt": test_std_opt,
        "tta_tau_050": test_tta_def,
        "tta_tau_opt": test_tta_opt
    }

    report_p = REPORTS_DIR / f"{Path(ckpt_path).stem}_evaluation.json"
    with open(report_p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation metrics to: {report_p}")

    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latest_ckpt = CHECKPOINTS_DIR / "resnet50_cbam_single_best.pth"
    if latest_ckpt.exists():
        evaluate_checkpoint(latest_ckpt, device)
    else:
        print(f"Checkpoint not found: {latest_ckpt}")
