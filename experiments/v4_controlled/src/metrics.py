#!/usr/bin/env python3
"""
Metrics and Threshold Optimization Module
Location: experiments/v4_controlled/src/metrics.py

Provides unified metrics and validation-only threshold optimization.
Standard Convention:
  Class 0 = UF (Uterine Fibroid) -> Clinical Positive
  Class 1 = NUF (Non-Uterine Fibroid) -> Clinical Negative
  Probability P(UF) = softmax(logits)[:, 0]
"""

import numpy as np

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

def compute_roc_auc(y_true_uf, p_uf):
    y_true = np.asarray(y_true_uf, dtype=int)
    y_scores = np.asarray(p_uf, dtype=float)
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

    return float(trapezoid(tpr, fpr))

def compute_pr_auc(y_true_uf, p_uf):
    y_true = np.asarray(y_true_uf, dtype=int)
    y_scores = np.asarray(p_uf, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0

    desc_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_indices]

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    recall = np.r_[0.0, tps / n_pos]
    precision = np.r_[1.0, tps / (tps + fps)]

    return float(trapezoid(precision, recall))

def evaluate_metrics_at_threshold(y_true_uf, p_uf, threshold_uf=0.50):
    y_true = np.asarray(y_true_uf, dtype=int)
    p_uf = np.asarray(p_uf, dtype=float)
    y_pred = (p_uf >= threshold_uf).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    youden_j = sensitivity + specificity - 1.0

    roc_auc = compute_roc_auc(y_true, p_uf)
    pr_auc = compute_pr_auc(y_true, p_uf)

    balanced_accuracy = (sensitivity + specificity) / 2.0

    return {
        "threshold": float(threshold_uf),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "uf_sensitivity": float(sensitivity),
        "nuf_specificity": float(specificity),
        "uf_precision": float(precision),
        "uf_f1": float(f1),
        "youden_j": float(youden_j),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "confusion_matrix": [[tp, fn], [fp, tn]]
    }

def optimize_threshold_on_val(y_true_uf, p_uf, objective="youden_j"):
    """
    Sweeps thresholds on validation predictions to maximize Youden's J statistic
    or balanced accuracy, completely isolated from test data.
    """
    best_thresh = 0.50
    best_score = -1.0
    best_metrics = None

    for thresh in np.arange(0.25, 0.76, 0.01):
        m = evaluate_metrics_at_threshold(y_true_uf, p_uf, threshold_uf=thresh)
        score = m["youden_j"] if objective == "youden_j" else (m["sensitivity"] + m["specificity"]) / 2.0
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)
            best_metrics = m

    return best_thresh, best_metrics
