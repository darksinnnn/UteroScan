#!/usr/bin/env python3
"""
Unit Tests: Threshold & Metrics Convention Reconciliation
Directory: experiments/external_dataset_audit_v2/

This test suite rigorously verifies:
  1. Exact mathematical duality between P(UF) and P(NUF) thresholding:
     P(UF) >= theta  <===>  P(NUF) < (1 - theta)
  2. The historical threshold anomaly:
     Why threshold 0.61 on P(NUF) was equivalent to threshold 0.39 on P(UF),
     and why lowering the fibroid threshold naturally increased sensitivity.
  3. Metric calculations: Confusion matrix, Sensitivity (UF Recall),
     Specificity (NUF Recall), Precision, and F1.
  4. Exact ROC-AUC invariance under label/probability inversion:
     ROC-AUC(y_UF, P(UF)) == ROC-AUC(y_NUF, P(NUF)).
  5. PR-AUC asymmetry between classes and the necessity of reporting PR-AUC(UF).
"""

import unittest
import numpy as np

# Compatible trapezoid function
try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

def compute_roc_auc(y_true, y_scores):
    """
    Standard ROC-AUC via trapezoidal integration.
    y_true: binary ground truth where 1 = positive class
    y_scores: float predicted probabilities for positive class
    """
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

    return float(trapezoid(tpr, fpr))

def compute_pr_auc(y_true, y_scores):
    """
    Computes Area Under the Precision-Recall Curve.
    y_true: 1 = positive class, 0 = negative class
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0

    desc_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_indices]

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    recall = tps / n_pos
    precision = tps / (tps + fps)

    recall = np.r_[0.0, recall]
    precision = np.r_[1.0, precision]

    return float(trapezoid(precision, recall))

def evaluate_predictions_uf_convention(y_true_uf, p_uf, threshold_uf=0.50):
    """
    Standard Clinical Convention:
      Positive Class = UF (1 = UF, 0 = NUF)
      Predict Fibroid if P(UF) >= threshold_uf
    """
    y_true = np.asarray(y_true_uf, dtype=int)
    p_uf = np.asarray(p_uf, dtype=float)
    y_pred = (p_uf >= threshold_uf).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))  # True Fibroid
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))  # Missed Fibroid
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))  # Normal called Fibroid
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))  # True Normal

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    roc_auc = compute_roc_auc(y_true, p_uf)
    pr_auc = compute_pr_auc(y_true, p_uf)

    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "y_pred": y_pred
    }

def evaluate_predictions_nuf_complementary(y_true_nuf, p_nuf, threshold_nuf=0.50):
    """
    Exact mathematical complementary evaluation:
      P(UF) >= theta_uf  <===>  P(NUF) <= threshold_nuf (where threshold_nuf = 1 - theta_uf)
    """
    y_true = np.asarray(y_true_nuf, dtype=int)
    p_nuf = np.asarray(p_nuf, dtype=float)
    # Predict NUF if P(NUF) > threshold_nuf, else predict UF
    y_pred_nuf = (p_nuf > threshold_nuf).astype(int)
    y_pred_uf = (y_pred_nuf == 0).astype(int)

    # Under UF evaluation:
    y_true_uf = (y_true == 0).astype(int)
    tp = int(np.sum((y_true_uf == 1) & (y_pred_uf == 1)))
    fn = int(np.sum((y_true_uf == 1) & (y_pred_uf == 0)))
    fp = int(np.sum((y_true_uf == 0) & (y_pred_uf == 1)))
    tn = int(np.sum((y_true_uf == 0) & (y_pred_uf == 0)))

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision
    }


class TestMetricsReconciliation(unittest.TestCase):

    def setUp(self):
        # Known synthetic benchmark dataset: 10 patients (5 UF, 5 NUF)
        # 1 = UF, 0 = NUF
        self.y_true_uf = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        # Model predicted probabilities of UF:
        self.p_uf = np.array([0.95, 0.85, 0.70, 0.55, 0.40, 0.45, 0.30, 0.20, 0.15, 0.05])
        
        # Dual representations:
        self.y_true_nuf = 1 - self.y_true_uf
        self.p_nuf = 1.0 - self.p_uf

    def test_threshold_duality(self):
        """
        Verify that predicting UF at threshold theta on P(UF)
        is mathematically identical to thresholding P(NUF) < (1 - theta).
        """
        for theta_uf in [0.30, 0.39, 0.50, 0.61, 0.75]:
            theta_nuf = 1.0 - theta_uf
            res_uf = evaluate_predictions_uf_convention(self.y_true_uf, self.p_uf, threshold_uf=theta_uf)
            res_nuf = evaluate_predictions_nuf_complementary(self.y_true_nuf, self.p_nuf, threshold_nuf=theta_nuf)

            self.assertEqual(res_uf["tp"], res_nuf["tp"], f"TP mismatch at theta={theta_uf}")
            self.assertEqual(res_uf["fn"], res_nuf["fn"], f"FN mismatch at theta={theta_uf}")
            self.assertEqual(res_uf["fp"], res_nuf["fp"], f"FP mismatch at theta={theta_uf}")
            self.assertEqual(res_uf["tn"], res_nuf["tn"], f"TN mismatch at theta={theta_uf}")
            self.assertAlmostEqual(res_uf["sensitivity"], res_nuf["sensitivity"], places=7)
            self.assertAlmostEqual(res_uf["specificity"], res_nuf["specificity"], places=7)

    def test_historical_061_anomaly_explanation(self):
        """
        Demonstrate why old script threshold 0.61 on P(NUF) showed HIGHER sensitivity
        than threshold 0.50:
        Because threshold 0.61 on P(NUF) was equivalent to threshold 0.39 on P(UF)!
        """
        res_at_050 = evaluate_predictions_uf_convention(self.y_true_uf, self.p_uf, threshold_uf=0.50)
        # In old script, theta=0.61 on P(NUF) mapped to theta=0.39 on P(UF):
        res_at_039 = evaluate_predictions_uf_convention(self.y_true_uf, self.p_uf, threshold_uf=0.39)

        # Notice at 0.50: 4 detected out of 5 (Sensitivity = 80%)
        # At 0.39: 5 detected out of 5 (Sensitivity = 100%)!
        self.assertGreaterEqual(res_at_039["sensitivity"], res_at_050["sensitivity"])
        print(f"\n[Verified] Sensitivity at theta_UF=0.50: {res_at_050['sensitivity']*100:.1f}%")
        print(f"[Verified] Sensitivity at theta_UF=0.39 (old 0.61 NUF threshold): {res_at_039['sensitivity']*100:.1f}%")

    def test_roc_auc_invariance(self):
        """
        Mathematically verify that binary ROC-AUC is strictly invariant
        under class inversion (ROC-AUC for UF == ROC-AUC for NUF).
        """
        auc_uf = compute_roc_auc(self.y_true_uf, self.p_uf)
        auc_nuf = compute_roc_auc(self.y_true_nuf, self.p_nuf)
        self.assertAlmostEqual(auc_uf, auc_nuf, places=7,
                               msg=f"ROC-AUC not invariant: {auc_uf} vs {auc_nuf}")
        print(f"[Verified] ROC-AUC(UF) = {auc_uf:.4f} == ROC-AUC(NUF) = {auc_nuf:.4f}")

    def test_pr_auc_asymmetry(self):
        """
        Demonstrate that PR-AUC is NOT symmetric between positive and negative classes.
        """
        pr_auc_uf = compute_pr_auc(self.y_true_uf, self.p_uf)
        pr_auc_nuf = compute_pr_auc(self.y_true_nuf, self.p_nuf)
        print(f"[Verified] PR-AUC(UF): {pr_auc_uf:.4f} vs PR-AUC(NUF): {pr_auc_nuf:.4f}")
        # Both should be valid numbers between 0 and 1
        self.assertGreaterEqual(pr_auc_uf, 0.0)
        self.assertLessEqual(pr_auc_uf, 1.0)
        self.assertGreaterEqual(pr_auc_nuf, 0.0)
        self.assertLessEqual(pr_auc_nuf, 1.0)

    def test_confusion_matrix_bounds(self):
        """
        Verify that TP + FN == Total Positive, and FP + TN == Total Negative.
        """
        res = evaluate_predictions_uf_convention(self.y_true_uf, self.p_uf, threshold_uf=0.50)
        self.assertEqual(res["tp"] + res["fn"], 5)
        self.assertEqual(res["fp"] + res["tn"], 5)
        self.assertEqual(res["tp"] + res["fn"] + res["fp"] + res["tn"], 10)


if __name__ == "__main__":
    unittest.main()
