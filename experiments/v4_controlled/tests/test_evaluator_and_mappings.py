#!/usr/bin/env python3
"""
Unit Tests: Class Mappings, Probability Extractions, Metrics, and Thresholds
Location: experiments/v4_controlled/tests/test_evaluator_and_mappings.py

This test suite rigorously validates:
  1. Class Ordering A: 0 = UF, 1 = NUF (UteroScan legacy format)
  2. Class Ordering B: 0 = NUF, 1 = UF (Standard ML positive=1 format)
  3. Correct extraction of P(UF) and P(NUF) from model logits under both orderings
  4. Metric invariance: Sensitivity (UF Recall), Specificity (NUF Recall),
     Accuracy, Precision, F1, ROC-AUC, and PR-AUC(UF)
  5. Decision rule equivalence: Predict UF if P(UF) >= threshold
"""

import unittest
import numpy as np
import torch
import torch.nn.functional as F

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

def compute_roc_auc_binary(y_true_binary, y_scores_positive):
    """
    Computes ROC-AUC via trapezoidal integration.
    y_true_binary: 1 for positive class, 0 for negative class
    y_scores_positive: predicted probability of positive class
    """
    y_true = np.asarray(y_true_binary, dtype=int)
    y_scores = np.asarray(y_scores_positive, dtype=float)
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

def compute_pr_auc_binary(y_true_binary, y_scores_positive):
    """
    Computes PR-AUC for the positive class.
    y_true_binary: 1 for positive class, 0 for negative class
    """
    y_true = np.asarray(y_true_binary, dtype=int)
    y_scores = np.asarray(y_scores_positive, dtype=float)
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

def evaluate_with_ordering(logits, ground_truth_raw, class_ordering="0_is_UF", threshold_uf=0.50):
    """
    Unified evaluator supporting both class orderings.
    class_ordering:
      "0_is_UF": logits[:, 0] is UF, logits[:, 1] is NUF; ground_truth 0=UF, 1=NUF
      "1_is_UF": logits[:, 1] is UF, logits[:, 0] is NUF; ground_truth 1=UF, 0=NUF
    """
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    gt = np.asarray(ground_truth_raw, dtype=int)

    if class_ordering == "0_is_UF":
        p_uf = probs[:, 0]
        y_true_uf = (gt == 0).astype(int)
    elif class_ordering == "1_is_UF":
        p_uf = probs[:, 1]
        y_true_uf = (gt == 1).astype(int)
    else:
        raise ValueError(f"Unknown class ordering: {class_ordering}")

    # Decision rule: Predict UF if P(UF) >= threshold_uf
    y_pred_uf = (p_uf >= threshold_uf).astype(int)

    tp = int(np.sum((y_true_uf == 1) & (y_pred_uf == 1)))
    fn = int(np.sum((y_true_uf == 1) & (y_pred_uf == 0)))
    fp = int(np.sum((y_true_uf == 0) & (y_pred_uf == 1)))
    tn = int(np.sum((y_true_uf == 0) & (y_pred_uf == 0)))

    total = len(y_true_uf)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    roc_auc = compute_roc_auc_binary(y_true_uf, p_uf)
    pr_auc = compute_pr_auc_binary(y_true_uf, p_uf)

    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "p_uf": p_uf,
        "y_true_uf": y_true_uf,
        "y_pred_uf": y_pred_uf
    }


class TestEvaluationAndMappings(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # 20 samples: 10 UF and 10 NUF
        # Let's create realistic logits
        self.num_samples = 20
        # True clinical conditions: 10 True Fibroids, 10 True Normals
        self.clinical_is_fibroid = np.array([1]*10 + [0]*10)

        # Logits under Ordering A (Col 0 = UF, Col 1 = NUF)
        # High logit in col 0 -> Fibroid
        logits_col0_uf = np.zeros((20, 2))
        for i in range(20):
            if self.clinical_is_fibroid[i] == 1:
                logits_col0_uf[i, 0] = np.random.uniform(1.0, 3.0)
                logits_col0_uf[i, 1] = np.random.uniform(-1.0, 0.5)
            else:
                logits_col0_uf[i, 0] = np.random.uniform(-1.0, 0.5)
                logits_col0_uf[i, 1] = np.random.uniform(1.0, 3.0)
        self.logits_order_a = logits_col0_uf
        self.labels_order_a = (1 - self.clinical_is_fibroid) # 0=UF, 1=NUF

        # Logits under Ordering B (Col 1 = UF, Col 0 = NUF)
        logits_col1_uf = np.zeros((20, 2))
        logits_col1_uf[:, 0] = logits_col0_uf[:, 1]
        logits_col1_uf[:, 1] = logits_col0_uf[:, 0]
        self.logits_order_b = logits_col1_uf
        self.labels_order_b = self.clinical_is_fibroid # 1=UF, 0=NUF

    def test_class_ordering_invariance(self):
        """
        Verify that evaluation results are 100% identical regardless of whether
        the model maps 0=UF, 1=NUF or 0=NUF, 1=UF.
        """
        for thresh in [0.35, 0.50, 0.65]:
            res_a = evaluate_with_ordering(self.logits_order_a, self.labels_order_a,
                                           class_ordering="0_is_UF", threshold_uf=thresh)
            res_b = evaluate_with_ordering(self.logits_order_b, self.labels_order_b,
                                           class_ordering="1_is_UF", threshold_uf=thresh)

            self.assertEqual(res_a["tp"], res_b["tp"])
            self.assertEqual(res_a["fn"], res_b["fn"])
            self.assertEqual(res_a["fp"], res_b["fp"])
            self.assertEqual(res_a["tn"], res_b["tn"])
            self.assertAlmostEqual(res_a["accuracy"], res_b["accuracy"], places=7)
            self.assertAlmostEqual(res_a["sensitivity"], res_b["sensitivity"], places=7)
            self.assertAlmostEqual(res_a["specificity"], res_b["specificity"], places=7)
            self.assertAlmostEqual(res_a["precision"], res_b["precision"], places=7)
            self.assertAlmostEqual(res_a["f1"], res_b["f1"], places=7)
            self.assertAlmostEqual(res_a["roc_auc"], res_b["roc_auc"], places=7)
            self.assertAlmostEqual(res_a["pr_auc"], res_b["pr_auc"], places=7)

    def test_threshold_monotonicity(self):
        """
        Verify that increasing threshold on P(UF) monotonically decreases (or maintains)
        UF Sensitivity while monotonically increasing (or maintaining) NUF Specificity.
        """
        res_low = evaluate_with_ordering(self.logits_order_a, self.labels_order_a,
                                         class_ordering="0_is_UF", threshold_uf=0.30)
        res_mid = evaluate_with_ordering(self.logits_order_a, self.labels_order_a,
                                         class_ordering="0_is_UF", threshold_uf=0.50)
        res_high = evaluate_with_ordering(self.logits_order_a, self.labels_order_a,
                                          class_ordering="0_is_UF", threshold_uf=0.70)

        # Sensitivity must decrease as bar for predicting UF rises:
        self.assertGreaterEqual(res_low["sensitivity"], res_mid["sensitivity"])
        self.assertGreaterEqual(res_mid["sensitivity"], res_high["sensitivity"])

        # Specificity must increase as fewer false UF calls are made:
        self.assertLessEqual(res_low["specificity"], res_mid["specificity"])
        self.assertLessEqual(res_mid["specificity"], res_high["specificity"])

    def test_probability_normalization(self):
        """
        Verify that P(UF) + P(NUF) == 1.0 for all samples.
        """
        probs = F.softmax(torch.tensor(self.logits_order_a), dim=1).numpy()
        sums = probs.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
