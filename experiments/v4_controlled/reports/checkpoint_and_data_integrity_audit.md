# UteroScan V4 — Checkpoint Selection and Training-Data Integrity Audit

**Author:** Antigravity AI (Pair Programming)  
**Date:** October 15, 2026  
**Audit Directory:** [`experiments/v4_controlled/reports/`](file:///d:/Project/UteroScan/experiments/v4_controlled/reports/)  
**Files Generated:**
- Contact Sheet Source Audit CSV: [`contact_sheet_source_audit.csv`](file:///d:/Project/UteroScan/experiments/v4_controlled/reports/contact_sheet_source_audit.csv)
- Training Set Doppler Leakage CSV: [`training_set_doppler_leakage.csv`](file:///d:/Project/UteroScan/experiments/v4_controlled/reports/training_set_doppler_leakage.csv)
- Contact Samples OCR Audit: [`contact_samples_ocr_audit.txt`](file:///d:/Project/UteroScan/experiments/v4_controlled/reports/contact_samples_ocr_audit.txt)
- Contact Samples Deep Inspection: [`contact_samples_deep_inspection.txt`](file:///d:/Project/UteroScan/experiments/v4_controlled/reports/contact_samples_deep_inspection.txt)

---

## 1. Executive Summary & Verdict

Following your instructions, we conducted an exhaustive, multi-part audit of:
1. **Checkpoint Selection & Objective Matching (Parts A & B):** Explaining the discrepancy between E2's peak epoch validation accuracy in logs (~96%) and the reported 93.97%, and comparing E0 vs E2 under identical, fair selection rules.
2. **Training Data Integrity & Visual Contamination (Part C):** Tracing the exact source files displayed on the caliper augmentation contact sheet, verifying dataloader paths against authoritative manifests, and individually inspecting all suspicious obstetric and Color Doppler scans.
3. **Evaluation Semantic Reconciliation (Part D):** Standardizing threshold definitions ($\tau=0.50$ fixed vs $\tau^*$ validation-optimized) and establishing strict boundaries for the exploratory 321-scan test cohort.

```
========================================================================================
AUDIT HEADLINE FINDINGS
========================================================================================
1. CHECKPOINT SELECTION DISCREPANCY RESOLVED:
   - E2 reported 93.97% accuracy because train_experiment.py selects checkpoints strictly
     by MINIMUM VALIDATION LOSS (Rule 1).
   - In E2, Fold 0, Fold 3, and Fold 4 achieved higher validation accuracy at different
     epochs than the minimum-loss epoch.
   - The mean of each fold's peak accuracy in E2 is indeed 95.24% (matching E0's reported 95.24%).
   - CRITICAL FAIRNESS INSIGHT: When the exact same peak-accuracy selection rule (Rule 2)
     is applied to E0, E0's mean accuracy ALSO rises to 95.38% (Sens: 98.48%, AUC: 0.9882).
   - VERDICT: Under BOTH selection rules, E0 Baseline outperforms E2 Caliper!
     * Rule 1 (Min Loss): E0 (95.24%) > E2 (93.97%)   [Delta: +1.27%]
     * Rule 2 (Max Acc) : E0 (95.38%) > E2 (95.24%)   [Delta: +0.14%]

2. CHECKPOINT WEIGHTS AVAILABILITY:
   - On disk, only minimum-loss checkpoints (*_fold_*_best.pth) were persisted.
   - Epoch weights corresponding to non-loss peak accuracies (e.g., E2 Fold 0 Ep 6,
     Fold 3 Ep 4, Fold 4 Ep 4; E0 Fold 2 Ep 5, Fold 3 Ep 3) were overwritten and are
     NOT available on disk.

3. CONTACT SHEET NUF CONTAMINATION CONFIRMED:
   - Quarantined files from data/excluded/ DID NOT inadvertently leak into the loader.
   - HOWEVER, the clean 1,224-scan training partition in datasets/datasets/train/ was
     incompletely sanitized by the historical heuristic cleaning sweep.
   - Sample 6 (11250322.2764469079.jpg, labeled NUF) IS A TRUE COLOR DOPPLER SCAN:
     It contains a vertical velocity colorbar (+18.5 cm/s scale) on the right edge,
     explicit OCR text 彩色血流 (Color Blood Flow), and 883 color flow pixels in the sector.
   - Samples 0 & 3 (labeled UF) contain explicit machine header text 中晚孕 (Mid/Late Pregnancy).
   - Samples 5 & 7 (labeled NUF) contain machine preset OB1 (Obstetric Ultrasound Preset).
   - Full dataset sweep reveals 67 scans in the 1,224 training set with Color Doppler features.
========================================================================================
```

---

## 2. Part A — Checkpoint Selection Audit (E0 vs E2)

### The Mechanism of Discrepancy
In `experiments/v4_controlled/src/train_experiment.py` (lines 221 & 238-245):
```python
is_best = val_loss < best_val_loss
if is_best:
    best_val_loss = val_loss
    best_val_auc = val_auc
    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    best_val_probs = p_val_uf
    best_val_targets = y_val_uf
```
Checkpoints are saved **exclusively when validation loss improves**. Intermediate checkpoints with higher classification accuracy but higher cross-entropy loss are not retained.

### E0 Baseline: Complete Epoch-by-Epoch Validation Trajectory
Parsed from execution log [`task-3341.log`](file:///C:/Users/ashis/.gemini/antigravity-ide/brain/af0e99e6-8ff9-4623-ad48-86540f6228c9/.system_generated/tasks/task-3341.log):

| Fold | Epoch | Train Loss | Val Loss | Val Acc (@0.50) | UF Sens | NUF Spec | ROC-AUC | Selected (Min Loss) | Selected (Max Acc) | Weights on Disk? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | Ep 01 | 0.5252 | 0.3203 | 87.6% | 97.6% | 75.7% | 0.9668 | No | No | No |
| Fold 0 | **Ep 02** | 0.2562 | **0.2030** | **93.1%** | 98.4% | 86.9% | 0.9835 | **YES (Rule 1)** | **YES (Rule 2)** | **YES (`e0_fold_0_best.pth`)** |
| Fold 0 | Ep 03 | 0.2099 | 0.2375 | 91.8% | 97.6% | 85.0% | 0.9801 | No | No | No |
| Fold 0 | Ep 04 | 0.1363 | 0.2650 | 91.4% | 93.7% | 88.8% | 0.9542 | No | No | No |
| Fold 0 | Ep 05 | 0.1181 | 0.2091 | 91.0% | 91.3% | 90.7% | 0.9780 | No | No | No |
| Fold 0 | Ep 06 | 0.0895 | 0.2501 | 91.8% | 96.0% | 86.9% | 0.9742 | No (Early Stop) | No | No |
| **Fold 1** | Ep 01 | 0.4811 | 0.2943 | 90.8% | 97.7% | 83.3% | 0.9744 | No | No | No |
| Fold 1 | Ep 02 | 0.2583 | 0.2341 | 91.2% | 85.3% | 97.5% | 0.9834 | No | No | No |
| Fold 1 | Ep 03 | 0.1809 | 0.1532 | 94.8% | 96.1% | 93.3% | 0.9879 | No | No | No |
| Fold 1 | Ep 04 | 0.1510 | 0.1625 | 95.6% | 97.7% | 93.3% | 0.9880 | No | No | No |
| Fold 1 | **Ep 05** | 0.1179 | **0.1387** | **95.6%** | 97.7% | 93.3% | 0.9877 | **YES (Rule 1)** | **YES (Rule 2)** | **YES (`e0_fold_1_best.pth`)** |
| Fold 1 | Ep 06 | 0.0976 | 0.1486 | 93.2% | 92.2% | 94.2% | 0.9871 | No | No | No |
| Fold 1 | Ep 07 | 0.0757 | 0.1487 | 93.6% | 94.6% | 92.5% | 0.9887 | No | No | No |
| Fold 1 | Ep 08 | 0.0690 | 0.1396 | 95.2% | 94.6% | 95.8% | 0.9861 | No | No | No |
| **Fold 2** | Ep 01 | 0.4834 | 0.3239 | 89.3% | 100.0% | 78.0% | 0.9827 | No | No | No |
| Fold 2 | Ep 02 | 0.2609 | 0.1817 | 96.0% | 100.0% | 91.9% | 0.9875 | No | No | No |
| Fold 2 | Ep 03 | 0.1876 | 0.1681 | 95.3% | 99.2% | 91.1% | 0.9861 | No | No | No |
| Fold 2 | Ep 04 | 0.1423 | 0.1209 | 95.3% | 97.7% | 92.7% | 0.9913 | No | No | No |
| Fold 2 | **Ep 05** | 0.1037 | 0.1346 | **96.0%** | 99.2% | 92.7% | 0.9923 | No | **YES (Rule 2)** | **NO (Overwritten)** |
| Fold 2 | Ep 06 | 0.1083 | 0.1861 | 92.9% | 99.2% | 86.2% | 0.9918 | No | No | No |
| Fold 2 | **Ep 07** | 0.1033 | **0.1183** | 95.7% | 96.9% | 94.3% | 0.9904 | **YES (Rule 1)** | No | **YES (`e0_fold_2_best.pth`)** |
| Fold 2 | Ep 08 | 0.0817 | 0.2041 | 95.7% | 100.0% | 91.1% | 0.9906 | No | No | No |
| **Fold 3** | Ep 01 | 0.5015 | 0.3683 | 83.9% | 99.3% | 66.4% | 0.9706 | No | No | No |
| Fold 3 | Ep 02 | 0.2504 | 0.2210 | 90.6% | 97.0% | 83.2% | 0.9777 | No | No | No |
| Fold 3 | **Ep 03** | 0.1898 | 0.1695 | **96.5%** | 100.0% | 92.4% | 0.9896 | No | **YES (Rule 2)** | **NO (Overwritten)** |
| Fold 3 | Ep 04 | 0.1397 | 0.1936 | 94.1% | 100.0% | 87.4% | 0.9862 | No | No | No |
| Fold 3 | Ep 05 | 0.1237 | 0.2243 | 91.7% | 97.8% | 84.9% | 0.9824 | No | No | No |
| Fold 3 | Ep 06 | 0.1079 | 0.2536 | 90.9% | 99.3% | 81.5% | 0.9839 | No | No | No |
| Fold 3 | **Ep 07** | 0.1035 | **0.1417** | 96.1% | 97.8% | 94.1% | 0.9879 | **YES (Rule 1)** | No | **YES (`e0_fold_3_best.pth`)** |
| Fold 3 | Ep 08 | 0.0620 | 0.1803 | 92.5% | 97.0% | 87.4% | 0.9821 | No | No | No |
| **Fold 4** | Ep 01 | 0.4936 | 0.2271 | 92.3% | 95.6% | 87.8% | 0.9791 | No | No | No |
| Fold 4 | Ep 02 | 0.2242 | 0.1733 | 94.0% | 95.6% | 91.8% | 0.9805 | No | No | No |
| Fold 4 | Ep 03 | 0.2096 | 0.1694 | 94.9% | 95.6% | 93.9% | 0.9862 | No | No | No |
| Fold 4 | Ep 04 | 0.1441 | 0.1506 | 95.3% | 95.6% | 94.9% | 0.9867 | No | No | No |
| Fold 4 | Ep 05 | 0.1341 | 0.1607 | 95.3% | 95.6% | 94.9% | 0.9833 | No | No | No |
| Fold 4 | Ep 06 | 0.0933 | 0.1654 | 94.9% | 94.9% | 94.9% | 0.9823 | No | No | No |
| Fold 4 | **Ep 07** | 0.0886 | **0.1347** | **95.7%** | 97.1% | 93.9% | 0.9880 | **YES (Rule 1)** | **YES (Rule 2)** | **YES (`e0_fold_4_best.pth`)** |
| Fold 4 | Ep 08 | 0.0635 | 0.1658 | 92.8% | 92.7% | 92.9% | 0.9861 | No | No | No |

---

### E2 Caliper: Complete Epoch-by-Epoch Validation Trajectory
Parsed from execution log [`task-3557.log`](file:///C:/Users/ashis/.gemini/antigravity-ide/brain/af0e99e6-8ff9-4623-ad48-86540f6228c9/.system_generated/tasks/task-3557.log):

| Fold | Epoch | Train Loss | Val Loss | Val Acc (@0.50) | UF Sens | NUF Spec | ROC-AUC | Selected (Min Loss) | Selected (Max Acc) | Weights on Disk? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | Ep 01 | 0.5297 | 0.3036 | 90.1% | 92.1% | 87.9% | 0.9663 | No | No | No |
| Fold 0 | **Ep 02** | 0.2923 | **0.2081** | 91.4% | 95.2% | 86.9% | 0.9790 | **YES (Rule 1)** | No | **YES (`e2_fold_0_best.pth`)** |
| Fold 0 | Ep 03 | 0.1982 | 0.2103 | 93.1% | 96.0% | 89.7% | 0.9761 | No | No | No |
| Fold 0 | Ep 04 | 0.1638 | 0.2496 | 91.8% | 95.2% | 87.9% | 0.9677 | No | No | No |
| Fold 0 | Ep 05 | 0.1210 | 0.2226 | 92.3% | 89.7% | 95.3% | 0.9800 | No | No | No |
| Fold 0 | **Ep 06** | 0.0789 | 0.2288 | **94.0%** | 96.0% | 91.6% | 0.9782 | No (Early Stop) | **YES (Rule 2)** | **NO (Overwritten)** |
| **Fold 1** | Ep 01 | 0.5119 | 0.3883 | 85.1% | 93.8% | 75.8% | 0.9444 | No | No | No |
| Fold 1 | Ep 02 | 0.2497 | 0.2448 | 91.6% | 96.1% | 86.7% | 0.9742 | No | No | No |
| Fold 1 | Ep 03 | 0.1762 | 0.2263 | 93.6% | 96.1% | 90.8% | 0.9731 | No | No | No |
| Fold 1 | Ep 04 | 0.1532 | 0.2752 | 91.6% | 98.4% | 84.2% | 0.9680 | No | No | No |
| Fold 1 | Ep 05 | 0.1400 | 0.4626 | 83.5% | 100.0% | 65.8% | 0.9770 | No | No | No |
| Fold 1 | Ep 06 | 0.1141 | 0.2400 | 92.0% | 92.2% | 91.7% | 0.9691 | No | No | No |
| Fold 1 | **Ep 07** | 0.0748 | **0.1593** | **94.8%** | 94.6% | 95.0% | 0.9875 | **YES (Rule 1)** | **YES (Rule 2)** | **YES (`e2_fold_1_best.pth`)** |
| Fold 1 | Ep 08 | 0.0554 | 0.2016 | 92.8% | 94.6% | 90.8% | 0.9823 | No | No | No |
| **Fold 2** | Ep 01 | 0.5233 | 0.3820 | 86.6% | 100.0% | 72.4% | 0.9816 | No | No | No |
| Fold 2 | Ep 02 | 0.2759 | 0.1979 | 93.7% | 100.0% | 87.0% | 0.9870 | No | No | No |
| Fold 2 | Ep 03 | 0.1853 | 0.1266 | 96.0% | 99.2% | 92.7% | 0.9879 | No | No | No |
| Fold 2 | Ep 04 | 0.1498 | 0.1301 | 95.7% | 98.5% | 92.7% | 0.9893 | No | No | No |
| Fold 2 | Ep 05 | 0.1024 | 0.1495 | 95.3% | 98.5% | 91.9% | 0.9917 | No | No | No |
| Fold 2 | **Ep 06** | 0.0929 | **0.0958** | **96.8%** | 99.2% | 94.3% | 0.9961 | **YES (Rule 1)** | **YES (Rule 2)** | **YES (`e2_fold_2_best.pth`)** |
| Fold 2 | Ep 07 | 0.0764 | 0.1135 | 96.4% | 98.5% | 94.3% | 0.9920 | No | No | No |
| Fold 2 | Ep 08 | 0.0697 | 0.1954 | 94.1% | 99.2% | 88.6% | 0.9922 | No | No | No |
| **Fold 3** | Ep 01 | 0.5073 | 0.3899 | 83.1% | 98.5% | 65.5% | 0.9716 | No | No | No |
| Fold 3 | Ep 02 | 0.2526 | 0.2219 | 91.7% | 87.4% | 96.6% | 0.9745 | No | No | No |
| Fold 3 | Ep 03 | 0.2074 | 0.2510 | 90.2% | 99.3% | 79.8% | 0.9828 | No | No | No |
| Fold 3 | **Ep 04** | 0.1438 | 0.1683 | **96.1%** | 98.5% | 93.3% | 0.9816 | No | **YES (Rule 2)** | **NO (Overwritten)** |
| Fold 3 | Ep 05 | 0.1203 | 0.2570 | 91.7% | 92.6% | 90.8% | 0.9635 | No | No | No |
| Fold 3 | **Ep 06** | 0.1076 | **0.1672** | 94.9% | 94.8% | 95.0% | 0.9806 | **YES (Rule 1)** | No | **YES (`e2_fold_3_best.pth`)** |
| Fold 3 | Ep 07 | 0.0994 | 0.1685 | 95.7% | 95.6% | 95.8% | 0.9818 | No | No | No |
| Fold 3 | Ep 08 | 0.0460 | 0.2088 | 95.7% | 96.3% | 95.0% | 0.9733 | No | No | No |
| **Fold 4** | Ep 01 | 0.5187 | 0.3632 | 85.1% | 97.8% | 67.3% | 0.9508 | No | No | No |
| Fold 4 | Ep 02 | 0.2782 | 0.2199 | 90.2% | 96.4% | 81.6% | 0.9772 | No | No | No |
| Fold 4 | Ep 03 | 0.2138 | 0.2331 | 88.9% | 87.6% | 90.8% | 0.9746 | No | No | No |
| Fold 4 | **Ep 04** | 0.1349 | 0.1725 | **94.5%** | 94.9% | 93.9% | 0.9809 | No | **YES (Rule 2)** | **NO (Overwritten)** |
| Fold 4 | Ep 05 | 0.0966 | 0.1878 | 92.8% | 96.4% | 87.8% | 0.9815 | No | No | No |
| Fold 4 | **Ep 06** | 0.0818 | **0.1687** | 91.9% | 97.1% | 84.7% | 0.9861 | **YES (Rule 1)** | No | **YES (`e2_fold_4_best.pth`)** |
| Fold 4 | Ep 07 | 0.0895 | 0.1862 | 91.9% | 94.2% | 88.8% | 0.9814 | No | No | No |
| Fold 4 | Ep 08 | 0.0751 | 0.4246 | 89.8% | 97.8% | 78.6% | 0.9659 | No | No | No |

---

## 3. Part B — Fair Model Comparison Under Identical Selection Rules

To compare models scientifically, the decision rule must be applied symmetrically to both candidates.

### Comparison 1: Rule 1 — Minimum Validation Loss (Existing Rule)
Checkpoints are chosen by `argmin(val_loss)`. Checkpoint weights for this rule are physically preserved on disk in [`experiments/v4_controlled/checkpoints/`](file:///d:/Project/UteroScan/experiments/v4_controlled/checkpoints/).

| Metric | E0 Baseline (Rule 1) | E2 Caliper (Rule 1) | Delta ($\Delta$) | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **Fold 0 Accuracy** | 93.1% (Ep 02) | 91.4% (Ep 02) | -1.7% | **E0** |
| **Fold 1 Accuracy** | 95.6% (Ep 05) | 94.8% (Ep 07) | -0.8% | **E0** |
| **Fold 2 Accuracy** | 95.7% (Ep 07) | 96.8% (Ep 06) | +1.1% | **E2** |
| **Fold 3 Accuracy** | 96.1% (Ep 07) | 94.9% (Ep 06) | -1.2% | **E0** |
| **Fold 4 Accuracy** | 95.7% (Ep 07) | 91.9% (Ep 06) | -3.8% | **E0** |
| **Mean CV Accuracy** | **95.24% $\pm$ 1.21%** | **93.96% $\pm$ 2.26%** | **-1.28%** | **E0 Baseline (+1.28%)** |
| **Mean UF Sensitivity** | **97.58% $\pm$ 0.60%** | **96.18% $\pm$ 1.96%** | **-1.40%** | **E0 Baseline (+1.40%)** |
| **Mean NUF Specificity**| **92.50% $\pm$ 3.15%** | **91.18% $\pm$ 4.98%** | **-1.32%** | **E0 Baseline (+1.32%)** |
| **Mean ROC-AUC** | **0.9875 $\pm$ 0.0025** | **0.9859 $\pm$ 0.0068** | **-0.0016** | **E0 Baseline (+0.0016)** |

> [!IMPORTANT]
> Under Rule 1 (the official pre-registered protocol), **E0 Baseline wins on 4 out of 5 folds** and is superior across every single summary metric (Accuracy, Sensitivity, Specificity, AUC).

---

### Comparison 2: Rule 2 — Maximum Validation Accuracy at $\tau=0.50$ (Val Loss Tiebreaker)
Checkpoints are chosen by `argmax(val_acc_050)`, using `argmin(val_loss)` to break any ties.

| Metric | E0 Baseline (Rule 2) | E2 Caliper (Rule 2) | Delta ($\Delta$) | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **Fold 0 Accuracy** | 93.1% (Ep 02) | 94.0% (Ep 06) | +0.9% | **E2** |
| **Fold 1 Accuracy** | 95.6% (Ep 05) | 94.8% (Ep 07) | -0.8% | **E0** |
| **Fold 2 Accuracy** | 96.0% (Ep 05) | 96.8% (Ep 06) | +0.8% | **E2** |
| **Fold 3 Accuracy** | 96.5% (Ep 03) | 96.1% (Ep 04) | -0.4% | **E0** |
| **Fold 4 Accuracy** | 95.7% (Ep 07) | 94.5% (Ep 04) | -1.2% | **E0** |
| **Mean CV Accuracy** | **95.38% $\pm$ 1.32%** | **95.24% $\pm$ 1.17%** | **-0.14%** | **E0 Baseline (+0.14%)** |
| **Mean UF Sensitivity** | **98.48% $\pm$ 1.16%** | **96.64% $\pm$ 2.10%** | **-1.84%** | **E0 Baseline (+1.84%)** |
| **Mean NUF Specificity**| **91.84% $\pm$ 2.82%** | **93.62% $\pm$ 1.29%** | **+1.78%** | **E2 Caliper (+1.78%)** |
| **Mean ROC-AUC** | **0.9882 $\pm$ 0.0032** | **0.9849 $\pm$ 0.0071** | **-0.0033** | **E0 Baseline (+0.0033)** |

> [!NOTE]
> When we look at E2's peak accuracy across folds, it averages **95.24%**.  
> But E0 was not static: when evaluated by the exact same peak-accuracy rule, **E0 Baseline reaches 95.38% accuracy**, with exceptional **98.48% UF sensitivity** and **0.9882 ROC-AUC**.  
> Therefore, **E0 remains the stronger model under both selection rules.** E2 does not beat E0; it merely shifted threshold balance slightly toward specificity at the cost of sensitivity and discriminative AUC.

---

## 4. Part C — Caliper Contact Sheet Trace & Data Integrity Audit

We traced all 8 image pairs displayed on [`caliper_augmentation_contact_sheet.png`](file:///d:/Project/UteroScan/experiments/v4_controlled/reports/caliper_augmentation_contact_sheet.png) back to their physical files, authoritative manifest entries, exclusion logs, and clinical OCR tags.

The exact audit records have been saved to [`contact_sheet_source_audit.csv`](file:///d:/Project/UteroScan/experiments/v4_controlled/reports/contact_sheet_source_audit.csv):

| Index | Label | Filename | Split / Source | Exclusion Log? | Clinical OCR & Machine Metadata | Verified Audit Finding |
| :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| **0** | **UF** | `12350016.1399125003.jpg` | Train (V1_Clean) | False (None) | `Affiniti 70 \| 中晚孕 \| TIB0.1` | **Obstetric Contamination**: Machine preset `中晚孕` (2nd/3rd trimester pregnancy). |
| **1** | **UF** | `11697203.3191604034.jpg` | Train (V1_Clean) | False (None) | `腹部常规 \| TIS0.2 \| C5-1` | **Clean**: Standard grayscale abdominal/gynecologic scan. |
| **2** | **UF** | `11733585.718194887.jpg` | Train (V1_Clean) | False (None) | `腹部常规 \| TIS0.3 \| C5-1` | **Clean**: Standard grayscale abdominal/gynecologic scan. |
| **3** | **UF** | `11365242.462725929.jpg` | Train (V1_Clean) | False (None) | `Affiniti 50 \| 中晚孕 \| TIB0.1` | **Obstetric Contamination**: Machine preset `中晚孕` (2nd/3rd trimester pregnancy). |
| **4** | **NUF** | `11368989.377816910.jpg` | Train (V1_Clean) | False (None) | `EPIQ7 \| XSP \| C5-1` | **Grayscale Frame from Contaminated Study**: Study cluster `11368989` had other scans quarantined for Doppler. |
| **5** | **NUF** | `11364923.1619662809.jpg` | Train (V1_Clean) | False (None) | `EPIQ7 \| OB1 \| TIB0.1` | **Obstetric Contamination**: Preset `OB1` (Obstetrics); visual gestational sac / fetal scan. |
| **6** | **NUF** | `11250322.2764469079.jpg` | Train (V1_Clean) | False (None) | `彩色血流 \| +18.5 cm/s \| PRF 1200Hz` | **TRUE COLOR DOPPLER SCAN**: Vertical color velocity scale (+18.5 cm/s) on right edge and 883 color flow pixels in sector! |
| **7** | **NUF** | `11364923.2046609818.jpg` | Train (V1_Clean) | False (None) | `EPIQ7 \| OB1 \| TIB0.1` | **Obstetric Contamination**: Preset `OB1` (Obstetrics); companion frame to sample 5. |

### Technical Analysis of Contamination Provenance
1. **Did quarantined files leak into the training loader?**  
   **No.** `ControlledUltrasoundDataset` strictly reads paths listed in `experiments/v4_controlled/data/folds.csv`, pointing to `datasets/datasets/train/UF/` and `datasets/datasets/train/NUF/`. No files from `data/excluded/` entered the dataloader.
2. **Why are contaminated files present in `datasets/datasets/train/`?**  
   The historical cleaning process (documented in `data/excluded/exclusion_log.csv`) used **keyword-matching OCR filters** (e.g. `GA=\d+`, `早孕`, `胎儿`) and a left-hand color Doppler detector.
   - Sample 6 escaped exclusion because its color velocity bar was located on the **far right edge** (columns 976–1006) rather than the left edge, and its flow label `彩色血流` was missed by earlier keyword filters.
   - Samples 0 and 3 escaped because the Chinese term **`中晚孕`** (Mid-to-Late Pregnancy) was not included in the earlier exclusion dictionary (which only checked `早孕` for early pregnancy).
   - Samples 5 and 7 escaped because they used the machine preset tag **`OB1`** without explicit millimeter caliper measurements (`GA=...`).
3. **Extent of Leakage Across the 1,224-Scan Training Cohort:**  
   A systematic color velocity and flow scan across all 1,224 scans in `folds.csv` revealed **67 scans with Color Doppler features** (18 labeled UF, 49 labeled NUF). The complete inventory is documented in [`training_set_doppler_leakage.csv`](file:///d:/Project/UteroScan/experiments/v4_controlled/reports/training_set_doppler_leakage.csv).

---

## 5. Part D — Evaluation Semantic Reconciliation

| Evaluation Component | Audit Status | Verification Details |
| :--- | :---: | :--- |
| **Class Label Encoding** | **VERIFIED** | `UF = 0`, `NUF = 1`. In metrics, `y_val_uf = (target == 0).astype(int)`. Output column 0 corresponds to $P(\text{UF})$. |
| **Metric Standard** | **VERIFIED** | Sensitivity measures UF detection rate; Specificity measures NUF non-fibroid rate. Unchanged between E0 and E2. |
| **Fixed Threshold ($\tau=0.50$)** | **VERIFIED** | Standardized fixed operating point reported across all folds for direct comparisons. |
| **Validation-Calibrated Threshold ($\tau^*$)** | **VERIFIED** | Tuned strictly on pooled out-of-fold validation predictions using Youden's J ($J = \text{Sens} + \text{Spec} - 1$). Never tuned on test scans. E0: $\tau^* = 0.64$; E2: $\tau^* = 0.62$. |
| **321 Clean Held-Out Cohort** | **VERIFIED** | Designated as **READ-ONLY EXPLORATORY**. Never used for model selection, epoch selection, or threshold calibration. |

---

## 6. Recommendations & Next Steps Before E3

> [!CAUTION]
> **Do not launch E3 (Auxiliary Localization Head) or retrain models until the following two clean-up actions are taken.**

### Action 1: Purge Residual Obstetric and Color Doppler Contamination from the Training Cohort
The discovery of 67 Color Doppler scans and residual obstetric scans (`中晚孕`, `OB1`) in `datasets/datasets/train/` explains persistent noise and spurious specificity drops across folds.
1. Run a comprehensive cleanup script to move all 67 Color Doppler scans and all confirmed obstetric scans (`中晚孕`, `OB1`) to `data/excluded/`.
2. Re-partition the remaining cleanly verified gynecological scans into balanced 5-fold cross-validation (`folds_clean_v5.csv`).

### Action 2: Model Selection Policy Recommendation
- **Model Choice:** **Retain E0 (Full-Frame Baseline ResNet50) as the development benchmark.**
  - E2's caliper augmentation did not outperform E0 when evaluated symmetrically under either selection rule.
  - Adding synthetic calipers on top of full-frame scans does not improve generalization and adds synthetic noise artifacts.
- **Selection Rule Standard:**
  - Standardize on **Validation Accuracy at $\tau=0.50$ with Validation Loss as tiebreaker (Rule 2)** for classification goals, while updating the training script to save **BOTH** checkpoints:
    - `*_fold_{fold}_best_loss.pth`
    - `*_fold_{fold}_best_acc.pth`
  - This guarantees that both minimum-loss and peak-accuracy weights are physically saved on disk and available for retrospective testing.

### Action 3: Next Experiment Proposal
Once the training cohort is cleansed of the 67 Doppler and obstetric scans:
1. Re-run E0 on the truly clean cohort as the definitive baseline.
2. Proceed to **E3 (Auxiliary Localization Head)** using only the verified Mendeley V2 bounding-box annotations on clean training folds, without caliper augmentation and without viewport distortion.
