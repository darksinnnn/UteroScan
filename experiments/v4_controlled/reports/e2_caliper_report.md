# UteroScan V4 — Experiment E2: Synthetic Caliper Augmentation Ablation Report

**Experiment Name:** `e2_caliper`  
**Execution Date:** 2026-10-15  
**Hardware:** NVIDIA GeForce RTX 3050 Ti Laptop GPU (`cuda`)  
**Artifacts Generated:**
- Checkpoints: `experiments/v4_controlled/checkpoints/e2_caliper_fold_[0-4]_best.pth`
- Report JSON: [`e2_caliper_report.json`](file:///d:/Project/UteroScan/experiments/v4_controlled/reports/e2_caliper_report.json)
- Contact Sheet: [`caliper_augmentation_contact_sheet.png`](file:///d:/Project/UteroScan/experiments/v4_controlled/reports/caliper_augmentation_contact_sheet.png)

---

## 1. Executive Summary & Decision Rule Verdict

In accordance with the UteroScan V4 controlled experimentation protocol, **Experiment E2** tested whether adding class-independent synthetic caliper augmentation ($p=0.25$) improves classification performance over the **E0 Full-Frame ResNet50 baseline** on the 1,224-scan clean development set across 5 study-proxy grouped folds.

### Decision Rule Outcome:
- **Result:** **Retain E0 as the leading development baseline.**
- **Empirical Rationale:** While E2 maintained high overall performance (**93.97% mean CV accuracy**, **0.9859 ROC-AUC**) and yielded a breakthrough in Fold 2 (reaching **96.84% accuracy** and **0.9961 AUC**), its average cross-validation accuracy across all 5 folds was **1.27% lower** than E0 (**95.24%** $\rightarrow$ **93.97%**), with 4 out of 5 folds showing mild sensitivity or specificity degradation.
- In strict adherence to the project protocol: **we do not retain an augmentation simply because it sounds clinically plausible.** E0 remains our benchmark.

---

## 2. Experimental Setup & Verification of Class Independence

- **Baseline Architecture:** ResNet50 (Full frame, uncropped, $224 \times 224$).
- **Seed & Partitions:** Exact matched random seed (`seed=42`) and identical 5-fold study-proxy grouped splits as E0 (0% patient/study leakage).
- **Augmentation Protocol:** Synthetic measurement crosses and dashed lines applied during **training only** with target $p=0.25$. Validation sets remained 100% clean and unaugmented.
- **Visual Verification:** Contact sheet generated showing authentic placement across 10 sample pairs:

![Caliper Augmentation Contact Sheet](file:///d:/Project/UteroScan/experiments/v4_controlled/reports/caliper_augmentation_contact_sheet.png)

### Class-Independence Audit Across Training Folds
To prove that the transform did not induce differential artifacts between classes:

| Fold | UF Augmented / Total | UF Aug % | NUF Augmented / Total | NUF Aug % | Delta (|UF% - NUF%|) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 802 / 3,186 | 25.17% | 704 / 2,760 | 25.51% | 0.34% |
| **Fold 1** | 1,113 / 4,224 | 26.35% | 855 / 3,576 | 23.91% | 2.44% |
| **Fold 2** | 1,059 / 4,216 | 25.12% | 955 / 3,552 | 26.89% | 1.77% |
| **Fold 3** | 1,049 / 4,176 | 25.12% | 879 / 3,584 | 24.53% | 0.59% |
| **Fold 4** | 998 / 4,160 | 23.99% | 929 / 3,752 | 24.76% | 0.77% |
| **Pooled Total** | **5,021 / 19,962** | **25.15%** | **4,322 / 17,224** | **25.09%** | **0.06%** |

*Verification:* Across nearly 40,000 training sample presentations, the augmentation rate differed by only **0.06%** between UF and NUF images, demonstrating empirical class independence.

---

## 3. Head-to-Head Comparison: E0 Baseline vs E2 Caliper

### A. 5-Fold Cross-Validation Summary

| Metric | E0 Baseline (Full Frame) | E2 Caliper ($p=0.25$) | Delta ($\Delta$) | Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Accuracy** | **95.24%** $\pm$ 1.06% | **93.97%** $\pm$ 2.02% | 🔻 **-1.27%** | E0 Favored |
| **Mean Balanced Accuracy** | **95.04%** $\pm$ 1.25% | **93.68%** $\pm$ 2.32% | 🔻 **-1.36%** | E0 Favored |
| **Mean UF Sensitivity** | **97.57%** $\pm$ 0.53% | **96.19%** $\pm$ 1.76% | 🔻 **-1.39%** | E0 Favored |
| **Mean NUF Specificity** | **92.51%** $\pm$ 2.82% | **91.18%** $\pm$ 4.45% | 🔻 **-1.34%** | E0 Favored |
| **Mean ROC-AUC** | **0.9875** $\pm$ 0.0022 | **0.9859** $\pm$ 0.0060 | 🔻 **-0.0016** | Comparable |
| **Mean UF F1-Score** | **95.66%** $\pm$ 0.89% | **94.55%** $\pm$ 1.61% | 🔻 **-1.11%** | E0 Favored |

---

### B. Per-Fold Granular Delta Breakdown

| Fold | Metric | E0 Baseline | E2 Caliper | Delta ($\Delta$) | Confusion Matrix (TP / FN / FP / TN) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Fold 0** | Accuracy<br>UF Sens<br>NUF Spec<br>ROC-AUC | 93.13%<br>98.41%<br>86.92%<br>0.9835 | 91.42%<br>95.24%<br>86.92%<br>0.9790 | -1.71%<br>-3.17%<br>0.00%<br>-0.0045 | TP: 120, FN: 6<br>FP: 14, TN: 93 |
| **Fold 1** | Accuracy<br>UF Sens<br>NUF Spec<br>ROC-AUC | 95.58%<br>97.67%<br>93.33%<br>0.9880 | 94.78%<br>94.57%<br>95.00%<br>0.9875 | -0.80%<br>-3.10%<br>+1.67%<br>-0.0005 | TP: 122, FN: 7<br>FP: 6, TN: 114 |
| **Fold 2** | Accuracy<br>UF Sens<br>NUF Spec<br>ROC-AUC | 95.65%<br>96.92%<br>94.31%<br>0.9904 | **96.84%**<br>**99.23%**<br>94.31%<br>**0.9961** | 🔼 **+1.19%**<br>🔼 **+2.31%**<br>0.00%<br>🔼 **+0.0057** | TP: 129, FN: 1<br>FP: 7, TN: 116 |
| **Fold 3** | Accuracy<br>UF Sens<br>NUF Spec<br>ROC-AUC | 96.06%<br>97.78%<br>94.12%<br>0.9879 | 94.88%<br>94.81%<br>94.96%<br>0.9806 | -1.18%<br>-2.97%<br>+0.84%<br>-0.0073 | TP: 128, FN: 7<br>FP: 6, TN: 113 |
| **Fold 4** | Accuracy<br>UF Sens<br>NUF Spec<br>ROC-AUC | 95.74%<br>97.08%<br>93.88%<br>0.9880 | 91.91%<br>97.08%<br>84.69%<br>0.9861 | -3.83%<br>0.00%<br>-9.19%<br>-0.0019 | TP: 133, FN: 4<br>FP: 15, TN: 83 |

---

### C. Pooled Out-of-Fold Validation (1,224 Scans)

| Operating Rule | Metric | E0 Baseline | E2 Caliper | Delta |
| :--- | :--- | :---: | :---: | :---: |
| **Default ($\tau=0.50$)** | Accuracy | **95.26%** | 94.04% | -1.22% |
| | UF Sensitivity | **97.56%** | 96.19% | -1.37% |
| | NUF Specificity | **92.59%** | 91.53% | -1.06% |
| | ROC-AUC | **0.9871** | 0.9850 | -0.0021 |
| | PR-AUC (UF) | **0.9880** | 0.9866 | -0.0014 |
| **Validation-Optimized ($\tau^*$)** | Optimal Threshold | $\tau^* = 0.64$ | $\tau^* = 0.62$ | -0.02 |
| | Accuracy | **95.83%** | 94.04% | -1.79% |
| | UF Sensitivity | 96.65% | 94.67% | -1.98% |
| | NUF Specificity | **94.89%** | 93.30% | -1.59% |
| | Youden's J Index | **0.9154** | 0.8797 | -0.0357 |

---

### D. Read-Only Exploratory Test Cohort (321 Scans, 5-Fold Logit Ensemble)

*Note: In accordance with protocol guidelines, this cohort is evaluated strictly as an exploratory indicator and was not used for model selection or threshold calibration.*

| Configuration | Test Accuracy | UF Sensitivity | NUF Specificity | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **E0 Ensemble ($\tau=0.50$)** | 95.02% | 98.16% | 91.77% | 0.9827 |
| **E2 Ensemble ($\tau=0.50$)** | 94.70% | 98.77% | 90.51% | **0.9903** |
| **E2 Ensemble ($\tau^*=0.62$)** | **96.26%** | 98.16% | 94.30% | **0.9903** |

*Observation:* On the exploratory test cohort, the E2 ensemble achieved higher test ROC-AUC (0.9903 vs 0.9827) and reached 96.26% accuracy when operating at $\tau^*=0.62$. However, because our decision rule is anchored strictly to cross-validation generalization on uninspected data, E0 remains the authoritative benchmark.

---

## 4. Key Scientific Insights

1. **Synthetic Calipers Do Not Harm Learning, but Do Not Provide a Broad Cross-Fold Gain:**  
   Unlike viewport cropping (E1), which severely damaged performance (-3.1% mean drop across all folds), synthetic caliper augmentation remained very close to baseline performance (~94% accuracy, ~0.986 AUC). However, it did not produce consistent gains across the folds (only 1 out of 5 folds improved).
2. **Why Calipers Did Not Lift Overall CV Performance:**  
   In clean, uncropped scans, ResNet50 already focuses strongly on genuine uterine acoustic features (hypoechoic acoustic shadowing, texture heterogeneity, capsule distortion). While synthetic calipers make the network invariant to crosshairs, the existing training set does not appear to suffer from enough spurious caliper correlation to justify adding synthetic line artifacts as a mandatory default transform.
3. **Implication for Next Steps:**  
   E0 (Full-Frame ResNet50 at **95.24% CV Accuracy**) remains our primary champion baseline. The next logical enhancement should focus on **Mendeley V2 focal localization supervision (auxiliary bounding-box regression head)**, testing whether teaching ResNet50 *where* the fibroid is located improves feature representations on the winning E0 baseline.
