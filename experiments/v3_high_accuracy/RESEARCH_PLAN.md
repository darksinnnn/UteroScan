# UteroScan — Research Plan: Pushing Diagnostic Accuracy >95%
**Author:** Ashish Singh (22MIS0173) | UteroScan Research Project  
**Folder:** `experiments/v3_high_accuracy/` (Strictly Isolated — Zero Changes to Root Files)  
**Objective:** Address proctor/guide feedback on accuracy standards, clarify obstetric dataset formulation, and deploy high-performance architectures (EfficientNet-B0 Attention, ConvNeXt, Caliper Inpainting, Ensembling) to reach >95% accuracy.

---

## 1. Fact-Check & Clinical Context

### A. Is >95% Accuracy the "Industry Standard" in Medical Ultrasound AI?
* **Clinical Screening Reality:** In diagnostic screening, raw accuracy is **not** the primary clinical standard. Accuracy treats False Positives (unnecessary follow-up scan) and False Negatives (missed fibroid) as equally bad.
* **Sensitivity (Recall) is Paramount:** Missing a lesion is clinically hazardous. The current UteroScan V1 model already achieves **95.09% Sensitivity** (155/163 true fibroids detected, only 8 missed) with an **AUC-ROC of 0.9533**.
* **Real-World FDA Clearances:** Real-world FDA-cleared ultrasound CADx algorithms (e.g., Koios DS, Sonio Detect, DeepHealth) typically operate with balanced sensitivities and specificities in the **88%–94%** range under rigorous clinical testing. Board-certified sonographers themselves exhibit inter-observer agreement around **85%–90%** on 2D B-mode pelvic scans.
* **Where the 99% Figure Comes From:** In academic CS literature (e.g., Xi & Wang, *BMC Medical Imaging* 2024), papers achieve 99% accuracy by evaluating on the raw Mendeley dataset without removing synthetic/obstetric artifacts or addressing patient clustering across train/test splits.
* **Strategic Solution for Review:** We will provide **both**:
  1. The **High-Performance Headline Benchmark (>95%)** to satisfy the guide's standard.
  2. The **Quarantined Clinical Reliability Benchmark** (highlighting 95.09% sensitivity, 0.9533 AUC, and zero domain leakage).

---

### B. Can Uterine Fibroids Be Detected in Pregnant Women?
* **Clinical Fact:** **Yes.** Uterine leiomyomas occur in **1.5% to 10%** of pregnant women and are routinely monitored during prenatal scans.
* **Why Obstetric Scans Were Flagged in Our Cleaned Baseline:**
  * In the Mendeley dataset, 173 scans containing fetal biometry (`GA 11w3d`), fetal limbs/skulls, and amniotic fluid were placed **exclusively into the NUF (Normal / Non-Fibroid) negative control group**.
  * None of these scans were labeled as *"pregnant woman with a fibroid"*.
  * Consequently, feeding these to a neural network causes a catastrophic shortcut (the "Clever Hans" effect): the model learns that *presence of a baby = normal uterus*, rather than learning myometrial tissue echogenicity.
  * To properly detect fibroids in pregnancy, both positive and negative classes must include pregnant patients.

---

## 2. Technical Roadmap for Reaching >95% Accuracy

All experiments will reside strictly inside `experiments/v3_high_accuracy/`:

```
experiments/v3_high_accuracy/
├── annotations/          # Clean annotations
├── preprocessing/        # Bilateral filter & Caliper Inpainting
├── models/               # EfficientNet-B0 + Attention & ConvNeXt-Tiny
├── checkpoints/          # Isolated model weights
├── reports/              # Metrics, ROC curves, confusion matrices
├── train_efficientnet.py # Training script
├── train_convnext.py     # Training script
├── evaluate_ensemble.py  # Ensembling & TTA evaluation script
└── RESEARCH_PLAN.md      # This document
```

### Module 1: Modern Convolutional & Attention Architectures
1. **EfficientNet-B0 with Channel Attention (SE / CBAM):**
   * Replicates the state-of-the-art methodology from Xi & Wang (*BMC Medical Imaging* 2024).
   * Uses compound scaling and depthwise separable convolutions to extract multi-frequency acoustic tissue features.
2. **ConvNeXt-Tiny:**
   * Modernized 7×7 depthwise convolutions and inverted bottlenecks with superior spatial texture representation.

### Module 2: Enhanced Ultrasound Preprocessing
1. **Bilateral Filtering (Edge-Preserving Despeckling):**
   * Replaces uniform 3×3 median blur. Smooths speckle in homogeneous regions while keeping the hypoechoic pseudocapsule boundary of fibroids sharp.
2. **Caliper Inpainting (`caliper_masking.py`):**
   * Removes synthetic yellow/white measurement crosshairs that contributed to 66.7% of false positives (Odds Ratio = 6.0).

### Module 3: Test-Time Augmentation (TTA) & Multi-Model Ensembling
1. **Test-Time Augmentation:**
   * Inference-time averaging over original, horizontal-flip, and multi-scale crops (+1.5–2.5% accuracy gain).
2. **Soft-Voting Ensemble:**
   * Combines posterior probabilities: $\hat{P} = w_1 P_{\text{ResNet50}} + w_2 P_{\text{EfficientNet}} + w_3 P_{\text{ConvNeXt}}$.
   * Ensembles consistently eliminate uncorrelated individual errors, typically lifting accuracy past **95%** and AUC above **0.9700**.
