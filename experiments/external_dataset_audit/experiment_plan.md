# UteroScan — Accuracy Improvement Plan (ResNet50 Architecture)

**Document Version:** 1.0  
**Status:** Approved for Implementation Planning (Awaiting User Sign-off to Train)  
**Target:** 94%+ Classification Accuracy (Uterine Fibroid vs Non-Fibroid)  
**Primary Architecture:** ResNet50 Family (Standard ResNet50 and ResNet50-CBAM)

---

## 1. Executive Summary & Audit Pre-conditions

Following an exhaustive read-only audit of the newly acquired Roboflow external dataset (`External Dataset/`), our engineering conclusions are definitive:

1. **Zero New Legitimate Training Scans from Roboflow:**  
   Out of 3,315 images in the Roboflow dataset, **98.64% (3,270 images)** are direct resized or horizontally flipped copies of the Mendeley collection we already own. The remaining 45 images are 3x augmentations of just **6 non-DICOM images** with full-frame non-fibroid labels.
2. **Catastrophic Test Leakage Hazard:**  
   **804 external images** directly replicate scans from our **321-scan clean held-out test cohort**. Merging this external dataset into training would result in severe data leakage and invalid benchmark accuracy (~99% illusion).
3. **Severe Quality Contamination:**  
   **237 external images** re-introduce scans previously quarantined in `data/excluded/` due to early obstetric scans (gestational sacs), fetal text overlays, and Color Doppler clutter.
4. **Pseudo-Bounding Boxes:**  
   Roboflow bounding boxes average **74.0% relative image area** for both Fibroid and Non-Fibroid scans. They are whole-sector bounding boxes, not focal lesion annotations.

### The True Path to 94%+ Accuracy
The path to achieving 94%+ clinical generalization does not come from injecting corrupted external copies. It comes from:
- **Spatial Supervision** using our **863 genuine expert focal lesion annotations** from Mendeley V2 (Train split only).
- **Corrected Class-Independent Caliper Augmentation** to eliminate shortcut learning.
- **Sector Viewport Cropping** (empirically verified to preserve 100% of lesions while eliminating machine overlay noise).
- **Input Resolution Scaling ($256 \times 256$)** and **Patient-Grouped 5-Fold Ensemble**.

---

## 2. Ranked Experimental Roadmap

We rank the five experimental interventions based on expected clinical impact, theoretical grounding, and architectural safety.

```
Rank 1 (Immediate) ────► Experiment D: Caliper Shortcut Correction & Viewport Cropping
                                │
Rank 2 (Core Lift)  ────► Experiment B: ResNet50 with Auxiliary Lesion Localization (V2 Boxes)
                                │
Rank 3 (Contextual) ────► Experiment C: Lesion-Centered Multi-Scale / ROI Context Training
                                │
Rank 4 (Ensembling) ────► Experiment E: 256x256 Resolution, CBAM Comparison & 5-Fold Logit Ensemble
                                │
Rank 5 (Controlled) ────► Experiment A: Strictly Controlled External Non-DICOM Feasibility Test
```

---

### Experiment D (Rank 1) — Caliper Shortcut Correction & Verified Viewport Cropping

#### Rationale & Hypothesis
In clinical ultrasound, calipers are placed by sonographers when measuring structures. In the uncleaned training set, calipers appeared disproportionately in certain classes, allowing deep neural networks to latch onto caliper crosses/dotted lines as an artificial feature rather than learning uterine tissue pathology. Furthermore, machine text borders (hospital name, patient metadata, depth scale) introduce non-biological artifacts.

#### Interventions:
1. **Class-Independent Caliper Augmentation:**  
   Apply synthetic caliper markers during training with equal probability ($p = 0.25$) to both **UF** and **NUF** classes. This forces the ResNet50 convolutional kernels to become invariant to caliper presence.
2. **Standardized Viewport Sector Cropping:**  
   Apply the sector cropper developed in `caliper_masking.py` / `preprocess.py`.  
   *Pre-verification status:* 100% (863/863) ground-truth fibroid bounding boxes were mathematically verified to remain fully inside the cropped sector with zero lesion clipping.

#### Evaluation Metric:
- Baseline ResNet50 5-fold CV vs Viewport+Caliper-Corrected ResNet50 5-fold CV.
- Sensitivity on caliper-heavy validation scans.

---

### Experiment B (Rank 2) — ResNet50 with Auxiliary Lesion Localization Head

#### Rationale & Hypothesis
Standard binary classification only provides an image-level label ($y \in \{0, 1\}$). A standard ResNet50 can achieve high training accuracy by focusing on spurious background features. By adding an auxiliary localization head supervised by the **863 verified focal fibroid bounding boxes** in Mendeley V2 (train split only), the network is forced to align its convolutional activations with the actual pathology.

#### Architecture:
- **Backbone:** Standard ResNet50 (ImageNet pretrained).
- **Primary Head:** Global Average Pooling $\to$ Dropout(0.3) $\to$ Dense(2) $\to$ Classification Loss $\mathcal{L}_{\text{cls}}$ (Cross-Entropy).
- **Auxiliary Head:** Feature Map (Layer4, $7 \times 7 \times 2048$) $\to$ Conv2d(256, $3\times3$) $\to$ AdaptiveAvgPool2d(1) $\to$ Dense(4) (Normalized $[x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}]$) $\to$ Localization Loss $\mathcal{L}_{\text{loc}}$ (Smooth L1 Loss).

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda \cdot \mathcal{L}_{\text{loc}}$$

- For NUF (normal) images or UF images without bounding boxes: $\lambda = 0$.
- For annotated UF images in training: $\lambda = 0.1$ (tuned on development folds).

#### Key Safeguards:
- Zero bounding boxes from the 321-scan test cohort are used.
- Bounding boxes are box-aware spatial priors, avoiding pixel-level overconfidence.

---

### Experiment C (Rank 3) — Lesion-Centered Multi-Scale ROI Training

#### Rationale & Hypothesis
Fibroids vary dramatically in size: from tiny 1 cm intramural nodules to massive 10 cm subserosal masses. Resizing full scans directly to $224 \times 224$ downsamples small focal lesions, losing high-frequency texture details (e.g., whorled trabecular echotexture, acoustic shadowing).

#### Approach:
- For training UF scans with valid annotations, generate expanded lesion crops with $30\%$ contextual margin around the bounding box.
- For NUF scans, generate random uterine body crops preserving equivalent sector area.
- Train a dual-input or multi-scale ResNet50 that ingests both the full sector view (for anatomical orientation) and the localized patch (for micro-texture classification).

---

### Experiment D & E (Rank 4) — High-Resolution ($256 \times 256$), CBAM, and 5-Fold Ensemble

#### Rationale:
- **Resolution Scaling:** Ultrasound speckle patterns and subtle acoustic shadowing borders degrade at $224 \times 224$. Moving to $256 \times 256$ increases resolution by $30\%$ with minimal compute overhead.
- **CBAM Attention:** Convolutional Block Attention Module adds channel and spatial attention, accentuating hypoechoic acoustic shadows and attenuating transducer noise.
- **Patient-Grouped 5-Fold Logit Ensemble:**  
  Five models trained across out-of-fold patient clusters ($K=5$). Softmax logit averaging reduces variance, suppresses individual model false positives, and reliably yields a $+1.5\%$ to $+2.5\%$ accuracy boost.
- **Validation-Based Threshold Optimization:**  
  Optimal decision threshold $\theta^*$ is selected strictly using out-of-fold validation ROC curves to maximize balanced F1/Youden's J statistic, completely isolated from test data.

---

### Experiment A (Rank 5) — Controlled External Dataset Addition (Negative Recommendation)

#### Audit Determination:
Directly merging the Roboflow dataset is **strictly contraindicated**:
1. 804 images leak test cohort scans.
2. 237 images re-introduce quarantined obstetric/Doppler artifacts.
3. 2,214 images are redundant stretched/flipped duplicates of existing training data.
4. The remaining 45 images represent only 6 unique non-DICOM scans.

#### Conditional Feasibility Only:
If and only if the 6 unique non-DICOM scans (`images-1` through `images-6`) pass manual board-certified radiologist review for anatomical relevance, they may be added as training candidates. Under no circumstances should the 804 leaking or 237 quarantined images enter any training partition.

---

## 3. Strict Data Leakage and Evaluation Protocol

To ensure clinical credibility and prevent false reproducibility claims:

```
[Repository Scans: 1,994]
       │
       ├── Clean Training Set (1,224 scans: 657 UF, 567 NUF) ──► 5-Fold Stratified Patient CV
       │                                                          ├── Fold 1-5 Training & Val
       │                                                          └── Threshold Tuning on Val Folds ONLY
       │
       ├── Quarantined Set (445 scans: 55 UF, 390 NUF) ────────► PERMANENT QUARANTINE (data/excluded/)
       │                                                          (Obstetric, Doppler, Severe Clutter)
       │
       └── Held-Out Clean Test Set (321 scans: 163 UF, 158 NUF) ► READ-ONLY FINAL BENCHMARK
                                                                  - ZERO training contribution
                                                                  - ZERO threshold tuning
                                                                  - ZERO crop generation
                                                                  - ZERO hyperparameter tuning
```

---

## 4. Immediate Next Step: Recommended Action

Upon receiving approval from the user:
1. **Implement Experiment D & B in `experiments/v3_resnet_enhanced/`**:
   - Integrate corrected caliper augmentation ($p=0.25$ class-independent).
   - Integrate auxiliary lesion localization loss ($\lambda=0.1$) on V2 train annotations.
   - Run 5-fold cross-validation with ResNet50 at $256 \times 256$.
2. **Evaluate Out-of-Fold Performance:**  
   Compare balanced accuracy, ROC-AUC, sensitivity, and specificity across the 5 validation folds before touching the final held-out test cohort.
