# UteroScan — Accuracy Improvement Plan V2 (ResNet50 Architecture)

**Version:** 2.0 (Audit-Corrected Specification)  
**Status:** Read-Only Plan (Awaiting Explicit User Approval Before Training)  
**Primary Architecture:** ResNet50 Binary Classifier (Torchvision Pretrained Backbone)  
**Target:** Rigorously Evaluate Hypotheses for Reaching 94%+ Clinical Classification Accuracy

---

## 1. Methodological Philosophy & Experimental Rigor

In strict adherence to medical AI validation standards, all proposed modifications in this document are **experimental hypotheses to be evaluated**, NOT guaranteed performance promises.

```
+---------------------------------------------------------------------------------------+
| CORE ENGINEERING POSTULATE                                                            |
+---------------------------------------------------------------------------------------+
| True generalization cannot be engineered by training on corrupted or leaking data.   |
| Diagnostic accuracy must be pushed forward by:                                       |
|   1. Removing artificial shortcuts (calipers, machine borders).                       |
|   2. Directing convolutional attention to true pathology via spatial supervision.     |
|   3. Ensembling out-of-fold models over patient-clustered distributions.             |
|   4. Setting operating thresholds strictly on development folds.                      |
+---------------------------------------------------------------------------------------+
```

---

## 2. Selection of Explicit Auxiliary Localization Objective

Per audit requirements, we select **Normalized Bounding-Box Regression with Smooth L1 Loss** as the single explicit auxiliary localization objective.

### Mathematical Formulation & Tensor Specifications

```
                     [Input Image X: 3 x 256 x 256]
                                   │
                                   ▼
                   [ResNet50 Backbone (Layers 1 to 4)]
                                   │
                                   ▼
             [Feature Map F (Layer 4): 2048 x 8 x 8]
                     │                           │
                     ▼                           ▼
        [Primary Classification Head]  [Auxiliary Localization Head]
        AdaptiveAvgPool2d(1)           Conv2d(2048, 256, kernel=3, pad=1)
        Flatten()                      ReLU()
        Dropout(p=0.3)                 AdaptiveAvgPool2d(1)
        Linear(2048, 2)                Flatten()
        Logits: z in R^2               Linear(256, 4)
             │                         Sigmoid()
             │                         Box Predictions: b_hat in [0, 1]^4
             ▼                                   │
      Cross-Entropy Loss                         ▼
            L_cls                        Smooth L1 Loss
                                              L_loc
             │                                   │
             └─────────────────┬─────────────────┘
                               ▼
            L_total = L_cls + lambda * I(has_box) * L_loc
```

#### Tensor Shapes & Layer Dimensions:
1. **Input Tensor:** $X \in \mathbb{R}^{B \times 3 \times 256 \times 256}$.
2. **Backbone Feature Map ($F$):** Output of ResNet50 `layer4` has shape $B \times 2048 \times 8 \times 8$.
3. **Primary Classification Head:**
   - $\text{GAP}(F) \to \mathbb{R}^{B \times 2048}$
   - $\text{Dropout}(0.3) \to \text{Linear}(2048, 2) \to \mathbf{z} \in \mathbb{R}^{B \times 2}$
   - Loss: $\mathcal{L}_{\text{cls}} = \text{CrossEntropyLoss}(\mathbf{z}, y)$, where $y \in \{0, 1\}$.
4. **Auxiliary Localization Head:**
   - $\text{Conv2d}(2048 \to 256, \text{kernel}=3, \text{padding}=1) \to \text{ReLU} \to \mathbb{R}^{B \times 256 \times 8 \times 8}$
   - $\text{AdaptiveAvgPool2d}(1) \to \mathbb{R}^{B \times 256 \times 1 \times 1} \to \text{Flatten} \to \mathbb{R}^{B \times 256}$
   - $\text{Linear}(256, 4) \to \text{Sigmoid} \to \hat{\mathbf{b}} = [\hat{x}_{\text{min}}, \hat{y}_{\text{min}}, \hat{x}_{\text{max}}, \hat{y}_{\text{max}}] \in [0, 1]^4$.
   - Target Box: $\mathbf{b}^* = [x^*_{\text{min}}, y^*_{\text{min}}, x^*_{\text{max}}, y^*_{\text{max}}] \in [0, 1]^4$ (normalized by post-crop width $W_{\text{crop}}$ and height $H_{\text{crop}}$).
   - Loss:
     $$\mathcal{L}_{\text{loc}} = \frac{1}{4} \sum_{k=1}^4 \text{SmoothL1}(\hat{b}_k, b^*_k, \beta=1.0)$$
     where:
     $$\text{SmoothL1}(u, v) = \begin{cases} 0.5 (u - v)^2, & \text{if } |u - v| < 1.0 \\ |u - v| - 0.5, & \text{otherwise} \end{cases}$$

#### Missing-Box and Negative Sample Handling:
- For Normal/Non-Fibroid (NUF) scans ($y=0$): No fibroid bounding box exists. The indicator $\mathbb{I}(\text{has\_box}) = 0$.
- For Uterine Fibroid (UF) scans without valid annotations: $\mathbb{I}(\text{has\_box}) = 0$.
- For UF scans with verified focal lesion annotations (Train split only): $\mathbb{I}(\text{has\_box}) = 1$.
- Total Loss Objective:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda \cdot \mathbb{I}(\text{has\_box}) \cdot \mathcal{L}_{\text{loc}}$$
  We select $\lambda = 0.1$ as the baseline auxiliary weighting. Zero gradients flow from the localization head for unannotated or normal scans.

#### Coordinate Transformation Rules:
All bounding boxes from Mendeley V2 are transformed strictly alongside the image:
1. **Sector Viewport Crop:** $[x_{\text{crop}}, y_{\text{crop}}] = [x_{\text{orig}} - x_{\text{sector\_min}}, y_{\text{orig}} - y_{\text{sector\_min}}]$.
2. **Normalization to $[0, 1]$:** $\mathbf{b}^* = [x_{\text{crop\_min}}/W_{\text{sector}}, y_{\text{crop\_min}}/H_{\text{sector}}, x_{\text{crop\_max}}/W_{\text{sector}}, y_{\text{crop\_max}}/H_{\text{sector}}]$.
3. **Horizontal Flip Augmentation (when applied with $p=0.5$):**
   $$[x^*_{\text{min, flip}}, x^*_{\text{max, flip}}] = [1.0 - x^*_{\text{max}}, 1.0 - x^*_{\text{min}}]$$

---

## 3. Prioritized Experimental Hypotheses

```
+----------------------------------------------------------------------------------------------------+
| RANKED EXPERIMENTAL ROADMAP                                                                        |
+-------+-----------------------------+------------------------------------+-------------------------+
| Rank  | Experiment                  | Primary Hypothesis                 | Evaluation Metric       |
+-------+-----------------------------+------------------------------------+-------------------------+
| 1     | Experiment D: Caliper &     | Eliminates shortcut learning and   | Val Folds Balanced Acc  |
|       | Sector Viewport Cropping    | non-biological border noise        | & Specificity           |
+-------+-----------------------------+------------------------------------+-------------------------+
| 2     | Experiment B: ResNet50 +    | Forces convolutional activations   | Val Folds ROC-AUC,      |
|       | Auxiliary Localization Head | onto true focal lesions            | Sens & Missed Fibroids  |
+-------+-----------------------------+------------------------------------+-------------------------+
| 3     | Experiment E: 256x256 &     | Preserves fine acoustic speckle;   | 5-Fold Ensemble Logit   |
|       | 5-Fold Logit Ensembling     | suppresses model variance          | AUC & Youden's J        |
+-------+-----------------------------+------------------------------------+-------------------------+
| 4     | Experiment C: Contextual    | Multi-scale resolution preserves   | False Positive Rate on  |
|       | Lesion-Centered ROI Crops   | focal micro-texture details        | Small (<2cm) Nodules    |
+-------+-----------------------------+------------------------------------+-------------------------+
| 5     | Experiment A: External Data | REJECTED: Catastrophic leakage     | Contraindicated by      |
|       | Direct Ingestion            | (804 files) & label contradictions | Audit V2 findings       |
+-------+-----------------------------+------------------------------------+-------------------------+
```

### Detailed Hypothesis Specifications:

#### Hypothesis 1 (Experiment D) — Shortcut Elimination
- **Intervention 1:** Standardize Sector Viewport Cropping. The sector cropper is pre-verified to keep 100.0% (863/863) of focal lesions intact while eliminating 35% of non-biological pixel area (hospital names, depth bars).
- **Intervention 2:** Class-Independent Synthetic Caliper Augmentation. Inject synthetic calipers with equal probability ($p=0.25$) on both UF and NUF images. This hypothesis posits that equalizing caliper frequency will prevent the network from using caliper presence as a cheat feature for either class.

#### Hypothesis 2 (Experiment B) — Auxiliary Localization Supervision
- Supervise the ResNet50 backbone using the **649 eligible clean training annotations** from Mendeley V2.
- **Hypothesis:** Multi-task spatial supervision regularizes Layer 4 feature maps, aligning Grad-CAM saliency with the hypoechoic acoustic shadow of the fibroid and reducing false-positive activations on bowel gas or normal myometrium.
- **Strict Guardrail:** 0 test set annotations and 0 quarantined annotations enter training.

#### Hypothesis 3 (Experiment E) — Resolution & Ensembling
- **Input Resolution Scaling:** Scale input from $224 \times 224$ to $256 \times 256$. Ultrasound speckle patterns and subtle acoustic shadowing margins degrade when heavily downsampled; $256 \times 256$ provides 30% more pixel information with negligible latency overhead.
- **Patient/Study Clustered 5-Fold Ensemble:** Train 5 out-of-fold ResNet50 models. Average softmax probabilities:
  $$P_{\text{ens}}(\text{UF}) = \frac{1}{5} \sum_{m=1}^5 P_m(\text{UF})$$
- **Validation Threshold Calibration:** Optimal operating threshold $\theta^*$ is selected strictly on pooled out-of-fold validation predictions to maximize Youden's J index ($J = \text{Sens} + \text{Spec} - 1$).

---

## 4. Strict Cohort Isolation & Validation Protocol

```
+----------------------------------------------------------------------------------------------------+
| DATASET POPULATION & ROLES                                                                         |
+--------------------------+-------+--------+--------------------------------------------------------+
| Cohort                   | Scans | Split  | Role in Study                                          |
+--------------------------+-------+--------+--------------------------------------------------------+
| Clean Training Set       | 1,224 | Train  | 5-Fold Clustered Cross-Validation & Threshold Tuning   |
| Clean Training Annotations|  649 | Train  | Auxiliary Localization Supervision (Lambda=0.1)        |
| Quarantined Scans        |   445 | Excl.  | PERMANENT QUARANTINE (Obstetric / Doppler / Artifacts)  |
| Held-Out Clean Test Set  |   321 | Test   | LOCKED FINAL BENCHMARK (100% Read-Only, Isolated)       |
| Roboflow External Dataset| 3,315 | Excl.  | PERMANENT QUARANTINE (Leakage, Contamination, Conflicting)|
+--------------------------+-------+--------+--------------------------------------------------------+
```

### Absolute Benchmark Safeguards:
1. The 321-scan clean held-out test cohort ($163 \text{ UF} / 158 \text{ NUF}$) is completely isolated:
   - **Zero** samples used in training or validation folds.
   - **Zero** threshold tuning (threshold $\theta^*$ is chosen on validation folds).
   - **Zero** crop fitting or normalization fitting.
2. Reporting metrics: Every evaluation must report:
   - Exact operating threshold $\theta^*$.
   - True Positives, False Positives, True Negatives, False Negatives.
   - Accuracy, UF Sensitivity (Recall), NUF Specificity, UF Precision, F1.
   - Exact ROC-AUC and PR-AUC(UF).
