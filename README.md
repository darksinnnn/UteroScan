# UteroScan: Deep Residual CADx Platform for Uterine Fibroid Sonography

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.2-61dafb.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8.3-646cff.svg)](https://vite.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

UteroScan is a deep learning computer-aided detection (CADx) research platform designed to distinguish **uterine leiomyomas (fibroids, UF)** from **normal physiological myometrium (non-uterine fibroid, NUF)** in 2D pelvic B-mode ultrasound imaging.

The authoritative development baseline is **Recovered E0: Full-Frame ResNet50 with uniform Adam optimization ($\eta = 1\times 10^{-4}$)**, achieving **94.12% ± 0.64% mean outer accuracy** and **0.9843 ROC-AUC** across 5-fold nested grouped cross-validation.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Development History & Ablations](#2-development-history--ablations)
3. [Recovered Development Cohort & Partitioning](#3-recovered-development-cohort--partitioning)
4. [E0 Model Architecture & Clinical Preprocessing](#4-e0-model-architecture--clinical-preprocessing)
5. [Authoritative Benchmark Metrics](#5-authoritative-benchmark-metrics)
6. [Production Deployment Architecture](#6-production-deployment-architecture)
7. [Running the Project](#7-running-the-project)
8. [Testing & Verification](#8-testing--verification)
9. [Documentation Links](#9-documentation-links)
10. [Investigational Disclaimer](#10-investigational-disclaimer)

---

## 1. Project Overview

### 1.1 The Clinical Challenge
Uterine leiomyomas (fibroids) are benign monoclonal tumors of the myometrium affecting up to 70% of reproductive-age women, frequently causing abnormal uterine bleeding, pelvic pain, and subfertility. While pelvic ultrasound is the primary diagnostic modality, sonographic interpretation is vulnerable to operator variability, acoustic shadowing artifacts, transducer gain fluctuations, and confusing non-focal myometrial heterogeneity.

### 1.2 Binary Classification Scope
UteroScan operates directly on single-frame 2D B-mode ultrasound images:
- **Class 0 — Uterine Fibroid (UF)**: Scans showing focal circumscribed, lobulated, or intramural hypoechoic/heterogeneous myometrial masses with characteristic acoustic margins.
- **Class 1 — Non-Uterine Fibroid (NUF)**: Scans exhibiting uniform physiological intermediate echogenicity, intact central endometrial stripe, and pelvic soft tissue free of focal leiomyomas.

### 1.3 Project Status & Limitations
- **Current Status**: Standalone production model trained on 1,474 development scans for 5 epochs (inner-validation median). Integrated into a high-performance FastAPI backend (`api/server.py`) and a modern interactive React 19 frontend (`frontend/`).
- **Clinical Limitations**: The underlying open dataset lacks true patient Hospital Medical Record Numbers (MRNs). While acquisition-study proxy grouping was strictly enforced, multi-session independence cannot be established without institutional patient records. UteroScan is an investigational research prototype, not a certified clinical device.

---

## 2. Development History & Ablations

UteroScan underwent rigorous empirical iterations before establishing E0 as the authoritative baseline:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Development Trajectory                          │
│                                                                        │
│  [V1 Historical Baseline] (~86.6% Acc; Contaminated by obstetric scans)│
│               │                                                        │
│               ▼                                                        │
│  [Comprehensive Audit] (Quarantined 64 fetal scans, resolved duplicates)│
│               │                                                        │
│               ▼                                                        │
│  [V3 Attention & Ensembles] (CBAM + 5-fold bagging; expressive but no  │
│                              significant gain over standard residual)  │
│               │                                                        │
│               ▼                                                        │
│  [E1 Viewport Cropping] (Acoustic context loss; degraded sensitivity)  │
│               │                                                        │
│               ▼                                                        │
│  [E2 Caliper Inpainting/Aug] (Caliper presence non-informative)        │
│               │                                                        │
│               ▼                                                        │
│  [E3 Auxiliary Localization Head] (Multi-task interference; acc dropped│
│                                    from 94.12% to 92.8%)               │
│               │                                                        │
│               ▼                                                        │
│  [E4/E5 Differential Learning Rates] (conv1-2: 1e-5, layer3-4: 5e-5;  │
│                                       acc dropped to 93.36%, -0.76%)   │
│               │                                                        │
│               ▼                                                        │
│  [E0 Baseline Locked] ──> Full-frame ResNet50, uniform Adam lr 1e-4    │
│                           94.12% ± 0.64% outer acc | 0.9843 ROC-AUC    │
└────────────────────────────────────────────────────────────────────────┘
```

- **V1 Baseline (Historical, Sept 2026)**: Initial two-stage fine-tuning achieved ~86.60% accuracy on early test sets, but audits discovered 202 unpunctuated obstetric images and non-standardized acoustic gains.
- **Image Audit & Recovery (Oct 2026)**: Systematic image-level review of all 1,994 unique candidate images recovered 1,474 valid development scans, quarantined 64 visible fetal scans, and isolated 456 scans into a held-out evaluation pool.
- **V3 Attention & Ensembles**: CBAM (Convolutional Block Attention Module) and 5-fold ensembling increased parameter overhead without improving cross-validation discriminability on despeckled inputs.
- **E1 Viewport Cropping**: Tight fan-beam cropping removed peripheral anatomical cues, degrading sensitivity.
- **E2 Caliper Augmentation**: Caliper masking and synthetic caliper overlays yielded no gain because clinical sonographers apply calipers to both normal anatomy and lesions.
- **E3 Auxiliary Localization**: Adding a bounding-box regression head trained on 607 annotations caused multi-task gradient interference that degraded classification accuracy from 94.12% to 92.8%.
- **E5 Differential Learning Rates**: Training with lower learning rates on early layers (`1e-5` on conv1–layer2, `5e-5` on layer3–4, `1e-4` on head) achieved **93.36% ± 0.99%** outer accuracy, a **0.76% degradation** compared to E0's **94.12% ± 0.64%**.
- **Decision**: E0 (uniform Adam lr `1e-4`) definitively remains the chosen development baseline.

---

## 3. Recovered Development Cohort & Partitioning

The authoritative dataset census established in `experiments/final_data_audit/master_review_manifest.csv`:

| Partition | Scans | Fibroid (UF) | Non-Fibroid (NUF) | Study Groups | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Development Cohort** | **1,474** | **685** | **789** | **917** | **Active training & cross-validation** |
| **Held-Out Evaluation Pool** | **456** | 224 | 232 | 312 | **Strictly quarantined (Zero access)** |
| **Visible Fetal Anatomy** | **64** | 0 | 64 | 42 | **Quarantined & excluded** |
| **Master Inventory Total** | **1,994** | **909** | **1,085** | **1,271** | **All reviewed unique scans** |

### Grouped Cross-Validation Partitioning
- **5-Fold Nested Grouped Partitioning**: Assigned in `experiments/final_data_audit/recovered_folds.csv`.
- **Acquisition-Study Independence**: All scans belonging to the same study proxy group are assigned to the same fold. Zero intra-study leakage across folds.
- **Evaluation Isolation**: The 456 held-out scans were never used for training, learning rate tuning, or checkpoint selection.

---

## 4. E0 Model Architecture & Clinical Preprocessing

### 4.1 Neural Network Configuration
- **Backbone**: Torchvision standard `ResNet50` initialized with ImageNet-1K V2 weights (`ResNet50_Weights.IMAGENET1K_V2`).
- **Classifier Head**: Standard linear layer `nn.Linear(2048, 2)`.
- **Optimizer**: Adam with uniform learning rate $\eta = 1\times 10^{-4}$, weight decay $\lambda = 1\times 10^{-4}$.
- **Loss Function**: Unweighted Cross-Entropy (`nn.CrossEntropyLoss`).
- **Batch Size**: 16 | **Seed**: 42.

### 4.2 Standard Clinical Preprocessing (Inference-Time)
Every input undergoes the exact clinical transform without stochastic augmentations:
1. **Luminance Ingestion**: Converted to single-channel grayscale (`cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)`).
2. **Area Resizing**: Resampled to $224 \times 224$ using `cv2.INTER_AREA` (preserves subtle acoustic speckle contrast without aliasing).
3. **Median Despeckling**: Filtered with a $3 \times 3$ median kernel (`cv2.medianBlur(gray, 3)`), attenuating transducer speckle.
4. **Contrast Standardization (CLAHE)**: Enhanced using Contrast-Limited Adaptive Histogram Equalization (`clipLimit=1.2`, tile grid $(8, 8)$), equalizing gain disparities across ultrasound manufacturers.
5. **3-Channel Replicate**: Stacked into RGB: `np.stack([clahe, clahe, clahe], axis=-1)`.
6. **ImageNet Normalization**: Scaled to $[0, 1]$ and normalized with ImageNet parameters ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### 4.3 Training-Only Augmentations
Applied stochastically during training only (never at test time):
- Random horizontal flip ($p = 0.5$).
- Random rotation $\pm 15^\circ$ with border reflection (`cv2.BORDER_REFLECT`).
- No vertical flip (preserves anatomical depth perspective).

---

## 5. Authoritative Benchmark Metrics

Evaluated via 5-fold nested grouped cross-validation across all 1,474 development scans:

### 5.1 Maximum Validation Accuracy Checkpoint Rule (Rule 2)

| Metric | Mean Outer-Fold | Std Dev | Pooled Cohort (n=1,474) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **94.12%** | **±0.64%** | **94.10%** (1,387 / 1,474) |
| **Balanced Accuracy** | 94.16% | ±0.55% | 94.14% |
| **Sensitivity (Recall)** | **94.75%** | **±1.50%** | **94.74%** (649 / 685) |
| **Specificity** | **93.57%** | **±1.93%** | **93.54%** (738 / 789) |
| **Precision (PPV)** | 92.78% | ±2.12% | 92.71% (649 / 700) |
| **F1-Score** | **0.9372** | **±0.0084** | **0.9372** |
| **ROC-AUC** | **0.9843** | **±0.0039** | **0.9833** |
| **PR-AUC (Average Precision)** | **0.9822** | **±0.0047** | **0.9799** |

*Rule 2 Confusion Matrix Counts*: TP: 649 | FN: 36 | FP: 51 | TN: 738.

### 5.2 Minimum Validation Loss Checkpoint Rule (Rule 1)
- Mean Outer Accuracy: **93.65% ± 0.74%** | Pooled Accuracy: **93.62%**
- Mean Sensitivity: **94.31% ± 2.48%** | Mean Specificity: **93.10% ± 2.18%**
- Mean ROC-AUC: **0.9847 ± 0.0035** | Mean F1: **0.9321 ± 0.0102**

*Note*: These are development cross-validation metrics on the recovered cohort, not independent clinical multicenter trial results.

---

## 6. Production Deployment Architecture

To transition from 5-fold cross-validation to a single deployment model:
1. **Production Retraining**: Trained on all 1,474 development scans using the median epoch count selected by Rule 2 across the 5 outer folds:
   $$\text{Selected Epochs} = [3, 5, 5, 5, 6] \implies \text{Median} = \mathbf{5\text{ epochs}}.$$
2. **Standalone Checkpoint**: Saved to `checkpoints/production/resnet50_e0_production.pth` (90.0 MB) with complete architecture and optimizer metadata.
3. **Dedicated Inference Engine (`api/inference.py`)**: Loads the checkpoint once into GPU memory on server initialization with parameter gradient freezing (`requires_grad=False`).
4. **FastAPI Server (`api/server.py`)**: Exposes REST endpoints serving predictions with image sanitization, dual-threshold calibration, and low latency (<20ms GPU inference).

---

## 7. Running the Project

### 7.1 Prerequisites
- Python 3.10+ with PyTorch and CUDA support.
- Node.js 18+ and npm.

### 7.2 Installation
```bash
# Clone the repository
git clone https://github.com/darksinnnn/UteroScan.git
cd UteroScan

# Install Python dependencies
pip install torch torchvision numpy pandas opencv-python Pillow fastapi uvicorn pydantic pytest httpx

# Install Frontend dependencies
npm --prefix frontend install
```

### 7.3 Reproduce Production Model Training
To train the standalone E0 production checkpoint from scratch on the 1,474 development scans for the median 5 epochs:
```bash
python scripts/train_production_e0.py
```

### 7.4 Start the Backend Server
```bash
python api/server.py
# Or using uvicorn directly:
uvicorn api.server:app --host 127.0.0.1 --port 8000
```
Backend will be available at `http://127.0.0.1:8000`. Interactive API documentation (Swagger UI) is at `http://127.0.0.1:8000/docs`.

### 7.5 Start the Frontend Application
```bash
npm --prefix frontend run dev
```
Frontend UI will be accessible at `http://localhost:3000`.

To run both backend and frontend concurrently in development mode from project root:
```bash
npm run dev
```

---

## 8. Testing & Verification

### 8.1 Run the Automated Test Suite
```bash
# Run the complete production backend and inference test suite
pytest tests/ -q

# Run with full verbosity
pytest tests/test_production_backend.py -v
```

### 8.2 Frontend Build & Lint Checks
```bash
# Verify production frontend build
npm --prefix frontend run build

# Run frontend linter
npm --prefix frontend run lint
```

### 8.3 Run an Inference Smoke Test
```bash
python -c "from api.inference import UteroScanInferenceEngine; engine = UteroScanInferenceEngine(); res = engine.predict(open('datasets/datasets/train/UF/1.2.826.0.1.3680043.2.461.11219581.1358591605.jpg', 'rb').read(), source_name='Smoke Test'); print(res['classification'])"
```

---

## 9. System Resources & Architecture

- **Interactive Diagnostic Workspace**: Run `npm --prefix frontend run dev` to launch the client GUI with live Grad-CAM visualizer, dual threshold triage slider, and empirical clinical analytics.
- **Production Inference Pipeline**: Fully implemented in `api/inference.py` and `api/server.py`.
- **Model Training & Cross-Validation**: Stored in `scripts/train_production_e0.py` and `experiments/final_data_audit/`.
- **Automated Verification Suite**: Located in `tests/test_production_backend.py`.

---

## 10. Investigational Disclaimer

> **IMPORTANT MEDICAL NOTICE**:  
> UteroScan is an investigational computer vision research platform developed for algorithmic benchmarking, data provenance study, and academic research in gynecologic oncology imaging. It is **not** an FDA-cleared, CE-marked, or certified medical diagnostic software device.  
> UteroScan predictions must never be utilized as an autonomous clinical diagnosis or substitute for expert interpretation by board-certified radiologists, sonographers, or gynecologists.
