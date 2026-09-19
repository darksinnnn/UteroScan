# UteroScan — Product Requirements Document (PRD)

**Project:** UteroScan: Deep Learning-Based Automated Detection of Uterine Fibroids in Ultrasound Images
**Course context:** SWE3004 — Software Design and Development Project (Review-1 baseline)
**Author:** 22MIS0173 — Ashish Singh | Guide: Dr. Uma K.
**Document status:** Source of truth for Antigravity implementation. Derived from the approved Review-1 report — scope, methodology, and metrics below are **locked** and must not be silently altered during implementation. Any deviation must be flagged back to the project owner before proceeding.

---

## 1. Problem & Goal

Transvaginal ultrasound (TVUS) diagnosis of uterine fibroids is operator-dependent, suffers from inter/intra-observer variability, and is degraded by acoustic speckle noise. UteroScan is a deep-learning-based **secondary, objective diagnostic aid** that classifies a 2D static TVUS image as showing a uterine fibroid (UF) or not (NUF), to reduce diagnostic subjectivity — not to replace clinical judgment.

This is an **academic proof-of-concept**, not a certified medical device.

---

## 2. Current Phase Scope (what we are building right now)

Per explicit instruction: **the web UI (Phase C / clinical interface) is ON HOLD.** Current build scope covers everything up to and including a working, evaluated classification model with a scriptable inference entry point (CLI/function call — no web frontend, no FastAPI/Flask server yet).

In scope now:
- Dataset verification & cleaning (**already completed** — see Section 6)
- Preprocessing pipeline (resize, speckle reduction, contrast enhancement)
- Data augmentation pipeline
- Model training (transfer learning, ResNet50 primary / VGG16 secondary)
- Evaluation against standard diagnostic metrics
- A single-image inference function/script (input: image path → output: label + confidence), callable from terminal — this is the seam the future UI will plug into, not a UI itself

Out of scope for now (deferred, not cancelled):
- Web interface (upload UI, results rendering, FastAPI backend)
- DICOM support (spec allows JPEG/PNG/DICOM long-term; current dataset is JPEG only — DICOM parsing deferred)
- Real-time video stream inference
- MRI/CT modalities
- Cloud deployment

---

## 3. Users

- **Primary:** Clinician / sonographer (future UI user — not interacting with the system yet in this phase)
- **Current phase operator:** Project owner + Antigravity, running everything via terminal/scripts

---

## 4. Functional Requirements (from Review-1, Section 8 — unchanged)

| ID | Requirement |
|---|---|
| FR-1 | System must accept ultrasound images in JPEG/PNG (DICOM deferred) |
| FR-2 | System must automatically preprocess images (resize, contrast enhancement, noise reduction) with no manual steps |
| FR-3 | System must run inference through the trained model and produce a prediction |
| FR-4 | Output must clearly present classification ("Fibroid Detected" / "Normal Uterus") **and** the confidence probability score |
| FR-5 | System must catch unsupported/corrupted files and fail with a clear, non-crashing error |

FR-4 and FR-5 currently apply at the **inference-function level** (return structured output / raise handled exceptions) since there is no UI yet to render them.

## 5. Non-Functional Requirements (from Review-1 — unchanged)

| ID | Requirement |
|---|---|
| NFR-1 | **Diagnostic reliability:** prioritize high sensitivity & specificity — false negatives (missed fibroids) are the clinically costlier error |
| NFR-2 | **Processing speed:** end-to-end inference under 3 seconds per image |
| NFR-3 | **Usability:** deferred to UI phase |
| NFR-4 | **Scalability:** code must stay modular so a larger dataset or cloud backend can be swapped in later without a rewrite |
| NFR-5 | **Privacy:** no patient-identifying metadata is stored or logged anywhere in the pipeline |

---

## 6. Dataset — Status: Verified & Cleaned (do not re-litigate unless new evidence appears)

- **Source:** Mendeley "uterine fibroid ultrasound images" — Tiantian Yang (2023), DOI `10.17632/552zbvzwrk.1`, CC BY 4.0. Used successfully in multiple peer-reviewed papers (Xi & Wang, BMC Medical Imaging 2024 — 99% accuracy with EfficientNetB0; others).
- **Structure on disk:** `datasets/datasets/{train,test}/{UF,NUF}/*.jpg`
- **Original size:** 1,990 images (train: 702 UF / 892 NUF; test: 173 UF / 223 NUF)
- **Class definition (important, do not restate incorrectly in any generated doc or report text):** UF = fibroid mass located within the uterus. NUF = no fibroid detected within the uterine cavity for the purposes of this classification task (includes normal myometrial tissue; a small number of extra-uterine-mass and non-gravid edge cases were present in the source data). This is a scope refinement of the original authors' NUF definition, not a correction of a dataset defect — document it as such in any generated report text, per the agreed methodology note (Section 10).
- **Cleaning already performed (do not repeat):**
  - 1 byte-identical duplicate image removed (`test/UF`)
  - 50 images containing confirmed obstetric biometry overlays (`GA=\d+` pattern, e.g. `GA=22w3d`) removed as out-of-scope pregnancy scans
  - Full audit trail: `data/excluded/exclusion_log.csv`
  - **Final clean counts:** train 700 UF / 857 NUF (1,557), test 172 UF / 210 NUF (382), **total 1,939**
- **Official companion code** (same Mendeley page, same author): `DCGAN&MobileNetV2/` — a PyTorch classification framework (backbones: MobileNetV2, ResNet, VGG, ViT, Swin) plus a DCGAN synthetic-augmentation module. Annotation files `cls_train.txt` / `cls_test.txt` have already been regenerated to match the cleaned 1,939-file set 1:1. `model_data/cls_classes.txt` defines `UF=0, NUF=1`.

**Do not re-run dataset verification, re-filter, or re-generate annotation files unless the project owner explicitly requests it.** This phase is closed.

---

## 7. Model Requirements

- **Primary architecture:** ResNet50 (transfer learning, ImageNet-pretrained), using `nets/resnet.py` from the official companion framework — **not** the framework's default MobileNetV2.
- **Secondary/fallback architecture:** VGG16, only if ResNet50 shows poor convergence — must be reported as a comparison, not a silent swap.
- Freeze initial convolutional blocks; fine-tune a dense classification head.
- Output layer: sigmoid, single unit (binary probability).
- Loss: binary cross-entropy. Optimizer: Adam.
- Two-stage training: (1) frozen base + head training, (2) partial unfreeze + low-LR fine-tuning.

## 8. Data Pipeline Requirements

**Preprocessing** (per Review-1 Methodology, Section 7.2 — locked):
- Resize to 224×224
- Intensity normalization
- Speckle noise reduction (adaptive histogram equalization / CLAHE)

**Augmentation** (per Review-1 Methodology, Section 7.3 — locked):
- Random rotation, horizontal flip, zoom, minor shear — applied dynamically during training
- No vertical flip (anatomically implausible for this probe orientation)

## 9. Evaluation Requirements (locked)

Report all of: **Accuracy, Sensitivity (Recall), Specificity, Precision, F1-score**, plus AUC and a confusion matrix (AUC/confusion matrix are useful additions beyond the original spec, not deviations from it). Benchmark final numbers against the literature-survey table in the Review-1 report (Section 6) where comparable.

## 10. Required Methodology Note for Any Generated Report Text

Any documentation, slide, or report section Antigravity generates that discusses the dataset **must** use this phrasing (or a close paraphrase) rather than calling NUF "healthy tissue" outright:

> *"NUF (Non-Uterine Fibroid) images represent scans where no fibroid is present within the uterine cavity, including cases of normal uterine tissue. A small number of images from pregnant patients present in the original public dataset were excluded from this work as outside the intended scope of non-gravid fibroid/non-fibroid differentiation — a scope refinement rather than a correction of the source dataset."*

## 11. Definition of Done for This Phase

- Clean data flows through preprocessing + augmentation without errors
- ResNet50-based model trains through both stages without crashing, with visible per-epoch progress
- Final evaluation report (all 5 core metrics + confusion matrix) generated and saved to disk
- A single callable inference function/script exists and returns correct structured output on a manually-checked sample image
- Nothing above required a UI
