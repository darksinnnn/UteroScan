# UteroScan — Architecture Document

Companion to `PRD.md`. Describes the actual, current-phase system structure (UI excluded per project decision).

---

## 1. High-Level Pipeline

```
Raw Dataset (Mendeley, cleaned)
        │
        ▼
┌─────────────────────┐
│  Preprocessing        │  resize → speckle reduction → CLAHE contrast
│  (preprocess.py)      │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Augmentation          │  rotation / h-flip / zoom / shear (train-time only)
│  (applied in DataLoader│
│   / train.py)          │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Model (ResNet50)      │  Stage 1: frozen base + head
│  nets/resnet.py        │  Stage 2: partial unfreeze, fine-tune
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Evaluation             │  Accuracy / Sensitivity / Specificity /
│  (eval.py)              │  Precision / F1 / AUC / Confusion Matrix
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Inference function     │  single image → {label, confidence}
│  (predict.py /          │  (no UI — CLI/importable function only,
│   classification.py)    │   this phase)
└─────────────────────┘
```

---

## 2. Directory Structure (actual, on disk — `D:\Project\UteroScan\`)

```
UteroScan/
│
├── datasets/
│   └── datasets/
│       ├── train/
│       │   ├── UF/                     # 700 images (cleaned)
│       │   └── NUF/                    # 857 images (cleaned)
│       └── test/
│           ├── UF/                     # 172 images (cleaned)
│           └── NUF/                    # 210 images (cleaned)
│
├── data/
│   └── excluded/
│       ├── exclusion_log.csv           # audit trail: filename, reason, source, moved_to
│       └── *.jpg                       # 51 excluded images (50 pregnancy + 1 duplicate)
│
├── DCGAN&MobileNetV2/
│   └── DCGAN&MobileNetV2/              # OFFICIAL companion code (same dataset author)
│       ├── nets/
│       │   ├── resnet.py               # ★ USE THIS — ResNet-18/34/50/101
│       │   ├── vgg.py                  # fallback architecture
│       │   ├── mobilenetv2.py          # framework default — NOT used per spec
│       │   ├── vision_transformer.py
│       │   └── swin_transformer.py
│       ├── model_data/
│       │   ├── cls_classes.txt         # UF=0, NUF=1
│       │   └── *.pth                   # ImageNet pretrained weights
│       ├── cls_train.txt               # 1,557 entries — regenerated, matches disk 1:1
│       ├── cls_test.txt                # 382 entries — regenerated, matches disk 1:1
│       ├── cls_train_ORIGINAL.txt      # preserved copy of author's original (reference only)
│       ├── cls_test_ORIGINAL.txt       # preserved copy of author's original (reference only)
│       ├── train.py                    # classification training entry point
│       ├── eval.py                     # test-set evaluation + confusion matrix
│       ├── predict.py / classification.py  # inference wrappers
│       ├── txt_annotation.py           # annotation-file generator (already run — don't re-run blindly)
│       └── dcgan-pytorch-main/         # optional DCGAN augmentation module (see Section 5)
│
├── verify_dataset.py                   # dataset integrity checks (hash/corrupt/balance) — DONE, keep for record
├── filter_dataset.py                   # pregnancy-scan filter (tightened regex) — DONE, keep for record
├── execute_filtering.py                # performed the actual move + annotation regen — DONE
├── diagnostic_results.csv              # full OCR/color/label diagnostic log — keep for record
│
├── logs/                               # NEW — training run logs (see Section 6)
├── checkpoints/                        # NEW — saved model weights per stage/epoch
├── reports/                            # NEW — evaluation outputs (metrics.json, confusion_matrix.png, etc.)
│
├── PRD.md
├── Architecture.md
├── Implementation_Plan.md
└── Agents.md
```

**App/UI folder intentionally does not exist yet** — do not create `app/`, `main.py`, `templates/` etc. in this phase.

---

## 3. Preprocessing Module

Reuses the already-built `preprocess.py` logic (resize 224×224 via `INTER_AREA`, median-filter speckle reduction, CLAHE contrast on grayscale or L-channel). This should be wired as a callable transform inside the training data loader, not just a standalone batch script — training needs it applied consistently to every image at load time.

## 4. Augmentation Module

Applied only at training time (never to validation/test data):
- Random rotation: ±15°
- Random horizontal flip
- Random zoom: ~0.9–1.1
- Minor shear
- **Explicitly excluded:** vertical flip, large rotations (anatomically implausible)

Implementation should live in `train.py`'s data loading section (e.g. `torchvision.transforms` since the official framework is PyTorch-based) — keep it declarative and easy to toggle per-technique for ablation later if needed.

## 5. Model Module

- Backbone: `nets/resnet.py`, ResNet50 variant, ImageNet-pretrained weights.
- Head: pooled features → dense layer(s) → dropout → single sigmoid output.
- Class mapping fixed by `model_data/cls_classes.txt`: `UF=0, NUF=1`.
- Two-stage training schedule (see `Implementation_Plan.md` Phase 3 for exact epoch/LR values).
- **DCGAN module (`dcgan-pytorch-main/`):** available but **not enabled by default this phase.** Only invoke it if Phase 3 evaluation shows the 700:857 class imbalance is materially hurting minority-class (UF) recall — treat as an opt-in remedy, not a default step. If used, log it explicitly as a deviation to report on.

## 6. Logging & Progress Visibility Requirement

**This is an architectural requirement, not a nice-to-have.** Every long-running process (training epochs, batch preprocessing, full-dataset evaluation) must:
- Print live progress to the visible terminal (e.g. `tqdm` progress bars, or per-epoch `print()` with loss/accuracy) — never run silently in the background with output only in a log file the operator can't see mid-run.
- Additionally persist the same progress to a timestamped file under `logs/` (e.g. `logs/train_stage1_YYYYMMDD_HHMMSS.log`) for later reference, but the terminal is the primary, real-time channel.
- See `Agents.md` for the exact reporting cadence expected during training.

## 7. Evaluation Module

`eval.py` (extend if needed) computes and saves to `reports/`:
- `metrics.json` — accuracy, sensitivity, specificity, precision, F1, AUC
- `confusion_matrix.png`
- Per-class breakdown (UF vs NUF performance separately, since this is a clinically asymmetric problem — recall on UF matters most)

## 8. Inference Module (this phase's "product," in lieu of a UI)

A single function, importable and CLI-runnable:
```
predict_image(image_path: str) -> {"label": "UF"|"NUF", "confidence": float}
```
This is the seam the future web UI (Phase C, on hold) will call into later — build it as a clean, dependency-light function now so wiring a FastAPI endpoint around it later is trivial, per NFR-4 (modularity/scalability) in the PRD.
