# UteroScan — Full Project Build Plan
**For execution by: Antigravity**
**Prepared by: Claude (planning/architecture pass)**
**Status: Ready for implementation — verification checkpoints included**

---

## 0. How to use this document

This is a step-by-step build plan for UteroScan, an AI-assisted uterine fibroid classifier from 2D transvaginal ultrasound images. Each phase below has:

- **Objective** — what this phase delivers
- **Tasks** — concrete implementation steps
- **Verification checklist** — things Antigravity should check and report back on *before* moving to the next phase. Some of these are genuine open questions I couldn't resolve without running code against the actual dataset.

Please execute phases in order (0 → 4). After each phase's verification checklist is run, report results back before proceeding — some downstream decisions (e.g. class weighting, architecture choice) depend on what the verification finds.

---

## 1. Dataset Decision (Already Made — Documented Here for Record)

Two versions of the Mendeley Uterine Fibroid dataset exist:

| Version | Structure | Fit for this project |
|---|---|---|
| **v1 (older)** | `train/UF/`, `train/NUF/`, `test/UF/`, `test/NUF/` — pre-sorted by class | ✅ **Selected** — matches binary classification + sigmoid output spec exactly |
| **v2 (newer, data.mendeley.com/datasets/n2zcmcypgb/2)** | `Annotations/` (Pascal VOC XML w/ bounding boxes), `JPEGImages/`, YOLOv3/EfficientNet detection code | ❌ Not used — built for object detection/localization, not binary classification. Would require a scope change to detection instead of classification. |

**Decision: Use v1.** Path on disk: `SSD 2 (D:)/Project/UteroScan/Version 1/datasets (1)/datasets/`

### ⚠️ Verification Task 1.1 — Duplicate Image Check (HIGH PRIORITY, DO FIRST)
During manual inspection, two example images shown from `UF/` and `NUF/` had **identical filenames and appeared visually identical**. This needs to be resolved before any training happens — if the same image exists in both class folders, it will corrupt every downstream metric.

**Antigravity: please run and report back:**
```python
import hashlib
from pathlib import Path

def hash_file(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

uf_dir = Path("data/raw/train/UF")
nuf_dir = Path("data/raw/train/NUF")

uf_hashes = {hash_file(p): p for p in uf_dir.glob("*.jpg")}
nuf_hashes = {hash_file(p): p for p in nuf_dir.glob("*.jpg")}

overlap = set(uf_hashes) & set(nuf_hashes)
print(f"UF images: {len(uf_hashes)}")
print(f"NUF images: {len(nuf_hashes)}")
print(f"Duplicate images across classes: {len(overlap)}")
for h in overlap:
    print(f"  {uf_hashes[h]}  <-->  {nuf_hashes[h]}")
```
Also check for duplicates *within* the same class folder and across `train/` vs `test/` (train/test leakage is just as damaging as cross-class leakage).

**Report back:** exact duplicate count, and whether they're byte-identical or just visually similar (near-duplicate frames from the same scan). If duplicates exist, they must be removed (keep one copy) before Phase A begins.

### Verification Task 1.2 — Class Balance
```python
print(f"UF train count: {len(list(uf_dir.glob('*.jpg')))}")
print(f"NUF train count: {len(list(nuf_dir.glob('*.jpg')))}")
# repeat for test/
```
**Report back:** the counts. If imbalance exceeds roughly 60/40, Phase B's `train.py` needs `class_weight` in `model.fit()` — flag this so it gets added.

### Verification Task 1.3 — Corrupt/Unreadable Files
```python
import cv2
bad_files = []
for p in list(uf_dir.glob("*.jpg")) + list(nuf_dir.glob("*.jpg")):
    img = cv2.imread(str(p))
    if img is None:
        bad_files.append(p)
print(f"Unreadable files: {len(bad_files)}")
```
**Report back:** any unreadable files so they can be excluded or re-downloaded.

---

## 2. Directory Structure

```
UteroScan/
│
├── data/
│   ├── raw/
│   │   ├── train/{UF,NUF}/
│   │   └── test/{UF,NUF}/
│   └── processed/              # output of preprocess.py, mirrors raw/ structure
│       ├── train/{UF,NUF}/
│       └── test/{UF,NUF}/
│
├── src/
│   ├── preprocess.py           # Phase A — DONE (see section 3)
│   ├── train.py                # Phase B — to build
│   ├── model.py                # architecture definition (optional split from train.py)
│   ├── inference.py            # loads trained model, single-image prediction
│   └── utils.py                # shared helpers (logging, config loading)
│
├── models/
│   └── uteroscan_resnet50.h5   # saved trained weights (output of train.py)
│
├── app/                         # Phase C
│   ├── main.py                  # Flask/FastAPI routes: upload, predict, results
│   ├── static/
│   └── templates/
│       └── index.html           # clinical dashboard UI
│
├── notebooks/
│   └── eda.ipynb                # class balance, duplicate scan, sample visualization
│
├── requirements.txt
├── config.yaml                  # image size, batch size, paths, thresholds
└── README.md
```

---

## 3. Phase A — Preprocessing Pipeline (`preprocess.py`)

**Status: Already written, attached separately as `preprocess.py`.** Antigravity should treat this as a first draft to validate, not rewrite from scratch, unless verification below finds a problem.

### What it does
1. **Resize** → 224×224 via `cv2.INTER_AREA` (best for downscaling without aliasing)
2. **Speckle reduction** → median filter by default (configurable to Gaussian or edge-preserving/anisotropic fallback)
3. **CLAHE** → contrast enhancement, applied on L-channel if color, direct if grayscale

### Verification Task 3.1 — Visual QA
Run `preprocess_image()` on ~10 sample images from each class and display before/after side by side (matplotlib grid). Confirm:
- Fibroid boundaries look *more* distinct after CLAHE, not washed out
- Speckle reduction isn't so aggressive it erases small fibroid masses
- No images end up mostly black/blown-out after CLAHE (can happen with `clipLimit` set too high for certain exposure levels in this dataset)

**Report back:** a few before/after image pairs (as files or inline) and whether `CLAHE_CLIP_LIMIT = 2.0` looks correct for this dataset, or needs tuning (try 1.5–3.0 range if not).

### Verification Task 3.2 — Run Full Batch
```bash
python src/preprocess.py --input_dir data/raw/train --output_dir data/processed/train
python src/preprocess.py --input_dir data/raw/test --output_dir data/processed/test
```
**Report back:** processed/failed/skipped counts from the script's own summary output. Investigate any failures.

---

## 4. Phase B — Deep Learning Engine (`train.py`) — TO BUILD

### Objective
Binary classifier (UF vs NUF) via transfer learning, per the original spec:

- **Base architecture:** ResNet50 (primary) — initialized with ImageNet weights, `include_top=False`
- **Freezing:** freeze all base conv blocks initially (feature extractor only)
- **Head:** `GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.5) → Dense(1, sigmoid)`
- **Loss:** Binary Crossentropy
- **Optimizer:** Adam, lr=1e-4 (typical starting point for frozen-base transfer learning)
- **Metrics:** accuracy, AUC, precision, recall (medical classification — recall on the positive/fibroid class matters more than raw accuracy, given false negatives are the costlier error clinically)

### Data Augmentation (per spec — dataset is small, must prevent overfitting)
- Random horizontal flip
- Rotation ±15°
- Shear
- Zoom range (~0.9–1.1)
- **Do NOT** apply vertical flip or large rotations — anatomically implausible for this probe orientation and would create unrealistic training samples

### Training Strategy
1. **Stage 1:** train only the classification head with base frozen, ~10–15 epochs, monitor val_loss with early stopping (patience=5)
2. **Stage 2 (fine-tuning):** unfreeze the last 20–30 layers of ResNet50, drop lr to 1e-5, continue training a further ~10 epochs
3. Use `ModelCheckpoint` to save best weights by `val_auc` (not just val_accuracy, given potential class imbalance)

### Verification Task 4.1 — Baseline Sanity Check
Before trusting real training results, run one quick pass to confirm the pipeline isn't broken:
- Train for 2 epochs on a tiny subset (~50 images/class)
- Confirm loss decreases and doesn't NaN out
- Confirm `predict()` on a held-out image returns a probability between 0–1, not garbage

**Report back:** loss/accuracy curve from this smoke test, and confirmation the pipeline runs end-to-end before committing to a full training run.

### Verification Task 4.2 — Full Training Run
After smoke test passes, run the full Stage 1 + Stage 2 training.

**Report back:**
- Final train/val accuracy, AUC, precision, recall
- Confusion matrix on the held-out test set
- Whether val_loss shows overfitting (diverging train/val curves) — if so, flag it, may need stronger augmentation or more dropout

---

## 5. Phase C — Clinical Web Interface — TO BUILD

### Objective
Physician-facing web app: upload image → see preprocessed image (optional) → see UF/NUF classification + confidence %.

### Backend (`app/main.py`)
- **Framework:** FastAPI (preferred over Flask for built-in request validation + easier async model serving)
- **Endpoints:**
  - `POST /predict` — accepts image upload, runs `preprocess.py` pipeline, runs inference, returns JSON `{label: "UF"|"NUF", confidence: float, preprocessed_image_url: str}`
  - `GET /health` — basic uptime check
- **Validation:** reject non-JPEG/PNG uploads with a clear 400 error before touching the ML pipeline (maps to the `Validate File Format` / `Handle Upload Errors` use cases from the UML diagrams)

### Frontend (`app/templates/index.html`)
- Simple drag-and-drop or file-picker upload
- Display: original image, preprocessed image (toggle/optional), diagnosis label, confidence % (maybe a simple progress-bar style visual)
- No patient identifiers stored or transmitted — image + result only (privacy-by-design, consistent with the note added to the earlier use case diagram)

### Verification Task 5.1 — End-to-End Test
Upload 5 sample test-set images (mix of UF/NUF) through the actual running web app (not just the Python function directly) and confirm the displayed result matches what `inference.py` returns when called directly.

**Report back:** screenshots or confirmation of matching results, plus response time per request (relevant if this needs to run on modest hardware for a demo).

---

## 6. Open Decisions Antigravity Should Flag Back to Me

These are things I can't resolve without seeing real results — please report findings so we can decide together before proceeding further:

1. **Duplicate image resolution (Task 1.1)** — if duplicates are found, should we just delete the duplicate, or is there a chance the "NUF" and "UF" folders themselves are mislabeled/swapped? Worth eyeballing a larger sample beyond the two duplicates.
2. **Class imbalance handling (Task 1.2)** — if imbalance is significant, do we use `class_weight`, oversampling, or both?
3. **ResNet50 vs VGG16** — spec allows either. Recommend defaulting to ResNet50 (generally better accuracy/parameter efficiency), but if Task 4.2 shows poor convergence, try VGG16 as a fallback and report comparison.
4. **Confidence threshold for clinical flagging** — not in original spec, but worth deciding: should low-confidence predictions (e.g. 45–55%) be visually flagged as "uncertain — recommend specialist review" in the UI, rather than presented as a confident binary answer? This is a clinically responsible addition worth considering once the model's real confidence distribution is known.

---

## 7. Suggested Execution Order

1. Run Section 1 verification tasks (1.1, 1.2, 1.3) — **do not skip**, blocks everything downstream
2. Fix any dataset issues found
3. Run Section 3 verification (preprocessing QA)
4. Build and smoke-test Phase B (4.1)
5. Run full Phase B training (4.2), report metrics
6. Build Phase C backend + frontend
7. Run end-to-end verification (5.1)
8. Report back full results — we'll review together and decide on any of the open items in Section 6 before calling this "done"
