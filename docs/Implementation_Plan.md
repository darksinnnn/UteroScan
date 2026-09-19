# UteroScan — Implementation Plan

Execute phases **in order**. Do not start a phase until the previous one's exit report has been reviewed and acknowledged by the project owner (see `Agents.md` for the report format). UI work is excluded from every phase below — see `PRD.md` Section 2.

---

## Phase 0 — Dataset Verification & Cleaning
**Status: ✅ ALREADY COMPLETE. Do not redo.**

For reference only:
- 1,990 → 1,939 images after removing 1 duplicate + 50 pregnancy scans
- `cls_train.txt` (1,557) / `cls_test.txt` (382) regenerated and disk-verified
- Original author annotation files preserved as `*_ORIGINAL.txt`

If Antigravity is asked to "start from Phase 0," treat it as a request to re-verify counts still match (quick sanity check only), not to re-run filtering logic from scratch.

---

## Phase 1 — Preprocessing Pipeline Integration

**Goal:** Wire the preprocessing steps (resize, speckle reduction, CLAHE) into the actual training data-loading path used by `train.py`, in a way that's consistent for train/val/test.

**Tasks:**
1. Confirm whether the official `DCGAN&MobileNetV2/train.py` already does its own resizing/normalization internally (check its data loader code) before adding a duplicate/conflicting preprocessing step.
2. Integrate speckle reduction (median filter) + CLAHE as a transform applied before/alongside whatever resizing the framework already does. Do not double-resize or double-normalize.
3. Run the pipeline on a small sample (~20 images across both classes) and visually confirm: fibroid boundaries look distinct, no images are blown out or over-darkened.

**Exit checklist (report all of these):**
- [ ] Confirmed whether framework had its own preprocessing, and how it was reconciled with ours
- [ ] Sample before/after images reviewed (describe or save a few side-by-side)
- [ ] No pipeline errors across a full dry run over all 1,939 images (just the transform, not training)

---

## Phase 2 — Data Augmentation Setup

**Goal:** Add the four specified augmentations (rotation ±15°, h-flip, zoom, shear) to the **training** data loader only.

**Tasks:**
1. Implement as `torchvision.transforms` (or the framework's existing augmentation hook if one already exists — check before adding a redundant one).
2. Explicitly verify vertical flip is NOT included anywhere in the pipeline.
3. Confirm validation/test loaders do NOT have augmentation applied (only resize/normalize/preprocess).

**Exit checklist:**
- [ ] Augmentation config shown/printed (which transforms, what parameters)
- [ ] Confirmed train vs. val/test loaders are configured differently (augmented vs. not)
- [ ] One batch visualized or logged to confirm augmentations are visibly applying (not silently no-op'd)

---

## Phase 3 — Model Training (ResNet50, Two-Stage)

**Goal:** Train the classifier per the locked methodology.

**Tasks:**
1. Point the framework at `nets/resnet.py` (ResNet50), not the default MobileNetV2. Confirm class mapping matches `model_data/cls_classes.txt` (`UF=0, NUF=1`).
2. **Smoke test first:** 2 epochs on a small subset (~50 images/class) to confirm the pipeline runs end-to-end without crashing and loss decreases. Report this before starting full training.
3. **Stage 1:** Freeze ResNet50 backbone, train classification head only. Suggested starting point: ~10–15 epochs, Adam, lr=1e-4, early stopping on val loss (patience=5), checkpoint best by val AUC (or val accuracy if AUC isn't wired up).
4. **Stage 2 (fine-tuning):** Unfreeze last ~20–30 layers of the backbone, drop lr to 1e-5, continue ~10 more epochs.
5. Save checkpoints to `checkpoints/` at minimum: best-so-far and final.
6. **Class imbalance (700 UF vs 857 NUF, ~55/45):** optional `class_weight` / weighted loss — apply it, but report the decision either way.

**Live progress requirement (see `Agents.md`):** every epoch must print train/val loss + accuracy to the terminal in real time. No silent multi-minute gaps.

**Exit checklist:**
- [ ] Smoke test results reported (loss trend, no NaNs/crashes)
- [ ] Stage 1 final metrics (train/val loss, accuracy, AUC)
- [ ] Stage 2 final metrics
- [ ] Confirmation of which architecture actually trained (ResNet50 — flag immediately if it silently fell back to something else)
- [ ] Whether class weighting was used and why

---

## Phase 4 — Evaluation

**Goal:** Produce the full metrics suite on the held-out test set (382 images), and only the test set — never report test metrics that leaked from validation-selected checkpoints without a final, separate test pass.

**Tasks:**
1. Run `eval.py` (or extend it) over `test/UF` + `test/NUF`.
2. Compute and save: Accuracy, Sensitivity (Recall), Specificity, Precision, F1-score, AUC, Confusion Matrix.
3. Break down UF-class recall specifically (clinically the most important single number — missed fibroids are the costly error).
4. Compare against the Review-1 literature table (e.g. Xi & Wang 99%, Shahzad et al. VGG16/ResNet50/InceptionV3 comparison) — state clearly if UteroScan's numbers are in the same range, notably lower, or notably higher, and offer a plausible reason either way (e.g. "our cleaned dataset removed 51 images the cited papers likely retained").

**Exit checklist:**
- [ ] `reports/metrics.json` and `reports/confusion_matrix.png` exist
- [ ] UF-class recall specifically called out
- [ ] Literature comparison paragraph written

---

## Phase 5 — Inference Function (No UI)

**Goal:** A single clean entry point other code (or a future UI) can call.

**Tasks:**
1. Implement/confirm `predict_image(image_path) -> {"label": ..., "confidence": ...}` using the best Stage 2 checkpoint.
2. Test manually on 5 images (mix of UF/NUF from test set) — confirm output matches known labels and confidence is sane (not stuck at exactly 0.5 or 1.0 for everything, which would indicate a bug).
3. Confirm processing time per image is under 3 seconds (NFR-2 in PRD) on the actual hardware being used.

**Exit checklist:**
- [ ] 5 manual test predictions shown with expected vs. actual label
- [ ] Timing reported
- [ ] Function/script location documented (file path, how to call it)

---

## What Happens After Phase 5

Stop and report. Do not proceed to building the web UI, FastAPI backend, or any Phase C work without an explicit new instruction — it is deliberately on hold per current project scope.
