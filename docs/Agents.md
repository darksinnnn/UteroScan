# Agents.md — Operating Instructions for Antigravity on UteroScan

This file governs *how* Antigravity works on this project, independent of *what* it's building (see `PRD.md`, `Architecture.md`, `Implementation_Plan.md` for that). Read this before starting any phase.

---

## 1. Core Working Principles

1. **Follow `Implementation_Plan.md` phase order.** Do not skip ahead, do not merge phases, do not start Phase 3 (training) because it "seems ready" if Phase 1/2 exit checklists haven't been reported yet.
2. **Never silently deviate from the locked spec.** Architecture (ResNet50 primary), preprocessing steps, and augmentation techniques are locked per `PRD.md` — they came from an already-submitted academic report and cannot be casually changed for convenience (e.g. switching to MobileNetV2 because it's the framework default, or dropping an augmentation because it's easier). If something in the spec seems technically wrong or suboptimal, **flag it and ask — do not just do the "better" thing silently.**
3. **Never overwrite original/reference files in place.** If regenerating a file that has authoritative prior content (annotation files, config files, anything from the original dataset author), always preserve a copy first (e.g. `filename_ORIGINAL.txt` or a timestamped backup) before writing over it. This is a hard rule — it was violated once already on this project (`cls_train.txt` was regenerated before its original was backed up, requiring a recovery step later). Do not repeat that.
4. **Verify before declaring done.** "The script ran without errors" is not the same as "the output is correct." Every phase's exit checklist includes an actual verification step (visual sample check, count reconciliation, manual test prediction) — do those, don't skip to the report.
5. **When something looks off, stop and report — don't quietly patch around it.** E.g., if OCR flags 100+ files as pregnancy scans and that seems too high, don't just accept it and move on — pull a sample and eyeball it, like was correctly done earlier in this project.

---

## 2. Live Progress Visibility (mandatory)

**Any process expected to run longer than ~30 seconds must show live, human-readable progress in the visible terminal — not run silently in the background.**

Specifically:
- **Training runs:** print per-epoch (or per-N-batches, for long epochs) status — current epoch/total, train loss, train accuracy, val loss, val accuracy, elapsed time, estimated time remaining. Use `tqdm` for batch-level progress bars within an epoch if the framework supports it.
- **Batch preprocessing / dataset scans:** print progress every N files (e.g. every 100), not just a final summary.
- **Any GPU-bound long task:** report device being used (confirm GPU is actually being utilized, not silently falling back to CPU) at the start of the run.
- Do not rely solely on writing to a `.log` file that the operator has to go open separately mid-run. The terminal output **is** the primary channel — logging to `logs/` is a secondary backup, not a replacement.
- If running something via a background/scheduled task tool, periodically surface a progress update back into the conversation (as was done correctly during the earlier OCR diagnostic pass — e.g. "1,200 / 1,990 scanned, ~3 min remaining") rather than going silent until full completion.

---

## 3. End-of-Phase Report Format (mandatory, every phase)

At the end of every phase in `Implementation_Plan.md`, produce a report in this structure before waiting for the next instruction:

```
## Phase [N] — [Name] — COMPLETE / BLOCKED

### What was done
- (bullet list of concrete actions taken — files created/modified, commands run)

### Exit checklist results
- [x] / [ ] for each item from Implementation_Plan.md's checklist for this phase,
      with the actual number/result next to each, not just a checkmark
      (e.g. "[x] Stage 1 val accuracy: 0.87, val AUC: 0.91" — not just "[x] done")

### Numbers that matter
- (any metric, count, or measurement produced this phase — always show real numbers)

### Deviations from plan (if any)
- (anything that didn't go exactly per PRD/Architecture/Implementation_Plan,
   why, and what was done instead — even small ones)

### Files changed/created this phase
- (explicit paths)

### Blockers / things needing a decision
- (anything Antigravity is NOT confident enough to decide unilaterally)

### Ready for next phase? YES / NO — waiting for review
```

**Do not proceed to the next phase after posting this report.** Wait for explicit go-ahead, exactly like previous phases in this project (dataset cleanup, annotation regeneration) were reviewed step by step before moving forward.

---

## 4. Escalation Rules — When to Stop and Ask Instead of Deciding Alone

Stop and ask (don't just proceed) when:
- A locked spec item (architecture, preprocessing steps, augmentation list, metrics list) seems to conflict with what the actual codebase/framework does by default
- A cleanup/filter/regeneration step would modify or move more than a handful of files, especially anything derived from the original dataset
- Training metrics look suspicious either way — too good (possible leakage/bug) or clearly broken (stuck loss, NaN, all-one-class predictions)
- Any step would take a long time (e.g. >20 minutes) and there's ambiguity about parameters — better to confirm before burning compute time on a possibly-wrong run
- Anything touching the dataset's class definitions, labels, or scope (this has been a recurring source of real issues on this project — see PRD Section 6 and 10)

Do NOT stop and ask for:
- Routine coding decisions within an already-approved phase (variable names, minor refactors, which specific plotting library for the confusion matrix)
- Things explicitly already decided in `PRD.md`/`Architecture.md` — re-litigating settled decisions wastes a turn; just follow what's written

---

## 5. Environment Notes

- Training hardware: confirm GPU availability at the start of Phase 3 and report which device is actually being used
- Framework: PyTorch (official `DCGAN&MobileNetV2` companion codebase) — use its existing scaffolding (`train.py`, `eval.py`, `predict.py`) rather than rewriting from scratch, extending it where the plan requires something it doesn't already do
- All file paths in reports should be absolute or clearly relative to `D:\Project\UteroScan\`, matching `Architecture.md`'s directory map
