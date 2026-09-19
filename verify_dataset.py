"""
UteroScan — Dataset Verification Script
Covers: Task 1.1 (duplicates), 1.2 (class balance), 1.3 (corrupt files),
        + Visual label sanity check (random sample inspection)
"""

import hashlib
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────
# CONFIG — adjust if needed
# ─────────────────────────────────────────────
BASE_DIR   = Path("d:/Project/UteroScan/datasets")
TRAIN_UF   = BASE_DIR / "train" / "UF"
TRAIN_NUF  = BASE_DIR / "train" / "NUF"
TEST_UF    = BASE_DIR / "test"  / "UF"
TEST_NUF   = BASE_DIR / "test"  / "NUF"
SAMPLE_N   = 8    # images to visually inspect per class
OUTPUT_DIR = Path("d:/Project/UteroScan/verify_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def collect_images(folder: Path) -> list[Path]:
    imgs = []
    for ext in EXTS:
        imgs.extend(folder.glob(ext))
    return imgs

def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def sep(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)

# ─────────────────────────────────────────────
# TASK 1.2 — CLASS BALANCE
# ─────────────────────────────────────────────
sep("TASK 1.2 — CLASS BALANCE")

folders = {
    "train/UF":  TRAIN_UF,
    "train/NUF": TRAIN_NUF,
    "test/UF":   TEST_UF,
    "test/NUF":  TEST_NUF,
}
counts = {}
for label, folder in folders.items():
    imgs = collect_images(folder)
    counts[label] = imgs
    print(f"  {label:<15}: {len(imgs):>5} images")

train_uf_n  = len(counts["train/UF"])
train_nuf_n = len(counts["train/NUF"])
test_uf_n   = len(counts["test/UF"])
test_nuf_n  = len(counts["test/NUF"])
train_total = train_uf_n + train_nuf_n
test_total  = test_uf_n  + test_nuf_n

print(f"\n  Train total : {train_total}")
print(f"  Test total  : {test_total}")
if train_total > 0:
    ratio = max(train_uf_n, train_nuf_n) / train_total
    print(f"  Train imbalance ratio (majority/total): {ratio:.1%}")
    if ratio > 0.60:
        print("  ⚠️  IMBALANCE > 60/40 — class_weight needed in train.py")
    else:
        print("  ✅  Class balance within 60/40 — class_weight optional")

# ─────────────────────────────────────────────
# TASK 1.1 — DUPLICATE DETECTION
# ─────────────────────────────────────────────
sep("TASK 1.1 — DUPLICATE DETECTION")

all_splits = {
    "train/UF":  counts["train/UF"],
    "train/NUF": counts["train/NUF"],
    "test/UF":   counts["test/UF"],
    "test/NUF":  counts["test/NUF"],
}

# Build hash maps per split
hash_maps: dict[str, dict[str, Path]] = {}
for split, paths in all_splits.items():
    hm = {}
    for p in paths:
        try:
            h = md5(p)
            hm[h] = p
        except Exception as e:
            print(f"  [WARN] Could not hash {p}: {e}")
    hash_maps[split] = hm
    print(f"  Hashed {len(hm):>5} files in {split}")

# Cross-class train leak (UF ↔ NUF same split)
print("\n  --- Cross-class duplicates (UF ↔ NUF within train) ---")
cross_train = set(hash_maps["train/UF"]) & set(hash_maps["train/NUF"])
print(f"  Byte-identical duplicates: {len(cross_train)}")
for h in cross_train:
    print(f"    UF: {hash_maps['train/UF'][h].name}  <-->  NUF: {hash_maps['train/NUF'][h].name}")

print("\n  --- Cross-class duplicates (UF ↔ NUF within test) ---")
cross_test = set(hash_maps["test/UF"]) & set(hash_maps["test/NUF"])
print(f"  Byte-identical duplicates: {len(cross_test)}")
for h in cross_test:
    print(f"    UF: {hash_maps['test/UF'][h].name}  <-->  NUF: {hash_maps['test/NUF'][h].name}")

# Train/test leakage (same image in train AND test)
print("\n  --- Train/Test leakage (same image in both splits) ---")
all_train_hashes = set(hash_maps["train/UF"]) | set(hash_maps["train/NUF"])
all_test_hashes  = set(hash_maps["test/UF"])  | set(hash_maps["test/NUF"])
leakage = all_train_hashes & all_test_hashes
print(f"  Byte-identical train↔test duplicates: {len(leakage)}")
train_all_map = {**hash_maps["train/UF"], **hash_maps["train/NUF"]}
test_all_map  = {**hash_maps["test/UF"],  **hash_maps["test/NUF"]}
for h in leakage:
    print(f"    train: {train_all_map[h]}  <-->  test: {test_all_map[h]}")

# Within-class duplicates
print("\n  --- Within-class duplicates ---")
for split, hm in hash_maps.items():
    all_paths = all_splits[split]
    # reverse: hash → list of all paths with that hash
    rev: dict[str, list[Path]] = {}
    for p in all_paths:
        try:
            h = md5(p)
            rev.setdefault(h, []).append(p)
        except:
            pass
    dups = {h: ps for h, ps in rev.items() if len(ps) > 1}
    print(f"  {split}: {len(dups)} duplicate groups")
    for h, ps in dups.items():
        names = ", ".join(p.name for p in ps)
        print(f"    {names}")

# ─────────────────────────────────────────────
# TASK 1.3 — CORRUPT / UNREADABLE FILES
# ─────────────────────────────────────────────
sep("TASK 1.3 — CORRUPT / UNREADABLE FILES")

bad_files = []
checked   = 0
for split, paths in all_splits.items():
    for p in paths:
        checked += 1
        img = cv2.imread(str(p))
        if img is None:
            bad_files.append((split, p))

print(f"  Total files checked : {checked}")
print(f"  Unreadable files    : {len(bad_files)}")
if bad_files:
    print("  Filenames:")
    for split, p in bad_files:
        print(f"    [{split}] {p.name}")
else:
    print("  ✅  All files readable by OpenCV")

# ─────────────────────────────────────────────
# VISUAL LABEL SANITY CHECK
# Build a side-by-side grid: SAMPLE_N images from UF + SAMPLE_N from NUF
# Saved to verify_output/ for inspection
# ─────────────────────────────────────────────
sep("VISUAL LABEL SANITY CHECK — random sample grid")

def make_sample_grid(paths: list[Path], n: int, label: str, split: str):
    """Save a grid of n random images from paths."""
    sample = random.sample(paths, min(n, len(paths)))
    imgs_resized = []
    stats = []
    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        # Basic stats
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        mean_bright = float(np.mean(gray))
        is_color    = len(img.shape) == 3 and img.shape[2] == 3
        stats.append({
            "file":    p.name,
            "h":       h,
            "w":       w,
            "color":   is_color,
            "mean_px": round(mean_bright, 1),
        })
        img_small = cv2.resize(img, (224, 224))
        imgs_resized.append(img_small)

    # Print stats
    print(f"\n  [{split}/{label}] Sample stats ({len(stats)} images):")
    for s in stats:
        mode = "COLOR" if s["color"] else "GRAY"
        print(f"    {s['file']:<40} {s['w']}x{s['h']}  {mode}  mean_px={s['mean_px']}")

    # Build grid (2 rows x 4 cols)
    if not imgs_resized:
        return
    cols = 4
    rows = (len(imgs_resized) + cols - 1) // cols
    # Pad to fill grid
    while len(imgs_resized) < rows * cols:
        imgs_resized.append(np.zeros((224, 224, 3), dtype=np.uint8))
    grid_rows = []
    for r in range(rows):
        row_imgs = imgs_resized[r * cols:(r + 1) * cols]
        grid_rows.append(np.hstack(row_imgs))
    grid = np.vstack(grid_rows)
    out_path = OUTPUT_DIR / f"sample_{split}_{label}.png"
    cv2.imwrite(str(out_path), grid)
    print(f"  → Grid saved: {out_path}")
    return stats

# Run for all four folders
for split_name, uf_paths, nuf_paths in [
    ("train", counts["train/UF"], counts["train/NUF"]),
    ("test",  counts["test/UF"],  counts["test/NUF"]),
]:
    make_sample_grid(uf_paths,  SAMPLE_N, "UF",  split_name)
    make_sample_grid(nuf_paths, SAMPLE_N, "NUF", split_name)

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
sep("FINAL SUMMARY")
print(f"  Train UF  : {train_uf_n}")
print(f"  Train NUF : {train_nuf_n}")
print(f"  Test  UF  : {test_uf_n}")
print(f"  Test  NUF : {test_nuf_n}")
print(f"  Cross-class dups (train) : {len(cross_train)}")
print(f"  Cross-class dups (test)  : {len(cross_test)}")
print(f"  Train↔Test leakage       : {len(leakage)}")
print(f"  Corrupt files            : {len(bad_files)}")
print(f"\n  Sample grids saved to: {OUTPUT_DIR}")
print("\n  ✅  Verification complete.")
