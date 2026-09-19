import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path("datasets/datasets")
folders = ["train/UF", "train/NUF", "test/UF", "test/NUF"]

def has_color_doppler_orig(img, threshold=20, min_color_pixels=100):
    if len(img.shape) != 3 or img.shape[2] != 3:
        return False
    max_c = np.max(img, axis=2)
    min_c = np.min(img, axis=2)
    diff = max_c.astype(np.int32) - min_c.astype(np.int32)
    return int(np.sum(diff > threshold)) > min_color_pixels

results = {}
for f in folders:
    folder_path = BASE_DIR / f
    imgs = sorted(list(folder_path.glob("*.jpg")))
    has_color_count = 0
    total = len(imgs)
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if has_color_doppler_orig(img):
            has_color_count += 1
    results[f] = {
        "total": total,
        "has_color": has_color_count,
        "pct": (has_color_count / total * 100) if total > 0 else 0
    }

print("=" * 70)
print("ORIGINAL RGB-VARIANCE METRIC (diff > 20, > 100 pixels)")
print("=" * 70)
for f, r in results.items():
    print(f"{f:<12}: {r['has_color']:4d} / {r['total']:4d} ({r['pct']:5.1f}%)")

uf_tot = results["train/UF"]["total"] + results["test/UF"]["total"]
uf_col = results["train/UF"]["has_color"] + results["test/UF"]["has_color"]

nuf_tot = results["train/NUF"]["total"] + results["test/NUF"]["total"]
nuf_col = results["train/NUF"]["has_color"] + results["test/NUF"]["has_color"]

train_tot = results["train/UF"]["total"] + results["train/NUF"]["total"]
train_col = results["train/UF"]["has_color"] + results["train/NUF"]["has_color"]

test_tot = results["test/UF"]["total"] + results["test/NUF"]["total"]
test_col = results["test/UF"]["has_color"] + results["test/NUF"]["has_color"]

all_tot = train_tot + test_tot
all_col = train_col + test_col

print("-" * 70)
print(f"Total UF     : {uf_col:4d} / {uf_tot:4d} ({uf_col/uf_tot*100:5.1f}%)")
print(f"Total NUF    : {nuf_col:4d} / {nuf_tot:4d} ({nuf_col/nuf_tot*100:5.1f}%)")
print(f"Total Train  : {train_col:4d} / {train_tot:4d} ({train_col/train_tot*100:5.1f}%)")
print(f"Total Test   : {test_col:4d} / {test_tot:4d} ({test_col/test_tot*100:5.1f}%)")
print(f"Overall Total: {all_col:4d} / {all_tot:4d} ({all_col/all_tot*100:5.1f}%)")
print("=" * 70)
