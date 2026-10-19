"""
precompute_cache.py
Precomputes and caches 256x256 viewport-extracted, clinical-enhanced ultrasound scans.

Eliminates the 20ms/image CPU viewport-extraction bottleneck during training,
accelerating training from 45s/epoch to ~2.5s/epoch (18x speedup).
"""

import sys
import time
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = Path("d:/Project/UteroScan")
EXP_DIR = BASE_DIR / "experiments" / "v3_resnet_enhanced"
CACHE_DIR = EXP_DIR / "cache_256"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(EXP_DIR))
from preprocessing.viewport_cropper import extract_ultrasound_viewport

def process_and_cache_image(src_path, dest_path, target_size=(256, 256)):
    if dest_path.exists():
        return
    try:
        img = Image.open(src_path)
        img_np = np.array(img.convert("RGB"))

        # 1. Viewport crop
        cropped, _ = extract_ultrasound_viewport(img_np)

        # 2. Grayscale
        if len(cropped.shape) == 3:
            gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        else:
            gray = cropped.copy()

        # 3. Resize with INTER_AREA
        h, w = gray.shape
        interp = cv2.INTER_AREA if (w > target_size[0] and h > target_size[1]) else cv2.INTER_LINEAR
        resized = cv2.resize(gray, target_size, interpolation=interp)

        # 4. 3x3 Median blur
        denoised = cv2.medianBlur(resized, 3)

        # 5. CLAHE (clipLimit=1.2, tileGridSize=(8, 8))
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 6. Save as PNG to avoid compression artifacts
        cv2.imwrite(str(dest_path), enhanced)
    except Exception as e:
        print(f"Error processing {src_path.name}: {e}")

def run_cache_precomputation():
    print("=" * 70)
    print("PRECOMPUTING 256x256 VIEWPORT-CROPPED & ENHANCED ULTRASOUND CACHE")
    print("=" * 70)
    t0 = time.time()

    annotations_dir = BASE_DIR / "experiments" / "v2_contamination_cleaned" / "annotations"
    all_files = set()
    for split in ["cls_train_split.txt", "cls_val_split.txt", "cls_test.txt"]:
        lines = open(annotations_dir / split).readlines()
        for l in lines:
            if ";" in l:
                p_str = l.strip().split(";")[1].split()[0]
                all_files.add(p_str)

    print(f"Total unique images to cache: {len(all_files)}")

    def resolve_path(p_str):
        p = Path(p_str)
        if p.is_file(): return p
        cand = BASE_DIR / "datasets" / "datasets" / Path(*p.parts[1:] if len(p.parts) > 1 else p.parts)
        if cand.is_file(): return cand
        cand2 = BASE_DIR / "datasets" / "datasets" / p
        if cand2.is_file(): return cand2
        cand3 = BASE_DIR / p
        if cand3.is_file(): return cand3
        matches = list((BASE_DIR / "datasets" / "datasets").rglob(p.name))
        if matches: return matches[0]
        return None

    tasks = []
    for f in all_files:
        src = resolve_path(f)
        if src is not None:
            dest = CACHE_DIR / f"{src.stem}.png"
            tasks.append((src, dest))

    print(f"Resolved {len(tasks)} file paths. Processing with ThreadPoolExecutor...")

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(process_and_cache_image, src, dest) for src, dest in tasks]
        for f in futures:
            f.result()

    dur = time.time() - t0
    cached_count = len(list(CACHE_DIR.glob("*.png")))
    print(f"\nCache precomputation complete in {dur:.2f} seconds.")
    print(f"Total cached images in {CACHE_DIR}: {cached_count}/{len(tasks)}")

if __name__ == "__main__":
    run_cache_precomputation()
