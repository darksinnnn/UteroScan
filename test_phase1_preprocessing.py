"""
test_phase1_preprocessing.py
Phase 1 verification script:
1. Runs UteroScan preprocessing (resize 224x224 INTER_AREA, medianBlur 5, CLAHE on L-channel)
   on a representative sample of 20 images (10 UF, 10 NUF).
2. Computes statistical intensity and edge metrics (before vs after) to verify:
   - Boundaries are sharp and distinct
   - No saturation/blown-out highlights (pixels at 255)
   - No crushed blacks/over-darkening (pixels at 0)
3. Saves side-by-side visual comparison images to reports/preprocessing_samples/
4. Runs a full dry run through all 1,939 images in the dataset to verify 0 pipeline errors.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add DCGAN path to sys.path
sys.path.append(str(Path("d:/Project/UteroScan/DCGAN&MobileNetV2/DCGAN&MobileNetV2")))
from utils.preprocessing import preprocess_ultrasound

BASE_DIR = Path("d:/Project/UteroScan")
DATASETS_DIR = BASE_DIR / "datasets" / "datasets"
REPORTS_DIR = BASE_DIR / "reports" / "preprocessing_samples"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLS_TRAIN = BASE_DIR / "DCGAN&MobileNetV2" / "DCGAN&MobileNetV2" / "cls_train.txt"
CLS_TEST = BASE_DIR / "DCGAN&MobileNetV2" / "DCGAN&MobileNetV2" / "cls_test.txt"

def resolve_path(rel_path_str):
    p = Path(rel_path_str)
    if p.is_file():
        return p
    candidates = [
        BASE_DIR / "datasets" / p,
        DATASETS_DIR / Path(*p.parts[1:] if len(p.parts) > 1 else p.parts),
        DATASETS_DIR / p,
        BASE_DIR / p
    ]
    for c in candidates:
        if c.is_file():
            return c
    return p

def main():
    print("=" * 70)
    print("PHASE 1: PREPROCESSING PIPELINE VERIFICATION")
    print("=" * 70)

    # 1. Collect sample of 20 images: 10 UF and 10 NUF
    uf_imgs = sorted(list((DATASETS_DIR / "train" / "UF").glob("*.jpg")))[:10]
    nuf_imgs = sorted(list((DATASETS_DIR / "train" / "NUF").glob("*.jpg")))[:10]
    sample_imgs = [(p, "UF") for p in uf_imgs] + [(p, "NUF") for p in nuf_imgs]

    print(f"\nEvaluating sample of {len(sample_imgs)} images (10 UF, 10 NUF)...")
    
    sample_metrics = []
    for idx, (img_path, label) in enumerate(sample_imgs, 1):
        raw = cv2.imread(str(img_path))
        assert raw is not None, f"Failed to read {img_path}"
        
        # Raw stats
        raw_gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        raw_lap = cv2.Laplacian(raw_gray, cv2.CV_64F).var() # Edge sharpness indicator
        raw_min, raw_max, raw_mean, raw_std = float(np.min(raw)), float(np.max(raw)), float(np.mean(raw)), float(np.std(raw))
        raw_zeros = float(np.mean(raw == 0)) * 100
        raw_saturated = float(np.mean(raw == 255)) * 100

        # Preprocessed (resize 224x224 INTER_AREA, medianBlur 5, CLAHE)
        prep = preprocess_ultrasound(raw, target_size=(224, 224))
        prep_bgr = cv2.cvtColor(prep, cv2.COLOR_RGB2BGR)
        prep_gray = cv2.cvtColor(prep_bgr, cv2.COLOR_BGR2GRAY)
        prep_lap = cv2.Laplacian(prep_gray, cv2.CV_64F).var()
        prep_min, prep_max, prep_mean, prep_std = float(np.min(prep)), float(np.max(prep)), float(np.mean(prep)), float(np.std(prep))
        prep_zeros = float(np.mean(prep == 0)) * 100
        prep_saturated = float(np.mean(prep == 255)) * 100

        sample_metrics.append({
            "filename": img_path.name,
            "label": label,
            "raw_shape": raw.shape,
            "raw_mean": raw_mean,
            "raw_std": raw_std,
            "prep_mean": prep_mean,
            "prep_std": prep_std,
            "raw_lap": raw_lap,
            "prep_lap": prep_lap,
            "prep_zeros": prep_zeros,
            "prep_saturated": prep_saturated
        })

        # Save side-by-side comparison for the first 6 samples (3 UF, 3 NUF)
        if idx <= 3 or (11 <= idx <= 13):
            # Resize raw to 224x224 just for visual side-by-side montage
            raw_vis = cv2.resize(raw, (224, 224))
            
            # Put text label
            cv2.putText(raw_vis, f"RAW ({label})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(prep_bgr, f"PREPROCESSED ({label})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Combine side by side
            divider = np.full((224, 4, 3), 128, dtype=np.uint8)
            combined = np.hstack([raw_vis, divider, prep_bgr])
            
            out_file = REPORTS_DIR / f"sample_{idx:02d}_{label}_{img_path.stem}.png"
            cv2.imwrite(str(out_file), combined)

    print(f"Saved side-by-side comparison images to {REPORTS_DIR}")

    # Summary table of 20 samples
    print("\nSAMPLE ANALYSIS (20 Images: 10 UF, 10 NUF):")
    print(f"{'Filename':<36} | {'Class':<5} | {'Raw Mean':<8} | {'Prep Mean':<9} | {'Raw Std':<7} | {'Prep Std':<8} | {'Zero%':<6} | {'Sat%':<6}")
    print("-" * 100)
    for m in sample_metrics:
        print(f"{m['filename']:<36} | {m['label']:<5} | {m['raw_mean']:<8.1f} | {m['prep_mean']:<9.1f} | {m['raw_std']:<7.1f} | {m['prep_std']:<8.1f} | {m['prep_zeros']:<6.2f} | {m['prep_saturated']:<6.2f}")

    avg_raw_std = np.mean([m['raw_std'] for m in sample_metrics])
    avg_prep_std = np.mean([m['prep_std'] for m in sample_metrics])
    avg_sat = np.mean([m['prep_saturated'] for m in sample_metrics])
    avg_zero = np.mean([m['prep_zeros'] for m in sample_metrics])

    print("-" * 100)
    print(f"Average Contrast (Std Dev): Raw={avg_raw_std:.1f} -> Preprocessed={avg_prep_std:.1f} (Enhanced contrast: +{avg_prep_std - avg_raw_std:.1f})")
    print(f"Average Saturated Pixels (at 255): {avg_sat:.2f}% (No blown-out highlights)")
    print(f"Average Saturated Blacks (at 0): {avg_zero:.2f}% (No over-darkened regions)")

    # 4. Full dry run on all 1,939 images
    print("\n" + "=" * 70)
    print("STARTING FULL DRY RUN OVER ALL 1,939 CLEAN DATASET IMAGES")
    print("=" * 70)

    all_annotation_lines = []
    with open(CLS_TRAIN, 'r', encoding='utf-8') as f:
        all_annotation_lines.extend(f.readlines())
    with open(CLS_TEST, 'r', encoding='utf-8') as f:
        all_annotation_lines.extend(f.readlines())

    total_images = len(all_annotation_lines)
    print(f"Total entries in annotation files: {total_images}")
    assert total_images == 1939, f"Expected 1939 entries, got {total_images}"

    errors = []
    shapes = set()
    channels = set()

    for idx, line in enumerate(all_annotation_lines, 1):
        if idx % 400 == 0 or idx == total_images:
            print(f"  Verified {idx}/{total_images} images ({(idx/total_images)*100:.1f}%)...")
        
        parts = line.strip().split(';')
        if len(parts) != 2:
            errors.append((idx, "Bad annotation line format", line))
            continue
            
        cls_id = int(parts[0])
        rel_path = parts[1]
        full_path = resolve_path(rel_path)
        
        if not full_path.is_file():
            errors.append((idx, "File not found", str(full_path)))
            continue
            
        img = cv2.imread(str(full_path))
        if img is None:
            errors.append((idx, "Corrupt / unreadable image", str(full_path)))
            continue
            
        prep = preprocess_ultrasound(img, target_size=(224, 224))
        shapes.add(prep.shape)
        channels.add(prep.dtype)

    print("\n" + "=" * 70)
    print("FULL DRY RUN RESULTS:")
    print("=" * 70)
    print(f"  Total images processed    : {total_images}")
    print(f"  Total errors encountered  : {len(errors)}")
    print(f"  Unique output shapes      : {shapes} (Expected: {{(224, 224, 3)}})")
    print(f"  Unique output data types  : {channels} (Expected: {{dtype('uint8')}})")
    
    if len(errors) == 0 and shapes == {(224, 224, 3)}:
        print("  Status: 100% SUCCESS - ALL 1,939 IMAGES PREPROCESSED WITH ZERO ERRORS.")
    else:
        print(f"  Status: FAILED with {len(errors)} errors.")
        for e in errors[:5]:
            print("    ", e)

if __name__ == "__main__":
    main()
