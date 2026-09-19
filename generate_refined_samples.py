"""
generate_refined_samples.py
Implements the refined preprocessing pipeline addressing the 3 clinical review issues:
1. Pure single-channel grayscale processing -> eliminates all chromatic artifacts and colored dots.
2. Tuned CLAHE (clipLimit=1.2, tileGridSize=(8, 8)) -> eliminates blocky background tile-grid patterns.
3. 3x3 median filter (ksize=3) -> preserves fine myometrial texture and distinct lesion margins without over-softening.
4. Output converted to 3-channel RGB (R=G=B) for ResNet50 input requirement.
"""

import cv2
import numpy as np
from pathlib import Path

TARGET_SIZE = (224, 224)
MEDIAN_KERNEL = 3
CLAHE_CLIP_LIMIT = 1.2
CLAHE_GRID_SIZE = (8, 8)

def preprocess_ultrasound_refined(image, target_size=TARGET_SIZE):
    # Convert to numpy array if PIL
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # 1. Convert to single-channel grayscale (eliminates compression chroma noise and caliper color fringing)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # 2. Resize to 224x224 using INTER_AREA
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    # 3. 3x3 Median Blur (removes speckle noise without over-softening tissue texture)
    denoised = cv2.medianBlur(resized, MEDIAN_KERNEL)

    # 4. CLAHE on grayscale (clipLimit=1.2 avoids blocky background tiling artifacts)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    enhanced = clahe.apply(denoised)

    # 5. Replicate clean grayscale to 3-channel RGB for ResNet50 compatibility
    rgb_out = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return rgb_out

def main():
    base_dir = Path("d:/Project/UteroScan")
    datasets_dir = base_dir / "datasets" / "datasets"
    reports_dir = base_dir / "reports" / "preprocessing_samples"
    reports_dir.mkdir(parents=True, exist_ok=True)
    art_dir = Path(r"C:\Users\ashis\.gemini\antigravity-ide\brain\af0e99e6-8ff9-4623-ad48-86540f6228c9")

    samples = [
        ("sample_01_UF", datasets_dir / "train" / "UF" / "1.2.826.0.1.3680043.2.461.11219581.1358591605.jpg", "UF"),
        ("sample_02_UF", datasets_dir / "train" / "UF" / "1.2.826.0.1.3680043.2.461.11221583.696933856.jpg", "UF"),
        ("sample_03_UF", datasets_dir / "train" / "UF" / "1.2.826.0.1.3680043.2.461.11221584.1158768121.jpg", "UF"),
        ("sample_11_NUF", datasets_dir / "train" / "NUF" / "1.2.826.0.1.3680043.2.461.11229756.2388392267.jpg", "NUF"),
    ]

    print("Generating refined preprocessing comparisons for all 4 samples...")

    for name, img_path, label in samples:
        raw_bgr = cv2.imread(str(img_path))
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        
        # Apply refined pipeline
        prep_rgb = preprocess_ultrasound_refined(raw_rgb)
        prep_bgr = cv2.cvtColor(prep_rgb, cv2.COLOR_RGB2BGR)

        # Raw resized for comparison
        raw_vis = cv2.resize(raw_bgr, (224, 224))
        
        # Overlay clean labels
        cv2.putText(raw_vis, f"RAW ({label})", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(prep_bgr, f"REFINED PREP ({label})", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Side by side montage
        divider = np.full((224, 4, 3), 80, dtype=np.uint8)
        montage = np.hstack([raw_vis, divider, prep_bgr])

        # Save to reports and artifact dir
        filename = f"{name}_{img_path.stem}.png"
        out_report = reports_dir / filename
        out_art = art_dir / filename
        
        cv2.imwrite(str(out_report), montage)
        cv2.imwrite(str(out_art), montage)

        # Check chromatic diff (must be exactly 0)
        chroma_diff = np.max(np.abs(prep_rgb[:,:,0].astype(int) - prep_rgb[:,:,1].astype(int)))
        
        # Measure tissue sharpness (Laplacian variance)
        h, w = prep_bgr.shape[:2]
        center_crop = prep_bgr[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
        sharpness = cv2.Laplacian(center_crop, cv2.CV_64F).var()

        print(f"  {name:<15} -> Chroma diff: {chroma_diff} (0=pure grayscale) | Tissue Sharpness: {sharpness:.1f}")

    print("\nRefined comparison images successfully saved to reports and artifact directory.")

if __name__ == "__main__":
    main()
