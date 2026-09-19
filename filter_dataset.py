import os
import shutil
import cv2
import numpy as np
import easyocr
import csv
from pathlib import Path

# Paths
DATASETS_DIR = Path("d:/Project/UteroScan/datasets")
EXCLUDED_DIR = Path("d:/Project/UteroScan/data/excluded")
CSV_LOG = EXCLUDED_DIR / "exclusion_log.csv"

# Make excluded dir
EXCLUDED_DIR.mkdir(parents=True, exist_ok=True)

# Initialize EasyOCR (English)
# Use GPU if available, easyocr handles it automatically
reader = easyocr.Reader(['en'], gpu=True)

FOLDERS = [
    DATASETS_DIR / "train" / "UF",
    DATASETS_DIR / "train" / "NUF",
    DATASETS_DIR / "test" / "UF",
    DATASETS_DIR / "test" / "NUF"
]

EXTS = (".jpg", ".jpeg", ".png")

# Biometry keywords
BIOMETRY_KEYWORDS = {"GA", "EFW", "BPD", "FL", "HC", "AC"}

def has_color_doppler(img, threshold=20, min_color_pixels=100):
    """
    Check if image has color overlays.
    Ultrasound B-mode is grayscale (R=G=B).
    We check if the max difference between R, G, B channels exceeds a threshold
    for a minimum number of pixels.
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        return False
    
    # Calculate max diff between channels
    max_c = np.max(img, axis=2)
    min_c = np.min(img, axis=2)
    diff = max_c.astype(np.int32) - min_c.astype(np.int32)
    
    color_pixels = np.sum(diff > threshold)
    return color_pixels > min_color_pixels

def has_biometry_text(img):
    """
    Check if image contains fetal biometry text.
    """
    results = reader.readtext(img)
    for (bbox, text, prob) in results:
        # Check if any biometry keyword is a distinct word in the detected text
        words = text.upper().replace(':', ' ').replace('=', ' ').split()
        if any(kw in words for kw in BIOMETRY_KEYWORDS):
            return True, text
    return False, ""

def process_datasets():
    log_entries = []
    stats = {}
    
    for folder in FOLDERS:
        folder_name = f"{folder.parent.name}/{folder.name}"
        if not folder.exists():
            print(f"Skipping {folder_name}, doesn't exist.")
            continue
            
        images = [p for p in folder.glob("*") if p.suffix.lower() in EXTS]
        total = len(images)
        stats[folder_name] = {"total": total, "color": 0, "biometry": 0, "both": 0}
        
        print(f"Processing {folder_name} ({total} images)...")
        
        # Create excluded subfolder to preserve structure
        excluded_subfolder = EXCLUDED_DIR / folder.parent.name / folder.name
        excluded_subfolder.mkdir(parents=True, exist_ok=True)
        
        for idx, img_path in enumerate(images):
            if idx > 0 and idx % 50 == 0:
                print(f"  ... {idx}/{total}")
                
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            is_color = has_color_doppler(img)
            is_bio, text_found = has_biometry_text(img)
            
            reasons = []
            if is_color:
                reasons.append("Color Doppler")
                stats[folder_name]["color"] += 1
            if is_bio:
                reasons.append(f"Biometry Text ({text_found})")
                stats[folder_name]["biometry"] += 1
            if is_color and is_bio:
                stats[folder_name]["both"] += 1
                
            if reasons:
                reason_str = " + ".join(reasons)
                dest_path = excluded_subfolder / img_path.name
                shutil.move(str(img_path), str(dest_path))
                
                log_entries.append({
                    "original_path": str(img_path),
                    "moved_to": str(dest_path),
                    "reason": reason_str
                })

    # Write CSV
    if log_entries:
        with open(CSV_LOG, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["original_path", "moved_to", "reason"])
            writer.writeheader()
            writer.writerows(log_entries)
            
    print("\n" + "="*50)
    print("FILTERING COMPLETE - SUMMARY")
    print("="*50)
    for folder_name, s in stats.items():
        total = s["total"]
        if total == 0:
            continue
        color_pct = (s['color'] / total) * 100
        bio_pct = (s['biometry'] / total) * 100
        print(f"{folder_name}:")
        print(f"  Total images      : {total}")
        print(f"  Color Doppler     : {s['color']} ({color_pct:.1f}%)")
        print(f"  Biometry Text     : {s['biometry']} ({bio_pct:.1f}%)")
        print(f"  Both              : {s['both']}")
        
    print(f"\nMoved {len(log_entries)} flagged images to {EXCLUDED_DIR}")
    print(f"Log saved to {CSV_LOG}")

if __name__ == "__main__":
    process_datasets()
