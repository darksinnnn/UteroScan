import cv2
import easyocr
import re
from pathlib import Path

EXCLUDED_DIR = Path("d:/Project/UteroScan/data/excluded")
DATASETS_DIR = Path("d:/Project/UteroScan/datasets")

BIOMETRY_KEYWORDS = {"GA", "EFW", "BPD", "FL", "HC", "AC"}

def has_color_doppler(img, threshold=20, min_color_pixels=100):
    if len(img.shape) != 3 or img.shape[2] != 3:
        return False
    max_c = np.max(img, axis=2)
    min_c = np.min(img, axis=2)
    diff = max_c.astype(np.int32) - min_c.astype(np.int32)
    return np.sum(diff > threshold) > min_color_pixels

def main():
    import numpy as np
    
    # Initialize OCR
    reader = easyocr.Reader(['en'], gpu=True)
    
    # 1. Reconstruct CSV logic since it wasn't saved before the server restart
    rows = []
    keyword_counts = {kw: 0 for kw in BIOMETRY_KEYWORDS}
    bio_flagged_files = set()
    all_flagged = set()
    
    EXTS = (".jpg", ".jpeg", ".png")
    
    print("Reconstructing matches from excluded folder...")
    for p in EXCLUDED_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            all_flagged.add(str(p))
            img = cv2.imread(str(p))
            if img is None: continue
            
            # Re-run the bio check to get the exact text snippet it matched
            results = reader.readtext(img)
            bio_found = False
            matched_kws = []
            exact_text = ""
            for (bbox, text, prob) in results:
                words = text.upper().replace(':', ' ').replace('=', ' ').split()
                matches = [kw for kw in BIOMETRY_KEYWORDS if kw in words]
                if matches:
                    matched_kws.extend(matches)
                    exact_text = text
                    bio_found = True
                    break
                    
            if bio_found:
                bio_flagged_files.add(p.name)
                for kw in matched_kws:
                    keyword_counts[kw] += 1
                rows.append((p.name, ",".join(matched_kws), exact_text))
                
    print("="*60)
    print("1. RAW MATCHES (First 30)")
    print(f"{'Filename':<45} | {'Keyword':<10} | {'Snippet'}")
    print("-" * 60)
    for r in rows[:30]:
        print(f"{r[0]:<45} | {r[1]:<10} | {r[2]}")
        
    print("\nKeyword Trigger Counts:")
    for kw, count in keyword_counts.items():
        print(f"  {kw}: {count}")
        
    print("\n="*60)
    print("2. CONFIRM REGEX VERSION")
    print("Version used: OLD LOOSE KEYWORD MATCH")
    print("Code confirmed: words = text.upper().replace(':', ' ').replace('=', ' ').split(); if any(kw in words for kw in BIOMETRY_KEYWORDS):")
    print("="*60)
    
    print("\nRunning OB OCR on all images (this may take a minute)...")
    
    all_images = []
    for base in [DATASETS_DIR, EXCLUDED_DIR]:
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTS:
                all_images.append(p)
                
    ob_count = 0
    ob_overlap_bio = 0
    
    for idx, img_path in enumerate(all_images):
        if idx > 0 and idx % 200 == 0:
            print(f"  ... {idx}/{len(all_images)}")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        crop = img[0:int(h*0.15), 0:int(w*0.30)]
        results = reader.readtext(crop)
        ob_found = False
        for (bbox, text, prob) in results:
            words = text.upper().split()
            if any(w in {"OB", "OB1", "OB2"} for w in words):
                ob_found = True
                break
                
        if ob_found:
            ob_count += 1
            if img_path.name in bio_flagged_files:
                ob_overlap_bio += 1

    print("="*60)
    print("3. OB1/OB2/OB LABEL SIGNAL")
    print(f"Total images scanned: {len(all_images)}")
    print(f"Images with OB/OB1/OB2 in top-left: {ob_count}")
    print(f"Overlap with biometry flagged files: {ob_overlap_bio}")
    print("="*60)
    
    print("\n="*60)
    print("4. CROSS-TABULATION")
    total_flagged = len(all_flagged)
    total_bio_flagged = len(bio_flagged_files)
    only_bio = total_bio_flagged - ob_overlap_bio
    both = ob_overlap_bio
    print(f"Total files flagged so far (color + bio): {total_flagged}")
    print(f"Files flagged for Biometry: {total_bio_flagged}")
    print(f"  ONLY Biometry (no OB label): {only_bio}")
    print(f"  BOTH (Biometry + OB label) : {both}")
    print("="*60)

if __name__ == "__main__":
    main()
