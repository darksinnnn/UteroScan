import csv
import cv2
import easyocr
import re
from pathlib import Path
from collections import defaultdict

EXCLUDED_DIR = Path("d:/Project/UteroScan/data/excluded")
CSV_LOG = EXCLUDED_DIR / "exclusion_log.csv"
DATASETS_DIR = Path("d:/Project/UteroScan/datasets")

BIOMETRY_KEYWORDS = {"GA", "EFW", "BPD", "FL", "HC", "AC"}

def parse_csv():
    if not CSV_LOG.exists():
        print(f"CSV log not found at {CSV_LOG}")
        return [], {}

    rows = []
    keyword_counts = {kw: 0 for kw in BIOMETRY_KEYWORDS}
    bio_flagged_files = set()
    all_flagged = set()

    with open(CSV_LOG, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original = row['original_path']
            all_flagged.add(original)
            reason = row['reason']
            
            # Extract biometry text if present
            # Format is: Biometry Text (exact text)
            if "Biometry Text" in reason:
                bio_flagged_files.add(original)
                match = re.search(r"Biometry Text \((.*?)\)", reason)
                if match:
                    text = match.group(1)
                    words = text.upper().replace(':', ' ').replace('=', ' ').split()
                    matched_kws = [kw for kw in BIOMETRY_KEYWORDS if kw in words]
                    for kw in matched_kws:
                        keyword_counts[kw] += 1
                        
                    filename = Path(original).name
                    kw_str = ",".join(matched_kws)
                    rows.append((filename, kw_str, text))

    return rows, keyword_counts, bio_flagged_files, all_flagged

def run_ob_scan(bio_flagged_files):
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], gpu=True)
    
    # We will scan the excluded folder AND the main datasets folder
    # because images were moved to excluded.
    # Actually, let's just scan everything to be sure, or just the excluded for diagnostic?
    # User said: "Run EasyOCR specifically looking for OB1, OB2, or standalone OB in the top-left corner region of all ~1990 images"
    
    # Gather all images
    all_images = []
    EXTS = (".jpg", ".jpeg", ".png")
    
    for base in [DATASETS_DIR, EXCLUDED_DIR]:
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTS:
                all_images.append(p)
                
    ob_count = 0
    ob_overlap_bio = 0
    
    for img_path in all_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        # Crop top-left corner (e.g., top 15%, left 30%)
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
            # We need to know if this was in bio_flagged_files
            # Wait, the path might be in excluded or dataset.
            # bio_flagged_files has the original path.
            # Let's match by filename to be safe, since names are unique DICOM IDs.
            if any(img_path.name == Path(bf).name for bf in bio_flagged_files):
                ob_overlap_bio += 1
                
    return ob_count, ob_overlap_bio, len(all_images)

def main():
    rows, kw_counts, bio_flagged_files, all_flagged = parse_csv()
    
    print("="*60)
    print("1. RAW MATCHES (First 30)")
    print(f"{'Filename':<55} | {'Keyword':<10} | {'Snippet'}")
    print("-" * 60)
    for r in rows[:30]:
        print(f"{r[0]:<55} | {r[1]:<10} | {r[2]}")
        
    print("\nKeyword Trigger Counts:")
    for kw, count in kw_counts.items():
        print(f"  {kw}: {count}")
        
    print("\n="*60)
    print("2. CONFIRM REGEX VERSION")
    print("Version used: OLD LOOSE KEYWORD MATCH")
    print("Code confirmed: words = text.upper().replace(':', ' ').replace('=', ' ').split(); if any(kw in words for kw in BIOMETRY_KEYWORDS):")
    print("="*60)
    
    print("\nRunning OB OCR on all images (this may take a few minutes)...")
    ob_count, ob_overlap_bio, total_images = run_ob_scan(bio_flagged_files)
    
    print("="*60)
    print("3. OB1/OB2/OB LABEL SIGNAL")
    print(f"Total images scanned: {total_images}")
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
