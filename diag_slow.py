import cv2
import easyocr
from pathlib import Path

EXCLUDED_DIR = Path("d:/Project/UteroScan/data/excluded")
DATASETS_DIR = Path("d:/Project/UteroScan/datasets")
BIOMETRY_KEYWORDS = {"GA", "EFW", "BPD", "FL", "HC", "AC"}

def get_bio_flagged_files():
    # Helper to just return set of names that are bio flagged
    reader = easyocr.Reader(['en'], gpu=True)
    EXTS = (".jpg", ".jpeg", ".png")
    excluded = list(EXCLUDED_DIR.rglob("*"))
    bio_flagged = set()
    all_flagged = set()
    
    for p in excluded:
        if p.is_file() and p.suffix.lower() in EXTS:
            all_flagged.add(str(p))
            img = cv2.imread(str(p))
            if img is None: continue
            results = reader.readtext(img)
            for (bbox, text, prob) in results:
                words = text.upper().replace(':', ' ').replace('=', ' ').split()
                if any(kw in words for kw in BIOMETRY_KEYWORDS):
                    bio_flagged.add(p.name)
                    break
    return bio_flagged, all_flagged

def main():
    print("Getting bio flagged files...", flush=True)
    bio_flagged_files, all_flagged = get_bio_flagged_files()
    
    print(f"Total flagged files (color+bio): {len(all_flagged)}")
    print(f"Bio flagged files: {len(bio_flagged_files)}")
    
    print("\nRunning OB OCR on all images...", flush=True)
    reader = easyocr.Reader(['en'], gpu=True)
    
    EXTS = (".jpg", ".jpeg", ".png")
    all_images = []
    for base in [DATASETS_DIR, EXCLUDED_DIR]:
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTS:
                all_images.append(p)
                
    ob_count = 0
    ob_overlap_bio = 0
    
    for idx, img_path in enumerate(all_images):
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(all_images)}...", flush=True)
        img = cv2.imread(str(img_path))
        if img is None: continue
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
    total_bio = len(bio_flagged_files)
    only_bio = total_bio - ob_overlap_bio
    both = ob_overlap_bio
    print(f"Total files flagged so far (color + bio): {total_flagged}")
    print(f"Files flagged for Biometry: {total_bio}")
    print(f"  ONLY Biometry (no OB label): {only_bio}")
    print(f"  BOTH (Biometry + OB label) : {both}")
    print("="*60)

if __name__ == "__main__":
    main()
