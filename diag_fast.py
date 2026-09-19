import cv2
import easyocr
import re
import sys
from pathlib import Path

EXCLUDED_DIR = Path("d:/Project/UteroScan/data/excluded")
BIOMETRY_KEYWORDS = {"GA", "EFW", "BPD", "FL", "HC", "AC"}

def main():
    print("Initializing OCR (CPU)...", flush=True)
    reader = easyocr.Reader(['en'], gpu=True)
    
    rows = []
    keyword_counts = {kw: 0 for kw in BIOMETRY_KEYWORDS}
    
    EXTS = (".jpg", ".jpeg", ".png")
    excluded_files = list(EXCLUDED_DIR.rglob("*"))
    excluded_imgs = [p for p in excluded_files if p.is_file() and p.suffix.lower() in EXTS]
    
    print(f"Scanning {len(excluded_imgs)} excluded images for text matches...", flush=True)
    
    for idx, p in enumerate(excluded_imgs):
        if idx % 10 == 0:
            print(f"  Processed {idx}/{len(excluded_imgs)}...", flush=True)
            
        img = cv2.imread(str(p))
        if img is None: continue
        
        results = reader.readtext(img)
        matched_kws = []
        exact_text = ""
        for (bbox, text, prob) in results:
            words = text.upper().replace(':', ' ').replace('=', ' ').split()
            matches = [kw for kw in BIOMETRY_KEYWORDS if kw in words]
            if matches:
                matched_kws.extend(matches)
                exact_text = text
                break
                
        if matched_kws:
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

if __name__ == "__main__":
    main()
