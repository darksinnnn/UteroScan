import cv2
import easyocr
import re
import csv
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("d:/Project/UteroScan/datasets/datasets")
FOLDERS = {
    "train/UF":  BASE_DIR / "train" / "UF",
    "train/NUF": BASE_DIR / "train" / "NUF",
    "test/UF":   BASE_DIR / "test"  / "UF",
    "test/NUF":  BASE_DIR / "test"  / "NUF"
}

BIOMETRY_KEYWORDS = {"GA", "EFW", "BPD", "FL", "HC", "AC"}
TIGHT_REGEX = re.compile(r'\b(GA|EFW|BPD|FL|HC|AC)\s*[:=]?\s*\d+', re.IGNORECASE)

def has_color_doppler(img, threshold=20, min_color_pixels=100):
    if len(img.shape) != 3 or img.shape[2] != 3:
        return False
    max_c = np.max(img, axis=2)
    min_c = np.min(img, axis=2)
    diff = max_c.astype(np.int32) - min_c.astype(np.int32)
    return int(np.sum(diff > threshold)) > min_color_pixels

def check_ob_label(texts):
    # Looking for OB, OB1, OB2, OB-1, OB/GYN, OBGYN
    ob_patterns = {"OB", "OB1", "OB2", "OB/GYN", "OBGYN", "OB-1", "OB-2"}
    for t in texts:
        words = re.split(r'[\s,;:|/]+', t.upper())
        for w in words:
            if w in ob_patterns or re.match(r'^OB[12]?$', w):
                return True, t
    return False, ""

def check_biometry_loose(texts):
    matched = []
    snippet = ""
    for t in texts:
        words = t.upper().replace(':', ' ').replace('=', ' ').split()
        hits = [kw for kw in BIOMETRY_KEYWORDS if kw in words]
        if hits:
            matched.extend(hits)
            if not snippet:
                snippet = t
    return list(set(matched)), snippet

def check_biometry_tight(texts):
    matched = []
    snippet = ""
    for t in texts:
        found = TIGHT_REGEX.findall(t)
        if found:
            matched.extend([f.upper() for f in found])
            if not snippet:
                snippet = t
    return list(set(matched)), snippet

def main():
    print("=" * 70)
    print("STARTING GPU DIAGNOSTIC PASS ON FULL DATASET (1,990 IMAGES)")
    print("=" * 70, flush=True)

    t_start = time.time()
    reader = easyocr.Reader(['en'], gpu=True)
    
    records = []
    folder_stats = {
        k: {
            "total": 0,
            "color_doppler": 0,
            "ob_label": 0,
            "bio_loose": 0,
            "bio_tight": 0,
            "both_ob_and_bio": 0
        } for k in FOLDERS
    }
    
    keyword_counts_loose = {kw: 0 for kw in BIOMETRY_KEYWORDS}
    keyword_counts_tight = {kw: 0 for kw in BIOMETRY_KEYWORDS}

    all_images = []
    for folder_key, folder_path in FOLDERS.items():
        imgs = sorted(list(folder_path.glob("*.jpg")))
        for p in imgs:
            all_images.append((folder_key, p))

    total_images = len(all_images)
    print(f"Total images collected: {total_images}", flush=True)

    for idx, (folder_key, img_path) in enumerate(all_images):
        if idx > 0 and idx % 200 == 0:
            elapsed = time.time() - t_start
            ips = idx / elapsed
            rem = (total_images - idx) / ips if ips > 0 else 0
            print(f"  Processed {idx}/{total_images} ({idx/total_images*100:.1f}%) - {ips:.1f} img/s - Est. remaining: {rem:.0f}s", flush=True)

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        folder_stats[folder_key]["total"] += 1

        # 1. Color Doppler
        is_color = has_color_doppler(img)
        if is_color:
            folder_stats[folder_key]["color_doppler"] += 1

        # 2. Top Banner / Header Crop for OB and telemetry text
        h, w = img.shape[:2]
        # Crop top 22% height and full width to catch both preset (top-left) and measurements/telemetry (top banner)
        top_crop = img[0:int(h * 0.22), :]
        top_texts_raw = reader.readtext(top_crop, mag_ratio=1.0)
        top_texts = [t[1] for t in top_texts_raw]

        is_ob, ob_snippet = check_ob_label(top_texts)
        if is_ob:
            folder_stats[folder_key]["ob_label"] += 1

        # Biometry loose check on top banner
        loose_kws, loose_snippet = check_biometry_loose(top_texts)
        # Biometry tight check on top banner
        tight_kws, tight_snippet = check_biometry_tight(top_texts)

        # If OB is found or if loose matched, also scan right sidebar where caliper tables live
        # (right 30% width, height 20% to 80%) to be 100% thorough
        right_crop = img[int(h*0.20):int(h*0.80), int(w*0.70):]
        right_texts_raw = reader.readtext(right_crop, mag_ratio=1.0)
        right_texts = [t[1] for t in right_texts_raw]
        
        r_loose_kws, r_loose_snip = check_biometry_loose(right_texts)
        r_tight_kws, r_tight_snip = check_biometry_tight(right_texts)

        if r_loose_kws:
            loose_kws = list(set(loose_kws + r_loose_kws))
            if not loose_snippet: loose_snippet = r_loose_snip
        if r_tight_kws:
            tight_kws = list(set(tight_kws + r_tight_kws))
            if not tight_snippet: tight_snippet = r_tight_snip

        is_loose = len(loose_kws) > 0
        is_tight = len(tight_kws) > 0

        if is_loose:
            folder_stats[folder_key]["bio_loose"] += 1
            for kw in loose_kws:
                keyword_counts_loose[kw] += 1

        if is_tight:
            folder_stats[folder_key]["bio_tight"] += 1
            for kw in tight_kws:
                keyword_counts_tight[kw] += 1

        if is_ob and is_loose:
            folder_stats[folder_key]["both_ob_and_bio"] += 1

        records.append({
            "filename": img_path.name,
            "folder": folder_key,
            "color_doppler": is_color,
            "ob_label": is_ob,
            "ob_snippet": ob_snippet,
            "bio_loose": is_loose,
            "loose_keywords": ",".join(loose_kws),
            "loose_snippet": loose_snippet,
            "bio_tight": is_tight,
            "tight_keywords": ",".join(tight_kws),
            "tight_snippet": tight_snippet
        })

    # Save to CSV
    csv_path = Path("d:/Project/UteroScan/diagnostic_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "folder", "color_doppler", "ob_label", "ob_snippet",
            "bio_loose", "loose_keywords", "loose_snippet",
            "bio_tight", "tight_keywords", "tight_snippet"
        ])
        writer.writeheader()
        writer.writerows(records)

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"DIAGNOSTIC SCAN FINISHED in {total_time:.1f}s ({total_time/60:.2f} min)")
    print("=" * 70)

    # 1. RAW MATCHES TABLE (FIRST 30)
    loose_matches = [r for r in records if r["bio_loose"]]
    print("\n1. RAW MATCHES TABLE (First 30 rows flagged by loose biometry):")
    print(f"{'Filename':<42} | {'Folder':<10} | {'Keyword':<12} | {'Exact Snippet'}")
    print("-" * 90)
    for r in loose_matches[:30]:
        print(f"{r['filename']:<42} | {r['folder']:<10} | {r['loose_keywords']:<12} | {r['loose_snippet']}")

    print("\nKeyword Trigger Counts (Loose):")
    for kw, cnt in sorted(keyword_counts_loose.items()):
        print(f"  {kw:<5}: {cnt}")

    print("\nKeyword Trigger Counts (Tightened Regex e.g. GA: 12w3d / FL: 24mm):")
    for kw, cnt in sorted(keyword_counts_tight.items()):
        print(f"  {kw:<5}: {cnt}")

    # 2. Breakdown per folder
    print("\n2. PER-FOLDER BREAKDOWN:")
    print(f"{'Folder':<12} | {'Total':<6} | {'ColorDoppler':<13} | {'OB Label':<10} | {'Bio Loose':<10} | {'Bio Tight':<10} | {'Both(OB+Bio)'}")
    print("-" * 85)
    for f_key, s in folder_stats.items():
        print(f"{f_key:<12} | {s['total']:<6} | {s['color_doppler']:<13} | {s['ob_label']:<10} | {s['bio_loose']:<10} | {s['bio_tight']:<10} | {s['both_ob_and_bio']}")

    # 3. Overall Totals and Overlap
    total_ob = sum(s["ob_label"] for s in folder_stats.values())
    total_loose = sum(s["bio_loose"] for s in folder_stats.values())
    total_tight = sum(s["bio_tight"] for s in folder_stats.values())
    total_both = sum(s["both_ob_and_bio"] for s in folder_stats.values())
    pure_loose_only = total_loose - total_both
    pure_ob_only = total_ob - total_both
    total_color = sum(s["color_doppler"] for s in folder_stats.values())

    print("\n3. OVERALL SIGNAL TOTALS & CROSS-TABULATION:")
    print(f"  Total Images in Dataset       : {total_images}")
    print(f"  Color Doppler images          : {total_color}")
    print(f"  Images with OB/OB1/OB2 Label  : {total_ob}")
    print(f"  Images with Loose Biometry    : {total_loose}")
    print(f"  Images with Tight Biometry    : {total_tight}")
    print(f"  Overlap (BOTH OB + Loose Bio) : {total_both}")
    print(f"  Purely Loose Bio (NO OB)      : {pure_loose_only}")
    print(f"  Purely OB (NO Bio detected)   : {pure_ob_only}")
    print("=" * 70)

if __name__ == "__main__":
    main()
