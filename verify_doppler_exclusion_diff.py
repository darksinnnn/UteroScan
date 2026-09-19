import csv
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

BASE_DIR = Path("d:/Project/UteroScan")
DATASETS_DIR = BASE_DIR / "datasets" / "datasets"
EXCLUDED_DIR = BASE_DIR / "data" / "excluded"
EXCLUSION_LOG = EXCLUDED_DIR / "exclusion_log.csv"

def get_excluded_doppler_from_log():
    doppler_files = {}
    with open(EXCLUSION_LOG, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["reason"] == "color_doppler_flow_map":
                doppler_files[row["filename"]] = row["source folder"]
    return doppler_files

def get_pregnancy_and_dup_files():
    preg_dup = set()
    with open(EXCLUSION_LOG, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["reason"] != "color_doppler_flow_map":
                preg_dup.add(row["filename"])
    return preg_dup

def run_color_doppler_detector_on_all_1939():
    """
    Scans all 1,939 clean images (which are currently either in datasets/datasets/
    or in data/excluded/ under the Doppler exclusion).
    Excludes the 51 pregnancy/duplicate scans.
    """
    preg_dup = get_pregnancy_and_dup_files()
    assert len(preg_dup) == 51, f"Expected 51 pregnancy/duplicate scans, got {len(preg_dup)}"

    folders = ["train/UF", "train/NUF", "test/UF", "test/NUF"]
    detected_doppler = {}

    for f in folders:
        # Search both current datasets directory and excluded directory for this folder
        sources = [DATASETS_DIR / f, EXCLUDED_DIR / f]
        seen_in_folder = set()
        for src in sources:
            if not src.exists():
                continue
            for p in sorted(list(src.glob("*.jpg"))):
                if p.name in preg_dup:
                    continue  # skip the 51 Phase 0 excluded files
                if p.name in seen_in_folder:
                    continue
                seen_in_folder.add(p.name)

                img = cv2.imread(str(p))
                if img is None:
                    continue
                h, w = img.shape[:2]
                roi = img[int(h * 0.15):int(h * 0.85), int(w * 0.10):int(w * 0.90)]
                b = roi[:, :, 0].astype(int)
                g = roi[:, :, 1].astype(int)
                r = roi[:, :, 2].astype(int)

                red_mask = (r > 120) & (r - b > 40) & (r - g > 30)
                blue_mask = (b > 100) & (b - r > 40)

                red_count = int(np.sum(red_mask))
                blue_count = int(np.sum(blue_mask))

                if red_count > 150 and blue_count > 80:
                    detected_doppler[p.name] = (f, red_count, blue_count)

    return detected_doppler

def main():
    print("=" * 75)
    print("EXACT FILENAME DIFF: EXCLUSION STEP vs. QUANTIFICATION PASS")
    print("=" * 75, flush=True)

    # 1. Filenames in exclusion log
    log_doppler = get_excluded_doppler_from_log()
    print(f"1. Filenames in exclusion log (reason='color_doppler_flow_map'): {len(log_doppler)}", flush=True)

    # 2. Re-detect on all 1,939 images using original detection logic
    print("2. Re-running detection logic across all 1,939 images...", flush=True)
    detected_doppler = run_color_doppler_detector_on_all_1939()
    print(f"   Detected Color Doppler images: {len(detected_doppler)}", flush=True)

    # 3. Check physical presence on disk in data/excluded/
    physical_on_disk = {}
    for f in ["train/UF", "train/NUF", "test/UF", "test/NUF"]:
        folder_p = EXCLUDED_DIR / f
        if folder_p.exists():
            for p in folder_p.glob("*.jpg"):
                if p.name in log_doppler:
                    physical_on_disk[p.name] = f
    print(f"3. Doppler files physically present in data/excluded/: {len(physical_on_disk)}", flush=True)

    # 4. Perform direct Set Diffs
    set_log = set(log_doppler.keys())
    set_detected = set(detected_doppler.keys())
    set_disk = set(physical_on_disk.keys())

    diff_detected_minus_log = set_detected - set_log
    diff_log_minus_detected = set_log - set_detected
    diff_log_minus_disk = set_log - set_disk
    diff_disk_minus_log = set_disk - set_log

    print("\n" + "-" * 75)
    print("DIFF RESULTS:")
    print("-" * 75)
    print(f"  Detected in Pass vs. In Exclusion Log: "
          f"Missing from Log = {len(diff_detected_minus_log)}, Unexpected in Log = {len(diff_log_minus_detected)}")
    print(f"  In Exclusion Log vs. Physically on Disk: "
          f"Missing from Disk = {len(diff_log_minus_disk)}, Unexpected on Disk = {len(diff_disk_minus_log)}")

    if diff_detected_minus_log:
        print("\n  Files detected in pass but NOT in exclusion log:")
        for fn in sorted(list(diff_detected_minus_log)):
            print(f"    - {fn}")

    if diff_log_minus_detected:
        print("\n  Files in exclusion log but NOT detected in pass:")
        for fn in sorted(list(diff_log_minus_detected)):
            print(f"    - {fn}")

    # Check folder classification consistency
    folder_mismatches = []
    for fn in set_log:
        f_log = log_doppler[fn]
        f_det = detected_doppler.get(fn, (None,))[0]
        if f_log != f_det:
            folder_mismatches.append((fn, f_log, f_det))

    print(f"  Folder/Split mismatches between log and detection: {len(folder_mismatches)}")
    if folder_mismatches:
        for fn, fl, fd in folder_mismatches:
            print(f"    - {fn}: Log={fl}, Det={fd}")

    # Breakdown by folder
    print("\n" + "-" * 75)
    print(f"{'Folder':<12} | {'Detected':<10} | {'In Exclusion Log':<18} | {'On Disk in data/excluded'}")
    print("-" * 75)
    for f in ["train/UF", "train/NUF", "test/UF", "test/NUF"]:
        cnt_det = sum(1 for v in detected_doppler.values() if v[0] == f)
        cnt_log = sum(1 for v in log_doppler.values() if v == f)
        cnt_dsk = sum(1 for v in physical_on_disk.values() if v == f)
        print(f"{f:<12} | {cnt_det:<10} | {cnt_log:<18} | {cnt_dsk}")
    print("-" * 75)
    print(f"{'TOTAL':<12} | {len(detected_doppler):<10} | {len(log_doppler):<18} | {len(physical_on_disk)}")
    print("=" * 75)

    if not diff_detected_minus_log and not diff_log_minus_detected and not folder_mismatches:
        print("\n>>> VERIFICATION SUCCESSFUL: 100% PERFECT 1-TO-1 MATCH (0 MISMATCHES) <<<")
    else:
        print("\n>>> WARNING: MISMATCHES DETECTED! <<<")

if __name__ == "__main__":
    main()
