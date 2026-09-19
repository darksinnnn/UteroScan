import os
import shutil
import csv
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

# Paths
BASE_DIR = Path("d:/Project/UteroScan")
DATASETS_DIR = BASE_DIR / "datasets" / "datasets"
EXCLUDED_DIR = BASE_DIR / "data" / "excluded"
EXCLUSION_LOG = EXCLUDED_DIR / "exclusion_log.csv"

CLS_DIR = BASE_DIR / "DCGAN&MobileNetV2" / "DCGAN&MobileNetV2"
CLS_TRAIN = CLS_DIR / "cls_train.txt"
CLS_TEST = CLS_DIR / "cls_test.txt"

def find_doppler_flow_images():
    folders = ["train/UF", "train/NUF", "test/UF", "test/NUF"]
    candidates = []
    
    for f in folders:
        folder_path = DATASETS_DIR / f
        imgs = sorted(list(folder_path.glob("*.jpg")))
        for p in imgs:
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            # Ultrasound sector area
            roi = img[int(h * 0.15):int(h * 0.85), int(w * 0.10):int(w * 0.90)]
            b = roi[:, :, 0].astype(int)
            g = roi[:, :, 1].astype(int)
            r = roi[:, :, 2].astype(int)
            
            # Red pixels: high R, lower B and G
            red_mask = (r > 120) & (r - b > 40) & (r - g > 30)
            # Blue pixels: high B, lower R and G
            blue_mask = (b > 100) & (b - r > 40)
            
            red_count = int(np.sum(red_mask))
            blue_count = int(np.sum(blue_mask))
            
            if red_count > 150 and blue_count > 80:
                candidates.append((p.name, f, p, red_count, blue_count))
                
    return candidates

def main():
    print("=" * 70)
    print("EXECUTING COLOR DOPPLER FLOW EXCLUSION (192 IMAGES)")
    print("=" * 70, flush=True)

    candidates = find_doppler_flow_images()
    print(f"Total Doppler flow images detected: {len(candidates)}", flush=True)
    counts = Counter([c[1] for c in candidates])
    for f in ["train/UF", "train/NUF", "test/UF", "test/NUF"]:
        print(f"  {f:<10}: {counts[f]}", flush=True)

    assert len(candidates) == 192, f"Expected 192 candidates, got {len(candidates)}"
    assert counts["train/UF"] == 5, f"Expected 5 in train/UF, got {counts['train/UF']}"
    assert counts["train/NUF"] == 163, f"Expected 163 in train/NUF, got {counts['train/NUF']}"
    assert counts["test/UF"] == 1, f"Expected 1 in test/UF, got {counts['test/UF']}"
    assert counts["test/NUF"] == 23, f"Expected 23 in test/NUF, got {counts['test/NUF']}"

    # Backup current cls_train.txt and cls_test.txt before editing
    backup_train = CLS_DIR / "cls_train_pre_doppler_exclusion.txt"
    backup_test = CLS_DIR / "cls_test_pre_doppler_exclusion.txt"
    shutil.copy2(str(CLS_TRAIN), str(backup_train))
    shutil.copy2(str(CLS_TEST), str(backup_test))
    print(f"\nCreated backups:")
    print(f"  {backup_train}")
    print(f"  {backup_test}", flush=True)

    # Backup existing exclusion_log.csv
    backup_log = EXCLUDED_DIR / "exclusion_log_pre_doppler.csv"
    shutil.copy2(str(EXCLUSION_LOG), str(backup_log))
    print(f"  {backup_log}", flush=True)

    # Move files and prepare new log entries
    new_log_entries = []
    excluded_names = set()

    for fname, folder, src_path, r_cnt, b_cnt in candidates:
        dest_dir = EXCLUDED_DIR / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / fname
        
        # Verify source file exists
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")
            
        shutil.move(str(src_path), str(dest_path))
        excluded_names.add(fname)
        
        # Consistent format matching existing exclusion log:
        # filename,reason,source folder,moved_to
        # moved_to uses relative path: data\excluded\folder\filename
        rel_dest = str(dest_path.relative_to(BASE_DIR))
        new_log_entries.append({
            "filename": fname,
            "reason": "color_doppler_flow_map",
            "source folder": folder,
            "moved_to": rel_dest
        })

    print(f"\nMoved {len(candidates)} files to {EXCLUDED_DIR}", flush=True)

    # Append to existing exclusion_log.csv
    # Read existing entries to count
    existing_entries = []
    if EXCLUSION_LOG.exists():
        with open(EXCLUSION_LOG, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            existing_entries = list(reader)

    print(f"Existing exclusion log entries: {len(existing_entries)}", flush=True)
    all_entries = existing_entries + new_log_entries

    with open(EXCLUSION_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "reason", "source folder", "moved_to"])
        writer.writeheader()
        writer.writerows(all_entries)

    print(f"Updated exclusion log written to {EXCLUSION_LOG} (Total: {len(all_entries)} rows)", flush=True)

    # Regenerate cls_train.txt
    with open(backup_train, "r", encoding="utf-8") as f:
        train_lines = f.readlines()

    clean_train_lines = []
    train_excluded = 0
    for line in train_lines:
        line_s = line.strip()
        if not line_s:
            continue
        fn = Path(line_s.split(";")[-1].split()[0]).name
        if fn in excluded_names:
            train_excluded += 1
        else:
            clean_train_lines.append(line)

    with open(CLS_TRAIN, "w", encoding="utf-8") as f:
        f.writelines(clean_train_lines)

    # Regenerate cls_test.txt
    with open(backup_test, "r", encoding="utf-8") as f:
        test_lines = f.readlines()

    clean_test_lines = []
    test_excluded = 0
    for line in test_lines:
        line_s = line.strip()
        if not line_s:
            continue
        fn = Path(line_s.split(";")[-1].split()[0]).name
        if fn in excluded_names:
            test_excluded += 1
        else:
            clean_test_lines.append(line)

    with open(CLS_TEST, "w", encoding="utf-8") as f:
        f.writelines(clean_test_lines)

    print(f"\nRegenerated Annotation Files:")
    print(f"  cls_train.txt: {len(train_lines)} -> {len(clean_train_lines)} (excluded {train_excluded})")
    print(f"  cls_test.txt : {len(test_lines)} -> {len(clean_test_lines)} (excluded {test_excluded})", flush=True)

    # Disk verification
    folders = ["train/UF", "train/NUF", "test/UF", "test/NUF"]
    disk_counts = {}
    total_disk = 0
    for fol in folders:
        p = DATASETS_DIR / fol
        cnt = len(list(p.glob("*.jpg")))
        disk_counts[fol] = cnt
        total_disk += cnt

    print("\n" + "=" * 70)
    print("FINAL POST-EXCLUSION CLEAN DATASET COUNTS PER CLASS / SPLIT:")
    print("=" * 70)
    print(f"  train/UF  : {disk_counts['train/UF']} (was 700, -5)")
    print(f"  train/NUF : {disk_counts['train/NUF']} (was 857, -163)")
    print(f"  test/UF   : {disk_counts['test/UF']} (was 172, -1)")
    print(f"  test/NUF  : {disk_counts['test/NUF']} (was 210, -23)")
    print("-" * 70)
    print(f"  Total Clean Images on Disk: {total_disk} (1,939 - 192 = 1,747)")
    print(f"  Total in cls_train.txt    : {len(clean_train_lines)} (1,557 - 168 = 1,389)")
    print(f"  Total in cls_test.txt     : {len(clean_test_lines)} (382 - 24 = 358)")
    print(f"  Total in Annotation Files : {len(clean_train_lines) + len(clean_test_lines)} (= {total_disk})")
    print(f"  Total in Exclusion Log    : {len(all_entries)} (51 + 192 = 243)")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
