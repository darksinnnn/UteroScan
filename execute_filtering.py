import os
import shutil
import csv
from pathlib import Path

# Paths
BASE_DIR = Path("d:/Project/UteroScan")
DATASETS_DIR = BASE_DIR / "datasets" / "datasets"
EXCLUDED_DIR = BASE_DIR / "data" / "excluded"
DIAG_CSV = BASE_DIR / "diagnostic_results.csv"

CLS_DIR = BASE_DIR / "DCGAN&MobileNetV2" / "DCGAN&MobileNetV2"
CLS_TRAIN = CLS_DIR / "cls_train.txt"
CLS_TEST = CLS_DIR / "cls_test.txt"

# 1. Duplicate file to exclude
DUPLICATE_FILE = "1.2.826.0.1.3680043.2.461.11899374.664449416.jpg"
DUPLICATE_FOLDER = "test/UF"
DUPLICATE_REASON = "Byte-identical duplicate of 1.2.826.0.1.3680043.2.461.11899374.3915671908.jpg"

def main():
    print("=" * 65)
    print("EXECUTING CLEANUP: MOVING 51 FILES TO data/excluded/")
    print("=" * 65)

    # Read diagnostic CSV for the 50 GA files
    ga_files = []
    with open(DIAG_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["bio_tight"] == "True":
                ga_files.append(row)

    print(f"Found {len(ga_files)} GA= regex matched files in diagnostic results.")
    assert len(ga_files) == 50, f"Expected 50 GA files, got {len(ga_files)}"

    # Prepare exclusion list: (filename, reason, source_folder, source_path)
    to_exclude = []
    for r in ga_files:
        fn = r["filename"]
        folder = r["folder"]  # e.g. "train/NUF"
        src_path = DATASETS_DIR / folder / fn
        reason = f"Fetal biometry overlay ({r['tight_snippet']})"
        to_exclude.append((fn, reason, folder, src_path))

    # Add duplicate file
    dup_src = DATASETS_DIR / DUPLICATE_FOLDER / DUPLICATE_FILE
    to_exclude.append((DUPLICATE_FILE, DUPLICATE_REASON, DUPLICATE_FOLDER, dup_src))

    print(f"Total files to exclude: {len(to_exclude)} (50 GA + 1 duplicate)")

    # Verify all source files exist
    for fn, reason, folder, src_path in to_exclude:
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")

    print("All 51 source files verified on disk.")

    # Create excluded directories
    EXCLUDED_DIR.mkdir(parents=True, exist_ok=True)
    log_rows = []
    excluded_names = set()

    for fn, reason, folder, src_path in to_exclude:
        dest_dir = EXCLUDED_DIR / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / fn
        
        # Move file
        shutil.move(str(src_path), str(dest_path))
        excluded_names.add(fn)
        
        log_rows.append({
            "filename": fn,
            "reason": reason,
            "source folder": folder,
            "moved_to": str(dest_path.relative_to(BASE_DIR))
        })

    # Write CSV log
    csv_log_path = EXCLUDED_DIR / "exclusion_log.csv"
    with open(csv_log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "reason", "source folder", "moved_to"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Successfully moved 51 files to {EXCLUDED_DIR}")
    print(f"Exclusion log written to {csv_log_path}")

    # Regenerate cls_train.txt and cls_test.txt
    print("\nRegenerating cls_train.txt and cls_test.txt...")

    # Filter cls_train.txt
    with open(CLS_TRAIN, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()
    
    clean_train_lines = []
    train_excluded_count = 0
    for line in train_lines:
        line_clean = line.strip()
        if not line_clean: continue
        fn = Path(line_clean.split(";")[-1]).name
        if fn in excluded_names:
            train_excluded_count += 1
        else:
            clean_train_lines.append(line)

    with open(CLS_TRAIN, 'w', encoding='utf-8') as f:
        f.writelines(clean_train_lines)

    # Filter cls_test.txt
    with open(CLS_TEST, 'r', encoding='utf-8') as f:
        test_lines = f.readlines()
    
    clean_test_lines = []
    test_excluded_count = 0
    for line in test_lines:
        line_clean = line.strip()
        if not line_clean: continue
        fn = Path(line_clean.split(";")[-1]).name
        if fn in excluded_names:
            test_excluded_count += 1
        else:
            clean_test_lines.append(line)

    with open(CLS_TEST, 'w', encoding='utf-8') as f:
        f.writelines(clean_test_lines)

    print(f"cls_train.txt: {len(train_lines)} -> {len(clean_train_lines)} (excluded {train_excluded_count})")
    print(f"cls_test.txt : {len(test_lines)} -> {len(clean_test_lines)} (excluded {test_excluded_count})")

    # Verify disk counts
    folders = ["train/UF", "train/NUF", "test/UF", "test/NUF"]
    disk_counts = {}
    total_clean = 0
    for fol in folders:
        p = DATASETS_DIR / fol
        cnt = len(list(p.glob("*.jpg")))
        disk_counts[fol] = cnt
        total_clean += cnt

    print("\n" + "=" * 65)
    print("FINAL CLEAN DATASET COUNTS PER CLASS / SPLIT:")
    print("=" * 65)
    print(f"  train/UF  : {disk_counts['train/UF']}")
    print(f"  train/NUF : {disk_counts['train/NUF']}")
    print(f"  test/UF   : {disk_counts['test/UF']}")
    print(f"  test/NUF  : {disk_counts['test/NUF']}")
    print("-" * 65)
    print(f"  Total Clean Images on Disk: {total_clean} (1,990 - 51 = 1,939)")
    print(f"  Total in cls_train.txt    : {len(clean_train_lines)}")
    print(f"  Total in cls_test.txt     : {len(clean_test_lines)}")
    print(f"  Total in Annotation Files : {len(clean_train_lines) + len(clean_test_lines)}")
    print("=" * 65)

if __name__ == "__main__":
    main()
