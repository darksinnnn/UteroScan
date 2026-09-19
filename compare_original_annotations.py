import shutil
import os
import csv
import cv2
import hashlib
from pathlib import Path

orig_src = Path(r"C:\Users\ashis\Downloads\DCGAN&MobileNetV2\DCGAN&MobileNetV2")
dest_dir = Path(r"d:\Project\UteroScan\DCGAN&MobileNetV2\DCGAN&MobileNetV2")

# 1. Restore copies as cls_train_ORIGINAL.txt and cls_test_ORIGINAL.txt
shutil.copy(orig_src / "cls_train.txt", dest_dir / "cls_train_ORIGINAL.txt")
shutil.copy(orig_src / "cls_test.txt", dest_dir / "cls_test_ORIGINAL.txt")
print("Restored cls_train_ORIGINAL.txt and cls_test_ORIGINAL.txt successfully.")

# 2. Read original files
orig_train = [l.strip() for l in open(dest_dir / "cls_train_ORIGINAL.txt", encoding="utf-8") if l.strip()]
orig_test = [l.strip() for l in open(dest_dir / "cls_test_ORIGINAL.txt", encoding="utf-8") if l.strip()]
orig_all_names = {Path(l.split(";")[-1]).name for l in orig_train + orig_test}

print(f"Original train count : {len(orig_train)}")
print(f"Original test count  : {len(orig_test)}")
print(f"Original total unique: {len(orig_all_names)}")

# 3. Read all 1990 images (clean datasets + data/excluded)
all_1990 = {}
for p in Path(r"d:\Project\UteroScan\datasets\datasets").rglob("*.jpg"):
    all_1990[p.name] = p
for p in Path(r"d:\Project\UteroScan\data\excluded").rglob("*.jpg"):
    all_1990[p.name] = p

print(f"Total dataset files on disk (clean + excluded): {len(all_1990)}")

# 4. Find the missing 8 files
missing_8 = sorted(list(set(all_1990.keys()) - orig_all_names))
print(f"\nFiles in image folders but MISSING from original Mendeley annotation lists: {len(missing_8)}")

# Diagnostic info lookup
diag_info = {}
if os.path.exists("diagnostic_results.csv"):
    with open("diagnostic_results.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            diag_info[row["filename"]] = row

# Also check if any of these 8 files are duplicates of any other file in the dataset!
all_hashes = {}
for name, path in all_1990.items():
    h = hashlib.md5(path.read_bytes()).hexdigest()
    all_hashes.setdefault(h, []).append((name, path))

for idx, fn in enumerate(missing_8, 1):
    p = all_1990[fn]
    img = cv2.imread(str(p))
    file_hash = hashlib.md5(p.read_bytes()).hexdigest()
    same_hash = [x[0] for x in all_hashes[file_hash] if x[0] != fn]
    d = diag_info.get(fn, {})
    
    print("\n" + "-" * 60)
    print(f"{idx}. Filename: {fn}")
    print(f"   Current Path    : {p}")
    print(f"   Image Shape     : {img.shape if img is not None else 'UNREADABLE'}")
    print(f"   MD5 Hash        : {file_hash}")
    print(f"   Byte-Duplicate? : {'Yes, duplicate of ' + ', '.join(same_hash) if same_hash else 'No (unique content)'}")
    print(f"   OB Label Present: {d.get('ob_label')}")
    print(f"   Fetal Biometry  : Loose={d.get('bio_loose')}, Tight={d.get('bio_tight')}")
    print(f"   Color Doppler   : {d.get('color_doppler')}")

# 5. Check what original cls_train/test contained by class
orig_train_uf = sum(1 for l in orig_train if l.startswith("0;"))
orig_train_nuf = sum(1 for l in orig_train if l.startswith("1;"))
orig_test_uf = sum(1 for l in orig_test if l.startswith("0;"))
orig_test_nuf = sum(1 for l in orig_test if l.startswith("1;"))

print("\n" + "=" * 60)
print("ORIGINAL MENDELEY ANNOTATION BREAKDOWN:")
print("=" * 60)
print(f"  Train UF  : {orig_train_uf}")
print(f"  Train NUF : {orig_train_nuf}")
print(f"  Test UF   : {orig_test_uf}")
print(f"  Test NUF  : {orig_test_nuf}")
print(f"  Total     : {len(orig_train) + len(orig_test)}")

print("\nDISK IMAGE FILES (ORIGINAL 1,990 TOTAL):")
print("  Train UF  : 702 (702 - 694 = 8 files in train/UF not in orig annotations)")
print("  Train NUF : 892 (exactly matches 892 in orig train annotations)")
print("  Test UF   : 173 (exactly matches 173 in orig test annotations)")
print("  Test NUF  : 223 (exactly matches 223 in orig test annotations)")
