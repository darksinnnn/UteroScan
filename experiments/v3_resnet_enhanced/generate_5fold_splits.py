import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = Path("d:/Project/UteroScan")
EXP_DIR = BASE_DIR / "experiments" / "v3_resnet_enhanced"
ANNOTATIONS_DIR = BASE_DIR / "experiments" / "v2_contamination_cleaned" / "annotations"

def extract_patient_id(filename: str) -> str:
    parts = Path(filename).name.split(".")
    if len(parts) >= 9:
        return parts[8]
    return Path(filename).stem

train_lines = [l.strip() for l in open(ANNOTATIONS_DIR / "cls_train_split.txt") if l.strip()]
val_lines = [l.strip() for l in open(ANNOTATIONS_DIR / "cls_val_split.txt") if l.strip()]

dev_lines = np.array(train_lines + val_lines)
y_dev = np.array([int(l.split(";")[0]) for l in dev_lines])
groups_dev = np.array([extract_patient_id(l.split(";")[1]) for l in dev_lines])

print(f"Total dev lines: {len(dev_lines)}, unique patients: {len(set(groups_dev))}")

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits_data = {}

for fold, (t_idx, v_idx) in enumerate(sgkf.split(dev_lines, y_dev, groups=groups_dev), 1):
    f_train = dev_lines[t_idx].tolist()
    f_val = dev_lines[v_idx].tolist()
    splits_data[f"fold_{fold}"] = {
        "train": f_train,
        "val": f_val
    }
    
    t_pts = {extract_patient_id(l.split(";")[1]) for l in f_train}
    v_pts = {extract_patient_id(l.split(";")[1]) for l in f_val}
    overlap = t_pts.intersection(v_pts)
    print(f"Fold {fold}: Train={len(f_train)}, Val={len(f_val)}, Overlap={len(overlap)}")

out_path = EXP_DIR / "5fold_splits.json"
with open(out_path, "w") as f:
    json.dump(splits_data, f, indent=2)

print(f"Saved 5-fold splits cleanly to: {out_path}")
