import cv2
import numpy as np
import pandas as pd
from pathlib import Path

proj = Path("d:/Project/UteroScan")
folds_df = pd.read_csv(proj / "experiments/v4_controlled/data/folds.csv")

doppler_list = []

print(f"Scanning all {len(folds_df)} scans for Color Doppler velocity bars and sector color maps...")

for idx, r in folds_df.iterrows():
    img_path = proj / r["image_path"]
    if not img_path.exists():
        continue
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    diff_rg = np.abs(img[:, :, 2].astype(int) - img[:, :, 1].astype(int))
    diff_rb = np.abs(img[:, :, 2].astype(int) - img[:, :, 0].astype(int))
    diff_gb = np.abs(img[:, :, 1].astype(int) - img[:, :, 0].astype(int))
    max_diff = np.maximum(diff_rg, np.maximum(diff_rb, diff_gb))
    
    # Left bar (x: 20..100) or Right bar (x: 950..1020)
    left_max = np.max(np.sum(max_diff[100:600, 20:100] > 50, axis=0)) if img.shape[1] > 100 else 0
    right_max = np.max(np.sum(max_diff[100:600, 950:1020] > 50, axis=0)) if img.shape[1] >= 1020 else 0
    
    # Sector color pixels (x: 150..850, y: 150..650)
    sector_color = np.sum(max_diff[150:650, 150:850] > 50)
    
    has_colorbar = (left_max > 120) or (right_max > 120)
    has_sector_flow = sector_color > 300
    
    if has_colorbar or has_sector_flow:
        doppler_list.append({
            "filename": r["filename"],
            "class_label": r["class_label"],
            "fold": r["fold"],
            "study_proxy_group": r["study_proxy_group"],
            "left_bar_max": int(left_max),
            "right_bar_max": int(right_max),
            "sector_color_px": int(sector_color),
            "has_colorbar": has_colorbar,
            "has_sector_flow": has_sector_flow
        })

df_d = pd.DataFrame(doppler_list)
out_csv = proj / "experiments/v4_controlled/reports/training_set_doppler_leakage.csv"
df_d.to_csv(out_csv, index=False)
print(f"Total scans with Color Doppler features in 1,224 training set: {len(df_d)}")
print(df_d.to_string())
