import cv2
import numpy as np
import pandas as pd
import easyocr
from pathlib import Path

proj = Path("d:/Project/UteroScan")
folds_df = pd.read_csv(proj / "experiments/v4_controlled/data/folds.csv")
reader = easyocr.Reader(['en', 'ch_sim'], gpu=True)

out_txt = proj / "experiments/v4_controlled/reports/full_training_set_contamination_scan.txt"
out_csv = proj / "experiments/v4_controlled/reports/full_training_set_contamination_scan.csv"

findings = []

print(f"Scanning all {len(folds_df)} scans in folds.csv for Obstetric and Doppler markers...")

# Keywords to search in header/metadata
ob_keywords_ch = ["中晚孕", "早孕", "胎儿", "羊水", "孕龄", "双顶径", "头臀长", "顶臀长", "股骨长", "腹围"]
ob_keywords_en = ["OB1", "OB2", "OB3", "OB/GYN", "OB-GYN", "GA=", "GA ", "EDD"]
doppler_keywords_ch = ["彩色血流", "彩鱼血", "收缩速度", "舒张末期", "阻力指数", "搏动指数"]
doppler_keywords_en = ["cm/s", "cmls", "PRF", "WF", "TIS", "TIB"]

for idx, r in folds_df.iterrows():
    img_path = proj / r["image_path"]
    if not img_path.exists():
        continue
        
    img = cv2.imread(str(img_path))
    if img is None:
        continue
        
    # Crop header (top 120 px)
    header = img[:120, :]
    
    # Read text with easyocr
    ocr_res = reader.readtext(header)
    header_text = " ".join([item[1] for item in ocr_res])
    
    reasons = []
    
    # Check OB keywords
    for kw in ob_keywords_ch:
        if kw in header_text:
            reasons.append(f"OB_CHINESE({kw})")
    for kw in ob_keywords_en:
        if kw in header_text:
            reasons.append(f"OB_EN({kw})")
            
    # Check Doppler keywords
    for kw in doppler_keywords_ch:
        if kw in header_text:
            reasons.append(f"DOPPLER_CHINESE({kw})")
            
    # Check Color bar in image
    diff_rg = np.abs(img[:, :, 2].astype(int) - img[:, :, 1].astype(int))
    diff_rb = np.abs(img[:, :, 2].astype(int) - img[:, :, 0].astype(int))
    diff_gb = np.abs(img[:, :, 1].astype(int) - img[:, :, 0].astype(int))
    max_diff = np.maximum(diff_rg, np.maximum(diff_rb, diff_gb))
    
    # Check if vertical colorbar exists on left (x: 20..100) or right (x: 950..1010)
    left_colorbar = np.max(np.sum(max_diff[100:600, 20:100] > 50, axis=0)) > 150
    right_colorbar = np.max(np.sum(max_diff[100:600, 950:1020] > 50, axis=0)) > 150
    
    if left_colorbar or right_colorbar:
        reasons.append("COLOR_DOPPLER_SCALE_BAR")
        
    if reasons:
        findings.append({
            "filename": r["filename"],
            "class_label": r["class_label"],
            "fold": r["fold"],
            "study_proxy_group": r["study_proxy_group"],
            "header_text": header_text,
            "contamination_flags": "; ".join(reasons)
        })

df_contam = pd.DataFrame(findings)
df_contam.to_csv(out_csv, index=False)
print(f"Total contaminated scans found in 1,224 training set: {len(df_contam)}")
if len(df_contam) > 0:
    print(df_contam["contamination_flags"].value_counts())
