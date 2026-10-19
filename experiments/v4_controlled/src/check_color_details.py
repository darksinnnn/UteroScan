import cv2
import numpy as np
import pandas as pd
from pathlib import Path

proj = Path("d:/Project/UteroScan")
csv_path = proj / "experiments/v4_controlled/reports/contact_sheet_source_audit.csv"
df = pd.read_csv(csv_path)

print("DETAILED COLOR BREAKDOWN FOR CONTACT SHEET SAMPLES:")
for idx, r in df.iterrows():
    img_path = proj / r["relative_path"]
    img = cv2.imread(str(img_path))
    h, w, _ = img.shape
    diff_rg = np.abs(img[:, :, 2].astype(int) - img[:, :, 1].astype(int))
    diff_rb = np.abs(img[:, :, 2].astype(int) - img[:, :, 0].astype(int))
    diff_gb = np.abs(img[:, :, 1].astype(int) - img[:, :, 0].astype(int))
    max_diff = np.maximum(diff_rg, np.maximum(diff_rb, diff_gb))
    
    # Top 80 pixels is PACS header
    ui_top = max_diff[:80, :] > 30
    sector_diff = max_diff[80:h-40, 80:w-80] > 30
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    sat_sector_high = sat[80:h-40, 80:w-80] > 80
    
    print(f"Idx {r['contact_sheet_index']} | {r['class_label']} | {r['filename']} | Top UI Color: {np.sum(ui_top)} | Sector Color: {np.sum(sector_diff)} | High Sat in Sector: {np.sum(sat_sector_high)}")
