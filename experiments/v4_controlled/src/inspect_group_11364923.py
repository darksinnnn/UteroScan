import cv2
import numpy as np
import pandas as pd
from pathlib import Path

proj = Path("d:/Project/UteroScan")
crops_dir = proj / "experiments/v4_controlled/reports/contact_sheet_crops"

auth_df = pd.read_csv(proj / "data/authoritative_manifest.csv")
group_scans = auth_df[auth_df["filename"].str.contains("11364923")]

print("ANALYSIS OF STUDY GROUP 11364923:")
for idx, r in group_scans.iterrows():
    img_path = proj / r["image_path"]
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"File not found: {img_path}")
        continue
    
    # Save image to crops_dir for inspection
    out_file = crops_dir / f"group_11364923_{r['filename']}.png"
    cv2.imwrite(str(out_file), img)
    
    # Check top text area (y: 0..100) and bottom text area (y: 700..768)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check for text in top
    top_text_crop = gray[:120, :]
    cv2.imwrite(str(crops_dir / f"group_11364923_{r['filename']}_top.png"), top_text_crop)
    
    print(f"Saved {r['filename']} (Split: {r['split']}, Class: {r['class_label']}) to {out_file}")
