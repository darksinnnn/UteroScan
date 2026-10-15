import cv2
import numpy as np
import pandas as pd
from pathlib import Path

proj = Path("d:/Project/UteroScan")
csv_path = proj / "experiments/v4_controlled/reports/contact_sheet_source_audit.csv"
df = pd.read_csv(csv_path)

out_dir = proj / "experiments/v4_controlled/reports/contact_sheet_crops"
out_dir.mkdir(parents=True, exist_ok=True)

print("INSPECTING INDIVIDUAL CONTACT SHEET IMAGES:")
for idx, r in df.iterrows():
    img_path = proj / r["relative_path"]
    img = cv2.imread(str(img_path))
    h, w, c = img.shape
    
    # Save a copy for visual artifact inspection
    stem = r["filename"].replace(".jpg", "")
    out_file = out_dir / f"sample_{idx}_{r['class_label']}_{r['study_proxy_group']}.png"
    cv2.imwrite(str(out_file), img)
    
    # Check text overlays via OCR or simple high-brightness connected components in margins
    # Check if there is color inside the ultrasound field
    # Specifically, check for Doppler color scale bar on the side
    # Doppler color scale bar is typically a narrow vertical bar with blue-red transition
    print(f"\n--- Sample {idx}: {r['filename']} ({r['class_label']}, Fold {r['fold']}, Group {r['study_proxy_group']}) ---")
    print(f"Path: {r['relative_path']}")
    
    # Check other files in same study_proxy_group
    auth_df = pd.read_csv(proj / "data/authoritative_manifest.csv")
    group_scans = auth_df[auth_df["filename"].str.contains(str(r['study_proxy_group']))]
    print(f"Total scans in this study group ({r['study_proxy_group']}) in authoritative manifest: {len(group_scans)}")
    print("Breakdown of study group in manifest:")
    print(group_scans[["filename", "split", "class_label", "exclusion_reason"]].to_string())
