import cv2
import numpy as np
import pandas as pd
import easyocr
from pathlib import Path

proj = Path("d:/Project/UteroScan")
csv_path = proj / "experiments/v4_controlled/reports/contact_sheet_source_audit.csv"
df = pd.read_csv(csv_path)

reader = easyocr.Reader(['en', 'ch_sim'], gpu=True)

out_txt = proj / "experiments/v4_controlled/reports/contact_samples_ocr_audit.txt"

with open(out_txt, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("OCR TEXT RECOGNITION AUDIT FOR CONTACT SHEET SAMPLES\n")
    f.write("=" * 80 + "\n\n")

    for idx, r in df.iterrows():
        img_path = proj / r["relative_path"]
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape
        
        # Read header (top 150 px)
        header = img[:150, :]
        res_header = reader.readtext(header)
        text_header = [item[1] for item in res_header]
        
        # Read footer/body (below 150 px)
        body = img[150:, :]
        res_body = reader.readtext(body)
        text_body = [item[1] for item in res_body]
        
        f.write(f"Sample {idx}: {r['filename']} | Class: {r['class_label']} | Fold: {r['fold']} | Group: {r['study_proxy_group']}\n")
        f.write(f"Header Text: {' // '.join(text_header)}\n")
        f.write(f"Body/Sector Text: {' // '.join(text_body)}\n")
        f.write("-" * 60 + "\n\n")

print(f"OCR audit saved to {out_txt}")
