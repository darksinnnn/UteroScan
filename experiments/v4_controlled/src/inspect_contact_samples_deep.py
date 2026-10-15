import cv2
import numpy as np
import pandas as pd
from pathlib import Path

proj = Path("d:/Project/UteroScan")
csv_path = proj / "experiments/v4_controlled/reports/contact_sheet_source_audit.csv"
df = pd.read_csv(csv_path)

out_txt = proj / "experiments/v4_controlled/reports/contact_samples_deep_inspection.txt"

auth_df = pd.read_csv(proj / "data/authoritative_manifest.csv")
excl_df = pd.read_csv(proj / "data/excluded/exclusion_log.csv")

with open(out_txt, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("DEEP CLINICAL INSPECTION OF CONTACT SHEET SAMPLES AND STUDY GROUPS\n")
    f.write("=" * 80 + "\n\n")

    for idx, r in df.iterrows():
        stem = r["filename"].replace(".jpg", "")
        f.write(f"### Contact Sheet Sample {idx}: {r['filename']} ({r['class_label']}, Fold {r['fold']})\n")
        f.write(f"- Disk Path: {r['relative_path']}\n")
        f.write(f"- Study Group ID: {r['study_proxy_group']}\n")
        
        # Manifest entry
        m = auth_df[auth_df["filename"] == r["filename"]]
        f.write(f"- Split in Manifest: {m['split'].values[0] if len(m) > 0 else 'NOT FOUND'}\n")
        f.write(f"- Source Dataset: {m['source_dataset'].values[0] if len(m) > 0 else 'NOT FOUND'}\n")
        f.write(f"- In Exclusion Log: {r['in_exclusion_log']}\n")
        
        # Check all scans with same study ID in authoritative manifest
        group_id = str(r['study_proxy_group'])
        matching_auth = auth_df[auth_df["filename"].str.contains(group_id)]
        f.write(f"\nAll Scans sharing Study Group `{group_id}` ({len(matching_auth)} total in manifest):\n")
        for _, g_row in matching_auth.iterrows():
            f.write(f"  * {g_row['filename']} | Split: {g_row['split']} | Class: {g_row['class_label']} | Exclusion Reason: {g_row['exclusion_reason']}\n")
            
        f.write("\n" + "-" * 70 + "\n\n")

print(f"Inspection written to {out_txt}")
