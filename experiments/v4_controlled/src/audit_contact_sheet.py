import os
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("d:/Project/UteroScan")
FOLDS_CSV = PROJECT_ROOT / "experiments" / "v4_controlled" / "data" / "folds.csv"
AUTH_MANIFEST_CSV = PROJECT_ROOT / "data" / "authoritative_manifest.csv"

folds_df = pd.read_csv(FOLDS_CSV)
auth_df = pd.read_csv(AUTH_MANIFEST_CSV)

num_pairs = 8
uf_samples = folds_df[folds_df["target_idx"] == 0].sample(num_pairs // 2, random_state=42)
nuf_samples = folds_df[folds_df["target_idx"] == 1].sample(num_pairs // 2, random_state=42)
sample_df = pd.concat([uf_samples, nuf_samples]).reset_index(drop=True)

print("=" * 80)
print("EXACT IMAGES SAMPLED IN CONTACT SHEET:")
print("=" * 80)
for idx, r in sample_df.iterrows():
    print(f"Index {idx} | Class: {r['class_label']} | Stem: {r['source_stem']}")
    print(f"  Path: {r['image_path']}")
    print(f"  Fold: {r.get('fold', 'N/A')} | Proxy Group: {r.get('study_proxy_group', 'N/A')}")
    # Cross reference with authoritative manifest
    auth_match = auth_df[auth_df["filename"] == f"{r['source_stem']}.jpg"]
    if len(auth_match) > 0:
        am = auth_match.iloc[0]
        print(f"  Auth Manifest Match -> Split: {am.get('split', 'N/A')} | Class: {am.get('class_label', 'N/A')} | Quarantined: {am.get('is_quarantined', 'N/A')} | Reason: {am.get('quarantine_reason', 'None')}")
    else:
        print("  Auth Manifest Match -> NOT FOUND!")
    print("-" * 60)
