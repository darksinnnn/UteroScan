#!/usr/bin/env python3
"""
Create Contact Sheet Source Audit CSV
Location: experiments/v4_controlled/e4/reports/e4_contact_sheet_source_audit.csv
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("d:/Project/UteroScan")
FOLDS_CSV = PROJECT_ROOT / "experiments/v4_controlled/data/clean_v5/folds_clean_v5.csv"
CLEAN_MANIFEST_CSV = PROJECT_ROOT / "experiments/v4_controlled/data/clean_v5/clean_manifest.csv"
AUTH_MANIFEST_CSV = PROJECT_ROOT / "data/authoritative_manifest.csv"
EXCLUSION_LOG_CSV = PROJECT_ROOT / "data/excluded/exclusion_log.csv"

df_folds = pd.read_csv(FOLDS_CSV)
df_clean = pd.read_csv(CLEAN_MANIFEST_CSV)
df_auth = pd.read_csv(AUTH_MANIFEST_CSV)
df_excl = pd.read_csv(EXCLUSION_LOG_CSV)

# The 4 images picked by validate_e4_components.py:
# Row 0: UF sample 0
# Row 1: UF sample 1
# Row 2: NUF sample 0
# Row 3: NUF sample 1

uf_samples = df_folds[df_folds["class_label"] == "UF"].head(2)
nuf_samples = df_folds[df_folds["class_label"] == "NUF"].head(2)
samples = pd.concat([uf_samples, nuf_samples]).reset_index(drop=True)

audit_rows = []

for idx, r in samples.iterrows():
    fn = r["filename"]
    rel_path = r["image_path"]
    cls_lbl = r["class_label"]
    fold_idx = r["fold"]
    grp = r["study_proxy_group"]
    
    in_folds = fn in df_folds["filename"].values
    in_clean = fn in df_clean["filename"].values
    clean_split = df_clean.loc[df_clean["filename"] == fn, "split"].values[0] if in_clean else "Not in clean manifest"
    in_excl = fn in df_excl["filename"].values
    excl_status = "Excluded in exclusion_log.csv" if in_excl else "Active (Not in exclusion_log.csv)"
    
    # Detailed findings from OCR audit
    if idx in [0, 1]:
        evidence = "Pelvic ultrasound (TIS soft-tissue thermal index, routine abdominal preset 腹部常规, caliper measurements of uterine fibroid). Genuine gynecological scan."
        disposition = "retained_verified_clean"
        contamination_type = "None"
    else:
        evidence = "Confirmed obstetric ultrasound contamination: Machine preset OB1 (recognized as '081' in header OCR), Thermal Index Bone TIB0.1 (fetal ossification acoustic preset on Philips EPIQ 7). Transabdominal fetal/obstetric anatomy."
        disposition = "confirmed_obstetric_contamination_quarantine"
        contamination_type = "Obstetric_Preset_OB1_TIB"

    dataloader_usage = f"Included in E4 training dataloaders for folds {[f for f in range(5) if f != fold_idx]} and outer validation for fold {fold_idx}"

    audit_rows.append({
        "contact_sheet_row": idx,
        "filename": fn,
        "source_path": rel_path,
        "class_label": cls_lbl,
        "split": clean_split,
        "assigned_fold": fold_idx,
        "study_proxy_group": grp,
        "in_folds_clean_v5": in_folds,
        "in_clean_manifest": in_clean,
        "in_exclusion_log": in_excl,
        "exclusion_status": excl_status,
        "e4_dataloader_status": dataloader_usage,
        "contamination_type": contamination_type,
        "evidence": evidence,
        "final_disposition": disposition
    })

df_out = pd.DataFrame(audit_rows)
out_csv = PROJECT_ROOT / "experiments/v4_controlled/e4/reports/e4_contact_sheet_source_audit.csv"
df_out.to_csv(out_csv, index=False)
print(f"Saved {out_csv.name} successfully.")
print(df_out[["contact_sheet_row", "class_label", "assigned_fold", "contamination_type", "final_disposition"]].to_string())
