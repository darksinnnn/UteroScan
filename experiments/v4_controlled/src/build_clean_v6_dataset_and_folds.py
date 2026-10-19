#!/usr/bin/env python3
"""
Clean V6 Dataset & Folds Rebuild: Full Obstetric Quarantining & Grouped 5-Fold Splits
Location: experiments/v4_controlled/src/build_clean_v6_dataset_and_folds.py

1. Reads clean_v5_obstetric_audit_full.csv to quarantine all confirmed obstetric scans
   and their associated study groups.
2. Updates clean manifest without modifying or deleting source image files on disk.
3. Generates verified 5-fold Stratified Group Splits (folds_clean_v6.csv) with zero leakage.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path("d:/Project/UteroScan")
CLEAN_V5_MANIFEST = PROJECT_ROOT / "experiments/v4_controlled/data/clean_v5/clean_manifest.csv"
AUDIT_CSV = PROJECT_ROOT / "experiments/v4_controlled/e4/reports/clean_v5_obstetric_audit_full.csv"

CLEAN_V6_DIR = PROJECT_ROOT / "experiments/v4_controlled/data/clean_v6"
CLEAN_V6_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_V6_MANIFEST = CLEAN_V6_DIR / "clean_manifest_v6.csv"
QUARANTINE_LOG_V6 = CLEAN_V6_DIR / "obstetric_quarantine_log_v6.csv"
FOLDS_V6_CSV = CLEAN_V6_DIR / "folds_clean_v6.csv"
VALIDATION_REPORT_V6 = CLEAN_V6_DIR / "split_validation_report_v6.json"

def rebuild_clean_v6():
    print("=" * 80)
    print("BUILDING CLEAN V6 DATASET & STRATIFIED GROUPED FOLDS")
    print("=" * 80)

    if not AUDIT_CSV.exists():
        print(f"Error: {AUDIT_CSV} does not exist. Run audit first.")
        return False

    df_audit = pd.read_csv(AUDIT_CSV)
    df_manifest_v5 = pd.read_csv(CLEAN_V5_MANIFEST)

    # 1. Identify all obstetric filenames and study groups
    flagged_df = df_audit[df_audit["study_group_flagged"] == True]
    ob_study_groups = set(flagged_df["study_proxy_group"].astype(str))
    ob_filenames = set(flagged_df["filename"])

    print(f"Directly/Group-flagged obstetric scans from audit: {len(flagged_df)}")
    print(f"Obstetric study groups to quarantine: {len(ob_study_groups)}")

    # 2. Build quarantine log
    flagged_df.to_csv(QUARANTINE_LOG_V6, index=False)
    print(f"Saved quarantine log to {QUARANTINE_LOG_V6.name}")

    # 3. Build Clean V6 Manifest
    manifest_rows = []
    quarantined_count = 0

    for idx, r in df_manifest_v5.iterrows():
        fn = r["filename"]
        split = r["split"]
        excl_reason = r["exclusion_reason"]
        grp = str(r["filename"]).split(".")[8] if len(str(r["filename"]).split(".")) > 8 else str(r["filename"])

        if fn in ob_filenames or grp in ob_study_groups:
            if split != "Excluded":
                quarantined_count += 1
            new_split = "Excluded"
            # Get specific reason if available
            match = df_audit[df_audit["filename"] == fn]
            if len(match) > 0 and match["reasons"].values[0] != "None":
                new_reason = f"V6_CLEANUP_OBSTETRIC: {match['reasons'].values[0]}"
            else:
                new_reason = f"V6_CLEANUP_OBSTETRIC: Group {grp} contains confirmed obstetric scans"
        else:
            new_split = split
            new_reason = excl_reason

        row_dict = r.to_dict()
        row_dict["split"] = new_split
        row_dict["exclusion_reason"] = new_reason
        manifest_rows.append(row_dict)

    df_manifest_v6 = pd.DataFrame(manifest_rows)
    df_manifest_v6.to_csv(CLEAN_V6_MANIFEST, index=False)
    print(f"Saved Clean V6 Manifest to {CLEAN_V6_MANIFEST.name}")
    print(f"Newly quarantined scans from development/test cohort: {quarantined_count}")

    # 4. Generate 5-Fold Stratified Group Splits
    df_train = df_manifest_v6[df_manifest_v6["split"] == "Train"].copy()
    df_test = df_manifest_v6[df_manifest_v6["split"].isin(["Held-Out Test", "Test"])].copy()
    df_excl = df_manifest_v6[df_manifest_v6["split"] == "Excluded"].copy()

    print(f"\nClean V6 Development Cohort Size: {len(df_train)} scans (UF: {len(df_train[df_train['class_label']=='UF'])}, NUF: {len(df_train[df_train['class_label']=='NUF'])})")
    print(f"Clean V6 Held-Out Test Cohort Size: {len(df_test)} scans (UF: {len(df_test[df_test['class_label']=='UF'])}, NUF: {len(df_test[df_test['class_label']=='NUF'])})")
    print(f"Clean V6 Excluded Cohort Size     : {len(df_excl)} scans")

    df_train["study_proxy_group"] = df_train["filename"].apply(lambda x: str(x).split(".")[8] if len(str(x).split(".")) > 8 else str(x))
    df_train["source_stem"] = df_train["filename"].str.replace(".jpg", "")
    df_train["target_idx"] = df_train["class_label"].apply(lambda x: 0 if x == "UF" else 1)

    df_test["study_proxy_group"] = df_test["filename"].apply(lambda x: str(x).split(".")[8] if len(str(x).split(".")) > 8 else str(x))

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    X = df_train["filename"].values
    y = df_train["target_idx"].values
    groups = df_train["study_proxy_group"].values

    df_train["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        df_train.iloc[val_idx, df_train.columns.get_loc("fold")] = fold

    # 5. Validation Assertions
    assertions_passed = True
    assertion_logs = []

    # All assigned
    if (df_train["fold"] == -1).any():
        assertion_logs.append("FAIL: Unassigned folds found in development cohort!")
        assertions_passed = False
    else:
        assertion_logs.append("PASS: All development scans assigned to folds [0, 4].")

    # Zero excluded scans in folds
    excl_filenames = set(df_excl["filename"])
    overlap_excl = set(df_train["filename"]).intersection(excl_filenames)
    if overlap_excl:
        assertion_logs.append(f"FAIL: {len(overlap_excl)} excluded scans found in training folds!")
        assertions_passed = False
    else:
        assertion_logs.append("PASS: Zero excluded scans present in training folds.")

    # Zero test scans in folds
    test_filenames = set(df_test["filename"])
    overlap_test = set(df_train["filename"]).intersection(test_filenames)
    if overlap_test:
        assertion_logs.append(f"FAIL: {len(overlap_test)} test scans found in training folds!")
        assertions_passed = False
    else:
        assertion_logs.append("PASS: Zero test scans present in training folds.")

    # Zero group leakage across folds
    group_leakage = False
    for f1 in range(5):
        g1 = set(df_train[df_train["fold"] == f1]["study_proxy_group"])
        for f2 in range(f1 + 1, 5):
            g2 = set(df_train[df_train["fold"] == f2]["study_proxy_group"])
            inter = g1.intersection(g2)
            if inter:
                assertion_logs.append(f"FAIL: Group leakage between Fold {f1} and Fold {f2}: {inter}")
                group_leakage = True
                assertions_passed = False
    if not group_leakage:
        assertion_logs.append("PASS: Strict zero study group leakage across all 5 folds.")

    df_train.to_csv(FOLDS_V6_CSV, index=False)
    print(f"Saved {FOLDS_V6_CSV.name}")

    report = {
        "status": "VALIDATED" if assertions_passed else "FAILED",
        "clean_v6_summary": {
            "total_development_scans": len(df_train),
            "uf_count": int(len(df_train[df_train["class_label"] == "UF"])),
            "nuf_count": int(len(df_train[df_train["class_label"] == "NUF"])),
            "distinct_study_groups": int(df_train["study_proxy_group"].nunique()),
            "held_out_test_scans": len(df_test),
            "total_quarantined_scans": len(df_excl),
            "obstetric_scans_quarantined_in_v6": int(len(flagged_df))
        },
        "per_fold_distribution": {
            f"fold_{f}": {
                "total": int(len(df_train[df_train["fold"] == f])),
                "uf": int(len(df_train[(df_train["fold"] == f) & (df_train["class_label"] == "UF")])),
                "nuf": int(len(df_train[(df_train["fold"] == f) & (df_train["class_label"] == "NUF")])),
                "study_groups": int(df_train[df_train["fold"] == f]["study_proxy_group"].nunique())
            } for f in range(5)
        },
        "assertions": assertion_logs
    }

    with open(VALIDATION_REPORT_V6, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved validation report to {VALIDATION_REPORT_V6.name}")
    print(json.dumps(report["clean_v6_summary"], indent=2))
    return assertions_passed

if __name__ == "__main__":
    rebuild_clean_v6()
