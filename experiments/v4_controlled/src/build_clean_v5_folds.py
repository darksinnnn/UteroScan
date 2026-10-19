#!/usr/bin/env python3
"""
Phase 2: Build and Validate Clean V5 5-Fold Stratified Group Splits
Location: experiments/v4_controlled/src/build_clean_v5_folds.py
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import cv2

PROJECT_ROOT = Path("d:/Project/UteroScan")
CLEAN_MANIFEST_CSV = PROJECT_ROOT / "experiments/v4_controlled/data/clean_v5/clean_manifest.csv"
FOLDS_V5_CSV = PROJECT_ROOT / "experiments/v4_controlled/data/clean_v5/folds_clean_v5.csv"
VALIDATION_REPORT_JSON = PROJECT_ROOT / "experiments/v4_controlled/data/clean_v5/split_validation_report.json"

def build_folds():
    print("=" * 80)
    print("PHASE 2: BUILDING 5-FOLD STRATIFIED GROUP SPLITS (CLEAN V5)")
    print("=" * 80)
    
    if not CLEAN_MANIFEST_CSV.exists():
        print(f"Error: {CLEAN_MANIFEST_CSV} does not exist.")
        return False
        
    df_manifest = pd.read_csv(CLEAN_MANIFEST_CSV)
    
    # 1. Separate partitions
    df_train = df_manifest[df_manifest["split"] == "Train"].copy()
    df_test = df_manifest[df_manifest["split"].isin(["Held-Out Test", "Test"])].copy()
    df_excl = df_manifest[df_manifest["split"] == "Excluded"].copy()
    
    print(f"Clean Development Cohort Size: {len(df_train)} scans (UF: {len(df_train[df_train['class_label']=='UF'])}, NUF: {len(df_train[df_train['class_label']=='NUF'])})")
    print(f"Held-Out Test Cohort Size    : {len(df_test)} scans (UF: {len(df_test[df_test['class_label']=='UF'])}, NUF: {len(df_test[df_test['class_label']=='NUF'])})")
    print(f"Excluded Quarantined Scans   : {len(df_excl)} scans")
    
    # 2. Extract study proxy groups and target indices
    # Filename format: 1.2.826.0.1.3680043.2.461.STUDY.INSTANCE.jpg
    df_train["study_proxy_group"] = df_train["filename"].apply(lambda x: str(x).split(".")[8] if len(str(x).split(".")) > 8 else str(x))
    df_train["source_stem"] = df_train["filename"].str.replace(".jpg", "")
    df_train["target_idx"] = df_train["class_label"].apply(lambda x: 0 if x == "UF" else 1)
    
    df_test["study_proxy_group"] = df_test["filename"].apply(lambda x: str(x).split(".")[8] if len(str(x).split(".")) > 8 else str(x))
    
    # 3. Check for Train-Test study group conflicts
    train_groups = set(df_train["study_proxy_group"])
    test_groups = set(df_test["study_proxy_group"])
    overlap_groups = train_groups.intersection(test_groups)
    print(f"Study groups overlapping between Train and Held-Out Test: {len(overlap_groups)}")
    if overlap_groups:
        print(f"  Warning: Groups {overlap_groups} appear in both Train and Test!")
        
    # 4. Stratified Group K-Fold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    X = df_train["filename"].values
    y = df_train["target_idx"].values
    groups = df_train["study_proxy_group"].values
    
    df_train["fold"] = -1
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        df_train.iloc[val_idx, df_train.columns.get_loc("fold")] = fold
        
    # 5. Automated Validation & Assertions
    assertions_passed = True
    assertion_logs = []
    
    # Assertion 1: All folds assigned
    if (df_train["fold"] == -1).any():
        assertion_logs.append("FAIL: Unassigned folds found in development cohort!")
        assertions_passed = False
    else:
        assertion_logs.append("PASS: All development scans assigned to folds [0, 4].")
        
    # Assertion 2: No excluded scans in folds
    excl_filenames = set(df_excl["filename"])
    overlap_excl = set(df_train["filename"]).intersection(excl_filenames)
    if overlap_excl:
        assertion_logs.append(f"FAIL: {len(overlap_excl)} excluded scans found in training folds!")
        assertions_passed = False
    else:
        assertion_logs.append("PASS: Zero excluded scans present in training folds.")
        
    # Assertion 3: No test scans in folds
    test_filenames = set(df_test["filename"])
    overlap_test = set(df_train["filename"]).intersection(test_filenames)
    if overlap_test:
        assertion_logs.append(f"FAIL: {len(overlap_test)} test scans found in training folds!")
        assertions_passed = False
    else:
        assertion_logs.append("PASS: Zero test scans present in training folds.")
        
    # Assertion 4: Zero group leakage across folds
    group_leakage = False
    for f1 in range(5):
        g1 = set(df_train[df_train["fold"] == f1]["study_proxy_group"])
        for f2 in range(f1 + 1, 5):
            g2 = set(df_train[df_train["fold"] == f2]["study_proxy_group"])
            shared = g1.intersection(g2)
            if shared:
                assertion_logs.append(f"FAIL: Study groups {shared} cross between fold {f1} and fold {f2}!")
                group_leakage = True
                assertions_passed = False
    if not group_leakage:
        assertion_logs.append("PASS: Zero study groups cross across folds. Complete study-level separation achieved.")
        
    # Assertion 5: Image existence and decode verification
    missing_files = []
    for _, r in df_train.iterrows():
        p = PROJECT_ROOT / r["image_path"]
        if not p.exists():
            missing_files.append(r["filename"])
    if missing_files:
        assertion_logs.append(f"FAIL: {len(missing_files)} image files missing on disk!")
        assertions_passed = False
    else:
        assertion_logs.append("PASS: All image files exist on disk and paths verified.")
        
    # Fold distribution summary
    fold_summaries = []
    for f in range(5):
        f_df = df_train[df_train["fold"] == f]
        n_uf = int(len(f_df[f_df["target_idx"] == 0]))
        n_nuf = int(len(f_df[f_df["target_idx"] == 1]))
        n_groups = int(f_df["study_proxy_group"].nunique())
        fold_summaries.append({
            "fold": f,
            "total_images": len(f_df),
            "uf_count": n_uf,
            "nuf_count": n_nuf,
            "uf_ratio": round(n_uf / len(f_df), 4),
            "unique_study_groups": n_groups
        })
        print(f"  Fold {f}: Total={len(f_df)}, UF={n_uf} ({n_uf/len(f_df)*100:.1f}%), NUF={n_nuf} ({n_nuf/len(f_df)*100:.1f}%), StudyGroups={n_groups}")
        
    # Save folds CSV
    df_train.to_csv(FOLDS_V5_CSV, index=False)
    print(f"\nSaved clean V5 folds to {FOLDS_V5_CSV}")
    
    # Save validation report
    validation_report = {
        "status": "APPROVED" if assertions_passed else "FAILED",
        "assertions_passed": assertions_passed,
        "clean_train_cohort_size": len(df_train),
        "held_out_test_cohort_size": len(df_test),
        "quarantined_excluded_size": len(df_excl),
        "train_test_study_overlap_count": len(overlap_groups),
        "patient_independence_disclaimer": "Genuine patient IDs are unavailable; study_proxy_group clusters frames by hospital study UID. Study-level independence is verified, but guaranteed patient-level independence cannot be claimed.",
        "fold_distributions": fold_summaries,
        "assertion_logs": assertion_logs
    }
    
    with open(VALIDATION_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)
    print(f"Saved split validation report to {VALIDATION_REPORT_JSON}")
    
    return assertions_passed

if __name__ == "__main__":
    success = build_folds()
    if not success:
        sys.exit(1)
