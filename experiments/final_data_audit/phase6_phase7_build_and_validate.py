#!/usr/bin/env python3
"""
Phase 6 & 7: Safe Evaluation Protocol, 5-Fold Stratified Group Splits & Data Integrity Validation
Location: experiments/final_data_audit/phase6_phase7_build_and_validate.py

1. Reads master_review_manifest.csv and duplicate_families.csv.
2. Enforces strict evaluation isolation:
   - Any scan sharing a duplicate family or study group with the held-out test set is evaluation-only.
   - Zero leakage into development folds.
3. Generates 5-fold Stratified Group Cross-Validation splits (recovered_folds.csv).
4. Executes full suite of validation assertions.
5. Generates data_integrity_report.json and split_validation_report.json.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path("d:/Project/UteroScan")
AUDIT_DIR = PROJECT_ROOT / "experiments" / "final_data_audit"
MASTER_MANIFEST_CSV = AUDIT_DIR / "master_review_manifest.csv"
DUPLICATE_FAMILIES_CSV = AUDIT_DIR / "duplicate_families.csv"

RECOVERED_FOLDS_CSV = AUDIT_DIR / "recovered_folds.csv"
DATA_INTEGRITY_REPORT_JSON = AUDIT_DIR / "data_integrity_report.json"
SPLIT_VALIDATION_REPORT_JSON = AUDIT_DIR / "split_validation_report.json"

def run_build_and_validate():
    print("=" * 80)
    print("PHASE 6 & 7: EVALUATION ISOLATION, GROUPED FOLDS & INTEGRITY VALIDATION")
    print("=" * 80)
    t0 = time.time()

    if not MASTER_MANIFEST_CSV.exists() or not DUPLICATE_FAMILIES_CSV.exists():
        print("Error: Master review manifest or duplicate families file does not exist yet.")
        return False

    df_master = pd.read_csv(MASTER_MANIFEST_CSV)
    df_fam = pd.read_csv(DUPLICATE_FAMILIES_CSV)
    print(f"Loaded {len(df_master)} master review records.")

    # 1. Identify all study groups and duplicate families present in the held-out test set
    held_out_df = df_master[df_master["new_disposition"] == "HELD_OUT_EVALUATION_ONLY"]
    held_out_groups = set(held_out_df["study_proxy_group"].astype(str))
    held_out_families = set(held_out_df["family_id"].astype(str))

    print(f"Held-out evaluation scans: {len(held_out_df)}")
    print(f"Distinct held-out study groups: {len(held_out_groups)}")
    print(f"Distinct held-out duplicate families: {len(held_out_families)}")

    # 2. Select eligible development candidates:
    # Must have disposition ELIGIBLE_UF or ELIGIBLE_NUF
    eligible_df = df_master[df_master["new_disposition"].isin(["ELIGIBLE_UF", "ELIGIBLE_NUF"])].copy()
    print(f"Raw eligible development candidates: {len(eligible_df)}")

    # 3. Check for any study group overlap with held-out test
    overlap_mask = eligible_df["study_proxy_group"].astype(str).isin(held_out_groups)
    overlap_count = overlap_mask.sum()
    if overlap_count > 0:
        print(f"Resolving conflict: {overlap_count} development candidates belong to study groups overlapping with held-out test.")
        # Move overlapping development scans to HELD_OUT_EVALUATION_ONLY to guarantee zero group leakage
        for idx_row in eligible_df[overlap_mask].index:
            df_master.loc[idx_row, "new_disposition"] = "HELD_OUT_EVALUATION_ONLY"
            df_master.loc[idx_row, "review_evidence"] += "; Moved to evaluation pool to resolve study group conflict with held-out test."
        # Update eligible_df
        eligible_df = df_master[df_master["new_disposition"].isin(["ELIGIBLE_UF", "ELIGIBLE_NUF"])].copy()
        # Save updated master manifest
        df_master.to_csv(MASTER_MANIFEST_CSV, index=False)
        print(f"Updated master manifest. Final eligible development candidates: {len(eligible_df)}")

    # 4. Stratified Group 5-Fold Cross-Validation on Recovered Development Cohort
    eligible_df["target_idx"] = eligible_df["verified_class"].apply(lambda x: 0 if x == "UF" else 1)
    eligible_df["study_proxy_group"] = eligible_df["study_proxy_group"].astype(str)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    X = eligible_df["canonical_image_id"].values
    y = eligible_df["target_idx"].values
    groups = eligible_df["study_proxy_group"].values

    eligible_df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        eligible_df.iloc[val_idx, eligible_df.columns.get_loc("fold")] = fold

    # 5. Validation Assertions
    assertion_logs = []
    passed = True

    # Check unassigned folds
    if (eligible_df["fold"] == -1).any():
        assertion_logs.append("FAIL: Some development scans were not assigned to folds.")
        passed = False
    else:
        assertion_logs.append("PASS: 100% of development scans assigned to folds [0, 4].")

    # Check study group leakage across folds
    group_leakage = False
    for f1 in range(5):
        g1 = set(eligible_df[eligible_df["fold"] == f1]["study_proxy_group"])
        for f2 in range(f1 + 1, 5):
            g2 = set(eligible_df[eligible_df["fold"] == f2]["study_proxy_group"])
            inter = g1.intersection(g2)
            if inter:
                assertion_logs.append(f"FAIL: Study group leakage between Fold {f1} and Fold {f2}: {inter}")
                group_leakage = True
                passed = False
    if not group_leakage:
        assertion_logs.append("PASS: Strict zero study group leakage across all 5 development folds.")

    # Check duplicate family leakage across folds
    fam_leakage = False
    for f1 in range(5):
        fam1 = set(eligible_df[eligible_df["fold"] == f1]["family_id"])
        for f2 in range(f1 + 1, 5):
            fam2 = set(eligible_df[eligible_df["fold"] == f2]["family_id"])
            inter = fam1.intersection(fam2)
            if inter:
                assertion_logs.append(f"FAIL: Duplicate family leakage between Fold {f1} and Fold {f2}: {inter}")
                fam_leakage = True
                passed = False
    if not fam_leakage:
        assertion_logs.append("PASS: Strict zero duplicate family leakage across all 5 development folds.")

    # Check overlap with held-out test
    dev_groups = set(eligible_df["study_proxy_group"])
    dev_families = set(eligible_df["family_id"])
    test_inter_g = dev_groups.intersection(held_out_groups)
    test_inter_f = dev_families.intersection(held_out_families)

    if test_inter_g:
        assertion_logs.append(f"FAIL: Study group overlap between development and held-out test: {test_inter_g}")
        passed = False
    else:
        assertion_logs.append("PASS: Strict zero study group overlap between development and held-out evaluation cohort.")

    if test_inter_f:
        assertion_logs.append(f"FAIL: Duplicate family overlap between development and held-out test: {test_inter_f}")
        passed = False
    else:
        assertion_logs.append("PASS: Strict zero duplicate family overlap between development and held-out evaluation cohort.")

    # Save recovered folds CSV
    eligible_df.to_csv(RECOVERED_FOLDS_CSV, index=False)
    print(f"Saved recovered folds to: {RECOVERED_FOLDS_CSV.name}")

    # Build validation reports
    split_report = {
        "status": "VALIDATED" if passed else "FAILED",
        "development_cohort": {
            "total_scans": len(eligible_df),
            "uf_count": int(len(eligible_df[eligible_df["verified_class"] == "UF"])),
            "nuf_count": int(len(eligible_df[eligible_df["verified_class"] == "NUF"])),
            "distinct_study_groups": int(eligible_df["study_proxy_group"].nunique()),
            "distinct_duplicate_families": int(eligible_df["family_id"].nunique())
        },
        "per_fold_distribution": {
            f"fold_{f}": {
                "total": int(len(eligible_df[eligible_df["fold"] == f])),
                "uf": int(len(eligible_df[(eligible_df["fold"] == f) & (eligible_df["verified_class"] == "UF")])),
                "nuf": int(len(eligible_df[(eligible_df["fold"] == f) & (eligible_df["verified_class"] == "NUF")])),
                "study_groups": int(eligible_df[eligible_df["fold"] == f]["study_proxy_group"].nunique())
            } for f in range(5)
        },
        "held_out_evaluation_cohort": {
            "total_scans": int(len(df_master[df_master["new_disposition"] == "HELD_OUT_EVALUATION_ONLY"])),
            "uf_count": int(len(df_master[(df_master["new_disposition"] == "HELD_OUT_EVALUATION_ONLY") & (df_master["verified_class"] == "UF")])),
            "nuf_count": int(len(df_master[(df_master["new_disposition"] == "HELD_OUT_EVALUATION_ONLY") & (df_master["verified_class"] == "NUF")]))
        },
        "assertions": assertion_logs
    }

    with open(SPLIT_VALIDATION_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(split_report, f, indent=2)
    print(f"Saved split validation report to: {SPLIT_VALIDATION_REPORT_JSON.name}")

    # Build comprehensive Data Integrity Report
    excluded_df = df_master[df_master["new_disposition"].str.startswith("EXCLUDE_")]
    recovered_df = df_master[df_master["review_evidence"].str.contains("RECOVERED:", na=False)]

    integrity_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_unique_canonical_scans": len(df_master),
        "disposition_summary": df_master["new_disposition"].value_counts().to_dict(),
        "recovered_scans_summary": {
            "total_recovered": len(recovered_df),
            "recovered_uf": int(len(recovered_df[recovered_df["verified_class"] == "UF"])),
            "recovered_nuf": int(len(recovered_df[recovered_df["verified_class"] == "NUF"]))
        },
        "excluded_scans_summary": {
            "total_excluded": len(excluded_df),
            "fetus_visible_excluded": int(len(df_master[df_master["new_disposition"] == "EXCLUDE_FETUS_VISIBLE"])),
            "unrelated_anatomy_excluded": int(len(df_master[df_master["new_disposition"] == "EXCLUDE_UNRELATED_ANATOMY"])),
            "corrupt_or_unreadable_excluded": int(len(df_master[df_master["new_disposition"] == "EXCLUDE_CORRUPT_OR_UNREADABLE"]))
        },
        "calipers_and_doppler_census": {
            "calipers_present_count": int(df_master["calipers_present"].sum()),
            "color_doppler_present_count": int(df_master["color_doppler_present"].sum())
        },
        "development_folds_status": "READY_FOR_E0_RETRAINING" if passed else "INTEGRITY_CHECK_FAILED"
    }

    with open(DATA_INTEGRITY_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(integrity_report, f, indent=2)
    print(f"Saved data integrity report to: {DATA_INTEGRITY_REPORT_JSON.name}")
    print(json.dumps(integrity_report, indent=2))

    return passed

if __name__ == "__main__":
    run_build_and_validate()
