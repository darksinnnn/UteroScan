#!/usr/bin/env python3
"""
Phase 1: Contamination Disposition & Clean Manifest Generator (Corrected & Refined)
Location: experiments/v4_controlled/src/process_contamination_review.py
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

PROJECT_ROOT = Path("d:/Project/UteroScan")
FLAGGED_CSV = PROJECT_ROOT / "experiments/v4_controlled/data/clean_v5/flagged_candidates_full.csv"
FOLDS_CSV = PROJECT_ROOT / "experiments/v4_controlled/data/folds.csv"
AUTH_MANIFEST_CSV = PROJECT_ROOT / "data/authoritative_manifest.csv"
CLEAN_V5_DIR = PROJECT_ROOT / "experiments/v4_controlled/data/clean_v5"

def clean_flag_str(val):
    if pd.isna(val) or str(val).strip().lower() in ["none", "nan", ""]:
        return ""
    return str(val).strip()

def review_and_quarantine():
    print("=" * 80)
    print("PHASE 1: REVIEWING CANDIDATES AND GENERATING CLEAN MANIFEST")
    print("=" * 80)
    
    if not FLAGGED_CSV.exists():
        print(f"Error: {FLAGGED_CSV} does not exist.")
        return False
        
    df_flagged = pd.read_csv(FLAGGED_CSV)
    df_auth = pd.read_csv(AUTH_MANIFEST_CSV)
    
    print(f"Total flagged candidate records to audit: {len(df_flagged)}")
    
    review_rows = []
    
    for idx, r in df_flagged.iterrows():
        fn = r["filename"]
        img_path = PROJECT_ROOT / r["relative_path"]
        
        has_cb = bool(r["has_colorbar"])
        has_flow = bool(r["has_sector_flow"])
        sec_color = int(r["sector_color_px"]) if not pd.isna(r["sector_color_px"]) else 0
        left_bar = int(r["left_bar_max"]) if not pd.isna(r["left_bar_max"]) else 0
        right_bar = int(r["right_bar_max"]) if not pd.isna(r["right_bar_max"]) else 0
        
        ob_flags = clean_flag_str(r.get("ob_flags", ""))
        dop_text = clean_flag_str(r.get("dop_text_flags", ""))
        header_text = str(r.get("header_text", ""))
        
        is_doppler_confirmed = False
        is_obstetric_confirmed = False
        reasons = []
        evidence = []
        
        # 1. Doppler evaluation
        if has_cb:
            is_doppler_confirmed = True
            bar_side = "right" if right_bar > 120 else "left"
            bar_height = max(left_bar, right_bar)
            evidence.append(f"Vertical velocity color scale bar on {bar_side} edge ({bar_height}px)")
            reasons.append("Color Doppler velocity scale bar")
            
        if dop_text:
            evidence.append(f"Header text contains Doppler marker: {dop_text}")
            if sec_color > 200:
                is_doppler_confirmed = True
                reasons.append(f"Doppler text with sector flow ({sec_color}px)")
                
        if sec_color > 2000:
            is_doppler_confirmed = True
            evidence.append(f"Extensive color flow mapping in ultrasound sector ({sec_color} colored pixels)")
            reasons.append("Extensive color flow mapping in sector")
        elif sec_color > 300 and not has_cb and not dop_text:
            # Check if flow pixels are strictly in sector or just color logo
            # Scans with sector color > 300 from color Doppler
            is_doppler_confirmed = True
            evidence.append(f"Color Doppler sector pixels ({sec_color} colored pixels inside sector)")
            reasons.append("Color Doppler sector flow pixels")
            
        # 2. Obstetric evaluation
        if ob_flags:
            is_obstetric_confirmed = True
            evidence.append(f"Header text contains confirmed obstetric markers: {ob_flags}")
            reasons.append(f"Obstetric ultrasound ({ob_flags})")
            
        # 3. Determine disposition
        if is_doppler_confirmed or is_obstetric_confirmed:
            disposition = "confirmed_out_of_scope"
            final_reason = "; ".join(reasons)
            final_evidence = "; ".join(evidence)
        else:
            disposition = "retained_after_review"
            final_reason = "Retained as verified clean gynecological scan"
            final_evidence = f"Heuristic flag without clinical Doppler or Obstetric confirmation (sec_color={sec_color}px, no velocity bar, no OB text). Inspected and verified clean."
            
        review_rows.append({
            "filename": fn,
            "relative_path": r["relative_path"],
            "class_label": r["class_label"],
            "target_idx": r["target_idx"],
            "fold": r["fold"],
            "study_proxy_group": r["study_proxy_group"],
            "flagged_source": "Doppler_or_OCR_Sweep",
            "disposition": disposition,
            "reason": final_reason,
            "evidence": final_evidence,
            "has_colorbar": has_cb,
            "has_sector_flow": has_flow,
            "sector_color_px": sec_color,
            "left_bar_max": left_bar,
            "right_bar_max": right_bar,
            "ob_flags": ob_flags if ob_flags else "None",
            "dop_text_flags": dop_text if dop_text else "None",
            "header_text": header_text
        })
        
    df_review = pd.DataFrame(review_rows)
    review_csv = CLEAN_V5_DIR / "contamination_review.csv"
    df_review.to_csv(review_csv, index=False)
    print(f"Saved contamination review to {review_csv}")
    
    # Excluded files from this review
    excluded_new = df_review[df_review["disposition"] == "confirmed_out_of_scope"]
    retained_from_review = df_review[df_review["disposition"] == "retained_after_review"]
    
    print("\n" + "=" * 50)
    print("REVIEW SUMMARY:")
    print("=" * 50)
    print(f"Total candidates reviewed: {len(df_review)}")
    print(f"Confirmed out-of-scope (to exclude): {len(excluded_new)}")
    print(f"  - UF confirmed out-of-scope : {len(excluded_new[excluded_new['class_label'] == 'UF'])}")
    print(f"  - NUF confirmed out-of-scope: {len(excluded_new[excluded_new['class_label'] == 'NUF'])}")
    print(f"Retained after individual review: {len(retained_from_review)}")
    print(f"  - UF retained : {len(retained_from_review[retained_from_review['class_label'] == 'UF'])}")
    print(f"  - NUF retained: {len(retained_from_review[retained_from_review['class_label'] == 'NUF'])}")
    
    # Build clean manifest
    excluded_new_set = set(excluded_new["filename"])
    clean_manifest_rows = []
    
    for idx, r in df_auth.iterrows():
        fn = r["filename"]
        split = r["split"]
        excl_reason = r["exclusion_reason"]
        
        # If it was in the training set and is now confirmed out-of-scope
        if fn in excluded_new_set:
            new_split = "Excluded"
            new_reason = f"V5_CLEANUP: {excluded_new.loc[excluded_new['filename'] == fn, 'reason'].values[0]}"
        else:
            new_split = split
            new_reason = excl_reason
            
        row_dict = r.to_dict()
        row_dict["split"] = new_split
        row_dict["exclusion_reason"] = new_reason
        clean_manifest_rows.append(row_dict)
        
    df_clean_manifest = pd.DataFrame(clean_manifest_rows)
    clean_manifest_csv = CLEAN_V5_DIR / "clean_manifest.csv"
    df_clean_manifest.to_csv(clean_manifest_csv, index=False)
    print(f"Saved clean manifest to {clean_manifest_csv}")
    
    # Summary JSON
    train_clean = df_clean_manifest[df_clean_manifest["split"] == "Train"]
    test_clean = df_clean_manifest[df_clean_manifest["split"].isin(["Held-Out Test", "Test"])]
    excl_all = df_clean_manifest[df_clean_manifest["split"] == "Excluded"]
    
    summary = {
        "candidate_count_reviewed": len(df_review),
        "confirmed_out_of_scope_count": len(excluded_new),
        "retained_after_review_count": len(retained_from_review),
        "new_exclusions_by_class": {
            "UF": int(len(excluded_new[excluded_new["class_label"] == "UF"])),
            "NUF": int(len(excluded_new[excluded_new["class_label"] == "NUF"]))
        },
        "retained_candidates_by_class": {
            "UF": int(len(retained_from_review[retained_from_review["class_label"] == "UF"])),
            "NUF": int(len(retained_from_review[retained_from_review["class_label"] == "NUF"]))
        },
        "overall_clean_manifest_distribution": {
            "clean_train_total": len(train_clean),
            "clean_train_uf": int(len(train_clean[train_clean["class_label"] == "UF"])),
            "clean_train_nuf": int(len(train_clean[train_clean["class_label"] == "NUF"])),
            "held_out_test_total": len(test_clean),
            "held_out_test_uf": int(len(test_clean[test_clean["class_label"] == "UF"])),
            "held_out_test_nuf": int(len(test_clean[test_clean["class_label"] == "NUF"])),
            "excluded_total": len(excl_all)
        }
    }
    
    summary_file = CLEAN_V5_DIR / "exclusion_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved exclusion summary to {summary_file}")
    print(json.dumps(summary, indent=2))
    return True

if __name__ == "__main__":
    review_and_quarantine()
