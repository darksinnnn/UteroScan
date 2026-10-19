#!/usr/bin/env python3
"""
Fast Comprehensive Obstetric & Doppler Contamination Audit of Clean V5 Cohort
Location: experiments/v4_controlled/e4/src/audit_obstetric_contamination.py
"""

import os
import sys
import json
import time
import re
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import easyocr

PROJECT_ROOT = Path("d:/Project/UteroScan")
FOLDS_CSV = PROJECT_ROOT / "experiments/v4_controlled/data/clean_v5/folds_clean_v5.csv"
CLEAN_MANIFEST_CSV = PROJECT_ROOT / "experiments/v4_controlled/data/clean_v5/clean_manifest.csv"
AUTH_MANIFEST_CSV = PROJECT_ROOT / "data/authoritative_manifest.csv"
EXCLUSION_LOG_CSV = PROJECT_ROOT / "data/excluded/exclusion_log.csv"

E4_REPORTS_DIR = PROJECT_ROOT / "experiments/v4_controlled/e4/reports"
E4_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OB_CH_KEYWORDS = ["中晚孕", "早孕", "胎儿", "羊水", "胎心", "胎头", "双顶径", "头臀长", "顶臀长", "股骨长", "腹围", "孕囊", "孕周", "预产期"]
OB_EN_KEYWORDS = ["OB1", "OB2", "OB3", "OB/GYN", "OB-GYN", "081", "082", "083", "BPD", "CRL", "GA=", "EDD=", "EFW"]

def run_audit():
    print("=" * 80, flush=True)
    print("STARTING FAST COMPREHENSIVE OBSTETRIC CONTAMINATION AUDIT", flush=True)
    print("=" * 80, flush=True)

    df_folds = pd.read_csv(FOLDS_CSV)
    print(f"Total scans in folds_clean_v5.csv: {len(df_folds)}", flush=True)

    reader = easyocr.Reader(['en', 'ch_sim'], gpu=True)

    results = []
    t0 = time.time()

    for idx, row in df_folds.iterrows():
        fn = row["filename"]
        rel_path = row["image_path"]
        img_full_path = PROJECT_ROOT / rel_path

        if (idx + 1) % 100 == 0 or idx == len(df_folds) - 1:
            elapsed = time.time() - t0
            print(f"[{idx+1}/{len(df_folds)}] Audited ({elapsed:.1f}s)...", flush=True)

        if not img_full_path.exists():
            continue

        img = cv2.imread(str(img_full_path))
        if img is None:
            continue

        # Top 80 pixels contains all PACS machine text
        header = img[:80, :]
        ocr_res = reader.readtext(header)
        header_text = " ".join([item[1] for item in ocr_res])

        reasons = []
        is_tib = False
        is_tis = False

        if "TIB" in header_text:
            is_tib = True
            reasons.append("TIB(Thermal_Index_Bone_Obstetric)")
        if "TIS" in header_text:
            is_tis = True

        for kw in OB_CH_KEYWORDS:
            if kw in header_text:
                reasons.append(f"OB_CH({kw})")

        for kw in OB_EN_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', header_text, re.IGNORECASE) or kw in header_text.split():
                reasons.append(f"OB_EN({kw})")

        if ("081" in header_text or "082" in header_text or "083" in header_text) and not any(f"OB_EN({kw})" in reasons for kw in ["081", "082", "083"]):
            reasons.append("OB_EN(081/082_OCR_Preset)")

        is_obstetric = len(reasons) > 0

        results.append({
            "filename": fn,
            "image_path": rel_path,
            "class_label": row["class_label"],
            "fold": row["fold"],
            "study_proxy_group": str(row["study_proxy_group"]),
            "is_obstetric": is_obstetric,
            "is_tib": is_tib,
            "is_tis": is_tis,
            "reasons": "; ".join(reasons) if reasons else "None",
            "header_text": header_text
        })

    df_res = pd.DataFrame(results)

    # Propagate obstetric flag across study_proxy_group
    ob_groups = set(df_res[df_res["is_obstetric"]]["study_proxy_group"])
    print(f"\nDirectly flagged obstetric images: {df_res['is_obstetric'].sum()}", flush=True)
    print(f"Distinct study groups with obstetric markers: {len(ob_groups)}", flush=True)

    df_res["study_group_flagged"] = df_res["study_proxy_group"].isin(ob_groups)
    total_study_flagged = df_res["study_group_flagged"].sum()
    print(f"Total scans in obstetric study groups: {total_study_flagged}", flush=True)

    # Output audit CSV
    audit_csv = E4_REPORTS_DIR / "clean_v5_obstetric_audit_full.csv"
    df_res.to_csv(audit_csv, index=False)
    print(f"Saved full audit to {audit_csv}", flush=True)

    # Summary table
    print("\n" + "=" * 50, flush=True)
    print("OBSTETRIC CONTAMINATION AUDIT SUMMARY:", flush=True)
    print("=" * 50, flush=True)
    print(df_res.groupby(["class_label", "study_group_flagged"]).size(), flush=True)

    return df_res

if __name__ == "__main__":
    run_audit()
