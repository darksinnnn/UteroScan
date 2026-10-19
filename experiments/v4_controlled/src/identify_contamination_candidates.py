import os
import sys
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import easyocr

PROJECT_ROOT = Path("d:/Project/UteroScan")
FOLDS_CSV = PROJECT_ROOT / "experiments/v4_controlled/data/folds.csv"
AUTH_MANIFEST_CSV = PROJECT_ROOT / "data/authoritative_manifest.csv"
EXCLUSION_LOG_CSV = PROJECT_ROOT / "data/excluded/exclusion_log.csv"
DOPPLER_CSV = PROJECT_ROOT / "experiments/v4_controlled/reports/training_set_doppler_leakage.csv"

CLEAN_V5_DIR = PROJECT_ROOT / "experiments/v4_controlled/data/clean_v5"
CLEAN_V5_DIR.mkdir(parents=True, exist_ok=True)

def find_all_candidates():
    print("=" * 80)
    print("STEP 1: IDENTIFYING ALL CONTAMINATION CANDIDATES")
    print("=" * 80)
    
    df_folds = pd.read_csv(FOLDS_CSV)
    df_doppler = pd.read_csv(DOPPLER_CSV)
    
    # Candidate set 1: 67 Doppler flagged records
    doppler_stems = set(df_doppler["filename"].str.replace(".jpg", ""))
    print(f"Candidate set 1 (Doppler flagged): {len(doppler_stems)} records")
    
    # Candidate set 2: Obstetric scans in training set
    # Let's inspect the headers of all 1,224 images for obstetric keywords
    reader = easyocr.Reader(['en', 'ch_sim'], gpu=True)
    
    ob_keywords_ch = ["中晚孕", "早孕", "胎儿", "羊水", "孕龄", "双顶径", "头臀长", "顶臀长", "股骨长", "腹围"]
    ob_keywords_en = ["OB1", "OB2", "OB3", "OB/GYN", "OB-GYN", "GA=", "GA ", "EDD="]
    
    candidates = {} # stem -> dict of info
    
    # Add Doppler candidates
    for _, r in df_doppler.iterrows():
        stem = r["filename"].replace(".jpg", "")
        candidates[stem] = {
            "stem": stem,
            "filename": r["filename"],
            "class_label": r["class_label"],
            "fold": r["fold"],
            "study_proxy_group": str(r["study_proxy_group"]),
            "flagged_source": "Doppler_Sweep",
            "doppler_left_max": r["left_bar_max"],
            "doppler_right_max": r["right_bar_max"],
            "sector_color_px": r["sector_color_px"],
            "has_colorbar": r["has_colorbar"],
            "has_sector_flow": r["has_sector_flow"]
        }
        
    print("Scanning headers of all 1,224 images for obstetric keywords...")
    for idx, r in df_folds.iterrows():
        stem = str(r["source_stem"])
        img_path = PROJECT_ROOT / r["image_path"]
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        header = img[:120, :]
        ocr_res = reader.readtext(header)
        header_text = " ".join([item[1] for item in ocr_res])
        
        reasons_found = []
        for kw in ob_keywords_ch:
            if kw in header_text:
                reasons_found.append(f"OB_CH({kw})")
        for kw in ob_keywords_en:
            if kw in header_text:
                reasons_found.append(f"OB_EN({kw})")
                
        if reasons_found:
            if stem in candidates:
                candidates[stem]["ob_flags"] = "; ".join(reasons_found)
                candidates[stem]["header_text"] = header_text
            else:
                candidates[stem] = {
                    "stem": stem,
                    "filename": r["filename"],
                    "class_label": r["class_label"],
                    "fold": r["fold"],
                    "study_proxy_group": str(r["study_proxy_group"]),
                    "flagged_source": "Obstetric_Sweep",
                    "doppler_left_max": 0,
                    "doppler_right_max": 0,
                    "sector_color_px": 0,
                    "has_colorbar": False,
                    "has_sector_flow": False,
                    "ob_flags": "; ".join(reasons_found),
                    "header_text": header_text
                }
                
    print(f"Total unique candidates identified: {len(candidates)}")
    return candidates

if __name__ == "__main__":
    candidates = find_all_candidates()
    # Save intermediate candidates
    import pickle
    with open(CLEAN_V5_DIR / "candidates_raw.pkl", "wb") as f:
        pickle.dump(candidates, f)
    print("Saved candidates_raw.pkl")
