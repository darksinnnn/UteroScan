import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import easyocr

PROJECT_ROOT = Path("d:/Project/UteroScan")
FOLDS_CSV = PROJECT_ROOT / "experiments/v4_controlled/data/folds.csv"
DOPPLER_CSV = PROJECT_ROOT / "experiments/v4_controlled/reports/training_set_doppler_leakage.csv"
CLEAN_V5_DIR = PROJECT_ROOT / "experiments/v4_controlled/data/clean_v5"
CLEAN_V5_DIR.mkdir(parents=True, exist_ok=True)

def run_fast_scan():
    print("=" * 80)
    print("RUNNING FAST COMPREHENSIVE TEXT & OVERLAY SCAN ACROSS 1,224 IMAGES")
    print("=" * 80)
    
    t0 = time.time()
    df_folds = pd.read_csv(FOLDS_CSV)
    df_doppler = pd.read_csv(DOPPLER_CSV)
    
    reader = easyocr.Reader(['en', 'ch_sim'], gpu=True)
    
    ob_keywords_ch = ["中晚孕", "早孕", "胎儿", "羊水", "孕龄", "双顶径", "头臀长", "顶臀长", "股骨长", "腹围"]
    ob_keywords_en = ["OB1", "OB2", "OB3", "OB/GYN", "OB-GYN", "GA=", "GA ", "EDD="]
    dop_keywords = ["彩色血流", "彩鱼血", "收缩速度", "舒张末期", "阻力指数", "搏动指数", "血流"]
    
    records = []
    
    # Pre-index the 67 Doppler candidates
    doppler_dict = {r["filename"]: r for _, r in df_doppler.iterrows()}
    
    print(f"Loaded {len(df_folds)} fold records and {len(df_doppler)} Doppler flagged records.")
    
    for idx, r in df_folds.iterrows():
        fn = r["filename"]
        img_path = PROJECT_ROOT / r["image_path"]
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w, c = img.shape
        # Top 80 pixels is the PACS header
        header_crop = img[:80, :]
        
        # Read text
        ocr_res = reader.readtext(header_crop)
        header_text = " // ".join([item[1] for item in ocr_res])
        
        # Check obstetric terms
        ob_found = []
        for kw in ob_keywords_ch:
            if kw in header_text:
                ob_found.append(f"OB_CH({kw})")
        for kw in ob_keywords_en:
            if kw in header_text:
                ob_found.append(f"OB_EN({kw})")
                
        # Check Doppler terms
        dop_text_found = []
        for kw in dop_keywords:
            if kw in header_text:
                dop_text_found.append(f"DOP_TEXT({kw})")
                
        # Check Doppler physical features
        is_in_doppler_csv = fn in doppler_dict
        dop_info = doppler_dict.get(fn, None)
        has_colorbar = dop_info["has_colorbar"] if dop_info is not None else False
        has_sector_flow = dop_info["has_sector_flow"] if dop_info is not None else False
        sector_color_px = dop_info["sector_color_px"] if dop_info is not None else 0
        left_bar_max = dop_info["left_bar_max"] if dop_info is not None else 0
        right_bar_max = dop_info["right_bar_max"] if dop_info is not None else 0
        
        # If any flag is triggered
        if ob_found or dop_text_found or is_in_doppler_csv:
            records.append({
                "filename": fn,
                "relative_path": r["image_path"],
                "class_label": r["class_label"],
                "target_idx": r["target_idx"],
                "fold": r["fold"],
                "study_proxy_group": str(r["study_proxy_group"]),
                "header_text": header_text,
                "ob_flags": "; ".join(ob_found) if ob_found else "None",
                "dop_text_flags": "; ".join(dop_text_found) if dop_text_found else "None",
                "is_in_doppler_csv": is_in_doppler_csv,
                "has_colorbar": has_colorbar,
                "has_sector_flow": has_sector_flow,
                "sector_color_px": sector_color_px,
                "left_bar_max": left_bar_max,
                "right_bar_max": right_bar_max
            })
            
        if (idx + 1) % 200 == 0 or (idx + 1) == len(df_folds):
            dur = time.time() - t0
            print(f"Processed {idx + 1}/{len(df_folds)} images in {dur:.1f}s ({len(records)} candidates flagged so far)...", flush=True)
            
    df_out = pd.DataFrame(records)
    out_csv = CLEAN_V5_DIR / "flagged_candidates_full.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"SUCCESS: Saved {len(df_out)} flagged candidates to {out_csv} in {time.time() - t0:.1f}s")
    return df_out

if __name__ == "__main__":
    run_fast_scan()
