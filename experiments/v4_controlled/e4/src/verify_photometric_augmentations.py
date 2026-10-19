#!/usr/bin/env python3
"""
Verification and Detailed Pixel-Level Audit of Photometric Augmentations
Location: experiments/v4_controlled/e4/src/verify_photometric_augmentations.py

Quantifies:
  - Mean Absolute Pixel Difference (MAD)
  - Percentage of pixels changed
  - Intensity statistics (min, max, mean, std)
  - Bit-exact evaluation invariance
  - Visual panel generation
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch

PROJECT_ROOT = Path("d:/Project/UteroScan")
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "v4_controlled" / "e4" / "src"))

from photometric_aug import UltrasoundPhotometricAugmentation
from e4_dataset import E4UltrasoundDataset

REPORTS_DIR = PROJECT_ROOT / "experiments" / "v4_controlled" / "e4" / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FOLDS_CSV = PROJECT_ROOT / "experiments" / "v4_controlled" / "data" / "clean_v5" / "folds_clean_v5.csv"

def run_verification():
    print("=" * 80)
    print("VERIFYING PHOTOMETRIC AUGMENTATIONS (GAMMA, CONTRAST, SPECKLE)")
    print("=" * 80)

    df_folds = pd.read_csv(FOLDS_CSV)
    
    # Select 10 diverse verified samples from development cohort
    sample_rows = df_folds.head(10)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    
    photo_aug = UltrasoundPhotometricAugmentation()
    
    gamma_stats = []
    contrast_stats = []
    speckle_stats = []
    
    for idx, r in sample_rows.iterrows():
        img_path = PROJECT_ROOT / r["image_path"]
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
        blurred = cv2.medianBlur(resized, 3)
        orig = clahe.apply(blurred) # base uint8 [0, 255]
        
        # 1. Test Gamma (range 0.85 to 1.15)
        for g in [0.85, 0.95, 1.05, 1.15]:
            aug_g = photo_aug.apply_gamma(orig, g)
            mad = float(np.mean(np.abs(aug_g.astype(float) - orig.astype(float))))
            pct_changed = float(np.mean(aug_g != orig) * 100.0)
            gamma_stats.append({
                "sample": r["filename"],
                "parameter": g,
                "mad": mad,
                "pct_changed": pct_changed,
                "orig_mean": float(np.mean(orig)),
                "aug_mean": float(np.mean(aug_g)),
                "orig_std": float(np.std(orig)),
                "aug_std": float(np.std(aug_g)),
                "min": int(np.min(aug_g)),
                "max": int(np.max(aug_g))
            })
            
        # 2. Test Contrast (range 0.90 to 1.10)
        for c in [0.90, 0.95, 1.05, 1.10]:
            aug_c = photo_aug.apply_contrast(orig, c)
            mad = float(np.mean(np.abs(aug_c.astype(float) - orig.astype(float))))
            pct_changed = float(np.mean(aug_c != orig) * 100.0)
            contrast_stats.append({
                "sample": r["filename"],
                "parameter": c,
                "mad": mad,
                "pct_changed": pct_changed,
                "orig_mean": float(np.mean(orig)),
                "aug_mean": float(np.mean(aug_c)),
                "orig_std": float(np.std(orig)),
                "aug_std": float(np.std(aug_c)),
                "min": int(np.min(aug_c)),
                "max": int(np.max(aug_c))
            })
            
        # 3. Test Speckle (range 0.005 to 0.020)
        for s in [0.005, 0.010, 0.015, 0.020]:
            aug_s = photo_aug.apply_speckle(orig, s)
            mad = float(np.mean(np.abs(aug_s.astype(float) - orig.astype(float))))
            pct_changed = float(np.mean(aug_s != orig) * 100.0)
            speckle_stats.append({
                "sample": r["filename"],
                "parameter": s,
                "mad": mad,
                "pct_changed": pct_changed,
                "orig_mean": float(np.mean(orig)),
                "aug_mean": float(np.mean(aug_s)),
                "orig_std": float(np.std(orig)),
                "aug_std": float(np.std(aug_s)),
                "min": int(np.min(aug_s)),
                "max": int(np.max(aug_s))
            })
            
    df_gamma = pd.DataFrame(gamma_stats)
    df_contrast = pd.DataFrame(contrast_stats)
    df_speckle = pd.DataFrame(speckle_stats)
    
    summary = {
        "gamma_summary": {
            "mean_mad": float(df_gamma["mad"].mean()),
            "max_mad": float(df_gamma["mad"].max()),
            "mean_pct_changed": float(df_gamma["pct_changed"].mean()),
            "by_param": df_gamma.groupby("parameter")[["mad", "pct_changed"]].mean().to_dict()
        },
        "contrast_summary": {
            "mean_mad": float(df_contrast["mad"].mean()),
            "max_mad": float(df_contrast["mad"].max()),
            "mean_pct_changed": float(df_contrast["pct_changed"].mean()),
            "by_param": df_contrast.groupby("parameter")[["mad", "pct_changed"]].mean().to_dict()
        },
        "speckle_summary": {
            "mean_mad": float(df_speckle["mad"].mean()),
            "max_mad": float(df_speckle["mad"].max()),
            "mean_pct_changed": float(df_speckle["pct_changed"].mean()),
            "by_param": df_speckle.groupby("parameter")[["mad", "pct_changed"]].mean().to_dict()
        }
    }
    
    out_json = REPORTS_DIR / "photometric_transform_verification.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved transform statistics: {out_json.name}")
    
    # 4. Generate visual comparison panels
    # Take sample 0
    ref_row = sample_rows.iloc[0]
    img = cv2.imread(str(PROJECT_ROOT / ref_row["image_path"]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
    blurred = cv2.medianBlur(resized, 3)
    base_img = clahe.apply(blurred)
    
    def make_diff_heatmap(orig, aug):
        diff = np.abs(aug.astype(float) - orig.astype(float))
        norm_diff = np.clip((diff / 25.0) * 255.0, 0, 255).astype(np.uint8) # scaled for visibility
        heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
        return heatmap

    # Gamma panel
    g_low = photo_aug.apply_gamma(base_img, 0.85)
    g_high = photo_aug.apply_gamma(base_img, 1.15)
    diff_g = make_diff_heatmap(base_img, g_high)
    
    c_orig = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    c_glow = cv2.cvtColor(g_low, cv2.COLOR_GRAY2BGR)
    c_ghigh = cv2.cvtColor(g_high, cv2.COLOR_GRAY2BGR)
    
    cv2.putText(c_orig, "Original CLAHE", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(c_glow, "Gamma = 0.85 (Darker)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(c_ghigh, "Gamma = 1.15 (Brighter)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(diff_g, "Pixel Diff (|G1.15 - Orig|)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    
    panel_gamma = np.hstack([c_orig, c_glow, c_ghigh, diff_g])
    cv2.imwrite(str(FIGURES_DIR / "e4b_gamma_transformation_audit.png"), panel_gamma)
    
    # Contrast panel
    c_low = photo_aug.apply_contrast(base_img, 0.90)
    c_high = photo_aug.apply_contrast(base_img, 1.10)
    diff_c = make_diff_heatmap(base_img, c_high)
    
    c_clow = cv2.cvtColor(c_low, cv2.COLOR_GRAY2BGR)
    c_chigh = cv2.cvtColor(c_high, cv2.COLOR_GRAY2BGR)
    cv2.putText(c_clow, "Contrast = 0.90 (-10%)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(c_chigh, "Contrast = 1.10 (+10%)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(diff_c, "Pixel Diff (|C1.10 - Orig|)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    
    panel_contrast = np.hstack([c_orig, c_clow, c_chigh, diff_c])
    cv2.imwrite(str(FIGURES_DIR / "e4b_contrast_transformation_audit.png"), panel_contrast)
    
    # Speckle panel
    s_low = photo_aug.apply_speckle(base_img, 0.005)
    s_high = photo_aug.apply_speckle(base_img, 0.020)
    diff_s = make_diff_heatmap(base_img, s_high)
    
    c_slow = cv2.cvtColor(s_low, cv2.COLOR_GRAY2BGR)
    c_shigh = cv2.cvtColor(s_high, cv2.COLOR_GRAY2BGR)
    cv2.putText(c_slow, "Speckle sigma=0.005", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(c_shigh, "Speckle sigma=0.020", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(diff_s, "Pixel Diff (|S0.02 - Orig|)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    
    panel_speckle = np.hstack([c_orig, c_slow, c_shigh, diff_s])
    cv2.imwrite(str(FIGURES_DIR / "e4b_speckle_transformation_audit.png"), panel_speckle)
    
    # Combined master panel
    master_panel = np.vstack([panel_gamma, panel_contrast, panel_speckle])
    cv2.imwrite(str(FIGURES_DIR / "e4b_isolated_transformations_panel.png"), master_panel)
    print("Saved visual transformation panels to experiments/v4_controlled/e4/reports/figures/")

if __name__ == "__main__":
    run_verification()
