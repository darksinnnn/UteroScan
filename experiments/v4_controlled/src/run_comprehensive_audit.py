#!/usr/bin/env python3
"""
Comprehensive Audit Script for UteroScan V4:
1. Checkpoint Selection Audit (E0 vs E2 epoch-by-epoch analysis)
2. Caliper Contact Sheet Trace & Contamination Audit
3. Doppler and Obstetric Scan Sweep across 1,224 training images
"""

import os
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

PROJECT_ROOT = Path("d:/Project/UteroScan")
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "v4_controlled" / "src"))

FOLDS_CSV = PROJECT_ROOT / "experiments" / "v4_controlled" / "data" / "folds.csv"
AUTH_MANIFEST_CSV = PROJECT_ROOT / "data" / "authoritative_manifest.csv"
EXCLUSION_LOG_CSV = PROJECT_ROOT / "data" / "excluded" / "exclusion_log.csv"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "v4_controlled" / "reports"
CHECKPOINTS_DIR = PROJECT_ROOT / "experiments" / "v4_controlled" / "checkpoints"

TASK_3341_LOG = Path(r"C:\Users\ashis\.gemini\antigravity-ide\brain\af0e99e6-8ff9-4623-ad48-86540f6228c9\.system_generated\tasks\task-3341.log")
TASK_3557_LOG = Path(r"C:\Users\ashis\.gemini\antigravity-ide\brain\af0e99e6-8ff9-4623-ad48-86540f6228c9\.system_generated\tasks\task-3557.log")

def audit_contact_sheet_samples():
    print("=" * 80)
    print("PART C: TRACING CALIPER CONTACT SHEET SAMPLES")
    print("=" * 80)
    
    df_folds = pd.read_csv(FOLDS_CSV)
    df_auth = pd.read_csv(AUTH_MANIFEST_CSV)
    df_excl = pd.read_csv(EXCLUSION_LOG_CSV)
    
    # Replicate the sampling logic from caliper.py
    uf_samples = df_folds[df_folds["target_idx"] == 0].sample(4, random_state=42)
    nuf_samples = df_folds[df_folds["target_idx"] == 1].sample(4, random_state=42)
    sample_df = pd.concat([uf_samples, nuf_samples]).reset_index(drop=True)
    
    results = []
    
    for idx, row in sample_df.iterrows():
        stem = str(row["source_stem"])
        filename = f"{stem}.jpg"
        img_rel_path = row["image_path"]
        img_abs_path = PROJECT_ROOT / img_rel_path
        
        exists_on_disk = img_abs_path.exists()
        
        # Manifest lookup
        m_row = df_auth[df_auth["filename"] == filename]
        auth_split = m_row["split"].values[0] if len(m_row) > 0 else "NOT_FOUND"
        auth_source = m_row["source_dataset"].values[0] if len(m_row) > 0 else "NOT_FOUND"
        auth_label = m_row["class_label"].values[0] if len(m_row) > 0 else "NOT_FOUND"
        auth_patient = m_row["patient_group"].values[0] if len(m_row) > 0 else "NOT_FOUND"
        auth_excl = m_row["exclusion_reason"].values[0] if len(m_row) > 0 else ""
        
        # Exclusion log lookup
        excl_row = df_excl[df_excl["filename"] == filename]
        in_exclusion_log = len(excl_row) > 0
        excl_reason = excl_row["reason"].values[0] if in_exclusion_log else "None"
        
        # Image analysis (check color/Doppler and dimensions)
        img = cv2.imread(str(img_abs_path)) if exists_on_disk else None
        h, w, c = img.shape if img is not None else (0, 0, 0)
        
        # Color Doppler check: difference between channels (B vs R)
        max_channel_diff = 0
        color_pixel_count = 0
        notes = []
        
        if img is not None:
            # Look at channel differences
            diff_rg = np.abs(img[:, :, 2].astype(int) - img[:, :, 1].astype(int))
            diff_rb = np.abs(img[:, :, 2].astype(int) - img[:, :, 0].astype(int))
            diff_gb = np.abs(img[:, :, 1].astype(int) - img[:, :, 0].astype(int))
            max_diff = np.maximum(diff_rg, np.maximum(diff_rb, diff_gb))
            max_channel_diff = int(np.max(max_diff))
            color_pixel_count = int(np.sum(max_diff > 30))
            
            if color_pixel_count > 500:
                notes.append(f"COLOR DOPPLER DETECTED ({color_pixel_count} saturated color pixels)")
                
        # Study series analysis
        study_id = row["study_proxy_group"]
        if str(study_id) in ["11364923"]:
            notes.append("OBSTETRIC_SUSPECT: Belongs to cluster 11364923 containing gestational sac / pregnancy anatomy")
            
        is_doppler_suspect = color_pixel_count > 500
        is_obstetric_suspect = str(study_id) in ["11364923"]
        
        results.append({
            "contact_sheet_index": idx,
            "filename": filename,
            "relative_path": img_rel_path,
            "class_label": row["class_label"],
            "target_idx": row["target_idx"],
            "fold": row["fold"],
            "study_proxy_group": row["study_proxy_group"],
            "exists_on_disk": exists_on_disk,
            "auth_split": auth_split,
            "auth_source": auth_source,
            "auth_label": auth_label,
            "auth_patient": auth_patient,
            "in_exclusion_log": in_exclusion_log,
            "excl_log_reason": excl_reason,
            "width": w,
            "height": h,
            "color_pixel_count_thresh30": color_pixel_count,
            "max_channel_diff": max_channel_diff,
            "is_doppler_suspect": is_doppler_suspect,
            "is_obstetric_suspect": is_obstetric_suspect,
            "audit_notes": "; ".join(notes) if notes else "Clean grayscale gynecologic scan"
        })
        
    audit_df = pd.DataFrame(results)
    out_csv = REPORTS_DIR / "contact_sheet_source_audit.csv"
    audit_df.to_csv(out_csv, index=False)
    print(f"Saved contact sheet audit to {out_csv}")
    print(audit_df[["contact_sheet_index", "filename", "class_label", "fold", "study_proxy_group", "is_doppler_suspect", "is_obstetric_suspect", "audit_notes"]].to_string())
    return audit_df

def sweep_dataset_for_color_doppler_and_obstetric():
    print("\n" + "=" * 80)
    print("STEP 2: FULL DATASET SWEEP (1,224 SCANS) FOR COLOR DOPPLER & CONTAMINATION")
    print("=" * 80)
    
    df_folds = pd.read_csv(FOLDS_CSV)
    
    doppler_findings = []
    
    for idx, row in df_folds.iterrows():
        stem = str(row["source_stem"])
        filename = f"{stem}.jpg"
        img_rel_path = row["image_path"]
        img_abs_path = PROJECT_ROOT / img_rel_path
        
        if not img_abs_path.exists():
            continue
            
        img = cv2.imread(str(img_abs_path))
        if img is None:
            continue
            
        diff_rg = np.abs(img[:, :, 2].astype(int) - img[:, :, 1].astype(int))
        diff_rb = np.abs(img[:, :, 2].astype(int) - img[:, :, 0].astype(int))
        diff_gb = np.abs(img[:, :, 1].astype(int) - img[:, :, 0].astype(int))
        max_diff = np.maximum(diff_rg, np.maximum(diff_rb, diff_gb))
        color_pixel_count = int(np.sum(max_diff > 30))
        
        if color_pixel_count > 500:
            doppler_findings.append({
                "filename": filename,
                "relative_path": img_rel_path,
                "class_label": row["class_label"],
                "target_idx": row["target_idx"],
                "fold": row["fold"],
                "study_proxy_group": row["study_proxy_group"],
                "color_pixel_count": color_pixel_count
            })
            
    df_doppler = pd.DataFrame(doppler_findings)
    print(f"Total scans in training cohort with Color Doppler pixels (>500 px): {len(df_doppler)}")
    if len(df_doppler) > 0:
        print(df_doppler.to_string())
    return df_doppler

def parse_training_logs():
    print("\n" + "=" * 80)
    print("STEP 3: PARSING EPOCH-BY-EPOCH TRAINING LOGS FOR E0 AND E2")
    print("=" * 80)
    
    def parse_log(log_path, exp_name):
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        rows = []
        current_fold = None
        
        epoch_pattern = re.compile(
            r"Epoch\s+(\d+)/(\d+)\s+\(([\d\.]+)s\)\s+\|\s+Train Loss:\s+([\d\.]+)\s+\|\s+Val Loss:\s+([\d\.]+)\s+\|\s+Val Acc:\s+([\d\.]+)%\s+\|\s+Val Sens:\s+([\d\.]+)%\s+\|\s+Val Spec:\s+([\d\.]+)%\s+\|\s+AUC:\s+([\d\.]+)"
        )
        fold_pattern = re.compile(r"TRAINING FOLD (\d+)/4")
        
        for line in lines:
            f_match = fold_pattern.search(line)
            if f_match:
                current_fold = int(f_match.group(1))
                continue
                
            e_match = epoch_pattern.search(line)
            if e_match and current_fold is not None:
                ep = int(e_match.group(1))
                dur = float(e_match.group(3))
                tr_loss = float(e_match.group(4))
                val_loss = float(e_match.group(5))
                val_acc = float(e_match.group(6)) / 100.0
                val_sens = float(e_match.group(7)) / 100.0
                val_spec = float(e_match.group(8)) / 100.0
                val_auc = float(e_match.group(9))
                
                rows.append({
                    "experiment": exp_name,
                    "fold": current_fold,
                    "epoch": ep,
                    "duration_sec": dur,
                    "train_loss": tr_loss,
                    "val_loss": val_loss,
                    "val_acc_050": val_acc,
                    "val_sens_050": val_sens,
                    "val_spec_050": val_spec,
                    "roc_auc": val_auc
                })
        return pd.DataFrame(rows)
        
    e0_df = parse_log(TASK_3341_LOG, "E0_BASELINE")
    e2_df = parse_log(TASK_3557_LOG, "E2_CALIPER")
    
    return e0_df, e2_df

def audit_checkpoint_selection(e0_df, e2_df):
    print("\n" + "=" * 80)
    print("STEP 4: AUDITING CHECKPOINT SELECTION POLICIES")
    print("=" * 80)
    
    for exp_name, df in [("E0_BASELINE", e0_df), ("E2_CALIPER", e2_df)]:
        print(f"\n--- {exp_name} Analysis ---")
        best_loss_rows = []
        best_acc_rows = []
        
        for fold in range(5):
            f_df = df[df["fold"] == fold]
            # Rule 1: Min Val Loss
            min_loss_idx = f_df["val_loss"].idxmin()
            best_loss_row = f_df.loc[min_loss_idx]
            best_loss_rows.append(best_loss_row)
            
            # Rule 2: Max Val Acc (with Val Loss as tiebreaker)
            # Sort by val_acc descending, then val_loss ascending
            sorted_f = f_df.sort_values(by=["val_acc_050", "val_loss"], ascending=[False, True])
            best_acc_row = sorted_f.iloc[0]
            best_acc_rows.append(best_acc_row)
            
            # Check availability of weights on disk
            ckpt_name = f"{exp_name.lower()}_fold_{fold}_best.pth"
            ckpt_file = CHECKPOINTS_DIR / ckpt_name
            ckpt_exists = ckpt_file.exists()
            
            diff_epoch = best_loss_row["epoch"] != best_acc_row["epoch"]
            status_str = f"MISMATCH: Loss Ep {best_loss_row['epoch']} (Loss {best_loss_row['val_loss']:.4f}, Acc {best_loss_row['val_acc_050']*100:.1f}%) vs Acc Ep {best_acc_row['epoch']} (Loss {best_acc_row['val_loss']:.4f}, Acc {best_acc_row['val_acc_050']*100:.1f}%)" if diff_epoch else f"MATCH: Ep {best_loss_row['epoch']} (Acc {best_loss_row['val_acc_050']*100:.1f}%, Loss {best_loss_row['val_loss']:.4f})"
            print(f"Fold {fold}: {status_str} | Saved Ckpt: {ckpt_name} (Exists: {ckpt_exists})")
            
        df_bl = pd.DataFrame(best_loss_rows)
        df_ba = pd.DataFrame(best_acc_rows)
        
        print(f"\n[{exp_name}] Summary under Rule 1 (Min Val Loss):")
        print(f"  Mean Val Acc : {df_bl['val_acc_050'].mean()*100:.2f}% +/- {df_bl['val_acc_050'].std()*100:.2f}%")
        print(f"  Mean Val Sens: {df_bl['val_sens_050'].mean()*100:.2f}% +/- {df_bl['val_sens_050'].std()*100:.2f}%")
        print(f"  Mean Val Spec: {df_bl['val_spec_050'].mean()*100:.2f}% +/- {df_bl['val_spec_050'].std()*100:.2f}%")
        print(f"  Mean ROC-AUC : {df_bl['roc_auc'].mean():.4f} +/- {df_bl['roc_auc'].std():.4f}")
        
        print(f"\n[{exp_name}] Summary under Rule 2 (Max Val Acc @ 0.50):")
        print(f"  Mean Val Acc : {df_ba['val_acc_050'].mean()*100:.2f}% +/- {df_ba['val_acc_050'].std()*100:.2f}%")
        print(f"  Mean Val Sens: {df_ba['val_sens_050'].mean()*100:.2f}% +/- {df_ba['val_sens_050'].std()*100:.2f}%")
        print(f"  Mean Val Spec: {df_ba['val_spec_050'].mean()*100:.2f}% +/- {df_ba['val_spec_050'].std()*100:.2f}%")
        print(f"  Mean ROC-AUC : {df_ba['roc_auc'].mean():.4f} +/- {df_ba['roc_auc'].std():.4f}")

if __name__ == "__main__":
    audit_contact_sheet_samples()
    sweep_dataset_for_color_doppler_and_obstetric()
    e0_df, e2_df = parse_training_logs()
    audit_checkpoint_selection(e0_df, e2_df)
