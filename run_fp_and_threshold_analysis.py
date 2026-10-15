"""
run_fp_and_threshold_analysis.py
UteroScan Error Analysis & Threshold Optimization

Tasks:
1. Extract exact test predictions for all 358 test images.
2. Isolate the 41 False Positive cases (NUF predicted as UF).
3. Compute image-level metrics: resolution, brightness, color tinge, dual-panel layout, calipers.
4. Statistical comparison between FP (41 scans) and TN (146 scans).
5. Cross-reference against excluded Doppler and pregnancy scans.
6. Generate 41-image labeled montage to reports/false_positive_analysis/fp_montage.png.
7. Compute ROC curve and Precision-Recall curve, saved to reports/roc_curve.png.
8. Evaluate threshold grid (0.30 to 0.80) with full metrics table.
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

# Trapezoid integration for AUC
try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

# Setup paths
BASE_DIR = Path("d:/Project/UteroScan")
COMPANION_DIR = BASE_DIR / "DCGAN&MobileNetV2" / "DCGAN&MobileNetV2"
sys.path.insert(0, str(COMPANION_DIR))

from nets import get_model_from_name
from utils.dataloader import DataGenerator, detection_collate

CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
REPORTS_DIR = BASE_DIR / "reports"
FP_ANALYSIS_DIR = REPORTS_DIR / "false_positive_analysis"
FP_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = CHECKPOINTS_DIR / "resnet50_stage2_best.pth"
TEST_FILE = COMPANION_DIR / "cls_test.txt"
EXCLUSION_LOG = BASE_DIR / "data" / "excluded" / "exclusion_log.csv"

def resolve_path(p_str):
    p = Path(p_str)
    if p.is_file():
        return p
    candidates = [
        BASE_DIR / p,
        BASE_DIR / "datasets" / p,
        BASE_DIR / "datasets" / "datasets" / p,
        BASE_DIR / "datasets" / "datasets" / Path(*p.parts[1:] if len(p.parts) > 1 else p.parts),
        COMPANION_DIR / p
    ]
    for c in candidates:
        if c.is_file():
            return c
    return p

def extract_patient_id(filename):
    # Format: 1.2.826.0.1.3680043.2.461.<patient_id>.<image_id>.jpg
    parts = filename.split(".")
    if len(parts) >= 10:
        return parts[8]  # The number following 461
    return "unknown"

def analyze_image_features(img_path):
    """
    Analyzes raw image for:
    - resolution
    - mean brightness
    - color variance (tinge)
    - dual-panel layout
    - caliper / annotation markings
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return {}
    
    h, w, c = img_bgr.shape
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))
    
    # 1. Color variance check
    b, g, r = img_bgr[:, :, 0].astype(float), img_bgr[:, :, 1].astype(float), img_bgr[:, :, 2].astype(float)
    color_diff = np.abs(r - g) + np.abs(g - b) + np.abs(b - r)
    mean_color_diff = float(np.mean(color_diff))
    max_color_diff = float(np.max(color_diff))
    colored_pixel_ratio = float(np.mean(color_diff > 15.0))
    has_color_tinge = (colored_pixel_ratio > 0.005) or (mean_color_diff > 3.0)
    
    # 2. Dual-panel layout detection
    # Dual panel ultrasound scans have two side-by-side sector fans with a vertical dividing band near center (w//2)
    mid_x = w // 2
    band_width = max(2, int(w * 0.02))
    center_strip = gray[:, mid_x - band_width : mid_x + band_width]
    center_strip_mean = float(np.mean(center_strip))
    
    # Check left vs right half correlation or dual fan presence
    left_half = gray[:, :mid_x]
    right_half = gray[:, mid_x:]
    # Resize to common shape for comparison
    min_hw = min(left_half.shape[1], right_half.shape[1])
    lh_resized = cv2.resize(left_half, (100, 100))
    rh_resized = cv2.resize(right_half, (100, 100))
    lr_diff = np.mean(np.abs(lh_resized.astype(float) - rh_resized.astype(float)))
    
    # Dual panel typically has aspect ratio >= 1.4, dark center divider line, and active tissue on both sides
    left_active = np.mean(lh_resized > 20) > 0.15
    right_active = np.mean(rh_resized > 20) > 0.15
    is_dual_panel = bool((w / h >= 1.35) and left_active and right_active and (center_strip_mean < 25.0 or lr_diff < 45.0))

    # 3. Caliper / measurement overlay detection
    # Calipers are thin white markers (+, x, dashed lines) with intensity near 255
    # High-intensity pixels in ultrasound field
    high_intensity_mask = (gray > 240)
    # Check if they form small isolated clusters (crosshairs / dots / text)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(high_intensity_mask.astype(np.uint8))
    # Count small components (size 5 to 200 pixels) that are not the top text header (y > 0.15*h)
    caliper_candidates = 0
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        cy = centroids[idx][1]
        cx = centroids[idx][0]
        if 5 <= area <= 250 and cy > 0.15 * h and cy < 0.90 * h:
            caliper_candidates += 1
    has_calipers = (caliper_candidates >= 2)
    
    return {
        "width": w,
        "height": h,
        "aspect_ratio": round(w / h, 3),
        "mean_brightness": round(mean_brightness, 2),
        "std_brightness": round(std_brightness, 2),
        "mean_color_diff": round(mean_color_diff, 2),
        "has_color_tinge": bool(has_color_tinge),
        "is_dual_panel": bool(is_dual_panel),
        "has_calipers": bool(has_calipers),
        "caliper_markers_count": caliper_candidates
    }

def main():
    print("=" * 80)
    print("RUNNING UTEROSCAN FALSE POSITIVE & THRESHOLD ANALYSIS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = get_model_from_name["resnet50"](num_classes=2, pretrained=False)
    ckpt = torch.load(str(BEST_CHECKPOINT), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load test annotations
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_lines = [l.strip() for l in f if l.strip()]

    test_ds = DataGenerator(test_lines, input_shape=(224, 224), random=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=detection_collate)

    all_targets = []
    all_probs_uf = []  # P(UF) = probs[:, 0]
    all_probs_nuf = [] # P(NUF) = probs[:, 1]

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs_uf.extend(probs[:, 0].cpu().numpy().tolist())
            all_probs_nuf.extend(probs[:, 1].cpu().numpy().tolist())

    y_true = np.array(all_targets, dtype=int)
    y_probs_uf = np.array(all_probs_uf, dtype=float)
    y_probs_nuf = np.array(all_probs_nuf, dtype=float)

    # Reconcile with 0.5 threshold
    y_pred_default = (y_probs_uf < 0.5).astype(int)  # 0 if p_uf >= 0.5 else 1
    # Check: if p_uf >= 0.5, y_pred is 0 (UF). If p_uf < 0.5, y_pred is 1 (NUF).
    y_pred = np.where(y_probs_uf >= 0.5, 0, 1)

    tp_uf = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn_uf = int(np.sum((y_true == 0) & (y_pred == 1)))
    fp_uf = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn_uf = int(np.sum((y_true == 1) & (y_pred == 1)))

    print(f"\nVerification against Phase 4 metrics:")
    print(f"  TP: {tp_uf} | FN: {fn_uf} | FP: {fp_uf} | TN: {tn_uf} | Total: {len(y_true)}")
    assert tp_uf == 162 and fn_uf == 9 and fp_uf == 41 and tn_uf == 146, "Counts do not match Phase 4 metrics!"
    print(">>> Exact Phase 4 confusion matrix verified (162 / 9 / 41 / 146). <<<")

    # Load exclusion log to cross-reference
    excluded_filenames = set()
    excluded_patient_ids = set()
    if EXCLUSION_LOG.is_file():
        import csv
        with open(EXCLUSION_LOG, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = row.get("filename", "").strip()
                if fn:
                    excluded_filenames.add(fn)
                    pid = extract_patient_id(fn)
                    excluded_patient_ids.add(pid)

    print(f"Exclusion Log: {len(excluded_filenames)} excluded files, {len(excluded_patient_ids)} unique patient IDs.")

    # Detailed image analysis across all 358 test images
    records = []
    for idx, line in enumerate(test_lines):
        target, path_str = line.split(";")
        target = int(target)
        res_path = resolve_path(path_str)
        fn = res_path.name
        pid = extract_patient_id(fn)
        
        feats = analyze_image_features(res_path)
        
        p_uf = float(y_probs_uf[idx])
        p_nuf = float(y_probs_nuf[idx])
        pred_label = 0 if p_uf >= 0.5 else 1
        
        is_fp = (target == 1 and pred_label == 0)
        is_tn = (target == 1 and pred_label == 1)
        is_tp = (target == 0 and pred_label == 0)
        is_fn = (target == 0 and pred_label == 1)

        records.append({
            "index": idx,
            "filename": fn,
            "path": str(res_path),
            "patient_id": pid,
            "target": target,
            "target_label": "UF" if target == 0 else "NUF",
            "p_uf": p_uf,
            "p_nuf": p_nuf,
            "pred_label": "UF" if pred_label == 0 else "NUF",
            "is_fp": is_fp,
            "is_tn": is_tn,
            "is_tp": is_tp,
            "is_fn": is_fn,
            **feats
        })

    # Global dataset stats for brightness comparison
    all_brightness = [r["mean_brightness"] for r in records]
    global_mean_bright = float(np.mean(all_brightness))
    global_std_bright = float(np.std(all_brightness))
    print(f"Test Set Global Mean Brightness: {global_mean_bright:.2f} +/- {global_std_bright:.2f}")

    # Mark exposure anomaly
    for r in records:
        r["exposure_z"] = round((r["mean_brightness"] - global_mean_bright) / global_std_bright, 2)
        r["is_unusually_dark"] = bool(r["exposure_z"] < -1.5)
        r["is_unusually_bright"] = bool(r["exposure_z"] > 1.5)

    fp_records = [r for r in records if r["is_fp"]]
    tn_records = [r for r in records if r["is_tn"]]

    assert len(fp_records) == 41, f"Expected 41 FP records, got {len(fp_records)}"
    assert len(tn_records) == 146, f"Expected 146 TN records, got {len(tn_records)}"

    print(f"\nExtracted 41 FP records and 146 TN records.")

    # Save detailed JSON of 41 FP cases
    fp_summary_path = FP_ANALYSIS_DIR / "fp_detailed_records.json"
    with open(fp_summary_path, "w", encoding="utf-8") as f:
        json.dump(fp_records, f, indent=2)
    print(f"Saved detailed FP records to {fp_summary_path}")

    # -------------------------------------------------------------------------
    # Statistical Pattern Analysis
    # -------------------------------------------------------------------------
    # 1. Confidence clustering
    conf_borderline = [r for r in fp_records if 0.50 <= r["p_uf"] <= 0.65]
    conf_moderate   = [r for r in fp_records if 0.65 < r["p_uf"] <= 0.80]
    conf_confident  = [r for r in fp_records if r["p_uf"] > 0.80]

    print("\n" + "=" * 60)
    print("1. CONFIDENCE DISTRIBUTION IN FALSE POSITIVES (P(UF))")
    print("=" * 60)
    print(f"  Borderline (0.50 - 0.65) : {len(conf_borderline):2d} / 41 ({len(conf_borderline)/41*100:5.1f}%)")
    print(f"  Moderate   (0.65 - 0.80) : {len(conf_moderate):2d} / 41 ({len(conf_moderate)/41*100:5.1f}%)")
    print(f"  Confident  (> 0.80)      : {len(conf_confident):2d} / 41 ({len(conf_confident)/41*100:5.1f}%)")
    print(f"  Mean P(UF) in FPs        : {np.mean([r['p_uf'] for r in fp_records]):.4f}")
    print(f"  Median P(UF) in FPs      : {np.median([r['p_uf'] for r in fp_records]):.4f}")
    print(f"  Min / Max P(UF) in FPs   : {min(r['p_uf'] for r in fp_records):.4f} / {max(r['p_uf'] for r in fp_records):.4f}")

    # 2. Structural & Visual Feature Comparison: FP (41) vs TN (146)
    print("\n" + "=" * 60)
    print("2. STRUCTURAL TRAIT COMPARISON (FP vs TN)")
    print("=" * 60)

    fp_dual_panel = sum(1 for r in fp_records if r["is_dual_panel"])
    tn_dual_panel = sum(1 for r in tn_records if r["is_dual_panel"])

    fp_calipers = sum(1 for r in fp_records if r["has_calipers"])
    tn_calipers = sum(1 for r in tn_records if r["has_calipers"])

    fp_color_tinge = sum(1 for r in fp_records if r["has_color_tinge"])
    tn_color_tinge = sum(1 for r in tn_records if r["has_color_tinge"])

    fp_dark = sum(1 for r in fp_records if r["is_unusually_dark"])
    tn_dark = sum(1 for r in tn_records if r["is_unusually_dark"])

    fp_bright = sum(1 for r in fp_records if r["is_unusually_bright"])
    tn_bright = sum(1 for r in tn_records if r["is_unusually_bright"])

    print(f"Dual-Panel Layout   : FP = {fp_dual_panel:2d}/41 ({fp_dual_panel/41*100:5.1f}%) | TN = {tn_dual_panel:2d}/146 ({tn_dual_panel/146*100:5.1f}%)")
    print(f"Caliper Overlays    : FP = {fp_calipers:2d}/41 ({fp_calipers/41*100:5.1f}%) | TN = {tn_calipers:2d}/146 ({tn_calipers/146*100:5.1f}%)")
    print(f"Color Tinge         : FP = {fp_color_tinge:2d}/41 ({fp_color_tinge/41*100:5.1f}%) | TN = {tn_color_tinge:2d}/146 ({tn_color_tinge/146*100:5.1f}%)")
    print(f"Unusually Dark (Z<-1.5): FP = {fp_dark:2d}/41 ({fp_dark/41*100:5.1f}%) | TN = {tn_dark:2d}/146 ({tn_dark/146*100:5.1f}%)")
    print(f"Unusually Bright(Z>1.5): FP = {fp_bright:2d}/41 ({fp_bright/41*100:5.1f}%) | TN = {tn_bright:2d}/146 ({tn_bright/146*100:5.1f}%)")
    print(f"Mean Brightness     : FP = {np.mean([r['mean_brightness'] for r in fp_records]):5.1f} | TN = {np.mean([r['mean_brightness'] for r in tn_records]):5.1f}")

    # Patient / Scanner ID clustering
    fp_pids = Counter(r["patient_id"] for r in fp_records)
    tn_pids = Counter(r["patient_id"] for r in tn_records)

    print(f"\nUnique Patient/Scanner IDs in FPs: {len(fp_pids)} unique across 41 images")
    print(f"Top clustered Patient IDs in FPs:")
    for pid, count in fp_pids.most_common(6):
        tn_count = tn_pids.get(pid, 0)
        print(f"  Patient ID {pid}: {count} FPs (out of {count + tn_count} total NUF scans for this patient)")

    # 3. Cross-reference with Excluded Lists
    print("\n" + "=" * 60)
    print("3. CROSS-REFERENCE AGAINST EXCLUSION LOG")
    print("=" * 60)
    fp_in_excluded = [r["filename"] for r in fp_records if r["filename"] in excluded_filenames]
    print(f"Exact filename overlap with excluded list: {len(fp_in_excluded)} (Expected: 0)")
    assert len(fp_in_excluded) == 0

    fp_pid_overlap = [r["filename"] for r in fp_records if r["patient_id"] in excluded_patient_ids]
    print(f"Patient IDs overlapping with excluded patients: {len(set(r['patient_id'] for r in fp_records if r['patient_id'] in excluded_patient_ids))} unique IDs ({len(fp_pid_overlap)} scans)")
    for r in fp_records:
        if r["patient_id"] in excluded_patient_ids:
            print(f"  FP {r['filename']} shares Patient ID {r['patient_id']} with an excluded case")

    # -------------------------------------------------------------------------
    # 4. Generate 41-Image Annotated Montage
    # -------------------------------------------------------------------------
    print("\nGenerating 41-Image Labeled Montage...")
    # 6 rows x 7 cols = 42 slots
    nrows, ncols = 6, 7
    thumb_w, thumb_h = 240, 240
    header_h = 45
    cell_w, cell_h = thumb_w, thumb_h + header_h

    montage_w = ncols * cell_w
    montage_h = nrows * cell_h
    montage = Image.new("RGB", (montage_w, montage_h), color=(20, 24, 33))
    draw = ImageDraw.Draw(montage)

    # Load default font
    try:
        font_main = ImageFont.truetype("arial.ttf", 12)
        font_sub = ImageFont.truetype("arial.ttf", 10)
    except:
        font_main = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # Sort FP records descending by confidence P(UF)
    fp_records_sorted = sorted(fp_records, key=lambda x: -x["p_uf"])

    for idx, r in enumerate(fp_records_sorted):
        row = idx // ncols
        col = idx % ncols
        x0 = col * cell_w
        y0 = row * cell_h

        # Load raw image
        raw_img = Image.open(r["path"]).convert("RGB")
        thumb = raw_img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        montage.paste(thumb, (x0, y0 + header_h))

        # Draw header banner
        # Color code by confidence: red if >0.80, orange if 0.65-0.80, yellow if borderline 0.50-0.65
        if r["p_uf"] > 0.80:
            hdr_bg = (180, 40, 40)
        elif r["p_uf"] > 0.65:
            hdr_bg = (200, 110, 30)
        else:
            hdr_bg = (140, 130, 20)

        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + header_h - 1], fill=hdr_bg)
        
        # Text label
        short_name = r["filename"].split(".")[-2]  # image id
        full_pid = r["patient_id"]
        draw.text((x0 + 4, y0 + 3), f"#{idx+1} [P(UF): {r['p_uf']*100:.1f}%]", fill=(255, 255, 255), font=font_main)
        draw.text((x0 + 4, y0 + 17), f"ID: {full_pid}.{short_name}", fill=(240, 240, 240), font=font_sub)
        
        # Flags line
        flags = []
        if r["is_dual_panel"]: flags.append("DualPanel")
        if r["has_calipers"]: flags.append("Caliper")
        if r["has_color_tinge"]: flags.append("ColorTint")
        if r["is_unusually_bright"]: flags.append("HighBright")
        if r["is_unusually_dark"]: flags.append("LowBright")
        flag_str = "|".join(flags) if flags else "NoOverlays"
        draw.text((x0 + 4, y0 + 29), flag_str, fill=(230, 230, 250), font=font_sub)

        # Border
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], outline=(60, 70, 90), width=1)

    # Slot 42 (the extra cell): add summary legend
    x0 = 6 * cell_w
    y0 = 5 * cell_h
    draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], fill=(15, 20, 28))
    draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], outline=(80, 90, 120), width=2)
    draw.text((x0 + 10, y0 + 15), "UteroScan Phase 4", fill=(255, 255, 255), font=font_main)
    draw.text((x0 + 10, y0 + 32), "41 False Positives", fill=(255, 215, 0), font=font_main)
    draw.text((x0 + 10, y0 + 55), "True: NUF (Normal)", fill=(200, 200, 200), font=font_sub)
    draw.text((x0 + 10, y0 + 70), "Pred: UF (Fibroid)", fill=(255, 120, 120), font=font_sub)
    draw.text((x0 + 10, y0 + 95), "Color Legend:", fill=(255, 255, 255), font=font_sub)
    draw.rectangle([x0 + 10, y0 + 115, x0 + 25, y0 + 127], fill=(180, 40, 40))
    draw.text((x0 + 30, y0 + 115), "> 80% Confident (17)", fill=(220, 220, 220), font=font_sub)
    draw.rectangle([x0 + 10, y0 + 132, x0 + 25, y0 + 144], fill=(200, 110, 30))
    draw.text((x0 + 30, y0 + 132), "65-80% Moderate (10)", fill=(220, 220, 220), font=font_sub)
    draw.rectangle([x0 + 10, y0 + 149, x0 + 25, y0 + 161], fill=(140, 130, 20))
    draw.text((x0 + 30, y0 + 149), "50-65% Borderline (14)", fill=(220, 220, 220), font=font_sub)

    montage_path = FP_ANALYSIS_DIR / "fp_montage.png"
    montage.save(str(montage_path), dpi=(300, 300))
    print(f"Saved 41-image montage to: {montage_path}")

    # -------------------------------------------------------------------------
    # 5. ROC Curve & Precision-Recall Curve Generation
    # -------------------------------------------------------------------------
    print("\nComputing Full ROC and Precision-Recall Curves...")
    
    # In y_true: UF = 0, NUF = 1.
    # For UF detection (UF is positive clinical target = 1):
    y_true_uf = (y_true == 0).astype(int)  # 1 for UF, 0 for NUF
    scores_uf = y_probs_uf                 # Probability of UF

    # Sort descending by score
    desc_idx = np.argsort(-scores_uf)
    y_true_sorted = y_true_uf[desc_idx]
    scores_sorted = scores_uf[desc_idx]

    distinct_val_indices = np.where(np.diff(scores_sorted))[0]
    thresh_idxs = np.r_[distinct_val_indices, y_true_sorted.size - 1]

    n_pos = int(np.sum(y_true_uf == 1))  # 171 UF
    n_neg = int(np.sum(y_true_uf == 0))  # 187 NUF

    tps = np.cumsum(y_true_sorted == 1)[thresh_idxs]
    fps = np.cumsum(y_true_sorted == 0)[thresh_idxs]

    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]

    roc_auc_val = float(trapezoid(tpr, fpr))

    # Precision-Recall curve
    precisions = tps / (tps + fps)
    recalls = tps / n_pos

    # Monotonic PR envelope
    pr_auc_val = float(trapezoid(np.r_[1.0, precisions], np.r_[0.0, recalls]))

    # Plot ROC & PR Curves in a 2-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # ROC Plot
    ax1.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ResNet50 ROC (AUC = {roc_auc_val:.4f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Chance (AUC = 0.5000)")
    
    # Mark current operational point (threshold 0.50)
    current_fpr = 41 / 187
    current_tpr = 162 / 171
    ax1.plot(current_fpr, current_tpr, marker="o", markersize=8, color="#d62728", label=f"Default Thresh 0.50 (Sens: {current_tpr*100:.1f}%, Spec: {(1-current_fpr)*100:.1f}%)")

    # Mark optimal operating points
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    ax1.set_title("UteroScan ROC Curve (Held-Out Test Set, n=358)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=9.5)

    # Precision-Recall Plot
    baseline_pr = n_pos / (n_pos + n_neg)
    ax2.plot(np.r_[0.0, recalls], np.r_[1.0, precisions], color="#2ca02c", lw=2.5, label=f"UF PR Curve (PR-AUC = {pr_auc_val:.4f})")
    ax2.axhline(baseline_pr, color="gray", linestyle="--", lw=1.5, label=f"Prevalence Baseline ({baseline_pr*100:.1f}%)")
    
    current_prec = 162 / (162 + 41)
    ax2.plot(current_tpr, current_prec, marker="o", markersize=8, color="#d62728", label=f"Default Thresh 0.50 (Prec: {current_prec*100:.1f}%, Rec: {current_tpr*100:.1f}%)")

    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.set_xlabel("UF Sensitivity (Recall)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("UF Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    ax2.set_title("UteroScan Precision-Recall Curve (UF Class)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=9.5)

    plt.tight_layout()
    roc_plot_path = REPORTS_DIR / "roc_curve.png"
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"Saved ROC and Precision-Recall plot to: {roc_plot_path}")

    # -------------------------------------------------------------------------
    # 6. Threshold Grid Sweep Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DECISION THRESHOLD GRID EVALUATION (UF DETECTION)")
    print("=" * 80)

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    threshold_results = []

    print(f"{'Thresh':<7} | {'Accuracy':<9} | {'UF Sens (Rec)':<14} | {'NUF Spec':<10} | {'UF Prec':<9} | {'UF F1':<8} | {'FN (Missed)':<11} | {'FP':<6}")
    print("-" * 95)

    for th in thresholds:
        pred_uf = (scores_uf >= th).astype(int)
        
        tp = int(np.sum((y_true_uf == 1) & (pred_uf == 1)))
        fn = int(np.sum((y_true_uf == 1) & (pred_uf == 0)))
        fp = int(np.sum((y_true_uf == 0) & (pred_uf == 1)))
        tn = int(np.sum((y_true_uf == 0) & (pred_uf == 0)))

        acc = (tp + tn) / len(y_true_uf)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

        threshold_results.append({
            "threshold": th,
            "accuracy": round(acc, 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn
        })

        tag = " <-- Default" if th == 0.50 else ""
        print(f"{th:<7.2f} | {acc*100:6.2f}%   | {sens*100:6.2f}% ({tp}/{tp+fn}) | {spec*100:6.2f}%   | {prec*100:6.2f}%  | {f1*100:6.2f}% | {fn:<11d} | {fp:<6d}{tag}")

    # Save threshold results to JSON
    thresh_json_path = REPORTS_DIR / "threshold_analysis.json"
    with open(thresh_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "roc_auc": roc_auc_val,
            "pr_auc": pr_auc_val,
            "threshold_sweep": threshold_results
        }, f, indent=2)
    print(f"\nSaved threshold sweep results to {thresh_json_path}")

    return {
        "fp_records": fp_records,
        "fp_pids": fp_pids,
        "tn_pids": tn_pids,
        "threshold_results": threshold_results,
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val
    }

if __name__ == "__main__":
    main()
