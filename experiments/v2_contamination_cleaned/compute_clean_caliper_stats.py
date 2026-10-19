import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from scipy.stats import fisher_exact, chi2_contingency

BASE_DIR = Path("d:/Project/UteroScan")
DATASETS_DIR = BASE_DIR / "datasets" / "datasets"
CLS_TEST_CLEAN = BASE_DIR / "experiments" / "v2_contamination_cleaned" / "annotations" / "cls_test.txt"
FP_DETAILED_JSON = BASE_DIR / "reports" / "false_positive_analysis" / "fp_detailed_records.json"

def extract_patient_id(filename: str) -> str:
    parts = filename.split(".")
    if len(parts) >= 9:
        return parts[8]
    return filename

def check_calipers(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return False, 0
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    high_intensity_mask = (gray > 240)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(high_intensity_mask.astype(np.uint8))
    caliper_candidates = 0
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        cy = centroids[idx][1]
        if 5 <= area <= 250 and cy > 0.15 * h and cy < 0.90 * h:
            caliper_candidates += 1
    return (caliper_candidates >= 2), caliper_candidates

def main():
    print("=" * 80)
    print("PATIENT-LEVEL CALIPER STATISTICAL TEST ON CLEAN TEST SET (321 IMAGES)")
    print("=" * 80)

    # 1. Load original 41 FP records to know which of the remaining 158 NUF files were predicted UF
    with open(FP_DETAILED_JSON, "r", encoding="utf-8") as f:
        fp_records = json.load(f)
    orig_fp_filenames = {r["filename"] for r in fp_records}

    # 2. Read clean test set
    with open(CLS_TEST_CLEAN, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    clean_nuf_lines = [l for l in lines if l.startswith("1;")]
    print(f"Total Clean NUF Scans in Test Set: {len(clean_nuf_lines)} (expected 158)")

    # Analyze scans per patient
    patient_data = defaultdict(lambda: {"total": 0, "fps": 0, "tns": 0, "calipers": 0, "files": []})

    total_fps = 0
    total_tns = 0

    for l in clean_nuf_lines:
        target, rel_path = l.split(";")
        p = DATASETS_DIR / rel_path.replace("datasets/", "")
        fn = p.name
        pid = extract_patient_id(fn)

        is_fp = (fn in orig_fp_filenames)
        if is_fp:
            total_fps += 1
        else:
            total_tns += 1

        has_cal, _ = check_calipers(p)

        patient_data[pid]["total"] += 1
        patient_data[pid]["files"].append(fn)
        if is_fp:
            patient_data[pid]["fps"] += 1
        else:
            patient_data[pid]["tns"] += 1
        if has_cal:
            patient_data[pid]["calipers"] += 1

    print(f"Scans Breakdown: Clean FPs = {total_fps}, Clean TNs = {total_tns}")
    print(f"Unique NUF Patients in Clean Test Set: {len(patient_data)}")

    # Contingency table:
    # Row 0: Patients with >= 1 FP
    # Row 1: Patients with 0 FP (Pure TN Patients)
    # Col 0: >= 1 Caliper Scan
    # Col 1: 0 Caliper Scans
    fp_pat_with_cal = 0
    fp_pat_no_cal = 0
    tn_pat_with_cal = 0
    tn_pat_no_cal = 0

    for pid, d in patient_data.items():
        has_fp = (d["fps"] > 0)
        has_cal = (d["calipers"] > 0)
        if has_fp:
            if has_cal:
                fp_pat_with_cal += 1
            else:
                fp_pat_no_cal += 1
        else:
            if has_cal:
                tn_pat_with_cal += 1
            else:
                tn_pat_no_cal += 1

    table = np.array([
        [fp_pat_with_cal, fp_pat_no_cal],
        [tn_pat_with_cal, tn_pat_no_cal]
    ])

    print("\n--- CLEAN DATASET PATIENT-LEVEL CONTINGENCY TABLE ---")
    print(f"                    | >=1 Caliper Scan | 0 Caliper Scans | Total")
    print(f"Patient with >=1 FP | {fp_pat_with_cal:16d} | {fp_pat_no_cal:15d} | {fp_pat_with_cal + fp_pat_no_cal}")
    print(f"Patient with 0 FP   | {tn_pat_with_cal:16d} | {tn_pat_no_cal:15d} | {tn_pat_with_cal + tn_pat_no_cal}")
    print(f"Total Patients      | {fp_pat_with_cal + tn_pat_with_cal:16d} | {fp_pat_no_cal + tn_pat_no_cal:15d} | {len(patient_data)}")

    odds_clean, p_fisher_clean = fisher_exact(table)
    chi2_clean, p_chi2_clean, _, _ = chi2_contingency(table, correction=True)

    fp_rate_clean = fp_pat_with_cal / (fp_pat_with_cal + fp_pat_no_cal)
    tn_rate_clean = tn_pat_with_cal / (tn_pat_with_cal + tn_pat_no_cal)

    print(f"\nStatistical Results on Clean Test Set:")
    print(f"  FP Patient Caliper Rate : {fp_pat_with_cal}/{fp_pat_with_cal + fp_pat_no_cal} ({fp_rate_clean*100:.2f}%)")
    print(f"  TN Patient Caliper Rate : {tn_pat_with_cal}/{tn_pat_with_cal + tn_pat_no_cal} ({tn_rate_clean*100:.2f}%)")
    print(f"  Clean Patient-Level Odds Ratio: {odds_clean:.3f}")
    print(f"  Clean Fisher's Exact p-value : {p_fisher_clean:.5e} (p = {p_fisher_clean:.6f})")
    print(f"  Clean Chi-Square p-value     : {p_chi2_clean:.5e} (p = {p_chi2_clean:.6f})")

    # Comparison to pre-cleaned test set
    print("\n--- COMPARISON TO PRE-CLEANED DATASET ---")
    print(f"Metric                       | Pre-Clean (n=92 pats) | Clean V2 (n={len(patient_data)} pats) | Delta")
    print("-" * 75)
    print(f"FP Patients with Calipers    | 66.67% (20/30)        | {fp_rate_clean*100:.2f}% ({fp_pat_with_cal}/{fp_pat_with_cal+fp_pat_no_cal})        | {fp_rate_clean*100 - 66.67:+.2f}%")
    print(f"TN Patients with Calipers    | 27.42% (17/62)        | {tn_rate_clean*100:.2f}% ({tn_pat_with_cal}/{tn_pat_with_cal+tn_pat_no_cal})        | {tn_rate_clean*100 - 27.42:+.2f}%")
    print(f"Patient-Level Odds Ratio     | 5.294                 | {odds_clean:.3f}                 | {odds_clean - 5.294:+.3f}")
    print(f"Fisher's Exact p-value       | 0.000555              | {p_fisher_clean:.6f}              | {'Strengthened' if p_fisher_clean < 0.000555 else 'Slight shift'}")

    # Save to JSON
    out_json = BASE_DIR / "experiments" / "v2_contamination_cleaned" / "reports" / "clean_caliper_stats.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "total_clean_nuf_scans": len(clean_nuf_lines),
            "total_clean_nuf_patients": len(patient_data),
            "clean_fps": total_fps,
            "clean_tns": total_tns,
            "contingency_table": {
                "fp_with_cal": fp_pat_with_cal,
                "fp_no_cal": fp_pat_no_cal,
                "tn_with_cal": tn_pat_with_cal,
                "tn_no_cal": tn_pat_no_cal
            },
            "fp_caliper_rate": fp_rate_clean,
            "tn_caliper_rate": tn_rate_clean,
            "odds_ratio": odds_clean,
            "fisher_p_value": p_fisher_clean,
            "chi2_p_value": p_chi2_clean
        }, f, indent=2)
    print(f"\nSaved statistical summary to: {out_json}")

if __name__ == "__main__":
    main()
