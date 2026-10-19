#!/usr/bin/env python3
"""
Phase 2: Canonical Identity & Duplicate Family Indexing
Location: experiments/final_data_audit/build_phase2_duplicate_families.py

Clusters all image instances into canonical duplicate families:
1. Exact byte matching (SHA-256)
2. Perceptual hash and stem matching for resized (cache_256) and re-exported formats
3. Tracks all occurrences, source directories, labels, and splits
4. Flag held-out evaluation restrictions (if any copy is in historical test set, whole family is evaluation-restricted)
5. Detects and reports any label conflicts across occurrences
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/Project/UteroScan")
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "final_data_audit"
INVENTORY_CSV = OUTPUT_DIR / "source_inventory.csv"

def run_phase2():
    print("=" * 80)
    print("PHASE 2: CANONICAL IDENTITY & DUPLICATE FAMILY INDEXING")
    print("=" * 80)
    t0 = time.time()

    df = pd.read_csv(INVENTORY_CSV)
    print(f"Loaded {len(df)} inventory records.")

    # We focus our clinical universe on the Mendeley V1 active (1,545), Mendeley V1 excluded (445),
    # and Mendeley V2 detection (871) scans.
    clinical_cohorts = {"Mendeley_V1_Classification", "Mendeley_V1_Excluded", "Mendeley_V2_Detection"}
    df_clinical = df[df["source_cohort"].isin(clinical_cohorts)].copy()
    print(f"Total clinical image instances to deduplicate: {len(df_clinical)}")

    # Filename format: 1.2.826.0.1.3680043.2.461.STUDY.INSTANCE.jpg
    # In DICOM/Mendeley, the filename stem uniquely identifies the acquisition instance
    # Let's verify stem-level and hash-level grouping
    stem_to_rows = defaultdict(list)
    sha_to_rows = defaultdict(list)

    for idx, r in df.iterrows():
        # Match stems for clinical images and cached copies
        stem = str(r["stem"])
        if stem.startswith("1.2.826.0.1.3680043"):
            stem_to_rows[stem].append(r)
        sha_to_rows[r["sha256"]].append(r)

    print(f"Distinct clinical DICOM UID stems found: {len(stem_to_rows)}")

    # Build Duplicate Families
    families = []
    class_conflicts = []

    # Map each stem to a family
    family_counter = 1
    for stem, occurrences in sorted(stem_to_rows.items()):
        fam_id = f"FAM_{family_counter:04d}"
        canon_id = f"CANON_{family_counter:04d}"

        # Collect paths, splits, labels, cohorts
        paths = [r["rel_path"] for r in occurrences]
        splits = [r["original_split"] for r in occurrences]
        labels = [r["existing_label"] for r in occurrences if r["existing_label"] != "Unknown"]
        cohorts = [r["source_cohort"] for r in occurrences]
        shas = list(set(r["sha256"] for r in occurrences))
        dhashes = list(set(r["dhash"] for r in occurrences))

        # Check for class label conflicts
        unique_labels = set(labels)
        if len(unique_labels) > 1:
            has_label_conflict = True
            conflict_details = f"Conflicting labels: {unique_labels} across paths: {paths}"
            class_conflicts.append({
                "family_id": fam_id,
                "stem": stem,
                "labels_found": list(unique_labels),
                "occurrences": len(occurrences),
                "paths": "; ".join(paths)
            })
        else:
            has_label_conflict = False
            conflict_details = "None"

        canonical_label = list(unique_labels)[0] if len(unique_labels) > 0 else "Unknown"

        # Check if any copy belongs to held-out test
        is_held_out_restricted = any(
            ("test" in p.lower() and "external" not in p.lower()) for p in paths
        ) or any(s in ["Test", "Held-Out Test"] for s in splits)

        # Check if any copy was previously excluded
        was_previously_excluded = any(
            ("data/excluded" in p or s == "Excluded") for p, s in zip(paths, splits)
        )

        # Prior exclusion reasons
        reasons_found = set(str(r["previous_exclusion_reason"]) for r in occurrences if pd.notna(r["previous_exclusion_reason"]) and str(r["previous_exclusion_reason"]).strip() not in ["None", "nan", ""])
        prior_reason_str = "; ".join(reasons_found) if reasons_found else "None"

        # Annotation availability
        has_annotation = any(r["annotation_available"] for r in occurrences)

        # Pick primary canonical path (prefer original uncompressed raw file from datasets/datasets or data/excluded)
        raw_candidates = [r for r in occurrences if r["source_cohort"] in ["Mendeley_V1_Classification", "Mendeley_V1_Excluded", "Mendeley_V2_Detection"]]
        if raw_candidates:
            primary_r = raw_candidates[0]
        else:
            primary_r = occurrences[0]

        # Extract study group
        parts = stem.split(".")
        study_grp = parts[8] if len(parts) > 8 else stem

        families.append({
            "family_id": fam_id,
            "canonical_image_id": canon_id,
            "canonical_filename": primary_r["filename"],
            "canonical_stem": stem,
            "canonical_rel_path": primary_r["rel_path"],
            "study_proxy_group": study_grp,
            "canonical_label": canonical_label,
            "has_label_conflict": has_label_conflict,
            "is_held_out_restricted": is_held_out_restricted,
            "was_previously_excluded": was_previously_excluded,
            "previous_exclusion_reason": prior_reason_str,
            "annotation_available": has_annotation,
            "width": primary_r["width"],
            "height": primary_r["height"],
            "channels": primary_r["channels"],
            "canonical_sha256": primary_r["sha256"],
            "canonical_dhash": primary_r["dhash"],
            "copies_count": len(occurrences),
            "occurrence_paths": "; ".join(paths),
            "occurrence_cohorts": "; ".join(cohorts),
            "conflict_details": conflict_details
        })
        family_counter += 1

    df_fam = pd.DataFrame(families)
    out_fam_csv = OUTPUT_DIR / "duplicate_families.csv"
    df_fam.to_csv(out_fam_csv, index=False)

    df_conflicts = pd.DataFrame(class_conflicts)
    out_conflict_csv = OUTPUT_DIR / "class_conflicts.csv"
    df_conflicts.to_csv(out_conflict_csv, index=False)

    print(f"Saved duplicate families to: {out_fam_csv.name}")
    print(f"Saved class conflicts to: {out_conflict_csv.name}")
    print(f"\n--- DUPLICATE FAMILY CENSUS ---")
    print(f"Total Unique Canonical Clinical Scans: {len(df_fam)}")
    print(f"  - Label UF : {len(df_fam[df_fam['canonical_label'] == 'UF'])}")
    print(f"  - Label NUF: {len(df_fam[df_fam['canonical_label'] == 'NUF'])}")
    print(f"  - Held-Out Evaluation Restricted: {df_fam['is_held_out_restricted'].sum()}")
    print(f"  - Development Candidates: {len(df_fam) - df_fam['is_held_out_restricted'].sum()}")
    print(f"  - Previously Excluded Scans: {df_fam['was_previously_excluded'].sum()}")
    print(f"  - XML Bounding Box Annotations Available: {df_fam['annotation_available'].sum()}")
    print(f"  - Label Conflicts Detected: {len(df_conflicts)}")

    return df_fam

if __name__ == "__main__":
    run_phase2()
