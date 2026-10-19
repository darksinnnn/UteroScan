#!/usr/bin/env python3
"""
Phase 1 & 2: Fast & Comprehensive Source Image Inventory & Deduplication
Location: experiments/final_data_audit/build_phase1_inventory.py

Recursively inventories every image file in the project repository:
- Prunes out .git, system, and cache directories in-place for fast traversal
- Computes SHA-256 for exact byte-level deduplication
- Computes difference hash (dhash) for perceptual matching
- Extracts dimensions, channels, decoding validity
- Cross-references with Mendeley V1, Mendeley V2 XMLs, exclusion logs, and historical manifests
"""

import os
import sys
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

PROJECT_ROOT = Path("d:/Project/UteroScan")
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "final_data_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXCLUDED_DIR_NAMES = {".git", ".gemini", ".system_generated", "node_modules", "__pycache__", ".agents"}

EXCLUSION_LOG_CSV = PROJECT_ROOT / "data" / "excluded" / "exclusion_log.csv"
AUTH_MANIFEST_CSV = PROJECT_ROOT / "data" / "authoritative_manifest.csv"
CLEAN_V5_MANIFEST = PROJECT_ROOT / "experiments" / "v4_controlled" / "data" / "clean_v5" / "clean_manifest.csv"
V2_ANNOTATIONS_DIR = PROJECT_ROOT / "datasets" / "new dataset" / "Annotations" / "Annotations"

def compute_dhash(img_gray, hash_size=8):
    """Computes difference hash (dhash) as a 64-bit hex string."""
    resized = cv2.resize(img_gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    decimal_val = 0
    hex_str = []
    for i, v in enumerate(diff.flatten()):
        if v:
            decimal_val += 2 ** (i % 4)
        if (i % 4) == 3:
            hex_str.append(hex(decimal_val)[2:])
            decimal_val = 0
    return "".join(hex_str)

def run_inventory():
    print("=" * 80, flush=True)
    print("PHASE 1: FAST RECURSIVE SOURCE INVENTORY ACROSS REPOSITORY", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()

    # Preload cross-referencing maps
    excl_reasons = {}
    if EXCLUSION_LOG_CSV.exists():
        df_excl = pd.read_csv(EXCLUSION_LOG_CSV)
        for _, r in df_excl.iterrows():
            excl_reasons[r["filename"]] = str(r["reason"])

    auth_splits = {}
    auth_classes = {}
    if AUTH_MANIFEST_CSV.exists():
        df_auth = pd.read_csv(AUTH_MANIFEST_CSV)
        for _, r in df_auth.iterrows():
            auth_splits[r["filename"]] = str(r["split"])
            auth_classes[r["filename"]] = str(r["class_label"])

    v2_xml_stems = set(p.stem for p in V2_ANNOTATIONS_DIR.glob("*.xml")) if V2_ANNOTATIONS_DIR.exists() else set()
    print(f"Preloaded {len(excl_reasons)} exclusion records, {len(auth_splits)} auth records, and {len(v2_xml_stems)} V2 XML annotations.", flush=True)

    # 1. Discover all image files using os.walk with in-place directory pruning
    discovered_paths = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Prune hidden or system directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES and not d.startswith(".")]

        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXTS:
                full_p = Path(root) / f
                discovered_paths.append(full_p)

    print(f"Discovered {len(discovered_paths)} image files in repository. Beginning metadata and hash extraction...", flush=True)

    records = []
    t_extract = time.time()

    for idx, p in enumerate(discovered_paths):
        if (idx + 1) % 500 == 0 or idx == len(discovered_paths) - 1:
            dur = time.time() - t_extract
            print(f"[{idx+1}/{len(discovered_paths)}] Indexed ({dur:.1f}s)...", flush=True)

        rel_path = p.relative_to(PROJECT_ROOT)
        parts = rel_path.parts
        fn = p.name
        stem = p.stem
        ext = p.suffix.lower()
        file_size = p.stat().st_size

        # Compute SHA-256
        with open(p, "rb") as f_in:
            file_bytes = f_in.read()
            sha256 = hashlib.sha256(file_bytes).hexdigest()

        # Attempt decoding
        img = cv2.imread(str(p))
        if img is None:
            decoding_status = "CORRUPT"
            h, w, c = 0, 0, 0
            dhash = "CORRUPT"
        else:
            decoding_status = "VALID"
            h, w = img.shape[:2]
            c = img.shape[2] if len(img.shape) > 2 else 1
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if c > 1 else img
            dhash = compute_dhash(gray)

        top_dir = parts[0]
        source_dir = str(rel_path.parent).replace("\\", "/")

        # Infer class label
        if "UF" in parts:
            existing_label = "UF"
        elif "NUF" in parts:
            existing_label = "NUF"
        elif fn in auth_classes:
            existing_label = auth_classes[fn]
        else:
            existing_label = "Unknown"

        # Infer split
        if "data/excluded" in source_dir:
            original_split = "Excluded"
        elif "train" in [pt.lower() for pt in parts]:
            original_split = "Train"
        elif "test" in [pt.lower() for pt in parts]:
            original_split = "Test"
        elif fn in auth_splits:
            original_split = auth_splits[fn]
        else:
            original_split = "Unassigned"

        prev_excl_reason = excl_reasons.get(fn, "None")
        has_annotation = (stem in v2_xml_stems)

        if top_dir == "datasets":
            if "new dataset" in source_dir:
                source_cohort = "Mendeley_V2_Detection"
            else:
                source_cohort = "Mendeley_V1_Classification"
        elif top_dir == "data" and "excluded" in source_dir:
            source_cohort = "Mendeley_V1_Excluded"
        elif top_dir == "External Dataset":
            source_cohort = "Roboflow_VOC_External"
        elif "experiments" in parts and "cache_256" in source_dir:
            source_cohort = "Cache_256_Preprocessed"
        elif "reports" in parts:
            source_cohort = "Report_Artifact_Images"
        else:
            source_cohort = "Other_Workspace_Images"

        records.append({
            "rel_path": str(rel_path).replace("\\", "/"),
            "filename": fn,
            "stem": stem,
            "extension": ext,
            "file_size_bytes": file_size,
            "width": w,
            "height": h,
            "channels": c,
            "decoding_status": decoding_status,
            "source_cohort": source_cohort,
            "source_directory": source_dir,
            "existing_label": existing_label,
            "original_split": original_split,
            "previous_exclusion_reason": prev_excl_reason,
            "annotation_available": has_annotation,
            "sha256": sha256,
            "dhash": dhash
        })

    df = pd.DataFrame(records)
    out_csv = OUTPUT_DIR / "source_inventory.csv"
    df.to_csv(out_csv, index=False)
    elapsed = time.time() - t0

    print("=" * 80, flush=True)
    print(f"PHASE 1 INVENTORY COMPLETE in {elapsed:.1f}s", flush=True)
    print("=" * 80, flush=True)
    print(f"Total files indexed: {len(df)}", flush=True)
    print(f"Readable images: {len(df[df['decoding_status'] == 'VALID'])}", flush=True)
    print(f"Corrupt images: {len(df[df['decoding_status'] == 'CORRUPT'])}", flush=True)
    print(f"Unique SHA-256 byte hashes: {df['sha256'].nunique()}", flush=True)
    print(f"Unique dhash perceptual hashes: {df['dhash'].nunique()}", flush=True)
    print(f"Saved inventory to: {out_csv.name}", flush=True)

    print("\n--- Source Cohort Breakdown ---", flush=True)
    print(df["source_cohort"].value_counts(), flush=True)

    return df

if __name__ == "__main__":
    run_inventory()
