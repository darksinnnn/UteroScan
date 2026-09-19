"""
preprocess.py
UteroScan - Phase A: Ultrasound Image Preprocessing Pipeline

Implements:
    1. Resizing to 224x224 (CNN input requirement)
    2. Speckle noise reduction (Median filter as default, with optional
       Gaussian blur or anisotropic diffusion)
    3. CLAHE contrast enhancement

Usage:
    Single image:
        img = preprocess_image("path/to/image.jpg")

    Batch process an entire dataset directory (mirrors folder structure):
        preprocess_dataset(
            input_dir="data/raw/train",
            output_dir="data/processed/train"
        )
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Literal


# ----------------------------------------------------------------------
# Config defaults — tune these if needed without touching function logic
# ----------------------------------------------------------------------
TARGET_SIZE = (224, 224)          # (width, height) — matches ResNet50/VGG16 input
MEDIAN_KERNEL_SIZE = 5            # must be odd
GAUSSIAN_KERNEL_SIZE = (5, 5)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ----------------------------------------------------------------------
# Step 1: Resizing
# ----------------------------------------------------------------------
def resize_image(image: np.ndarray, target_size: tuple = TARGET_SIZE) -> np.ndarray:
    """Resize image to target_size using area interpolation (best for downscaling
    ultrasound frames without introducing aliasing artifacts)."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------
# Step 2: Speckle Noise Reduction
# ----------------------------------------------------------------------
def reduce_speckle_noise(
    image: np.ndarray,
    method: Literal["median", "gaussian", "anisotropic"] = "median"
) -> np.ndarray:
    """
    Reduce acoustic speckle noise inherent to ultrasound images while
    attempting to preserve myometrial tissue edges.

    - "median": Median blur. Good general-purpose speckle suppressor,
      preserves edges better than Gaussian for salt-and-pepper-like noise.
    - "gaussian": Simple Gaussian blur. Faster, but smooths edges more
      aggressively — use only if median is too slow for your pipeline.
    - "anisotropic": Anisotropic diffusion (Perona-Malik). Best edge
      preservation but slower; falls back to OpenCV's edge-preserving
      filter as a practical approximation.
    """
    if method == "median":
        return cv2.medianBlur(image, MEDIAN_KERNEL_SIZE)

    elif method == "gaussian":
        return cv2.GaussianBlur(image, GAUSSIAN_KERNEL_SIZE, sigmaX=0)

    elif method == "anisotropic":
        # cv2.ximgproc has true anisotropic diffusion in opencv-contrib-python.
        # If unavailable, fall back to edge-preserving filter (still respects
        # myometrial boundaries much better than a plain blur).
        try:
            return cv2.ximgproc.anisotropicDiffusion(
                image, alpha=0.1, K=15, niters=10
            )
        except AttributeError:
            return cv2.edgePreservingFilter(image, flags=cv2.RECURS_FILTER,
                                              sigma_s=40, sigma_r=0.2)
    else:
        raise ValueError(f"Unknown speckle reduction method: {method}")


# ----------------------------------------------------------------------
# Step 3: CLAHE Contrast Enhancement
# ----------------------------------------------------------------------
def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization to enhance
    tissue boundary and shadowing contrast.

    Works on grayscale. If a 3-channel image is passed in (e.g. JPEG loaded
    with color channels but visually grayscale), CLAHE is applied on the
    L-channel in LAB space to avoid introducing color artifacts.
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)

    if len(image.shape) == 2:
        # Already single-channel grayscale
        return clahe.apply(image)

    # 3-channel image — apply CLAHE only on lightness channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ----------------------------------------------------------------------
# Full Pipeline — Single Image
# ----------------------------------------------------------------------
def preprocess_image(
    image_path: str,
    target_size: tuple = TARGET_SIZE,
    speckle_method: str = "median",
    grayscale: bool = True
) -> np.ndarray:
    """
    Run the full Phase A pipeline on a single image file:
    load -> resize -> speckle reduction -> CLAHE.

    Returns a numpy array ready to be fed into the CNN preprocessing
    step (e.g. ResNet50 preprocess_input) in train.py / inference.py.
    """
    read_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(image_path, read_flag)

    if image is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")

    image = resize_image(image, target_size)
    image = reduce_speckle_noise(image, method=speckle_method)
    image = apply_clahe(image)

    return image


# ----------------------------------------------------------------------
# Full Pipeline — Batch Dataset Processing
# ----------------------------------------------------------------------
def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    target_size: tuple = TARGET_SIZE,
    speckle_method: str = "median",
    grayscale: bool = True,
    overwrite: bool = False
) -> None:
    """
    Walks input_dir (expects class-subfolder structure, e.g. UF/ and NUF/),
    applies the full preprocessing pipeline to every valid image, and saves
    results to output_dir with the same relative folder structure.

    Example:
        preprocess_dataset("data/raw/train", "data/processed/train")
        -> data/processed/train/UF/*.jpg
        -> data/processed/train/NUF/*.jpg
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    image_paths = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in VALID_EXTENSIONS
    ]

    if not image_paths:
        print(f"[WARNING] No valid images found under {input_dir}")
        return

    print(f"[INFO] Found {len(image_paths)} images under {input_dir}. Starting preprocessing...")

    success_count = 0
    fail_count = 0

    for img_path in image_paths:
        relative_path = img_path.relative_to(input_dir)
        out_path = output_dir / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            continue

        try:
            processed = preprocess_image(
                str(img_path),
                target_size=target_size,
                speckle_method=speckle_method,
                grayscale=grayscale
            )
            cv2.imwrite(str(out_path), processed)
            success_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {img_path}: {e}")
            fail_count += 1

    print(f"[DONE] Processed: {success_count} | Failed: {fail_count} | "
          f"Skipped (already exist): {len(image_paths) - success_count - fail_count}")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UteroScan Phase A - Image Preprocessing")
    parser.add_argument("--input_dir", type=str, required=True,
                         help="Path to raw dataset directory (e.g. data/raw/train)")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Path to save processed images (e.g. data/processed/train)")
    parser.add_argument("--speckle_method", type=str, default="median",
                         choices=["median", "gaussian", "anisotropic"])
    parser.add_argument("--color", action="store_true",
                         help="Process as color images instead of grayscale")
    parser.add_argument("--overwrite", action="store_true",
                         help="Overwrite existing processed images")

    args = parser.parse_args()

    preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        speckle_method=args.speckle_method,
        grayscale=not args.color,
        overwrite=args.overwrite
    )
