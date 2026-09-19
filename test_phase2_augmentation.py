"""
test_phase2_augmentation.py
Verification and Visualization Script for UteroScan Phase 2 (Data Augmentation Setup)

Exit Checklist Requirements:
1. Augmentation config shown/printed (which transforms, what parameters).
2. Confirmed train vs. val/test loaders are configured differently (augmented vs. not).
3. Explicitly verify vertical flip is NOT included anywhere in the pipeline.
4. One batch visualized or logged to confirm augmentations are visibly applying (not silently no-op'd).
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Add companion codebase to path
SYS_PATH = Path("d:/Project/UteroScan/DCGAN&MobileNetV2/DCGAN&MobileNetV2")
sys.path.insert(0, str(SYS_PATH))

from utils.dataloader import DataGenerator, detection_collate
from utils.augmentation import (
    get_ultrasound_train_transforms,
    apply_augmentation,
    ROTATION_DEGREES,
    HFLIP_PROB,
    ZOOM_SCALE,
    SHEAR_DEGREES
)
from utils.preprocessing import preprocess_ultrasound

BASE_DIR = Path("d:/Project/UteroScan")
REPORTS_DIR = BASE_DIR / "reports" / "augmentation_samples"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR = Path(r"C:\Users\ashis\.gemini\antigravity-ide\brain\af0e99e6-8ff9-4623-ad48-86540f6228c9")

def test_1_print_config_and_verify_no_vflip():
    print("=" * 80)
    print("TEST 1: AUGMENTATION CONFIGURATION & VERTICAL FLIP AUDIT")
    print("=" * 80)

    aug = get_ultrasound_train_transforms()
    print("Augmentation Pipeline Components:")
    for idx, t in enumerate(aug.transforms, 1):
        print(f"  [{idx}] {t.__class__.__name__}: {t.__dict__}")

    print("\nParameters Summary:")
    print(f"  - Random Rotation       : [-{ROTATION_DEGREES} deg, +{ROTATION_DEGREES} deg] (Bilinear, fill=0)")
    print(f"  - Random Horizontal Flip: p = {HFLIP_PROB} (Left-Right flip)")
    print(f"  - Random Zoom / Scale   : {ZOOM_SCALE} (90% to 110% scale)")
    print(f"  - Random Shear          : [{SHEAR_DEGREES[0]} deg, {SHEAR_DEGREES[1]} deg] (Bilinear, fill=0)")

    # Audit for vertical flip
    has_vflip = any(isinstance(t, transforms.RandomVerticalFlip) for t in aug.transforms)
    print(f"\nAudit Check: transforms.RandomVerticalFlip present? -> {has_vflip}")
    assert not has_vflip, "CRITICAL ERROR: RandomVerticalFlip found in pipeline!"

    # Physical test with asymmetric upward arrow
    arrow_img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.arrowedLine(arrow_img, (112, 180), (112, 40), (255, 255, 255), 6, tipLength=0.3)
    
    # Run 100 stochastic augmentations and verify tip remains in upper hemisphere
    pil_arrow = Image.fromarray(arrow_img)
    flips_down = 0
    for _ in range(100):
        aug_arrow = np.array(aug(pil_arrow))
        # Upper hemisphere brightness vs lower hemisphere brightness
        upper_sum = np.sum(aug_arrow[:112, :])
        lower_sum = np.sum(aug_arrow[112:, :])
        # Upward arrow must have more pixels in upper half (tip) than lower half
        # Rotation is only +-15 deg, so tip never rotates into lower half unless flipped vertically
        if lower_sum > upper_sum * 1.5:
            flips_down += 1

    print(f"Stochastic Orientation Test (100 trials): Inverted downward frames -> {flips_down} / 100")
    assert flips_down == 0, "CRITICAL ERROR: Vertical inversion detected in stochastic test!"
    print(">>> PASS: Vertical flip is strictly absent from the pipeline. <<<\n")

def test_2_train_vs_val_loader_configuration():
    print("=" * 80)
    print("TEST 2: TRAIN VS. VAL/TEST LOADER CONFIGURATION AUDIT")
    print("=" * 80)

    train_file = SYS_PATH / "cls_train.txt"
    test_file = SYS_PATH / "cls_test.txt"

    with open(train_file, "r", encoding="utf-8") as f:
        train_lines = [l.strip() for l in f if l.strip()]
    with open(test_file, "r", encoding="utf-8") as f:
        val_lines = [l.strip() for l in f if l.strip()]

    train_ds = DataGenerator(train_lines, input_shape=(224, 224), random=True)
    val_ds = DataGenerator(val_lines, input_shape=(224, 224), random=False)

    print(f"Train Dataset: random = {train_ds.random}, train_transforms = {train_ds.train_transforms.__class__.__name__}")
    print(f"Val Dataset  : random = {val_ds.random}, train_transforms = {val_ds.train_transforms}")

    assert train_ds.random is True, "Train dataset should have random=True"
    assert train_ds.train_transforms is not None, "Train dataset must have train_transforms active"
    assert val_ds.random is False, "Val dataset should have random=False"
    assert val_ds.train_transforms is None, "Val dataset train_transforms must be None"

    # Verify Validation Determinism: fetching item 0 twice must yield exact bitwise identical tensors
    val_tensor_1, y1 = val_ds[0]
    val_tensor_2, y2 = val_ds[0]
    diff_val = np.max(np.abs(val_tensor_1 - val_tensor_2))
    print(f"Validation Determinism Check (Item 0 pass 1 vs pass 2 max diff): {diff_val}")
    assert diff_val == 0.0, "CRITICAL ERROR: Validation loader is not deterministic!"

    # Verify Training Stochasticity: fetching item 0 across 5 passes must yield variations
    train_diffs = []
    base_train, _ = train_ds[0]
    for _ in range(5):
        t_sample, _ = train_ds[0]
        train_diffs.append(np.mean(np.abs(t_sample - base_train)))

    avg_train_diff = np.mean(train_diffs)
    print(f"Training Stochasticity Check (Item 0 variations over 5 calls): mean diff = {avg_train_diff:.4f}")
    assert avg_train_diff > 0.01, "CRITICAL ERROR: Training loader augmentations appear to be silently no-op'd!"
    print(">>> PASS: Train and Val loaders are strictly separated (Augmented vs. Deterministic). <<<\n")

def test_3_visualize_sample_batch():
    print("=" * 80)
    print("TEST 3: VISUALIZING AUGMENTATION SAMPLES ON REAL ULTRASOUND IMAGES")
    print("=" * 80)

    train_file = SYS_PATH / "cls_train.txt"
    with open(train_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    # Pick 4 representative samples (2 UF and 2 NUF)
    uf_lines = [l for l in lines if l.startswith("0;")]
    nuf_lines = [l for l in lines if l.startswith("1;")]

    selected_lines = uf_lines[:2] + nuf_lines[:2]
    aug = get_ultrasound_train_transforms()

    montages = []
    for idx, line in enumerate(selected_lines, 1):
        label_id, path_str = line.split(";")
        label_name = "UF" if label_id == "0" else "NUF"
        p_str = path_str.split()[0]
        
        # Resolve path
        ds_path = BASE_DIR / "datasets" / "datasets" / Path(*Path(p_str).parts[1:])
        bgr = cv2.imread(str(ds_path))
        if bgr is None:
            continue
            
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_ultrasound(rgb, target_size=(224, 224))
        
        # Generate 3 augmented variations
        variations = [preprocessed]
        for _ in range(3):
            aug_v = apply_augmentation(preprocessed, aug)
            variations.append(aug_v)

        # Annotate and stack
        vis_row = []
        titles = ["Base (Preprocessed)", "Aug 1 (Rot/Scale)", "Aug 2 (H-Flip/Shear)", "Aug 3 (Combined)"]
        for t_title, v_img in zip(titles, variations):
            v_bgr = cv2.cvtColor(v_img, cv2.COLOR_RGB2BGR)
            # Add small label overlay
            cv2.putText(v_bgr, t_title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
            vis_row.append(v_bgr)

        divider = np.full((224, 3, 3), 80, dtype=np.uint8)
        row_montage = np.hstack([vis_row[0], divider, vis_row[1], divider, vis_row[2], divider, vis_row[3]])
        
        out_name = f"augmentation_sample_{idx:02d}_{label_name}_{Path(p_str).stem}.png"
        out_path = REPORTS_DIR / out_name
        art_path = ART_DIR / out_name
        cv2.imwrite(str(out_path), row_montage)
        cv2.imwrite(str(art_path), row_montage)
        print(f"  Generated Sample {idx}: {out_name} -> {out_path}")
        montages.append(out_name)

    print(">>> PASS: Augmentation visualization complete and saved to reports and artifact dir. <<<\n")

def test_4_dataloader_batch_execution():
    print("=" * 80)
    print("TEST 4: FULL DATALOADER BATCH EXECUTION TEST")
    print("=" * 80)

    train_file = SYS_PATH / "cls_train.txt"
    test_file = SYS_PATH / "cls_test.txt"

    with open(train_file, "r", encoding="utf-8") as f:
        train_lines = [l.strip() for l in f if l.strip()]
    with open(test_file, "r", encoding="utf-8") as f:
        val_lines = [l.strip() for l in f if l.strip()]

    train_ds = DataGenerator(train_lines, input_shape=(224, 224), random=True)
    val_ds = DataGenerator(val_lines, input_shape=(224, 224), random=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=detection_collate)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=detection_collate)

    # Fetch one train batch
    train_imgs, train_targets = next(iter(train_loader))
    print(f"Train Batch: images shape = {train_imgs.shape}, targets shape = {train_targets.shape}")
    print(f"  Image value range: min = {train_imgs.min():.2f}, max = {train_imgs.max():.2f}, mean = {train_imgs.mean():.2f}")
    print(f"  Target classes in batch: {train_targets.tolist()}")

    # Fetch one val batch
    val_imgs, val_targets = next(iter(val_loader))
    print(f"Val Batch  : images shape = {val_imgs.shape}, targets shape = {val_targets.shape}")
    print(f"  Image value range: min = {val_imgs.min():.2f}, max = {val_imgs.max():.2f}, mean = {val_imgs.mean():.2f}")
    print(f"  Target classes in batch: {val_targets.tolist()}")

    assert train_imgs.shape == (16, 3, 224, 224), f"Unexpected train batch shape: {train_imgs.shape}"
    assert val_imgs.shape == (16, 3, 224, 224), f"Unexpected val batch shape: {val_imgs.shape}"
    print(">>> PASS: DataLoader batch execution verified with 0 errors. <<<\n")

if __name__ == "__main__":
    test_1_print_config_and_verify_no_vflip()
    test_2_train_vs_val_loader_configuration()
    test_3_visualize_sample_batch()
    test_4_dataloader_batch_execution()
    print("=" * 80)
    print("ALL PHASE 2 AUGMENTATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)
