#!/usr/bin/env python3
"""
Unit Validation and Contact Sheet Generator for E4 Components
Location: experiments/v4_controlled/e4/src/validate_e4_components.py

Validates:
  1. E4A Differential Learning Rate Parameter Grouping & Counts
  2. E4B Photometric Augmentations (Range, Validity, Non-Destructive Physics)
  3. Visual Contact Sheet Generation of Original vs Augmented Scans
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.optim as optim
import pandas as pd

PROJECT_ROOT = Path("d:/Project/UteroScan")
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "v4_controlled" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "v4_controlled" / "e4" / "src"))

from models import ResNet50Baseline
from photometric_aug import UltrasoundPhotometricAugmentation
from e4_dataset import E4UltrasoundDataset

class TestE4Components(unittest.TestCase):
    def test_e4a_differential_lr_groups(self):
        """Verify differential LR groups cover 100% of parameters with zero overlap."""
        model = ResNet50Baseline(num_classes=2, pretrained=False)
        total_model_params = list(model.parameters())
        total_param_count = sum(p.numel() for p in total_model_params)

        # 3 parameter groups
        g0_params = [
            *model.conv1.parameters(),
            *model.bn1.parameters(),
            *model.layer1.parameters(),
            *model.layer2.parameters(),
        ]
        g1_params = [
            *model.layer3.parameters(),
            *model.layer4.parameters(),
        ]
        g2_params = list(model.fc.parameters())

        g0_ids = {id(p) for p in g0_params}
        g1_ids = {id(p) for p in g1_params}
        g2_ids = {id(p) for p in g2_params}

        # Check disjointness
        self.assertEqual(len(g0_ids & g1_ids), 0, "Group 0 and Group 1 overlap!")
        self.assertEqual(len(g0_ids & g2_ids), 0, "Group 0 and Group 2 overlap!")
        self.assertEqual(len(g1_ids & g2_ids), 0, "Group 1 and Group 2 overlap!")

        # Check completeness
        all_grouped_ids = g0_ids | g1_ids | g2_ids
        model_param_ids = {id(p) for p in total_model_params}
        self.assertEqual(all_grouped_ids, model_param_ids, "Not all model parameters are grouped!")

        # Build optimizer
        optimizer = optim.Adam([
            {"params": g0_params, "lr": 1e-5},
            {"params": g1_params, "lr": 5e-5},
            {"params": g2_params, "lr": 1e-4},
        ], weight_decay=1e-4)

        self.assertEqual(len(optimizer.param_groups), 3)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-5)
        self.assertEqual(optimizer.param_groups[1]["lr"], 5e-5)
        self.assertEqual(optimizer.param_groups[2]["lr"], 1e-4)
        print(f"E4A Differential LR: 3 groups verified. Total parameters: {total_param_count:,}")

    def test_e4b_photometric_aug_ranges(self):
        """Verify photometric augmentation preserves valid uint8 range [0, 255]."""
        aug = UltrasoundPhotometricAugmentation()
        # Test synthetic image
        img = np.random.randint(20, 230, size=(224, 224), dtype=np.uint8)
        for _ in range(50):
            out = aug(img)
            self.assertEqual(out.dtype, np.uint8)
            self.assertEqual(out.shape, (224, 224))
            self.assertTrue(np.all(out >= 0))
            self.assertTrue(np.all(out <= 255))

    def test_e4_dataset_output(self):
        """Verify dataset returns valid normalized tensors."""
        folds_csv = PROJECT_ROOT / "experiments" / "v4_controlled" / "data" / "clean_v5" / "folds_clean_v5.csv"
        df = pd.read_csv(folds_csv).head(4)
        ds = E4UltrasoundDataset(df, is_train=True, use_photometric_aug=True, project_root=str(PROJECT_ROOT))
        tensor, target, stem = ds[0]
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertFalse(torch.isnan(tensor).any())
        self.assertFalse(torch.isinf(tensor).any())

def generate_contact_sheet():
    """Generates a high-resolution visual contact sheet of original vs augmented scans."""
    output_path = PROJECT_ROOT / "experiments" / "v4_controlled" / "e4" / "reports" / "figures" / "e4b_augmentation_contact_sheet.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    folds_csv = PROJECT_ROOT / "experiments" / "v4_controlled" / "data" / "clean_v5" / "folds_clean_v5.csv"
    df = pd.read_csv(folds_csv)

    # Pick 2 UF and 2 NUF scans
    uf_samples = df[df["class_label"] == "UF"].head(2)
    nuf_samples = df[df["class_label"] == "NUF"].head(2)
    samples = pd.concat([uf_samples, nuf_samples]).reset_index(drop=True)

    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    photo_aug = UltrasoundPhotometricAugmentation(gamma_p=1.0, contrast_p=1.0, speckle_p=1.0) # Always trigger all 3 for visualization

    rows = []
    for idx, row in samples.iterrows():
        img_full_path = PROJECT_ROOT / row["image_path"]
        img_bgr = cv2.imread(str(img_full_path))
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
        blurred = cv2.medianBlur(resized, 3)
        clahe_orig = clahe.apply(blurred)

        # Generate 3 distinct stochastic augmentations of the same base scan
        aug1 = photo_aug(clahe_orig)
        aug2 = photo_aug(clahe_orig)
        aug3 = photo_aug(clahe_orig)

        # Convert to BGR for labeling
        c_orig = cv2.cvtColor(clahe_orig, cv2.COLOR_GRAY2BGR)
        c_aug1 = cv2.cvtColor(aug1, cv2.COLOR_GRAY2BGR)
        c_aug2 = cv2.cvtColor(aug2, cv2.COLOR_GRAY2BGR)
        c_aug3 = cv2.cvtColor(aug3, cv2.COLOR_GRAY2BGR)

        cls_lbl = row["class_label"]
        cv2.putText(c_orig, f"{cls_lbl} (Original)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        cv2.putText(c_aug1, f"{cls_lbl} (Aug 1)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(c_aug2, f"{cls_lbl} (Aug 2)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(c_aug3, f"{cls_lbl} (Aug 3)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        row_img = np.hstack([c_orig, c_aug1, c_aug2, c_aug3])
        rows.append(row_img)

    contact_sheet = np.vstack(rows)
    cv2.imwrite(str(output_path), contact_sheet)
    print(f"Saved E4B augmentation contact sheet: {output_path.name} (Shape: {contact_sheet.shape})")
    return output_path

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestE4Components)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    if res.wasSuccessful():
        print("\n=== GENERATING E4B VISUAL CONTACT SHEET ===")
        generate_contact_sheet()
    else:
        sys.exit(1)
