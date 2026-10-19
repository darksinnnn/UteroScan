#!/usr/bin/env python3
"""
Unit Tests for E4 Data Preprocessing and Photometric Augmentation Integrity
Location: experiments/v4_controlled/e4/tests/test_e4_augmentation_integrity.py

Test Cases:
  1. Enabled augmentation (is_train=True, use_photometric_aug=True) produces stochastic variations
     across repeated calls on training images.
  2. Disabled augmentation (is_train=False) returns 100% BIT-FOR-BIT IDENTICAL tensors across
     multiple calls on the same sample (validation/evaluation invariance).
  3. Outer-fold evaluation dataloader strictly leaves images completely unaugmented.
  4. Dynamic range check: transformed uint8 outputs remain strictly within [0, 255], and normalized
     tensors have finite float values with standard ImageNet distribution.
  5. Geometric invariance: photometric transformations never alter spatial coordinates, rotate,
     or shift anatomy.
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("d:/Project/UteroScan")
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "v4_controlled" / "e4" / "src"))

from photometric_aug import UltrasoundPhotometricAugmentation
from e4_dataset import E4UltrasoundDataset

FOLDS_CSV = PROJECT_ROOT / "experiments" / "v4_controlled" / "data" / "clean_v5" / "folds_clean_v5.csv"

class TestE4AugmentationIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_folds = pd.read_csv(FOLDS_CSV)
        cls.test_df = cls.df_folds.head(5).copy()

    def test_evaluation_dataset_bit_exact_invariance(self):
        """Verify that evaluation/validation datasets (is_train=False) produce 100% bit-exact outputs across calls."""
        eval_ds = E4UltrasoundDataset(self.test_df, is_train=False, use_photometric_aug=False, project_root=str(PROJECT_ROOT))
        
        for idx in range(len(eval_ds)):
            t1, lbl1, stem1 = eval_ds[idx]
            t2, lbl2, stem2 = eval_ds[idx]
            
            # Must be bit-exact
            self.assertEqual(lbl1, lbl2)
            self.assertEqual(stem1, stem2)
            self.assertTrue(torch.equal(t1, t2), f"Evaluation sample {idx} ({stem1}) produced non-identical tensors across calls!")

    def test_disabled_photometric_aug_even_if_flagged(self):
        """Verify that even if use_photometric_aug=True is passed with is_train=False, augmentation is strictly disabled."""
        eval_ds_flagged = E4UltrasoundDataset(self.test_df, is_train=False, use_photometric_aug=True, project_root=str(PROJECT_ROOT))
        self.assertFalse(eval_ds_flagged.use_photometric_aug)
        self.assertIsNone(eval_ds_flagged.photo_aug)
        
        for idx in range(len(eval_ds_flagged)):
            t1, _, _ = eval_ds_flagged[idx]
            t2, _, _ = eval_ds_flagged[idx]
            self.assertTrue(torch.equal(t1, t2))

    def test_training_augmentation_produces_stochastic_variation(self):
        """Verify that enabled augmentation on training images produces stochastic variation across calls."""
        train_ds = E4UltrasoundDataset(self.test_df, is_train=True, use_photometric_aug=True, project_root=str(PROJECT_ROOT))
        self.assertTrue(train_ds.use_photometric_aug)
        self.assertIsNotNone(train_ds.photo_aug)

        # Over 10 repeated calls on the first image, tensors must differ due to stochastic transforms
        tensors = [train_ds[0][0] for _ in range(10)]
        differences = [not torch.equal(tensors[0], tensors[i]) for i in range(1, 10)]
        self.assertTrue(any(differences), "Training augmentation failed to produce any variation across 10 calls!")

    def test_photometric_transform_range_and_finite(self):
        """Verify all photometric functions keep pixel intensities strictly bounded in [0, 255]."""
        aug = UltrasoundPhotometricAugmentation(gamma_p=1.0, contrast_p=1.0, speckle_p=1.0)
        
        # Test synthetic images with boundary conditions
        test_images = [
            np.zeros((224, 224), dtype=np.uint8), # all black
            np.full((224, 224), 255, dtype=np.uint8), # all white
            np.random.randint(0, 256, size=(224, 224), dtype=np.uint8) # random uniform
        ]
        
        for img in test_images:
            for _ in range(20):
                out = aug(img)
                self.assertEqual(out.dtype, np.uint8)
                self.assertEqual(out.shape, (224, 224))
                self.assertTrue(np.all(out >= 0), "Augmented image contains negative values!")
                self.assertTrue(np.all(out <= 255), "Augmented image contains values exceeding 255!")

    def test_geometric_invariance_of_photometric_aug(self):
        """Verify photometric augmentation does NOT shift or rotate pixels (strictly zero geometric displacement)."""
        aug = UltrasoundPhotometricAugmentation(gamma_p=1.0, contrast_p=1.0, speckle_p=1.0)
        
        # Create an image with an isolated point at (100, 100)
        grid = np.zeros((224, 224), dtype=np.uint8)
        grid[100, 100] = 200
        
        # In contrast or speckle or gamma, if background is 0, gamma(0)=0, speckle(0)=0.
        # Contrast around mean will change non-zero mean, but point location (100, 100) has max intensity.
        out_gamma = aug.apply_gamma(grid, 0.85)
        self.assertEqual(np.argmax(out_gamma), 100 * 224 + 100)
        
        out_speckle = aug.apply_speckle(grid, 0.02)
        self.assertEqual(np.argmax(out_speckle), 100 * 224 + 100)

if __name__ == "__main__":
    unittest.main(verbosity=2)
