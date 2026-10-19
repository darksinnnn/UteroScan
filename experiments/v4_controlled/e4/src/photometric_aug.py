#!/usr/bin/env python3
"""
Ultrasound-Preserving Photometric Augmentation Module for E4B
Location: experiments/v4_controlled/e4/src/photometric_aug.py

Implements mild, physics-preserving acoustic augmentations:
  1. Gamma Adjustment: gamma in [0.85, 1.15], p=0.5 (simulates dynamic range / gain curve)
  2. Contrast Scaling: alpha in [0.90, 1.10] (+-10%), p=0.5 (simulates tissue acoustic attenuation)
  3. Multiplicative Speckle Noise: N(0, sigma^2) with sigma in [0.005, 0.020] (<=0.02), p=0.3
"""

import random
import cv2
import numpy as np

class UltrasoundPhotometricAugmentation:
    def __init__(self,
                 gamma_p=0.5,
                 gamma_range=(0.85, 1.15),
                 contrast_p=0.5,
                 contrast_range=(0.90, 1.10),
                 speckle_p=0.3,
                 speckle_max_std=0.02):
        self.gamma_p = gamma_p
        self.gamma_range = gamma_range
        self.contrast_p = contrast_p
        self.contrast_range = contrast_range
        self.speckle_p = speckle_p
        self.speckle_max_std = speckle_max_std

        # Precompute lookup tables for common gammas for speed, or compute dynamically
        self.inv255 = 1.0 / 255.0

    def apply_gamma(self, img_uint8, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i * self.inv255) ** inv_gamma) * 255 for i in range(256)]).clip(0, 255).astype(np.uint8)
        return cv2.LUT(img_uint8, table)

    def apply_contrast(self, img_uint8, alpha):
        img_f = img_uint8.astype(np.float32)
        mean_val = np.mean(img_f)
        out = mean_val + alpha * (img_f - mean_val)
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    def apply_speckle(self, img_uint8, sigma):
        img_f = img_uint8.astype(np.float32)
        noise = np.random.normal(0.0, sigma, size=img_uint8.shape)
        out = img_f * (1.0 + noise)
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    def __call__(self, img_uint8):
        """
        img_uint8: Grayscale or BGR uint8 image in range [0, 255].
        Returns: Augmented uint8 image in range [0, 255].
        """
        out = img_uint8.copy()

        # 1. Gamma adjustment (p=0.5)
        if random.random() < self.gamma_p:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            out = self.apply_gamma(out, gamma)

        # 2. Contrast scaling around mean (p=0.5)
        if random.random() < self.contrast_p:
            alpha = random.uniform(self.contrast_range[0], self.contrast_range[1])
            out = self.apply_contrast(out, alpha)

        # 3. Multiplicative speckle noise (p=0.3)
        if random.random() < self.speckle_p:
            sigma = random.uniform(0.005, self.speckle_max_std)
            out = self.apply_speckle(out, sigma)

        return out
