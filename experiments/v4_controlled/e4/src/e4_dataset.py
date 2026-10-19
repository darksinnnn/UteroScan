#!/usr/bin/env python3
"""
Controlled Ultrasound Dataset for E4 Experiments
Location: experiments/v4_controlled/e4/src/e4_dataset.py

Supports:
  - Standard E0 full-frame pipeline: Grayscale -> 224x224 -> Median Blur -> CLAHE -> 3ch RGB -> ImageNet Norm
  - Standard geometric training augmentations: H-Flip (p=0.5), Rotation (+-15 deg)
  - Optional Ultrasound-Preserving Photometric Augmentations (E4B training only):
      Gamma (0.85-1.15, p=0.5), Contrast (+-10%, p=0.5), Multiplicative Speckle (sigma<=0.02, p=0.3)
"""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from photometric_aug import UltrasoundPhotometricAugmentation

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class E4UltrasoundDataset(Dataset):
    def __init__(self,
                 records_df,
                 input_size=(224, 224),
                 is_train=False,
                 use_photometric_aug=False,
                 project_root="."):
        self.records = records_df.reset_index(drop=True)
        self.input_size = input_size
        self.is_train = is_train
        self.use_photometric_aug = use_photometric_aug and is_train
        self.project_root = project_root

        self.clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        self.photo_aug = UltrasoundPhotometricAugmentation() if self.use_photometric_aug else None
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.cache = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if idx in self.cache:
            img_bgr, target_idx, stem = self.cache[idx]
            img_bgr = img_bgr.copy()
        else:
            row = self.records.iloc[idx]
            img_rel_path = row["image_path"]
            full_img_path = os.path.join(self.project_root, img_rel_path)

            img_bgr = cv2.imread(full_img_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Image not found at {full_img_path}")

            target_idx = int(row["target_idx"]) # 0=UF, 1=NUF
            stem = str(row["source_stem"])
            self.cache[idx] = (img_bgr.copy(), target_idx, stem)

        h, w = img_bgr.shape[:2]

        # Geometric Augmentations during Training
        if self.is_train:
            # Horizontal Flip (p=0.5)
            if random.random() < 0.5:
                img_bgr = cv2.flip(img_bgr, 1)

            # Rotation (+- 15 degrees)
            angle = random.uniform(-15.0, 15.0)
            if abs(angle) > 1.0:
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                img_bgr = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Clinical Preprocessing Transform
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.input_size, interpolation=cv2.INTER_AREA)
        blurred = cv2.medianBlur(resized, 3)
        clahe_out = self.clahe.apply(blurred)

        # Mild Photometric Augmentation (E4B training only)
        if self.use_photometric_aug and self.photo_aug is not None:
            clahe_out = self.photo_aug(clahe_out)

        # 3-channel RGB replicate
        rgb = np.stack([clahe_out, clahe_out, clahe_out], axis=-1)

        # Convert to Tensor [3, H, W] in [0, 1]
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0

        # ImageNet normalization
        tensor = self.normalize(tensor)

        return tensor, torch.tensor(target_idx, dtype=torch.long), stem
