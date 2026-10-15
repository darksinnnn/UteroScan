"""
enhanced_dataloader.py
High-Accuracy Anatomy-Aware Ultrasound Data Loader for ResNet50-CBAM

Key Features:
1. Ultrasound Viewport Extraction: Strips non-acoustic borders and machine UI.
2. Clinical Denoising & Enhancement: 3x3 Median Blur + CLAHE (clipLimit=1.2) at 256x256.
3. Caliper-Invariance Augmentation: Injects synthetic crosshairs & measurement lines on training data.
4. Rigid Anatomical Preserving Augmentations: Random rotation [-15°, +15°], horizontal flip, zoom [0.9, 1.1], shear [-10°, +10°]. Strictly NO vertical flip.
"""

import sys
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from pathlib import Path

BASE_DIR = Path("d:/Project/UteroScan")
EXP_DIR = BASE_DIR / "experiments" / "v3_resnet_enhanced"
sys.path.insert(0, str(EXP_DIR))

from preprocessing.viewport_cropper import extract_ultrasound_viewport
from preprocessing.caliper_augmentation import SyntheticCaliperAugmentation

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_anatomy_ultrasound(image, target_size=(256, 256), use_viewport=True):
    """
    Full anatomical ultrasound preprocessing pipeline:
    1. Viewport extraction (optional, recommended)
    2. Grayscale conversion (eliminates chroma artifacts)
    3. Resize via cv2.INTER_AREA to target_size (e.g., 256x256)
    4. 3x3 Median blur (preserves subtle acoustic borders)
    5. CLAHE contrast enhancement
    6. 3-channel RGB replicate
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()

    # 1. Viewport extraction
    if use_viewport:
        img_np, _ = extract_ultrasound_viewport(img_np)

    # 2. Grayscale conversion
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    elif len(img_np.shape) == 2:
        gray = img_np.copy()
    else:
        gray = img_np[:, :, 0]

    # 3. Downscale / upscale using INTER_AREA or INTER_CUBIC
    h, w = gray.shape
    interp = cv2.INTER_AREA if (w > target_size[0] and h > target_size[1]) else cv2.INTER_LINEAR
    resized = cv2.resize(gray, target_size, interpolation=interp)

    # 4. 3x3 Median filter
    denoised = cv2.medianBlur(resized, 3)

    # 5. CLAHE (clipLimit=1.2, tileGridSize=(8, 8))
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 6. Replicate to 3 channels RGB
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return rgb

CACHE_DIR = EXP_DIR / "cache_256"

class EnhancedUltrasoundDataset(data.Dataset):
    def __init__(self, annotation_lines, input_shape=(256, 256), is_train=True, use_viewport=True, caliper_aug_p=0.4):
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.is_train = is_train
        self.use_viewport = use_viewport
        self.caliper_aug = SyntheticCaliperAugmentation(p=caliper_aug_p) if is_train else None

    def __len__(self):
        return len(self.annotation_lines)

    def _resolve_path(self, path_str):
        p = Path(path_str)
        if p.is_file():
            return str(p)
        candidates = [
            BASE_DIR / "datasets" / "datasets" / Path(*p.parts[1:] if len(p.parts) > 1 else p.parts),
            BASE_DIR / "datasets" / "datasets" / p,
            BASE_DIR / p,
            Path("datasets/datasets") / p.name
        ]
        for c in candidates:
            if c.is_file():
                return str(c)
        matches = list((BASE_DIR / "datasets" / "datasets").rglob(p.name))
        if matches:
            return str(matches[0])
        return path_str

    def __getitem__(self, index):
        line = self.annotation_lines[index].strip()
        parts = line.split(";")
        y = int(parts[0])
        raw_path = parts[1].split()[0]
        stem = Path(raw_path).stem

        # 1. Fast path: Check 256x256 precomputed cache
        cached_p = CACHE_DIR / f"{stem}.png"
        if cached_p.exists() and self.input_shape == (256, 256):
            gray = cv2.imread(str(cached_p), cv2.IMREAD_GRAYSCALE)
            preprocessed = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            resolved_path = self._resolve_path(raw_path)
            image = Image.open(resolved_path)
            img_np = np.array(image.convert("RGB"))
            preprocessed = preprocess_anatomy_ultrasound(
                img_np,
                target_size=self.input_shape,
                use_viewport=self.use_viewport
            )

        # Training-time data augmentations
        if self.is_train:
            # 1. Synthetic caliper invariance
            if self.caliper_aug is not None:
                preprocessed = self.caliper_aug(preprocessed)

            # 2. Geometric augmentations (OpenCV for high speed)
            h, w = preprocessed.shape[:2]
            
            # Random horizontal flip
            if np.random.rand() > 0.5:
                preprocessed = cv2.flip(preprocessed, 1)

            # Random slight rotation [-15, +15] and scale [0.92, 1.08]
            angle = np.random.uniform(-15, 15)
            scale = np.random.uniform(0.92, 1.08)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            preprocessed = cv2.warpAffine(preprocessed, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Normalize with ImageNet mean/std
        img_norm = (preprocessed.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        # Transpose to (C, H, W)
        tensor_img = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        return tensor_img, y

def custom_collate(batch):
    images = torch.stack([item[0] for item in batch])
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, targets

if __name__ == "__main__":
    split_file = BASE_DIR / "experiments" / "v2_contamination_cleaned" / "annotations" / "cls_train_split.txt"
    lines = open(split_file).readlines()[:10]
    ds = EnhancedUltrasoundDataset(lines, input_shape=(256, 256), is_train=True)
    loader = data.DataLoader(ds, batch_size=4, shuffle=True, collate_fn=custom_collate)
    
    for imgs, tgts in loader:
        print(f"Loaded batch: Images {imgs.shape}, Targets {tgts.shape}, Dtype: {imgs.dtype}")
        print(f"Min: {imgs.min():.2f}, Max: {imgs.max():.2f}, Mean: {imgs.mean():.2f}")
        break
