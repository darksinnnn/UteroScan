"""
utils/augmentation.py
UteroScan Clinical Ultrasound Data Augmentation Pipeline (Phase 2)

Locked Specification per PRD.md Section 8 and Implementation_Plan.md Phase 2:
- Applied ONLY to Training DataLoader (random=True).
- Validation and Test loaders NEVER receive data augmentation (random=False).
- Transforms:
    1. Random Rotation: [-15 deg, +15 deg] (Bilinear, fill=0)
    2. Random Horizontal Flip: p = 0.5 (Flip Left-Right)
    3. Random Zoom / Scale: [0.9, 1.1] (90% to 110% scale)
    4. Random Shear: [-10 deg, +10 deg] (Bilinear, fill=0)
    5. STRICTLY NO Vertical Flip (anatomically invalid for acoustic probe orientation).
"""

import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Augmentation Parameters (Locked Spec)
ROTATION_DEGREES = 15
HFLIP_PROB = 0.5
ZOOM_SCALE = (0.9, 1.1)
SHEAR_DEGREES = (-10, 10)

def get_ultrasound_train_transforms():
    """
    Returns the locked torchvision Compose pipeline for clinical ultrasound training data.
    """
    aug_pipeline = transforms.Compose([
        transforms.RandomRotation(
            degrees=ROTATION_DEGREES,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0
        ),
        transforms.RandomHorizontalFlip(p=HFLIP_PROB),
        transforms.RandomAffine(
            degrees=0,
            scale=ZOOM_SCALE,
            shear=SHEAR_DEGREES,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0
        )
    ])
    
    # Assert vertical flip is strictly absent
    for t in aug_pipeline.transforms:
        assert not isinstance(t, transforms.RandomVerticalFlip), "Vertical flip is strictly forbidden in UteroScan!"
        
    return aug_pipeline

def apply_augmentation(image, train_transforms=None):
    """
    Applies the clinical ultrasound augmentation pipeline to an image.
    Accepts: PIL Image or numpy ndarray (H, W, 3)
    Returns: numpy ndarray (H, W, 3) uint8
    """
    if train_transforms is None:
        train_transforms = get_ultrasound_train_transforms()

    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    else:
        pil_img = image

    aug_pil = train_transforms(pil_img)
    return np.array(aug_pil)
