"""
utils/preprocessing.py
UteroScan Preprocessing Transform Module (Refined Phase 1 Pipeline)

Clinical Preprocessing Pipeline:
1. Grayscale conversion: Converts input to single-channel grayscale to eliminate
   JPEG chroma compression artifacts, color-channel phase separation, and discolored caliper fringing.
2. Downscale to 224x224 using cv2.INTER_AREA (avoids aliasing during ultrasound downsampling).
3. 3x3 Median Blur: Suppresses rogue acoustic speckle noise while preserving fine myometrial
   striations and small focal lesion boundaries (avoiding the over-softening caused by 5x5 filters).
4. CLAHE on grayscale (clipLimit=1.2, tileGridSize=(8, 8)): Enhances tissue boundary contrast
   without causing blocky tile-grid artifacts in dark acoustic shadowing or background regions.
5. 3-Channel RGB Replicate: Converts the clean enhanced grayscale to a 3-channel (R=G=B) image
   satisfying ImageNet pretrained CNN backbone (ResNet50) requirements.
"""

import cv2
import numpy as np
from PIL import Image

TARGET_SIZE = (224, 224)  # (width, height)
MEDIAN_KERNEL = 3
CLAHE_CLIP_LIMIT = 1.2
CLAHE_GRID_SIZE = (8, 8)

def preprocess_ultrasound(image, target_size=TARGET_SIZE):
    """
    Applies the refined UteroScan clinical preprocessing pipeline to an ultrasound image.
    Accepts:
        image: numpy.ndarray (H, W, 3) or (H, W), or PIL Image
    Returns:
        numpy.ndarray (224, 224, 3) uint8 RGB image (pure grayscale replicated across 3 channels)
    """
    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # 1. Convert to single-channel grayscale (eliminates compression chroma noise and caliper color fringing)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 2:
        gray = image.copy()
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        gray = image.copy()

    # 2. Resize to target_size (224x224) using INTER_AREA (best for downscaling ultrasound)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    # 3. 3x3 Median Blur (removes acoustic speckle spikes without over-softening fine tissue margins)
    denoised = cv2.medianBlur(resized, MEDIAN_KERNEL)

    # 4. CLAHE contrast enhancement on single-channel grayscale (clipLimit=1.2 prevents blocky background tiling)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    enhanced = clahe.apply(denoised)

    # 5. Replicate clean enhanced grayscale to 3-channel RGB for ResNet50 input requirements
    preprocessed_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return preprocessed_rgb
