#!/usr/bin/env python3
"""
Sector Viewport Cropping Module
Location: experiments/v4_controlled/src/viewport.py

Isolates active ultrasound acoustic sector/cone and crops out extraneous machine UI margins.
"""

import cv2
import numpy as np
from PIL import Image

def extract_ultrasound_viewport(image, min_intensity_thresh=15, margin_pad=5):
    """
    Locates the active ultrasound acoustic cone/fan and crops out extraneous machine UI margins.
    Accepts:
        image: numpy.ndarray (H, W, 3) or (H, W) or PIL.Image
    Returns:
        cropped: numpy.ndarray of the cropped ultrasound viewport
        bbox: (ymin, xmin, ymax, xmax) relative to original image
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()

    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()

    h, w = gray.shape

    # 1. Mask ultrasound fan pixels (pixels with actual acoustic echo)
    _, binary = cv2.threshold(gray, min_intensity_thresh, 255, cv2.THRESH_BINARY)

    # 2. Morphological closing to bridge speckle holes within the uterine sector
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # 3. Find connected components to locate the main ultrasound acoustic cone
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

    if num_labels <= 1:
        return img_np, (0, 0, h, w)

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = np.argmax(areas) + 1

    x = stats[max_idx, cv2.CC_STAT_LEFT]
    y = stats[max_idx, cv2.CC_STAT_TOP]
    bw = stats[max_idx, cv2.CC_STAT_WIDTH]
    bh = stats[max_idx, cv2.CC_STAT_HEIGHT]

    if bw < w * 0.20 or bh < h * 0.20:
        return img_np, (0, 0, h, w)

    ymin = max(0, y - margin_pad)
    xmin = max(0, x - margin_pad)
    ymax = min(h, y + bh + margin_pad)
    xmax = min(w, x + bw + margin_pad)

    cropped = img_np[ymin:ymax, xmin:xmax]
    return cropped, (ymin, xmin, ymax, xmax)
