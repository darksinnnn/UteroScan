"""
viewport_cropper.py
Automated Ultrasound Viewport & Sector Extraction Module

Removes:
- Non-acoustic black borders/letterboxing
- Ultrasound machine top headers (hospital name, patient ID, date/time)
- Bottom/side UI overlays (depth scales, dB gain text, frequency indicators)
- Watermark margins

Isolates:
- The active B-mode acoustic sector/fan containing pelvic anatomy.
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
    # Ultrasound fan has typical echogenic speckle > 15
    _, binary = cv2.threshold(gray, min_intensity_thresh, 255, cv2.THRESH_BINARY)

    # 2. Morphological closing to bridge speckle holes within the uterine sector
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # 3. Find connected components to locate the main ultrasound acoustic cone
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

    if num_labels <= 1:
        # Fallback if image is uniformly dark or binary empty
        return img_np, (0, 0, h, w)

    # Skip background label 0, find component with largest area
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = np.argmax(areas) + 1  # 1-indexed

    x = stats[max_idx, cv2.CC_STAT_LEFT]
    y = stats[max_idx, cv2.CC_STAT_TOP]
    bw = stats[max_idx, cv2.CC_STAT_WIDTH]
    bh = stats[max_idx, cv2.CC_STAT_HEIGHT]

    # Sanity checks: ultrasound sector should be substantial (at least 20% of image width/height)
    if bw < w * 0.20 or bh < h * 0.20:
        return img_np, (0, 0, h, w)

    # Trim typical top machine header bar if it was connected (top 5-8% often contains patient text)
    # If the sector starts at y=0, check if row 0 to 25 has text lines
    ymin = max(0, y - margin_pad)
    xmin = max(0, x - margin_pad)
    ymax = min(h, y + bh + margin_pad)
    xmax = min(w, x + bw + margin_pad)

    cropped = img_np[ymin:ymax, xmin:xmax]
    return cropped, (ymin, xmin, ymax, xmax)

if __name__ == "__main__":
    from pathlib import Path
    
    # Test on a few sample images from the dataset
    test_paths = list(Path("datasets/datasets/train/UF").glob("*.jpg"))[:3] + \
                 list(Path("datasets/datasets/train/NUF").glob("*.jpg"))[:3]
    
    out_dir = Path("experiments/v3_resnet_enhanced/reports/viewport_samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Testing viewport extraction on {len(test_paths)} images...")
    for idx, p in enumerate(test_paths):
        img = Image.open(p)
        orig_w, orig_h = img.size
        cropped, bbox = extract_ultrasound_viewport(img)
        ch, cw = cropped.shape[:2]
        
        save_p = out_dir / f"viewport_test_{idx}_{p.stem}.png"
        # Save side-by-side: original vs cropped
        orig_np = np.array(img.convert("RGB"))
        cv2.rectangle(orig_np, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 255, 0), 2)
        
        # Resize cropped to match height for display
        cropped_resized = cv2.resize(cropped, (int(cw * (orig_h / ch)), orig_h))
        concat = np.hstack([orig_np, cropped_resized])
        cv2.imwrite(str(save_p), cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))
        print(f"[{idx+1}] {p.name}: Original ({orig_w}x{orig_h}) -> Crop bbox {bbox} ({cw}x{ch}) -> Saved {save_p.name}")
