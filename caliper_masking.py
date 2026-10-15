"""
caliper_masking.py
Detects and inpaints caliper/measurement-marker artifacts (crosshairs, dashed
calipers, distance-measurement dashes) from ultrasound B-mode images, while
leaving true grayscale tissue echogenicity untouched.

Core principle: true anatomy is grayscale (R≈G≈B). Caliper markers and
measurement text are rendered in white or yellow/orange with a slight but
detectable chromatic signature. We isolate small, chromatic, high-contrast
connected components and inpaint them.
"""

import cv2
import numpy as np

# ---- Tunable thresholds ----
HEADER_EXCLUDE_FRAC = 0.12   # top strip is the machine metadata banner — skip entirely,
                              # it's not part of the anatomical image and shouldn't be inpainted
CHROMA_DIFF_THRESH = 15       # min |R-G|, |G-B|, or |R-B| to call a pixel "chromatic" (not grayscale)
BRIGHTNESS_THRESH = 140       # calipers/measurement marks are bright; filters out darker chromatic noise
MIN_BLOB_AREA = 4             # px — below this is likely JPEG chroma noise, not a real marker
MAX_BLOB_AREA = 180           # px — above this it's probably a real colored region (e.g. Doppler), not a caliper
MAX_ASPECT_RATIO = 4.0        # crosshairs/dashes are compact or short line segments, not long shapes
DILATE_KERNEL_SIZE = 5        # 5x5 kernel fully eliminates anti-aliased font ghosting and marker edges
INPAINT_RADIUS = 5
INPAINT_METHOD = cv2.INPAINT_TELEA  # generally better for thin, elongated regions like dashes


def detect_caliper_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Returns a binary mask (uint8, 0/255) of detected caliper/marker pixels."""
    h, w = image_bgr.shape[:2]
    b = image_bgr[:, :, 0].astype(int)
    g = image_bgr[:, :, 1].astype(int)
    r = image_bgr[:, :, 2].astype(int)

    # Chromatic test: real tissue is grayscale, calipers/text are not
    chroma = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(r - b))
    is_chromatic = chroma > CHROMA_DIFF_THRESH

    # Brightness test: calipers/measurement marks are bright, not dark chroma artifacts
    brightness = (r + g + b) / 3.0
    is_bright = brightness > BRIGHTNESS_THRESH

    raw_mask = (is_chromatic & is_bright).astype(np.uint8) * 255

    # Exclude the top metadata banner entirely — it's not anatomy, don't touch it
    raw_mask[: int(h * HEADER_EXCLUDE_FRAC), :] = 0

    # Connected-component filtering: keep only small, compact/short-line blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    clean_mask = np.zeros_like(raw_mask)
    for i in range(1, num_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA and aspect <= MAX_ASPECT_RATIO:
            clean_mask[labels == i] = 255

    # Slight dilation so inpainting covers anti-aliased marker edges completely
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE))
    clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)

    return clean_mask


def remove_calipers(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Returns (inpainted_image, mask_used, num_blobs_removed).
    Call this BEFORE the existing grayscale/median/CLAHE pipeline, on the
    original color image, so chromatic information still exists to detect —
    the current pipeline converts to grayscale immediately, which would
    destroy this signal.
    """
    mask = detect_caliper_mask(image_bgr)
    num_blobs = cv2.connectedComponents(mask)[0] - 1
    if num_blobs == 0:
        return image_bgr.copy(), mask, 0
    inpainted = cv2.inpaint(image_bgr, mask, INPAINT_RADIUS, INPAINT_METHOD)
    return inpainted, mask, num_blobs
