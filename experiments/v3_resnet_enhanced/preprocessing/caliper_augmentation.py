"""
caliper_augmentation.py
Synthetic Caliper-Invariance Data Augmentation Module

Attacks the #1 shortcut learning failure mode identified in clinical audit:
- 66.7% of false positives had caliper overlays (Odds Ratio = 6.0).
- Models learn "caliper crosshairs => fibroid" instead of diagnostic tissue features.

This transform injects realistic synthetic ultrasound calipers into images during training:
1. Dotted / dashed measurement lines.
2. Crosshair endpoints ('+', 'x', or dot markers).
3. Realistic ultrasound colors (high-intensity white, machine yellow, cyan).
4. Occasional synthetic measurement distance text (e.g., "D1: 34.2mm").

By showing the network calipers on BOTH positive (UF) and negative (NUF) scans,
the network is mathematically forced to become invariant to measurement artifacts.
"""

import cv2
import random
import numpy as np
from PIL import Image

class SyntheticCaliperAugmentation:
    def __init__(self, p=0.5):
        """
        p: Probability of applying caliper augmentation to a training image.
        """
        self.p = p

    def __call__(self, img):
        """
        Accepts: PIL Image or numpy ndarray (H, W, 3) or (H, W)
        Returns: Same type with synthetic calipers drawn
        """
        if random.random() > self.p:
            return img

        is_pil = isinstance(img, Image.Image)
        if is_pil:
            arr = np.array(img).copy()
        else:
            arr = img.copy()

        h, w = arr.shape[:2]

        # Caliper colors in B-mode / clinical ultrasound:
        # 1. Bright white (240-255, 240-255, 240-255)
        # 2. Machine yellow (230-255, 230-255, 0-50)
        # 3. Cyan (0-50, 230-255, 230-255)
        color_choices = [
            (250, 250, 250), # White
            (255, 255, 30),  # Yellow
            (240, 240, 240), # Pale white
            (30, 240, 240),  # Cyan
        ]
        color = random.choice(color_choices)

        # Decide caliper type:
        # Type 1: Two crosshairs with a dotted line between them (classic caliper)
        # Type 2: An ellipse measurement with major/minor axes
        # Type 3: Simple paired '+' crosshairs
        caliper_type = random.choice(["linear", "paired_cross", "linear_with_text"])

        # Select random endpoints inside the tissue region (central 20% - 80%)
        margin_x = int(w * 0.20)
        margin_y = int(h * 0.20)
        pt1 = (random.randint(margin_x, w - margin_x), random.randint(margin_y, h - margin_y))
        
        # Second point at a realistic fibroid distance (30px to 100px away)
        dist = random.randint(int(min(h, w) * 0.15), int(min(h, w) * 0.45))
        angle = random.uniform(0, 2 * np.pi)
        pt2 = (
            int(np.clip(pt1[0] + dist * np.cos(angle), 10, w - 10)),
            int(np.clip(pt1[1] + dist * np.sin(angle), 10, h - 10))
        )

        def draw_crosshair(canvas, pt, c, arm_len=4, thickness=1):
            x, y = pt
            cv2.line(canvas, (x - arm_len, y), (x + arm_len, y), c, thickness)
            cv2.line(canvas, (x, y - arm_len), (x, y + arm_len), c, thickness)

        def draw_dashed_line(canvas, p1, p2, c, thickness=1, dash_len=4):
            dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if dist < 1:
                return
            num_dashes = int(dist / (2 * dash_len))
            for i in range(max(1, num_dashes)):
                t0 = (2 * i) * dash_len / dist
                t1 = min(1.0, (2 * i + 1) * dash_len / dist)
                s = (int(p1[0] + t0 * (p2[0] - p1[0])), int(p1[1] + t0 * (p2[1] - p1[1])))
                e = (int(p1[0] + t1 * (p2[0] - p1[0])), int(p1[1] + t1 * (p2[1] - p1[1])))
                cv2.line(canvas, s, e, c, thickness)

        if caliper_type in ["linear", "linear_with_text"]:
            draw_dashed_line(arr, pt1, pt2, color, thickness=1, dash_len=random.randint(3, 5))
            draw_crosshair(arr, pt1, color, arm_len=random.randint(3, 5))
            draw_crosshair(arr, pt2, color, arm_len=random.randint(3, 5))
            
            if caliper_type == "linear_with_text" and w >= 150:
                meas_num = random.choice([1, 2])
                dist_val = random.uniform(15.0, 55.0)
                text = f"{meas_num} {dist_val:.1f}mm"
                text_pos = (max(5, pt2[0] + 5), max(15, pt2[1] - 5))
                cv2.putText(arr, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        elif caliper_type == "paired_cross":
            draw_crosshair(arr, pt1, color, arm_len=random.randint(4, 6))
            draw_crosshair(arr, pt2, color, arm_len=random.randint(4, 6))

        if is_pil:
            return Image.fromarray(arr)
        return arr

if __name__ == "__main__":
    from pathlib import Path
    
    test_paths = list(Path("datasets/datasets/train/UF").glob("*.jpg"))[:2] + \
                 list(Path("datasets/datasets/train/NUF").glob("*.jpg"))[:2]
    
    out_dir = Path("experiments/v3_resnet_enhanced/reports/caliper_aug_samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    aug = SyntheticCaliperAugmentation(p=1.0)
    print(f"Generating synthetic caliper demonstration on {len(test_paths)} images...")
    for idx, p in enumerate(test_paths):
        img = Image.open(p).convert("RGB")
        img_np = np.array(img)
        aug_img = aug(img)
        aug_np = np.array(aug_img)
        
        # Concat original and augmented
        concat = np.hstack([img_np, aug_np])
        save_p = out_dir / f"synthetic_caliper_{idx}_{p.stem}.png"
        cv2.imwrite(str(save_p), cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))
        print(f"[{idx+1}] Saved {save_p.name}")
