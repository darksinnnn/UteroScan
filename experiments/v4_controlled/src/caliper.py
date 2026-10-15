#!/usr/bin/env python3
"""
Class-Independent Synthetic Caliper Augmentation
Location: experiments/v4_controlled/src/caliper.py

Applies realistic ultrasound caliper markers (crosshairs, dotted measurement lines,
or distance calipers) with equal probability p=0.25 to both UF and NUF images.
Tracks augmentation counts by class to guarantee empirical class-independence.
"""

import os
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image

class SyntheticCaliperAugmentation:
    def __init__(self, p=0.25):
        """
        p: Probability of applying caliper augmentation (applied class-independently).
        """
        self.p = p
        self.stats = {
            "uf_total": 0,
            "uf_aug": 0,
            "nuf_total": 0,
            "nuf_aug": 0
        }

    def reset_stats(self):
        self.stats = {
            "uf_total": 0,
            "uf_aug": 0,
            "nuf_total": 0,
            "nuf_aug": 0
        }

    def get_stats(self):
        return dict(self.stats)

    def draw_caliper(self, arr):
        h, w = arr.shape[:2]
        
        # Clinical ultrasound color palette:
        # High-intensity white, machine pale yellow, pale cyan
        color_choices = [
            (255, 255, 255), # Bright white
            (245, 245, 245), # Off white
            (250, 240, 70),  # Ultrasound yellow
            (70, 240, 240),  # Ultrasound cyan
        ]
        color = random.choice(color_choices)
        caliper_type = random.choice(["linear", "paired_cross", "linear_tick"])

        margin_x = int(w * 0.18)
        margin_y = int(h * 0.18)
        pt1 = (
            random.randint(margin_x, max(margin_x + 1, w - margin_x)),
            random.randint(margin_y, max(margin_y + 1, h - margin_y))
        )
        
        # Distance between caliper endpoints: typical uterine/fibroid scale (15% to 45% of dimension)
        dist = random.randint(int(min(h, w) * 0.15), int(min(h, w) * 0.45))
        angle = random.uniform(0, 2 * np.pi)
        pt2 = (
            int(np.clip(pt1[0] + dist * np.cos(angle), 15, w - 15)),
            int(np.clip(pt1[1] + dist * np.sin(angle), 15, h - 15))
        )

        def draw_crosshair(canvas, pt, c, arm_len=4, thickness=1, style="plus"):
            x, y = pt
            if style == "plus":
                cv2.line(canvas, (x - arm_len, y), (x + arm_len, y), c, thickness, cv2.LINE_AA)
                cv2.line(canvas, (x, y - arm_len), (x, y + arm_len), c, thickness, cv2.LINE_AA)
            else: # "x" cross
                d = int(arm_len * 0.707)
                cv2.line(canvas, (x - d, y - d), (x + d, y + d), c, thickness, cv2.LINE_AA)
                cv2.line(canvas, (x - d, y + d), (x + d, y - d), c, thickness, cv2.LINE_AA)

        def draw_dotted_line(canvas, p1, p2, c, thickness=1, dash_len=4):
            d = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if d < 1:
                return
            num_dashes = int(d / (2 * dash_len))
            for i in range(max(1, num_dashes)):
                t0 = (2 * i) * dash_len / d
                t1 = min(1.0, (2 * i + 1) * dash_len / d)
                s = (int(p1[0] + t0 * (p2[0] - p1[0])), int(p1[1] + t0 * (p2[1] - p1[1])))
                e = (int(p1[0] + t1 * (p2[0] - p1[0])), int(p1[1] + t1 * (p2[1] - p1[1])))
                cv2.line(canvas, s, e, c, thickness, cv2.LINE_AA)

        cross_style = random.choice(["plus", "x"])
        arm_len = random.randint(3, 5)

        if caliper_type in ["linear", "linear_tick"]:
            draw_dotted_line(arr, pt1, pt2, color, thickness=1, dash_len=random.randint(3, 5))
            draw_crosshair(arr, pt1, color, arm_len=arm_len, style=cross_style)
            draw_crosshair(arr, pt2, color, arm_len=arm_len, style=cross_style)
        else: # paired crosshairs without central dashed line
            draw_crosshair(arr, pt1, color, arm_len=arm_len, style=cross_style)
            draw_crosshair(arr, pt2, color, arm_len=arm_len, style=cross_style)

        return arr

    def __call__(self, img, target_idx=None, force=False):
        do_augment = force or (random.random() <= self.p)

        if target_idx is not None:
            if target_idx == 0: # UF
                self.stats["uf_total"] += 1
                if do_augment:
                    self.stats["uf_aug"] += 1
            else: # NUF
                self.stats["nuf_total"] += 1
                if do_augment:
                    self.stats["nuf_aug"] += 1

        if not do_augment:
            return img

        is_pil = isinstance(img, Image.Image)
        if is_pil:
            arr = np.array(img).copy()
        else:
            arr = img.copy()

        arr = self.draw_caliper(arr)

        if is_pil:
            return Image.fromarray(arr)
        return arr

def generate_caliper_contact_sheet(folds_df, project_root, output_path, num_pairs=8):
    """
    Generates side-by-side contact sheet of original vs caliper-augmented training scans.
    Shows balanced UF and NUF examples for visual inspection.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    aug = SyntheticCaliperAugmentation(p=1.0) # force augmentation

    # Pick samples: half UF, half NUF
    uf_samples = folds_df[folds_df["target_idx"] == 0].sample(num_pairs // 2, random_state=42)
    nuf_samples = folds_df[folds_df["target_idx"] == 1].sample(num_pairs // 2, random_state=42)
    sample_df = pd.concat([uf_samples, nuf_samples]).reset_index(drop=True)

    rows = []
    tile_h, tile_w = 200, 240

    for _, row in sample_df.iterrows():
        img_path = os.path.join(project_root, row["image_path"])
        orig = cv2.imread(img_path)
        if orig is None:
            continue
        orig_resized = cv2.resize(orig, (tile_w, tile_h))
        
        # Augmented version
        aug_copy = orig.copy()
        aug_copy = aug(aug_copy, force=True)
        aug_resized = cv2.resize(aug_copy, (tile_w, tile_h))

        # Add label headers
        label = row["class_label"]
        stem = str(row["source_stem"])[:18]

        header_orig = np.zeros((30, tile_w, 3), dtype=np.uint8)
        cv2.putText(header_orig, f"{label} | Orig: {stem}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        orig_tile = np.vstack([header_orig, orig_resized])

        header_aug = np.zeros((30, tile_w, 3), dtype=np.uint8)
        cv2.putText(header_aug, f"{label} | Caliper Aug", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 240, 80), 1, cv2.LINE_AA)
        aug_tile = np.vstack([header_aug, aug_resized])

        pair = np.hstack([orig_tile, aug_tile])
        rows.append(pair)

    if not rows:
        print("Warning: No contact sheet rows generated.")
        return

    # Stack in 2 columns of pairs (4 pairs per column)
    half = len(rows) // 2
    col1 = np.vstack(rows[:half])
    col2 = np.vstack(rows[half:])
    sheet = np.hstack([col1, col2])

    cv2.imwrite(str(output_path), sheet)
    print(f"[SAVED] Contact sheet saved to {output_path}")

if __name__ == "__main__":
    from pathlib import Path
    proj = Path("d:/Project/UteroScan")
    df = pd.read_csv(proj / "experiments" / "v4_controlled" / "data" / "folds.csv")
    out = proj / "experiments" / "v4_controlled" / "reports" / "caliper_augmentation_contact_sheet.png"
    generate_caliper_contact_sheet(df, str(proj), str(out), num_pairs=8)
