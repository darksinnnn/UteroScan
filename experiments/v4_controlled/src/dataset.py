#!/usr/bin/env python3
"""
Controlled Ultrasound Dataset Module
Location: experiments/v4_controlled/src/dataset.py

Implements:
  - Clinical preprocessing pipeline:
      Grayscale -> 224x224 (or target res) -> 3x3 Median Blur -> CLAHE (1.2) -> 3-channel RGB -> ImageNet Norm
  - Optional Viewport Cropping (with strict bounding box coordinate adjustments)
  - Optional Class-Independent Synthetic Caliper Augmentation (applied equally to UF & NUF)
  - Geometric Augmentations during training (Rotation +-15 deg, H-Flip p=0.5)
  - Normalized Bounding-Box Targets for Auxiliary Localization Head
"""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET

from caliper import SyntheticCaliperAugmentation
from viewport import extract_ultrasound_viewport

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def parse_focal_bounding_box(xml_path):
    """
    Parses genuine focal lesion bounding box from Pascal VOC XML.
    Returns: (has_box, [xmin, ymin, xmax, ymax])
    """
    if not os.path.exists(xml_path):
        return False, [0.0, 0.0, 0.0, 0.0]
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objs = root.findall("object")
        if not objs:
            return False, [0.0, 0.0, 0.0, 0.0]
        boxes = []
        for obj in objs:
            bb = obj.find("bndbox")
            if bb is not None:
                boxes.append([
                    float(bb.find("xmin").text),
                    float(bb.find("ymin").text),
                    float(bb.find("xmax").text),
                    float(bb.find("ymax").text)
                ])
        if not boxes:
            return False, [0.0, 0.0, 0.0, 0.0]
        # Envelope over all focal lesions
        b_arr = np.array(boxes)
        xmin = float(np.min(b_arr[:, 0]))
        ymin = float(np.min(b_arr[:, 1]))
        xmax = float(np.max(b_arr[:, 2]))
        ymax = float(np.max(b_arr[:, 3]))
        return True, [xmin, ymin, xmax, ymax]
    except Exception:
        return False, [0.0, 0.0, 0.0, 0.0]

class ControlledUltrasoundDataset(Dataset):
    def __init__(self,
                 records_df,
                 input_size=(224, 224),
                 is_train=False,
                 use_viewport=False,
                 use_caliper_aug=False,
                 caliper_p=0.25,
                 project_root="."):
        """
        records_df: DataFrame with columns: [image_path, class_label, target_idx, source_stem, ...]
        """
        self.records = records_df.reset_index(drop=True)
        self.input_size = input_size
        self.is_train = is_train
        self.use_viewport = use_viewport
        self.use_caliper_aug = use_caliper_aug
        self.project_root = project_root
        
        self.clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        self.caliper_aug = SyntheticCaliperAugmentation(p=caliper_p) if use_caliper_aug else None
        
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.cache = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if idx in self.cache:
            img_bgr, has_box, box_coords, crop_h, crop_w, target_idx, stem = self.cache[idx]
            img_bgr = img_bgr.copy()
            box_coords = list(box_coords)
        else:
            row = self.records.iloc[idx]
            img_rel_path = row["image_path"]
            full_img_path = os.path.join(self.project_root, img_rel_path)
            
            # Load image (BGR)
            img_bgr = cv2.imread(full_img_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Image not found at {full_img_path}")
                
            orig_h, orig_w = img_bgr.shape[:2]
            target_idx = int(row["target_idx"]) # 0=UF, 1=NUF
            stem = str(row["source_stem"])
            
            # Check for focal bounding box annotation in Mendeley V2
            xml_path = os.path.join(self.project_root, "datasets", "new dataset", "Annotations", "Annotations", f"{stem}.xml")
            has_box, box_coords = parse_focal_bounding_box(xml_path)
            
            # Only UF images in train split can have positive box supervision
            if target_idx != 0: # NUF
                has_box = False
                box_coords = [0.0, 0.0, 0.0, 0.0]

            # 1. Viewport Cropping (if enabled)
            if self.use_viewport:
                cropped_bgr, (ymin, xmin, ymax, xmax) = extract_ultrasound_viewport(img_bgr)
                crop_h, crop_w = cropped_bgr.shape[:2]
                img_bgr = cropped_bgr
                
                # Adjust box coordinates relative to cropped sector
                if has_box:
                    x1, y1, x2, y2 = box_coords
                    x1_c = max(0.0, x1 - xmin)
                    y1_c = max(0.0, y1 - ymin)
                    x2_c = min(float(crop_w), x2 - xmin)
                    y2_c = min(float(crop_h), y2 - ymin)
                    if x2_c > x1_c and y2_c > y1_c:
                        box_coords = [x1_c, y1_c, x2_c, y2_c]
                    else:
                        has_box = False
                        box_coords = [0.0, 0.0, 0.0, 0.0]
            else:
                crop_h, crop_w = orig_h, orig_w

            # Store in cache
            self.cache[idx] = (img_bgr.copy(), has_box, list(box_coords), crop_h, crop_w, target_idx, stem)

        # 2. Synthetic Caliper Augmentation (Class-Independent, Training only)
        if self.is_train and self.use_caliper_aug:
            img_bgr = self.caliper_aug(img_bgr, target_idx=target_idx)

        # 3. Geometric Augmentations during Training
        h_flipped = False
        if self.is_train:
            # Horizontal Flip (p=0.5)
            if random.random() < 0.5:
                img_bgr = cv2.flip(img_bgr, 1)
                h_flipped = True
                if has_box:
                    x1, y1, x2, y2 = box_coords
                    box_coords = [crop_w - x2, y1, crop_w - x1, y2]
                    
            # Rotation (+- 15 degrees)
            angle = random.uniform(-15.0, 15.0)
            if abs(angle) > 1.0:
                M = cv2.getRotationMatrix2D((crop_w / 2, crop_h / 2), angle, 1.0)
                img_bgr = cv2.warpAffine(img_bgr, M, (crop_w, crop_h), borderMode=cv2.BORDER_REFLECT)
                # Bounding box rotates slightly; we expand bounding box to cover rotated corners if annotated
                if has_box:
                    x1, y1, x2, y2 = box_coords
                    corners = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]])
                    rot_corners = (M @ corners.T).T
                    rx1 = float(np.clip(np.min(rot_corners[:, 0]), 0, crop_w))
                    ry1 = float(np.clip(np.min(rot_corners[:, 1]), 0, crop_h))
                    rx2 = float(np.clip(np.max(rot_corners[:, 0]), 0, crop_w))
                    ry2 = float(np.clip(np.max(rot_corners[:, 1]), 0, crop_h))
                    if rx2 > rx1 and ry2 > ry1:
                        box_coords = [rx1, ry1, rx2, ry2]
                    else:
                        has_box = False
                        box_coords = [0.0, 0.0, 0.0, 0.0]

        # 4. Normalize Bounding Box to [0, 1]^4
        if has_box:
            norm_box = [
                box_coords[0] / crop_w,
                box_coords[1] / crop_h,
                box_coords[2] / crop_w,
                box_coords[3] / crop_h
            ]
            norm_box = [max(0.0, min(1.0, b)) for b in norm_box]
        else:
            norm_box = [0.0, 0.0, 0.0, 0.0]

        # 5. Clinical Preprocessing Transform
        # Grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Resize to input resolution
        resized = cv2.resize(gray, self.input_size, interpolation=cv2.INTER_AREA)
        # 3x3 Median Blur (speckle attenuation)
        blurred = cv2.medianBlur(resized, 3)
        # CLAHE (contrast limited adaptive histogram equalization)
        clahe_out = self.clahe.apply(blurred)
        # 3-channel RGB replicate
        rgb = np.stack([clahe_out, clahe_out, clahe_out], axis=-1)
        
        # Convert to Tensor [3, H, W] in [0, 1]
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        # ImageNet normalization
        tensor = self.normalize(tensor)

        return (
            tensor,
            torch.tensor(target_idx, dtype=torch.long),
            torch.tensor(1.0 if has_box else 0.0, dtype=torch.float32),
            torch.tensor(norm_box, dtype=torch.float32),
            stem
        )
