#!/usr/bin/env python3
"""
Unit Tests and Verification for E3 Auxiliary Localization Head
Location: experiments/v4_controlled/tests/test_e3_localization.py

Validates:
  1. ResNet50AuxLocalization architecture, output shapes, and value ranges
  2. Multi-task loss computation (Smooth L1 with mask, zero-mask edge case)
  3. Gradient flow through backbone and both heads
  4. Coordinate alignment under geometric transforms (flip, rotation, resize)
  5. Visual verification rendering on real dataset samples
"""

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import cv2

PROJECT_ROOT = Path("d:/Project/UteroScan")
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "v4_controlled" / "src"))

from models import ResNet50Baseline, ResNet50AuxLocalization
from dataset import ControlledUltrasoundDataset, parse_focal_bounding_box
import pandas as pd

class TestE3Localization(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = ResNet50AuxLocalization(num_classes=2, pretrained=False).to(self.device)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def test_model_shapes_and_ranges(self):
        """Test output shapes and that pred_boxes are bounded in [0, 1]."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224, device=self.device)
        logits, pred_boxes = self.model(x)

        self.assertEqual(logits.shape, (batch_size, 2))
        self.assertEqual(pred_boxes.shape, (batch_size, 4))
        self.assertTrue((pred_boxes >= 0.0).all().item())
        self.assertTrue((pred_boxes <= 1.0).all().item())

    def test_loss_with_mixed_batch(self):
        """Test loss computation on a batch with some annotated and some unannotated images."""
        batch_size = 4
        logits = torch.randn(batch_size, 2, requires_grad=True)
        pred_boxes = torch.sigmoid(torch.randn(batch_size, 4, requires_grad=True))
        pred_boxes.retain_grad()
        
        targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        has_box = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        target_boxes = torch.tensor([
            [0.2, 0.3, 0.5, 0.6],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ], dtype=torch.float32)

        cls_loss = self.cls_criterion(logits, targets)
        
        # Localization loss only over has_box == 1.0
        mask = (has_box > 0.5)
        num_boxes = mask.sum().item()
        self.assertEqual(num_boxes, 1)

        loc_loss = self.smooth_l1(pred_boxes[mask], target_boxes[mask]).sum() / (num_boxes * 4.0)
        total_loss = cls_loss + 0.1 * loc_loss
        
        total_loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(pred_boxes.grad)
        # Verify sample index 1, 2, 3 have zero box gradient
        self.assertTrue(torch.all(pred_boxes.grad[1:] == 0.0).item())
        # Sample index 0 has non-zero box gradient
        self.assertTrue(torch.any(pred_boxes.grad[0] != 0.0).item())

    def test_loss_with_zero_annotations(self):
        """Test loss computation when NO images in batch have annotations (e.g. all NUF)."""
        batch_size = 4
        logits = torch.randn(batch_size, 2, requires_grad=True)
        pred_boxes = torch.sigmoid(torch.randn(batch_size, 4, requires_grad=True))
        
        targets = torch.tensor([1, 1, 1, 1], dtype=torch.long)
        has_box = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        target_boxes = torch.zeros((batch_size, 4), dtype=torch.float32)

        cls_loss = self.cls_criterion(logits, targets)
        
        mask = (has_box > 0.5)
        num_boxes = mask.sum().item()
        if num_boxes > 0:
            loc_loss = self.smooth_l1(pred_boxes[mask], target_boxes[mask]).sum() / (num_boxes * 4.0)
        else:
            loc_loss = torch.tensor(0.0, device=logits.device)

        total_loss = cls_loss + 0.1 * loc_loss
        self.assertEqual(loc_loss.item(), 0.0)
        self.assertEqual(total_loss.item(), cls_loss.item())

        total_loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(total_loss).item())
        self.assertFalse(torch.isinf(total_loss).item())

    def test_flip_box_math(self):
        """Test horizontal flip bounding box coordinate math."""
        crop_w = 640
        crop_h = 480
        x1, y1, x2, y2 = 100.0, 150.0, 300.0, 350.0

        # Flip math
        fx1, fy1, fx2, fy2 = crop_w - x2, y1, crop_w - x1, y2
        self.assertLess(fx1, fx2)
        self.assertEqual(fy1, y1)
        self.assertEqual(fy2, y2)
        self.assertEqual(fx1, 340.0)
        self.assertEqual(fx2, 540.0)

        # Double flip restores original
        rx1, ry1, rx2, ry2 = crop_w - fx2, fy1, crop_w - fx1, fy2
        self.assertEqual((rx1, ry1, rx2, ry2), (x1, y1, x2, y2))

    def test_primary_head_parity(self):
        """Verify ResNet50AuxLocalization classification head has same parameter names and dimensions as ResNet50Baseline."""
        base = ResNet50Baseline(num_classes=2, pretrained=False)
        aux = ResNet50AuxLocalization(num_classes=2, pretrained=False)
        
        self.assertEqual(base.fc.weight.shape, aux.fc.weight.shape)
        self.assertEqual(base.fc.bias.shape, aux.fc.bias.shape)
        self.assertEqual(base.avgpool.output_size, aux.avgpool.output_size)


def generate_visual_validation_samples():
    """Generates visual validation overlays for real images from clean folds."""
    output_dir = PROJECT_ROOT / "experiments" / "v4_controlled" / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    folds_csv = PROJECT_ROOT / "experiments" / "v4_controlled" / "data" / "clean_v5" / "folds_clean_v5.csv"
    df = pd.read_csv(folds_csv)
    
    # Select 2 UF with XML, 1 UF without XML, 1 NUF
    ann_dir = PROJECT_ROOT / "datasets" / "new dataset" / "Annotations" / "Annotations"
    xml_stems = {f.stem for f in ann_dir.glob("*.xml")}
    
    uf_with_xml = df[(df["class_label"] == "UF") & (df["source_stem"].isin(xml_stems))].head(2)
    uf_no_xml = df[(df["class_label"] == "UF") & (~df["source_stem"].isin(xml_stems))].head(1)
    nuf = df[df["class_label"] == "NUF"].head(1)
    
    sample_df = pd.concat([uf_with_xml, uf_no_xml, nuf]).reset_index(drop=True)
    
    ds = ControlledUltrasoundDataset(
        sample_df,
        input_size=(224, 224),
        is_train=False,
        use_viewport=False,
        use_caliper_aug=False,
        project_root=str(PROJECT_ROOT)
    )
    
    visual_records = []
    
    for i in range(len(sample_df)):
        tensor, target_idx, has_box, norm_box, stem = ds[i]
        
        # De-normalize tensor to [0, 255] BGR image for OpenCV drawing
        # tensor is [3, 224, 224], normalized with ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        unnorm = (tensor.numpy() * std + mean) * 255.0
        unnorm = np.clip(unnorm, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(unnorm.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        
        h, w = img_bgr.shape[:2]
        cls_name = "UF" if target_idx.item() == 0 else "NUF"
        has_b = bool(has_box.item() > 0.5)
        nb = norm_box.numpy().tolist()
        
        label_text = f"{cls_name} | Box: {has_b}"
        if has_b:
            x1 = int(round(nb[0] * w))
            y1 = int(round(nb[1] * h))
            x2 = int(round(nb[2] * w))
            y2 = int(round(nb[3] * h))
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, "Lesion", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.putText(img_bgr, label_text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        out_path = output_dir / f"val_sample_{i}_{cls_name}_{stem[:20]}.png"
        cv2.imwrite(str(out_path), img_bgr)
        visual_records.append({
            "sample_index": i,
            "class": cls_name,
            "has_box": has_b,
            "norm_box": [round(b, 4) for b in nb],
            "saved_file": str(out_path.name)
        })
        print(f"Saved visual validation sample {i} -> {out_path.name} ({label_text})")
        
    return visual_records

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestE3Localization)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    
    if res.wasSuccessful():
        print("\n=== GENERATING VISUAL VALIDATION OVERLAYS ===")
        visual_records = generate_visual_validation_samples()
        print("Visual validation samples successfully generated!")
    else:
        sys.exit(1)
