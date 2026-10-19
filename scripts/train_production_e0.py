#!/usr/bin/env python3
"""
UteroScan Production Model Training Script (E0 Final Model)
Location: scripts/train_production_e0.py

Retrains a single standalone production ResNet50 model on the entire
recovered development cohort (1,474 eligible scans) using the verified E0
hyperparameters and the median epoch count selected by the inner-validation
maximum-accuracy rule (Epoch 5).

Strict Safeguards:
  - Trains exclusively on the 1,474 development scans in recovered_folds.csv.
  - Zero access to the 456 held-out evaluation pool scans (HELD_OUT_EVALUATION_ONLY).
  - Zero access to quarantined fetal scans (EXCLUDE_FETUS_VISIBLE).
  - Fixed epoch count = 5 (median of inner-validation selected epochs [3, 5, 5, 5, 6]).
  - Fixed seed = 42 for complete reproducibility.
  - Checkpoint saved to checkpoints/production/resnet50_e0_production.pth.
"""

import os
import sys
import time
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "experiments" / "final_data_audit" / "recovered_folds.csv"
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "production"
OUTPUT_CKPT = OUTPUT_DIR / "resnet50_e0_production.pth"
OUTPUT_META = OUTPUT_DIR / "model_metadata.json"

SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
TARGET_EPOCHS = 5  # Median inner-val selected epoch from Rule 2 (Max Accuracy)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class ProductionUltrasoundDataset(Dataset):
    """
    Dataset loader for production training. Implements exact E0 augmentations and preprocessing.
    """
    def __init__(self, records_df: pd.DataFrame, is_train: bool = True):
        self.records = records_df.reset_index(drop=True)
        self.is_train = is_train
        self.clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        rel_p = row["rel_path"]
        full_p = PROJECT_ROOT / rel_p
        img_bgr = cv2.imread(str(full_p))
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found at {full_p}")
        target_idx = int(row["target_idx"])  # 0=UF, 1=NUF
        stem = str(row["canonical_image_id"])

        h, w = img_bgr.shape[:2]

        # Training-only geometric augmentations
        if self.is_train:
            if random.random() < 0.5:
                img_bgr = cv2.flip(img_bgr, 1)
            angle = random.uniform(-15.0, 15.0)
            if abs(angle) > 1.0:
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                img_bgr = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Standard Clinical Preprocessing (Identical to E0 Inference Pipeline)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
        blurred = cv2.medianBlur(resized, 3)
        clahe_out = self.clahe.apply(blurred)

        # 3-channel RGB replicate
        rgb = np.stack([clahe_out, clahe_out, clahe_out], axis=-1)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        tensor = self.normalize(tensor)

        return tensor, torch.tensor(target_idx, dtype=torch.long), stem


def build_resnet50_model():
    """Builds torchvision ResNet50 with ImageNet V2 weights and 2-class linear head."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model


def main():
    print(f"=== UteroScan Production E0 Retraining ===")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Target Epochs: {TARGET_EPOCHS} (Rule 2 inner-val median: [3, 5, 5, 5, 6])")
    print(f"Manifest: {MANIFEST_PATH}")

    seed_everything(SEED)

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Required manifest not found: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)
    assert len(df) == 1474, f"Expected 1474 development scans, found {len(df)}"
    assert (df["target_idx"] == 0).sum() == 685, "Expected 685 UF scans"
    assert (df["target_idx"] == 1).sum() == 789, "Expected 789 NUF scans"
    print(f"Loaded {len(df)} eligible development scans (UF: 685, NUF: 789). Integrity verified.")

    dataset = ProductionUltrasoundDataset(df, is_train=True)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Safe cross-platform
        pin_memory=torch.cuda.is_available()
    )

    model = build_resnet50_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    epoch_logs = []
    start_time = time.time()

    for epoch in range(1, TARGET_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for inputs, targets, _ in loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        elapsed = time.time() - t0

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": round(epoch_loss, 4),
            "train_acc": round(epoch_acc, 4),
            "elapsed_seconds": round(elapsed, 2)
        })
        print(f"Epoch {epoch}/{TARGET_EPOCHS} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} ({elapsed:.1f}s)")

    total_training_time = time.time() - start_time
    print(f"Training completed in {total_training_time:.1f}s.")

    # Save Checkpoint
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "architecture": "torchvision.models.resnet50",
        "weights": "ResNet50_Weights.IMAGENET1K_V2",
        "num_classes": 2,
        "classes": ["UF", "NUF"],
        "class_to_idx": {"UF": 0, "NUF": 1},
        "idx_to_class": {0: "UF", 1: "NUF"},
        "epochs_trained": TARGET_EPOCHS,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "training_cohort_size": 1474,
        "selection_rule": "Median Rule 2 Max Validation Accuracy Epoch (5 epochs)",
        "training_history": epoch_logs,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    torch.save(checkpoint_payload, str(OUTPUT_CKPT))
    print(f"Saved production checkpoint to: {OUTPUT_CKPT} ({OUTPUT_CKPT.stat().st_size / (1024*1024):.1f} MB)")

    # Save Metadata JSON
    metadata = {
        "model_name": "UteroScan ResNet50 E0 Production Model",
        "model_version": "E0_ResNet50_Production_v1",
        "architecture": "torchvision.models.resnet50",
        "checkpoint_file": "resnet50_e0_production.pth",
        "classes": ["UF", "NUF"],
        "class_mapping": {"0": "UF", "1": "NUF"},
        "epochs_trained": TARGET_EPOCHS,
        "selection_rule": "Median Rule 2 Max Validation Accuracy Epoch across 5 Outer Folds ([3, 5, 5, 5, 6] -> median=5)",
        "training_cohort_size": 1474,
        "cohort_breakdown": {"UF": 685, "NUF": 789},
        "optimizer": "Adam",
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "seed": SEED
        },
        "input_pipeline": {
            "resolution": [224, 224],
            "color_mode": "Grayscale to 3-channel RGB replicate",
            "interpolation": "cv2.INTER_AREA",
            "filter": "3x3 Median Blur",
            "contrast": "CLAHE (clipLimit=1.2, tileGridSize=(8,8))",
            "normalization": {
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD
            }
        },
        "published_cv_benchmark": {
            "protocol": "5-Fold Nested Grouped Cross-Validation (917 study proxy groups)",
            "mean_outer_accuracy": 0.9412,
            "mean_outer_accuracy_std": 0.0064,
            "pooled_accuracy": 0.9410,
            "mean_sensitivity": 0.9475,
            "mean_specificity": 0.9357,
            "mean_f1": 0.9372,
            "mean_roc_auc": 0.9843
        },
        "disclaimer": "Development cross-validation results from recovered cohort. Not an independent hospital patient-level clinical validation.",
        "epoch_logs": epoch_logs,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {OUTPUT_META}")


if __name__ == "__main__":
    main()
