import os
import sys
import time
import json
import random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# -------------------------------------------------------------
# Configuration & Seed
# -------------------------------------------------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path("d:/Project/UteroScan")
DATASETS_DIR = BASE_DIR / "datasets" / "datasets"
BATCH_SIZE = 16
NUM_EPOCHS = 2
LR = 1e-3
NUM_CLASSES = 2  # UF=0, NUF=1

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------------------------------------------
# Preprocessing Transforms
# -------------------------------------------------------------
def preprocess_variant_a(img_bgr):
    """Current Refined: Resize + 3x3 Median Blur + CLAHE (clip=1.2, grid=(8,8))"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
    denoised = cv2.medianBlur(resized, 3)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return rgb

def preprocess_variant_b(img_bgr):
    """Light Variant: Resize + CLAHE (clip=1.2, grid=(8,8)), NO median filter"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return rgb

def preprocess_variant_c(img_bgr):
    """Raw Baseline: Resize only (Grayscale -> RGB), NO median, NO CLAHE"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return rgb

def to_tensor_normalized(rgb_img):
    # ImageNet normalization
    arr = rgb_img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    # HWC -> CHW
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    return tensor

# -------------------------------------------------------------
# Dataset
# -------------------------------------------------------------
class SmokeTestDataset(Dataset):
    def __init__(self, samples, transform_fn):
        self.samples = samples  # list of (path, label)
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        rgb = self.transform_fn(bgr)
        tensor = to_tensor_normalized(rgb)
        return tensor, torch.tensor(label, dtype=torch.long)

# -------------------------------------------------------------
# Data Partitioning (50/class train, 25/class val)
# -------------------------------------------------------------
def build_subsets():
    set_seed(SEED)
    train_uf_files = sorted(list((DATASETS_DIR / "train" / "UF").glob("*.jpg")))
    train_nuf_files = sorted(list((DATASETS_DIR / "train" / "NUF").glob("*.jpg")))
    test_uf_files = sorted(list((DATASETS_DIR / "test" / "UF").glob("*.jpg")))
    test_nuf_files = sorted(list((DATASETS_DIR / "test" / "NUF").glob("*.jpg")))

    # Randomly select 50 train UF, 50 train NUF
    random.shuffle(train_uf_files)
    random.shuffle(train_nuf_files)
    train_samples = [(p, 0) for p in train_uf_files[:50]] + [(p, 1) for p in train_nuf_files[:50]]

    # Randomly select 25 val UF, 25 val NUF from test split
    random.shuffle(test_uf_files)
    random.shuffle(test_nuf_files)
    val_samples = [(p, 0) for p in test_uf_files[:25]] + [(p, 1) for p in test_nuf_files[:25]]

    return train_samples, val_samples

# -------------------------------------------------------------
# Training Routine
# -------------------------------------------------------------
def train_variant(variant_name, transform_fn, train_samples, val_samples, pretrained_weights_path=None):
    print("\n" + "=" * 75)
    print(f"STARTING SMOKE TEST: {variant_name.upper()}")
    print("=" * 75, flush=True)

    set_seed(SEED)
    train_ds = SmokeTestDataset(train_samples, transform_fn)
    val_ds = SmokeTestDataset(val_samples, transform_fn)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize ResNet50
    set_seed(SEED)
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except Exception as e:
        print(f"  Note: downloading DEFAULT weights ({e}), retrying...", flush=True)
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze backbone (Stage 1 per Phase 3 spec)
    for param in model.parameters():
        param.requires_grad = False

    # Replace head
    in_features = model.fc.in_features
    # Fixed initialization for head
    set_seed(SEED)
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    history = []
    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_t0 = time.time()
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += imgs.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_idx, (imgs, targets) in enumerate(val_loader):
                imgs = imgs.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += imgs.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_dur = time.time() - epoch_t0

        print(f"  Epoch [{epoch}/{NUM_EPOCHS}] ({epoch_dur:.2f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:5.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:5.1f}%", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": float(epoch_train_loss),
            "train_acc": float(epoch_train_acc),
            "val_loss": float(epoch_val_loss),
            "val_acc": float(epoch_val_acc),
            "epoch_seconds": float(epoch_dur)
        })

    tot_time = time.time() - t_start
    print(f"Completed {variant_name} in {tot_time:.2f}s")
    return history

def main():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Target Device  : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU Model      : {torch.cuda.get_device_name(0)}")

    train_samples, val_samples = build_subsets()
    print(f"Prepared Subsets:")
    print(f"  Train: {len(train_samples)} images (50 UF, 50 NUF)")
    print(f"  Val  : {len(val_samples)} images (25 UF, 25 NUF)")

    variants = [
        ("Current Refined (Resize + 3x3 Median + CLAHE clip=1.2)", preprocess_variant_a),
        ("Light Variant (Resize + CLAHE clip=1.2, No Median)", preprocess_variant_b),
        ("Raw Baseline (Resize Only, No Median, No CLAHE)", preprocess_variant_c)
    ]

    all_results = {}
    for name, fn in variants:
        hist = train_variant(name, fn, train_samples, val_samples)
        all_results[name] = hist

    # Summary Table
    print("\n" + "=" * 85)
    print("ABLATION SMOKE-TEST SUMMARY (2 Epochs, 50 img/class Train, 25 img/class Val)")
    print("=" * 85)
    print(f"{'Variant':<45} | {'Val Loss (E1 -> E2)':<20} | {'Val Acc (E1 -> E2)'}")
    print("-" * 85)
    for name, hist in all_results.items():
        e1_loss, e2_loss = hist[0]["val_loss"], hist[1]["val_loss"]
        e1_acc, e2_acc = hist[0]["val_acc"] * 100, hist[1]["val_acc"] * 100
        loss_str = f"{e1_loss:.4f} -> {e2_loss:.4f}"
        acc_str = f"{e1_acc:.1f}% -> {e2_acc:.1f}%"
        print(f"{name:<45} | {loss_str:<20} | {acc_str}")
    print("=" * 85)

    # Save to reports
    out_json = BASE_DIR / "reports" / "ablation_smoketest_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved detailed metrics to {out_json}")

if __name__ == "__main__":
    main()
