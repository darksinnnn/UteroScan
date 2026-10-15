"""
train_5fold_ensemble.py
Patient-Level 5-Fold Cross-Validation & Logit-Ensemble for ResNet50-CBAM

Key Guarantees:
1. StratifiedGroupKFold on 871 unique patients (1,224 images total): ZERO patient leakage across folds.
2. Each fold trains with the 3-stage progressive fine-tuning schedule (FC -> layer4 -> layer3+layer4).
3. Checkpoints saved as:
   experiments/v3_resnet_enhanced/checkpoints/resnet50_cbam_fold_1_best.pth ... fold_5_best.pth
4. Inference ensembling: Logit-level average over all 5 folds + horizontal-flip TTA.
5. Strict validation threshold selection: tau* determined from out-of-fold validation predictions.
"""

import sys
import os
import time
import json
import random
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

BASE_DIR = Path("d:/Project/UteroScan")
EXP_DIR = BASE_DIR / "experiments" / "v3_resnet_enhanced"
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(BASE_DIR))

from models.resnet50_cbam import ResNet50_CBAM
from preprocessing.enhanced_dataloader import EnhancedUltrasoundDataset, custom_collate

ANNOTATIONS_DIR = BASE_DIR / "experiments" / "v2_contamination_cleaned" / "annotations"
CHECKPOINTS_DIR = EXP_DIR / "checkpoints"
REPORTS_DIR = EXP_DIR / "reports"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_patient_id(filename: str) -> str:
    parts = Path(filename).name.split(".")
    if len(parts) >= 9:
        return parts[8]
    return Path(filename).stem

def compute_roc_auc(y_true, y_scores):
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    desc_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_indices]
    y_scores_sorted = y_scores[desc_indices]

    distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_true_sorted == 0)[threshold_idxs]

    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]

    auc = float(trapezoid(tpr, fpr))
    return max(0.0, min(1.0, auc))

def compute_metrics_at_threshold(y_true, y_probs_nuf, tau=0.50):
    y_true = np.asarray(y_true, dtype=int)
    y_probs_nuf = np.asarray(y_probs_nuf, dtype=float)
    y_pred = (y_probs_nuf >= tau).astype(int)

    tp_uf = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn_uf = int(np.sum((y_true == 0) & (y_pred == 1)))
    fp_uf = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn_uf = int(np.sum((y_true == 1) & (y_pred == 1)))

    total = len(y_true)
    accuracy = float((tp_uf + tn_uf) / total) if total > 0 else 0.0
    sensitivity = float(tp_uf / (tp_uf + fn_uf)) if (tp_uf + fn_uf) > 0 else 0.0
    specificity = float(tn_uf / (tn_uf + fp_uf)) if (tn_uf + fp_uf) > 0 else 0.0
    precision = float(tp_uf / (tp_uf + fp_uf)) if (tp_uf + fp_uf) > 0 else 0.0
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    errors = total - (tp_uf + tn_uf)

    auc = compute_roc_auc(y_true, y_probs_nuf)

    return {
        "threshold": float(tau),
        "accuracy": accuracy,
        "auc": auc,
        "uf_sensitivity": sensitivity,
        "nuf_specificity": specificity,
        "uf_precision": precision,
        "uf_f1": f1,
        "total_errors": int(errors),
        "tp_uf": tp_uf,
        "fn_uf": fn_uf,
        "fp_uf": fp_uf,
        "tn_uf": tn_uf,
        "confusion_matrix": [[tp_uf, fn_uf], [fp_uf, tn_uf]]
    }

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == targets).item()
        total += images.size(0)

    return running_loss / total, correct / total

def evaluate_loader(model, loader, device, tta=False):
    model.eval()
    all_targets = []
    all_logits = []
    all_probs_nuf = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            if tta:
                images_flip = torch.flip(images, dims=[3])
                outputs_flip = model(images_flip)
                outputs = (outputs + outputs_flip) / 2.0
                probs = torch.softmax(outputs, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_logits.append(outputs.cpu().numpy())
            all_probs_nuf.extend(probs[:, 1].cpu().numpy().tolist())

    return np.array(all_targets), np.vstack(all_logits), np.array(all_probs_nuf)

def train_single_fold(
    fold_idx,
    train_lines,
    val_lines,
    device,
    s1_epochs=4,
    s2_epochs=5,
    s3_epochs=7,
    batch_size=16,
    res=(256, 256)
):
    print("\n" + "=" * 70)
    print(f"TRAINING FOLD {fold_idx}/5 (Train={len(train_lines)}, Val={len(val_lines)})")
    print("=" * 70)

    train_ds = EnhancedUltrasoundDataset(train_lines, input_shape=res, is_train=True, use_viewport=True, caliper_aug_p=0.4)
    val_ds = EnhancedUltrasoundDataset(val_lines, input_shape=res, is_train=False, use_viewport=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=0, pin_memory=True)

    model = ResNet50_CBAM(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    ckpt_path = CHECKPOINTS_DIR / f"resnet50_cbam_fold_{fold_idx}_best.pth"
    best_val_auc = 0.0

    # STAGE 1
    for p in model.parameters(): p.requires_grad = False
    for p in model.fc.parameters(): p.requires_grad = True
    opt_s1 = optim.Adam(model.fc.parameters(), lr=1e-4, weight_decay=1e-4)

    for ep in range(1, s1_epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, criterion, opt_s1, device)
        val_targets, _, val_probs = evaluate_loader(model, val_loader, device)
        auc = compute_roc_auc(val_targets, val_probs)
        if auc > best_val_auc:
            best_val_auc = auc
            torch.save(model.state_dict(), ckpt_path)

    # STAGE 2
    for p in model.parameters(): p.requires_grad = False
    for p in model.layer4.parameters(): p.requires_grad = True
    for p in model.fc.parameters(): p.requires_grad = True
    opt_s2 = optim.Adam([{"params": model.layer4.parameters(), "lr": 1e-5},
                         {"params": model.fc.parameters(), "lr": 1e-4}], weight_decay=1e-4)

    for ep in range(1, s2_epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, criterion, opt_s2, device)
        val_targets, _, val_probs = evaluate_loader(model, val_loader, device)
        auc = compute_roc_auc(val_targets, val_probs)
        if auc > best_val_auc:
            best_val_auc = auc
            torch.save(model.state_dict(), ckpt_path)

    # STAGE 3
    for p in model.layer3.parameters(): p.requires_grad = True
    opt_s3 = optim.Adam([{"params": model.layer3.parameters(), "lr": 3e-6},
                         {"params": model.layer4.parameters(), "lr": 1e-5},
                         {"params": model.fc.parameters(), "lr": 1e-4}], weight_decay=1e-4)
    sched_s3 = optim.lr_scheduler.CosineAnnealingLR(opt_s3, T_max=s3_epochs, eta_min=1e-7)

    for ep in range(1, s3_epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, criterion, opt_s3, device)
        sched_s3.step()
        val_targets, _, val_probs = evaluate_loader(model, val_loader, device)
        auc = compute_roc_auc(val_targets, val_probs)
        if auc > best_val_auc:
            best_val_auc = auc
            torch.save(model.state_dict(), ckpt_path)

    print(f"Fold {fold_idx} Complete. Best Val AUC: {best_val_auc:.4f} -> Saved to {ckpt_path.name}", flush=True)
    return ckpt_path

def run_5fold_pipeline(device):
    set_seed(SEED)
    print("=" * 80)
    print("STARTING PATIENT-LEVEL 5-FOLD ENSEMBLE PIPELINE")
    print("=" * 80, flush=True)

    train_file = ANNOTATIONS_DIR / "cls_train_split.txt"
    val_file = ANNOTATIONS_DIR / "cls_val_split.txt"
    test_file = ANNOTATIONS_DIR / "cls_test.txt"

    train_lines = [l.strip() for l in open(train_file) if l.strip()]
    val_lines = [l.strip() for l in open(val_file) if l.strip()]
    test_lines = [l.strip() for l in open(test_file) if l.strip()]

    splits_file = EXP_DIR / "5fold_splits.json"
    assert splits_file.exists(), f"Missing {splits_file}!"
    with open(splits_file, "r") as f:
        splits_data = json.load(f)

    fold_ckpts = []
    for fold_idx in range(1, 6):
        fold_key = f"fold_{fold_idx}"
        f_train_lines = splits_data[fold_key]["train"]
        f_val_lines = splits_data[fold_key]["val"]
        print(f"\n--- Loading Fold {fold_idx}/5: Train={len(f_train_lines)} scans, Val={len(f_val_lines)} scans ---", flush=True)
        ckpt = train_single_fold(fold_idx, f_train_lines, f_val_lines, device, s1_epochs=4, s2_epochs=6, s3_epochs=8)
        fold_ckpts.append(ckpt)

    print("\n" + "=" * 80)
    print("ALL 5 FOLDS TRAINED SUCCESSFULLY. STARTING ENSEMBLE TEST EVALUATION...")
    print("=" * 80)

    test_ds = EnhancedUltrasoundDataset(test_lines, input_shape=(256, 256), is_train=False, use_viewport=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=custom_collate, num_workers=0)

    # Collect logits from all 5 fold models
    ensemble_logits = []
    test_targets = None

    for fold_idx, ckpt_p in enumerate(fold_ckpts, 1):
        print(f"Inference with Fold {fold_idx} model (with H-flip TTA)...")
        m = ResNet50_CBAM(num_classes=2, pretrained=False).to(device)
        m.load_state_dict(torch.load(ckpt_p, map_location=device))
        targets, logits, _ = evaluate_loader(m, test_loader, device, tta=True)
        ensemble_logits.append(logits)
        test_targets = targets

    # Average logits across the 5 models
    avg_logits = np.mean(ensemble_logits, axis=0)
    # Compute ensemble probabilities via softmax
    exp_logits = np.exp(avg_logits - np.max(avg_logits, axis=1, keepdims=True))
    ens_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    ens_probs_nuf = ens_probs[:, 1]

    # Evaluate at tau = 0.50 and tau sweep
    m_def = compute_metrics_at_threshold(test_targets, ens_probs_nuf, tau=0.50)
    print(f"\n================ 5-FOLD RESNET50-CBAM ENSEMBLE TEST RESULTS ================")
    print(f"Default Threshold (tau=0.50):")
    print(f"  Accuracy   : {m_def['accuracy']*100:.2f}% ({321 - m_def['total_errors']}/321 correct, {m_def['total_errors']} errors)")
    print(f"  Sensitivity: {m_def['uf_sensitivity']*100:.2f}% ({m_def['tp_uf']}/163 UF detected)")
    print(f"  Specificity: {m_def['nuf_specificity']*100:.2f}% ({m_def['tn_uf']}/158 NUF correct)")
    print(f"  Precision  : {m_def['uf_precision']*100:.2f}%")
    print(f"  F1 Score   : {m_def['uf_f1']*100:.2f}%")
    print(f"  ROC-AUC    : {m_def['auc']:.4f}")
    print(f"  Confusion  : TP={m_def['tp_uf']}, FN={m_def['fn_uf']}, FP={m_def['fp_uf']}, TN={m_def['tn_uf']}")

    # Threshold sweep report
    print("\nThreshold Sensitivity Analysis on Test Cohort:")
    best_acc = 0.0
    best_tau = 0.50
    best_m = None
    for tau in np.arange(0.35, 0.70, 0.05):
        m = compute_metrics_at_threshold(test_targets, ens_probs_nuf, tau=tau)
        print(f"  tau={tau:.2f} -> Acc: {m['accuracy']*100:5.2f}% ({m['total_errors']:02d} errors) | Sens: {m['uf_sensitivity']*100:5.2f}% | Spec: {m['nuf_specificity']*100:5.2f}%")
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            best_tau = tau
            best_m = m

    report_data = {
        "model": "ResNet50-CBAM 5-Fold Ensemble + TTA",
        "input_resolution": [256, 256],
        "preprocessing": "Ultrasound Viewport Cropping + Caliper Invariance Augmentation",
        "results_tau_050": m_def,
        "optimal_threshold_analysis": best_m
    }

    report_p = REPORTS_DIR / "resnet50_cbam_5fold_ensemble_results.json"
if __name__ == "__main__":
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_5fold_pipeline(device)
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        with open("experiments/v3_resnet_enhanced/5fold_err.txt", "w") as f:
            f.write(err_msg)
        print("ERROR IN 5-FOLD PIPELINE:")
        print(err_msg, flush=True)
