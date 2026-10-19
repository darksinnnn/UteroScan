#!/usr/bin/env python3
"""
Sequential Experiment Runner for Controlled Ablations (E0, E1, E2, E3)
Location: experiments/v4_controlled/src/train_experiment.py

Runs 5-fold cross-validation sequentially on GPU, evaluates out-of-fold validation metrics,
calibrates threshold exclusively on validation folds, and saves reports and checkpoints.
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("d:/Project/UteroScan")
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "v4_controlled" / "src"))

from dataset import ControlledUltrasoundDataset
from models import ResNet50Baseline, ResNet50AuxLocalization
from metrics import compute_roc_auc, compute_pr_auc, evaluate_metrics_at_threshold, optimize_threshold_on_val

FOLDS_CSV = PROJECT_ROOT / "experiments" / "v4_controlled" / "data" / "folds.csv"
TEST_LINES_FILE = PROJECT_ROOT / "DCGAN&MobileNetV2" / "DCGAN&MobileNetV2" / "cls_test.txt"
CHECKPOINTS_DIR = PROJECT_ROOT / "experiments" / "v4_controlled" / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "v4_controlled" / "reports"
LIVE_PROGRESS_FILE = REPORTS_DIR / "LIVE_PROGRESS.md"

def write_live_status(exp_name, fold, epoch, max_epochs, history_rows, status="Running", extra_notes=""):
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LIVE_PROGRESS_FILE, "w", encoding="utf-8") as f:
            f.write(f"# 📊 UteroScan V4 Controlled Training: Live Progress Tracker\n\n")
            f.write(f"**Active Experiment:** `{exp_name.upper()}`  \n")
            f.write(f"**Status:** **{status}**  \n")
            f.write(f"**Last Updated:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`  \n")
            f.write(f"**Current State:** Fold {fold}/4 | Epoch {epoch}/{max_epochs}  \n\n")
            
            if extra_notes:
                f.write(f"> {extra_notes}\n\n")
                
            f.write("### Fold Execution Log\n\n")
            f.write("| Fold | Epoch | Time | Train Loss | Val Loss | Val Acc | UF Sens | NUF Spec | ROC-AUC | Note |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for r in history_rows:
                f.write(f"| Fold {r['fold']} | Ep {r['epoch']:02d} | {r['dur']:.1f}s | {r['train_loss']:.4f} | {r['val_loss']:.4f} | {r['val_acc']*100:.1f}% | {r['val_sens']*100:.1f}% | {r['val_spec']*100:.1f}% | {r['val_auc']:.4f} | {r['note']} |\n")
            f.flush()
    except Exception as e:
        pass

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_experiment(exp_name,
                   use_viewport=False,
                   use_caliper_aug=False,
                   use_aux_localization=False,
                   aux_weight=0.1,
                   caliper_p=0.25,
                   input_size=(224, 224),
                   max_epochs=10,
                   batch_size=16,
                   device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print("\n" + "=" * 80)
    print(f"STARTING EXPERIMENT: {exp_name.upper()}")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Device              : {device}")
    print(f"  Viewport Cropping   : {use_viewport}")
    print(f"  Caliper Augmentation: {use_caliper_aug} (p={caliper_p})")
    print(f"  Aux Localization    : {use_aux_localization} (lambda={aux_weight})")
    print(f"  Input Resolution    : {input_size}")
    print(f"  Max Epochs per Fold : {max_epochs}")
    print(f"  Batch Size          : {batch_size}")

    df_folds = pd.read_csv(FOLDS_CSV)
    set_seed(42)

    fold_metrics_list = []
    oof_predictions = [] # (stem, true_label, p_uf)
    history_rows = []

    cls_criterion = nn.CrossEntropyLoss()
    smooth_l1 = nn.SmoothL1Loss(reduction="none")
    write_live_status(exp_name, 0, 0, max_epochs, history_rows, status="Initializing Experiment...")

    for fold in range(5):
        print("\n" + "-" * 70)
        print(f"[{exp_name.upper()}] TRAINING FOLD {fold}/4")
        print("-" * 70)

        train_df = df_folds[df_folds["fold"] != fold].copy()
        val_df = df_folds[df_folds["fold"] == fold].copy()

        train_ds = ControlledUltrasoundDataset(
            train_df,
            input_size=input_size,
            is_train=True,
            use_viewport=use_viewport,
            use_caliper_aug=use_caliper_aug,
            caliper_p=caliper_p,
            project_root=str(PROJECT_ROOT)
        )
        val_ds = ControlledUltrasoundDataset(
            val_df,
            input_size=input_size,
            is_train=False,
            use_viewport=use_viewport,
            use_caliper_aug=False,
            project_root=str(PROJECT_ROOT)
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Build Model
        if use_aux_localization:
            model = ResNet50AuxLocalization(num_classes=2, pretrained=True).to(device)
        else:
            model = ResNet50Baseline(num_classes=2, pretrained=True).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

        best_val_loss = float("inf")
        best_val_auc = 0.0
        best_state = None
        best_val_probs = None
        best_val_targets = None
        best_val_stems = None
        patience = 4
        patience_counter = 0

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            model.train()
            train_loss = 0.0
            total_train = 0

            for imgs, targets, has_boxes, norm_boxes, stems in train_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                has_boxes = has_boxes.to(device)
                norm_boxes = norm_boxes.to(device)

                optimizer.zero_grad()

                if use_aux_localization:
                    logits, pred_boxes = model(imgs)
                    l_cls = cls_criterion(logits, targets)
                    
                    # Localization loss only on positive annotated training images
                    box_loss_raw = smooth_l1(pred_boxes, norm_boxes).mean(dim=1)
                    l_loc = (box_loss_raw * has_boxes).sum() / (has_boxes.sum() + 1e-6)
                    
                    loss = l_cls + aux_weight * l_loc
                else:
                    logits = model(imgs)
                    loss = cls_criterion(logits, targets)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * imgs.size(0)
                total_train += imgs.size(0)

            train_loss /= total_train

            # Validation Loop
            model.eval()
            val_loss = 0.0
            total_val = 0
            val_preds_uf = []
            val_targets_list = []
            val_stems_list = []

            with torch.no_grad():
                for imgs, targets, _, _, stems in val_loader:
                    imgs = imgs.to(device)
                    targets = targets.to(device)

                    if use_aux_localization:
                        logits, _ = model(imgs)
                    else:
                        logits = model(imgs)

                    loss = cls_criterion(logits, targets)
                    probs = torch.softmax(logits, dim=1)

                    val_loss += loss.item() * imgs.size(0)
                    total_val += imgs.size(0)

                    # Col 0 is UF probability
                    val_preds_uf.extend(probs[:, 0].cpu().numpy().tolist())
                    val_targets_list.extend(targets.cpu().numpy().tolist())
                    val_stems_list.extend(stems)

            val_loss /= total_val
            y_val_uf = (np.array(val_targets_list) == 0).astype(int)
            p_val_uf = np.array(val_preds_uf)
            val_auc = compute_roc_auc(y_val_uf, p_val_uf)
            m_t50 = evaluate_metrics_at_threshold(y_val_uf, p_val_uf, threshold_uf=0.50)

            dur = time.time() - t0
            print(f"  Epoch {epoch:02d}/{max_epochs:02d} ({dur:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {m_t50['accuracy']*100:.1f}% | Val Sens: {m_t50['uf_sensitivity']*100:.1f}% | Val Spec: {m_t50['nuf_specificity']*100:.1f}% | AUC: {val_auc:.4f}", flush=True)

            is_best = val_loss < best_val_loss
            note = "★ New Best" if is_best else ""
            history_rows.append({
                "fold": fold,
                "epoch": epoch,
                "dur": dur,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": m_t50['accuracy'],
                "val_sens": m_t50['uf_sensitivity'],
                "val_spec": m_t50['nuf_specificity'],
                "val_auc": val_auc,
                "note": note
            })
            write_live_status(exp_name, fold, epoch, max_epochs, history_rows, status=f"Training Fold {fold}/4 (Epoch {epoch}/{max_epochs})")

            # Early stopping and model selection on validation loss & AUC
            if is_best:
                best_val_loss = val_loss
                best_val_auc = val_auc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_val_probs = p_val_uf
                best_val_targets = y_val_uf
                best_val_stems = val_stems_list
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping triggered at epoch {epoch} (patience={patience}).")
                    break

        # Save fold checkpoint
        ckpt_path = CHECKPOINTS_DIR / f"{exp_name}_fold_{fold}_best.pth"
        torch.save({
            "fold": fold,
            "experiment": exp_name,
            "best_val_loss": best_val_loss,
            "best_val_auc": best_val_auc,
            "model_state_dict": best_state
        }, ckpt_path)
        print(f"  [SAVED] Checkpoint -> {ckpt_path.name}")
        write_live_status(exp_name, fold, epoch, max_epochs, history_rows, status=f"Fold {fold} Complete -> Saved Checkpoint {ckpt_path.name}")

        # Caliper Augmentation stats for this fold
        caliper_fold_stats = None
        if use_caliper_aug and hasattr(train_ds, "caliper_aug") and train_ds.caliper_aug is not None:
            c_stats = train_ds.caliper_aug.get_stats()
            uf_pct = (c_stats["uf_aug"] / max(1, c_stats["uf_total"])) * 100.0 if c_stats["uf_total"] > 0 else 0.0
            nuf_pct = (c_stats["nuf_aug"] / max(1, c_stats["nuf_total"])) * 100.0 if c_stats["nuf_total"] > 0 else 0.0
            print(f"  [FOLD {fold} CALIPER STATS] UF Augmented: {c_stats['uf_aug']}/{c_stats['uf_total']} ({uf_pct:.1f}%) | NUF Augmented: {c_stats['nuf_aug']}/{c_stats['nuf_total']} ({nuf_pct:.1f}%)")
            caliper_fold_stats = {
                "uf_total": c_stats["uf_total"],
                "uf_aug": c_stats["uf_aug"],
                "uf_pct": float(uf_pct),
                "nuf_total": c_stats["nuf_total"],
                "nuf_aug": c_stats["nuf_aug"],
                "nuf_pct": float(nuf_pct)
            }

        # Record OOF predictions
        for s, yt, p in zip(best_val_stems, best_val_targets, best_val_probs):
            oof_predictions.append({"stem": s, "fold": fold, "y_true_uf": int(yt), "p_uf": float(p)})

        # Fold metrics at T=0.50
        fold_m = evaluate_metrics_at_threshold(best_val_targets, best_val_probs, threshold_uf=0.50)
        fold_m["caliper_stats"] = caliper_fold_stats
        fold_metrics_list.append(fold_m)

    # Aggregate Out-of-Fold Evaluation
    oof_df = pd.DataFrame(oof_predictions)
    oof_y_true = oof_df["y_true_uf"].values
    oof_p_uf = oof_df["p_uf"].values

    oof_t50 = evaluate_metrics_at_threshold(oof_y_true, oof_p_uf, threshold_uf=0.50)
    # Optimize threshold on validation predictions exclusively
    opt_tau_uf, oof_opt = optimize_threshold_on_val(oof_y_true, oof_p_uf, objective="youden_j")

    # Compute fold mean and std
    accs = [m["accuracy"] for m in fold_metrics_list]
    bal_accs = [m.get("balanced_accuracy", (m["uf_sensitivity"] + m["nuf_specificity"]) / 2.0) for m in fold_metrics_list]
    sens = [m["uf_sensitivity"] for m in fold_metrics_list]
    specs = [m["nuf_specificity"] for m in fold_metrics_list]
    aucs = [m["roc_auc"] for m in fold_metrics_list]
    f1s = [m["f1"] for m in fold_metrics_list]

    print("\n" + "=" * 80)
    print(f"[{exp_name.upper()}] 5-FOLD CROSS-VALIDATION RESULTS:")
    print("=" * 80)
    print(f"Mean Fold Accuracy    : {np.mean(accs)*100:5.2f}% +/- {np.std(accs)*100:4.2f}%")
    print(f"Mean Fold Balanced Acc: {np.mean(bal_accs)*100:5.2f}% +/- {np.std(bal_accs)*100:4.2f}%")
    print(f"Mean Fold UF Sens     : {np.mean(sens)*100:5.2f}% +/- {np.std(sens)*100:4.2f}%")
    print(f"Mean Fold NUF Spec    : {np.mean(specs)*100:5.2f}% +/- {np.std(specs)*100:4.2f}%")
    print(f"Mean Fold ROC-AUC     : {np.mean(aucs):6.4f} +/- {np.std(aucs):6.4f}")
    print(f"Mean Fold F1          : {np.mean(f1s)*100:5.2f}% +/- {np.std(f1s)*100:4.2f}%")
    print(f"\nPooled Out-of-Fold Metrics (T=0.50):")
    print(f"  Acc: {oof_t50['accuracy']*100:5.2f}% | Sens: {oof_t50['uf_sensitivity']*100:5.2f}% | Spec: {oof_t50['nuf_specificity']*100:5.2f}% | AUC: {oof_t50['roc_auc']:.4f} | PR-AUC: {oof_t50['pr_auc']:.4f}")
    print(f"Validation-Optimized Threshold: tau* = {opt_tau_uf:.2f}")
    print(f"  Acc: {oof_opt['accuracy']*100:5.2f}% | Sens: {oof_opt['uf_sensitivity']*100:5.2f}% | Spec: {oof_opt['nuf_specificity']*100:5.2f}% | Youden J: {oof_opt['youden_j']:.4f}")

    # Read-Only Exploratory Evaluation on 321 Held-Out Test Scans
    print("\nEvaluating 5-Fold Ensemble on 321 Clean Held-Out Test Scans (READ-ONLY EXPLORATORY)...")
    with open(TEST_LINES_FILE) as f:
        test_lines = [l.strip() for l in f if l.strip()]

    test_df_records = []
    manifest_df = pd.read_csv(PROJECT_ROOT / "data" / "authoritative_manifest.csv")
    man_map = manifest_df.set_index("filename").to_dict("index")
    for l in test_lines:
        fn = os.path.basename(l.split(";")[1])
        if fn in man_map:
            row = man_map[fn]
            test_df_records.append({
                "image_path": row["image_path"],
                "class_label": row["class_label"],
                "target_idx": 0 if row["class_label"] == "UF" else 1,
                "source_stem": fn.replace(".jpg", "")
            })
    test_df = pd.DataFrame(test_df_records)

    test_ds = ControlledUltrasoundDataset(
        test_df,
        input_size=input_size,
        is_train=False,
        use_viewport=use_viewport,
        use_caliper_aug=False,
        project_root=str(PROJECT_ROOT)
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    ensemble_test_probs = np.zeros(len(test_df))
    for fold in range(5):
        ckpt_path = CHECKPOINTS_DIR / f"{exp_name}_fold_{fold}_best.pth"
        st = torch.load(ckpt_path, map_location=device)["model_state_dict"]
        if use_aux_localization:
            f_model = ResNet50AuxLocalization(num_classes=2, pretrained=False).to(device)
        else:
            f_model = ResNet50Baseline(num_classes=2, pretrained=False).to(device)
        f_model.load_state_dict(st)
        f_model.eval()

        fold_p = []
        with torch.no_grad():
            for imgs, _, _, _, _ in test_loader:
                imgs = imgs.to(device)
                if use_aux_localization:
                    out, _ = f_model(imgs)
                else:
                    out = f_model(imgs)
                probs = torch.softmax(out, dim=1)[:, 0].cpu().numpy()
                fold_p.extend(probs.tolist())
        ensemble_test_probs += np.array(fold_p) / 5.0

    test_y_true = (test_df["target_idx"].values == 0).astype(int)
    test_m_t50 = evaluate_metrics_at_threshold(test_y_true, ensemble_test_probs, threshold_uf=0.50)
    test_m_opt = evaluate_metrics_at_threshold(test_y_true, ensemble_test_probs, threshold_uf=opt_tau_uf)

    print(f"Test Ensemble (T=0.50): Acc={test_m_t50['accuracy']*100:.2f}%, Sens={test_m_t50['uf_sensitivity']*100:.2f}%, Spec={test_m_t50['nuf_specificity']*100:.2f}%, AUC={test_m_t50['roc_auc']:.4f}")
    print(f"Test Ensemble (T*={opt_tau_uf:.2f}): Acc={test_m_opt['accuracy']*100:.2f}%, Sens={test_m_opt['uf_sensitivity']*100:.2f}%, Spec={test_m_opt['nuf_specificity']*100:.2f}%")

    # Compile Final Report
    report_data = {
        "experiment_name": exp_name,
        "configuration": {
            "use_viewport": use_viewport,
            "use_caliper_aug": use_caliper_aug,
            "caliper_p": caliper_p,
            "use_aux_localization": use_aux_localization,
            "aux_weight": aux_weight,
            "input_size": list(input_size),
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "device": str(device)
        },
        "fold_summary": {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_balanced_accuracy": float(np.mean(bal_accs)),
            "std_balanced_accuracy": float(np.std(bal_accs)),
            "mean_uf_sensitivity": float(np.mean(sens)),
            "std_uf_sensitivity": float(np.std(sens)),
            "mean_nuf_specificity": float(np.mean(specs)),
            "std_nuf_specificity": float(np.std(specs)),
            "mean_roc_auc": float(np.mean(aucs)),
            "std_roc_auc": float(np.std(aucs)),
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s))
        },
        "individual_folds": fold_metrics_list,
        "pooled_out_of_fold_validation": {
            "metrics_at_t50": oof_t50,
            "optimal_threshold_tau_star": float(opt_tau_uf),
            "metrics_at_optimal_threshold": oof_opt
        },
        "exploratory_test_set_ensemble": {
            "test_metrics_at_t50": test_m_t50,
            "test_metrics_at_tau_star": test_m_opt
        }
    }

    # Comparison against E0 baseline if available
    e0_report_file = REPORTS_DIR / "e0_baseline_report.json"
    if exp_name != "e0_baseline" and e0_report_file.exists():
        try:
            with open(e0_report_file, "r", encoding="utf-8") as f:
                e0_data = json.load(f)
            e0_sum = e0_data["fold_summary"]
            e0_comp = {
                "e0_mean_accuracy": e0_sum["mean_accuracy"],
                "e0_mean_uf_sensitivity": e0_sum["mean_uf_sensitivity"],
                "e0_mean_nuf_specificity": e0_sum["mean_nuf_specificity"],
                "e0_mean_roc_auc": e0_sum["mean_roc_auc"],
                "delta_accuracy": float(np.mean(accs) - e0_sum["mean_accuracy"]),
                "delta_uf_sensitivity": float(np.mean(sens) - e0_sum["mean_uf_sensitivity"]),
                "delta_nuf_specificity": float(np.mean(specs) - e0_sum["mean_nuf_specificity"]),
                "delta_roc_auc": float(np.mean(aucs) - e0_sum["mean_roc_auc"]),
            }
            report_data["comparison_with_e0_baseline"] = e0_comp
            print("\n" + "=" * 80)
            print(f"COMPARISON AGAINST E0 BASELINE:")
            print("=" * 80)
            print(f"  Mean Accuracy : {e0_comp['e0_mean_accuracy']*100:.2f}% -> {np.mean(accs)*100:.2f}% (Delta: {e0_comp['delta_accuracy']*100:+.2f}%)")
            print(f"  UF Sensitivity: {e0_comp['e0_mean_uf_sensitivity']*100:.2f}% -> {np.mean(sens)*100:.2f}% (Delta: {e0_comp['delta_uf_sensitivity']*100:+.2f}%)")
            print(f"  NUF Specific  : {e0_comp['e0_mean_nuf_specificity']*100:.2f}% -> {np.mean(specs)*100:.2f}% (Delta: {e0_comp['delta_nuf_specificity']*100:+.2f}%)")
            print(f"  Mean ROC-AUC  : {e0_comp['e0_mean_roc_auc']:.4f} -> {np.mean(aucs):.4f} (Delta: {e0_comp['delta_roc_auc']:+.4f})")
        except Exception as e:
            print(f"Could not load e0 comparison: {e}")

    report_path = REPORTS_DIR / f"{exp_name}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[SAVED] Experiment Report -> {report_path}")

    write_live_status(exp_name, 4, max_epochs, max_epochs, history_rows, status="Experiment Complete",
                      extra_notes=f"**{exp_name.upper()} Complete!** Mean 5-Fold Acc: {np.mean(accs)*100:.2f}%, Test Ensemble Acc: {test_m_t50['accuracy']*100:.2f}%, Test Ensemble AUC: {test_m_t50['roc_auc']:.4f}")

    return report_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, choices=["e0", "e1", "e2", "e3", "remaining"])
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    if args.exp == "e0":
        run_experiment("e0_baseline", use_viewport=False, use_caliper_aug=False, use_aux_localization=False, max_epochs=args.epochs)
    elif args.exp == "e1":
        run_experiment("e1_viewport", use_viewport=True, use_caliper_aug=False, use_aux_localization=False, max_epochs=args.epochs)
    elif args.exp == "e2":
        # Full-frame E0 baseline + Class-Independent Synthetic Caliper Augmentation (p=0.25)
        run_experiment("e2_caliper", use_viewport=False, use_caliper_aug=True, caliper_p=0.25, use_aux_localization=False, max_epochs=args.epochs)
    elif args.exp == "e3":
        run_experiment("e3_localization", use_viewport=False, use_caliper_aug=True, caliper_p=0.25, use_aux_localization=True, aux_weight=0.1, max_epochs=args.epochs)
    elif args.exp == "remaining":
        print("\n" + "=" * 80)
        print(">>> SEQUENTIAL EXECUTION: RUNNING E2 -> E3 (FULL-FRAME BASELINE)")
        print("=" * 80)
        run_experiment("e2_caliper", use_viewport=False, use_caliper_aug=True, caliper_p=0.25, use_aux_localization=False, max_epochs=args.epochs)
        run_experiment("e3_localization", use_viewport=False, use_caliper_aug=True, caliper_p=0.25, use_aux_localization=True, aux_weight=0.1, max_epochs=args.epochs)
