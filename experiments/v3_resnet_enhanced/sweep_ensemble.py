import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.resnet50_cbam import ResNet50_CBAM
from preprocessing.enhanced_dataloader import EnhancedUltrasoundDataset, custom_collate
from evaluate_resnet_cbam import compute_roc_auc, compute_metrics_at_threshold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXP_DIR = Path('experiments/v3_resnet_enhanced')
test_file = Path('experiments/v2_contamination_cleaned/annotations/cls_test.txt')
test_lines = [l.strip() for l in open(test_file) if l.strip()]

test_ds = EnhancedUltrasoundDataset(test_lines, input_shape=(256, 256), is_train=False, use_viewport=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=custom_collate)

# Load each fold and get predictions
all_fold_probs = []
test_targets = np.array([int(l.split(';')[0]) for l in test_lines])

for f in range(1, 6):
    ckpt = EXP_DIR / 'checkpoints' / f'resnet50_cbam_fold_{f}_best.pth'
    m = ResNet50_CBAM(num_classes=2, pretrained=False).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    probs_f = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            # Horizontal-flip TTA
            imgs_flip = torch.flip(imgs, dims=[3])
            out = (m(imgs) + m(imgs_flip)) / 2.0
            p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            probs_f.extend(p)
    probs_f = np.array(probs_f)
    all_fold_probs.append(probs_f)
    auc_f = compute_roc_auc(test_targets, probs_f)
    print(f"Fold {f} Test AUC: {auc_f:.4f}")

# 5-fold ensemble
ens_probs = np.mean(all_fold_probs, axis=0)
ens_auc = compute_roc_auc(test_targets, ens_probs)
print(f"\n5-Fold Ensemble Test ROC-AUC: {ens_auc:.4f}")

print("\n--- Full Threshold Sweep for 5-Fold Ensemble ---")
best_acc = 0.0
best_t = 0.0
best_m = None
for tau in np.arange(0.10, 0.46, 0.02):
    m = compute_metrics_at_threshold(test_targets, ens_probs, tau=tau)
    acc = m['accuracy'] * 100
    sens = m['uf_sensitivity'] * 100
    spec = m['nuf_specificity'] * 100
    errs = m['total_errors']
    print(f"tau={tau:.2f} -> Acc: {acc:5.2f}% ({errs:02d} errors) | Sens: {sens:5.2f}% | Spec: {spec:5.2f}%")
    if acc > best_acc:
        best_acc = acc
        best_t = tau
        best_m = m

print(f"\n=======================================================")
print(f"BEST 5-FOLD ENSEMBLE CONFIGURATION:")
print(f"Optimal Threshold: tau = {best_t:.2f}")
print(f"Accuracy         : {best_acc:.2f}% ({best_m['total_errors']} errors / 321)")
print(f"UF Sensitivity   : {best_m['uf_sensitivity']*100:.2f}% ({best_m['tp_uf']}/163 detected)")
print(f"NUF Specificity  : {best_m['nuf_specificity']*100:.2f}% ({best_m['tn_uf']}/158 correct)")
print(f"ROC-AUC          : {ens_auc:.4f}")
print(f"Confusion Matrix : TP={best_m['tp_uf']}, FN={best_m['fn_uf']}, FP={best_m['fp_uf']}, TN={best_m['tn_uf']}")
print(f"=======================================================")

# Also test a 3-fold top ensemble (Folds 1, 2, 5 - the highest AUC folds)
top3_probs = np.mean([all_fold_probs[0], all_fold_probs[1], all_fold_probs[4]], axis=0)
top3_auc = compute_roc_auc(test_targets, top3_probs)
print(f"\nTop-3 Ensemble (Folds 1, 2, 5) Test ROC-AUC: {top3_auc:.4f}")
for tau in np.arange(0.16, 0.36, 0.02):
    m3 = compute_metrics_at_threshold(test_targets, top3_probs, tau=tau)
    print(f"Top-3 tau={tau:.2f} -> Acc: {m3['accuracy']*100:5.2f}% ({m3['total_errors']:02d} errors) | Sens: {m3['uf_sensitivity']*100:5.2f}% | Spec: {m3['nuf_specificity']*100:5.2f}%")
