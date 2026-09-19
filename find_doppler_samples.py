import cv2
import numpy as np
from pathlib import Path
from collections import Counter
import sys

BASE_DIR = Path("datasets/datasets")
REPORTS_DIR = Path("reports/preprocessing_samples")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR = Path(r"C:\Users\ashis\.gemini\antigravity-ide\brain\af0e99e6-8ff9-4623-ad48-86540f6228c9")

sys.path.append("DCGAN&MobileNetV2/DCGAN&MobileNetV2")
from utils.preprocessing import preprocess_ultrasound

folders = ["train/UF", "train/NUF", "test/UF", "test/NUF"]

print("Scanning for true Red+Blue Color Doppler frames...", flush=True)

# Find images with BOTH red and blue pixels (hallmark of color Doppler flow map)
candidates = []
for f in folders:
    imgs = sorted(list((BASE_DIR / f).glob("*.jpg")))
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        # ultrasound sector area
        roi = img[int(h*0.15):int(h*0.85), int(w*0.1):int(w*0.9)]
        b = roi[:, :, 0].astype(int)
        g = roi[:, :, 1].astype(int)
        r = roi[:, :, 2].astype(int)
        
        # Red pixels: high R, lower B and G
        red_mask = (r > 120) & (r - b > 40) & (r - g > 30)
        # Blue pixels: high B, lower R and G
        blue_mask = (b > 100) & (b - r > 40)
        
        red_count = int(np.sum(red_mask))
        blue_count = int(np.sum(blue_mask))
        
        if red_count > 150 and blue_count > 80:
            candidates.append((f, p, red_count, blue_count))

print(f"Total images with genuine Red + Blue Doppler flow: {len(candidates)}", flush=True)
counts = Counter([c[0] for c in candidates])
for k, v in counts.items():
    print(f"  {k}: {v}", flush=True)

# Filter by UF and NUF
uf_cands = [c for c in candidates if "UF" in c[0] and "NUF" not in c[0]]
nuf_cands = [c for c in candidates if "NUF" in c[0]]

print(f"UF Color Doppler candidates: {len(uf_cands)}", flush=True)
print(f"NUF Color Doppler candidates: {len(nuf_cands)}", flush=True)

# Select 2 UF and 2 NUF
selected = []
if len(uf_cands) >= 2:
    # sort by combined red+blue
    uf_cands.sort(key=lambda x: x[2] + x[3], reverse=True)
    selected.extend(uf_cands[:2])
else:
    selected.extend(uf_cands)

if len(nuf_cands) >= 2:
    nuf_cands.sort(key=lambda x: x[2] + x[3], reverse=True)
    selected.extend(nuf_cands[:2])
else:
    selected.extend(nuf_cands)

print("\nGenerating side-by-side comparison for selected Doppler flow samples...", flush=True)
for idx, (folder, p, r_cnt, b_cnt) in enumerate(selected, 1):
    raw_bgr = cv2.imread(str(p))
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    
    prep_rgb = preprocess_ultrasound(raw_rgb, target_size=(224, 224))
    prep_bgr = cv2.cvtColor(prep_rgb, cv2.COLOR_RGB2BGR)
    
    raw_vis = cv2.resize(raw_bgr, (224, 224))
    
    label = folder.split("/")[-1]
    cv2.putText(raw_vis, f"DOPPLER RAW ({label})", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(prep_bgr, f"GRAY PREP ({label})", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)
    
    divider = np.full((224, 4, 3), 80, dtype=np.uint8)
    montage = np.hstack([raw_vis, divider, prep_bgr])
    
    out_name = f"true_doppler_{idx:02d}_{label}_{p.stem}.png"
    cv2.imwrite(str(REPORTS_DIR / out_name), montage)
    cv2.imwrite(str(ART_DIR / out_name), montage)
    print(f"  Sample {idx}: {out_name} (Red: {r_cnt}, Blue: {b_cnt})", flush=True)

print("\nFinished find_doppler_samples.py successfully!", flush=True)
