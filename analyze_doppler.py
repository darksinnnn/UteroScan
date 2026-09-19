import cv2
import numpy as np
from pathlib import Path

DATASETS_DIR = Path("datasets/datasets")
REPORTS_DIR = Path("reports/preprocessing_samples")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR = Path(r"C:\Users\ashis\.gemini\antigravity-ide\brain\af0e99e6-8ff9-4623-ad48-86540f6228c9")

folders = ['train/UF', 'train/NUF', 'test/UF', 'test/NUF']

# Scan all images
stats = {f: {'total': 0, 'doppler_flow': 0, 'text_color_only': 0, 'pure_gray': 0} for f in folders}
doppler_samples = []

for f in folders:
    imgs = sorted(list((DATASETS_DIR / f).glob("*.jpg")))
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        diff = np.max(img, axis=2).astype(int) - np.min(img, axis=2).astype(int)
        total_colored = int(np.sum(diff > 20))
        
        # Sector ROI: middle 75% height, middle 80% width (where Doppler color box / ultrasound anatomy is)
        sector_diff = diff[int(h*0.15):int(h*0.90), int(w*0.10):int(w*0.90)]
        sector_colored = int(np.sum(sector_diff > 20))
        
        # Real Doppler flow has thousands of colored pixels in the anatomy sector
        is_doppler = (sector_colored > 800)
        
        stats[f]['total'] += 1
        if is_doppler:
            stats[f]['doppler_flow'] += 1
            doppler_samples.append((p, f, sector_colored))
        elif total_colored > 100:
            stats[f]['text_color_only'] += 1
        else:
            stats[f]['pure_gray'] += 1

print("=" * 80)
print("COLOR DOPPLER & OVERLAY QUANTIFICATION (CLEAN DATASET, 1,939 IMAGES)")
print("=" * 80)
print(f"{'Folder':<12} | {'Total':<6} | {'Genuine Doppler Flow':<22} | {'Colored Text/UI Only':<22} | {'Pure Grayscale'}")
print("-" * 80)
for f in folders:
    s = stats[f]
    dop_pct = (s['doppler_flow'] / s['total']) * 100
    txt_pct = (s['text_color_only'] / s['total']) * 100
    gry_pct = (s['pure_gray'] / s['total']) * 100
    print(f"{f:<12} | {s['total']:<6} | {s['doppler_flow']:<4} ({dop_pct:5.1f}%)           | {s['text_color_only']:<4} ({txt_pct:5.1f}%)         | {s['pure_gray']:<4} ({gry_pct:5.1f}%)")

tot_all = sum(s['total'] for s in stats.values())
tot_dop = sum(s['doppler_flow'] for s in stats.values())
tot_txt = sum(s['text_color_only'] for s in stats.values())
tot_gry = sum(s['pure_gray'] for s in stats.values())

print("-" * 80)
print(f"{'TOTAL':<12} | {tot_all:<6} | {tot_dop:<4} ({tot_dop/tot_all*100:5.1f}%)           | {tot_txt:<4} ({tot_txt/tot_all*100:5.1f}%)         | {tot_gry:<4} ({tot_gry/tot_all*100:5.1f}%)")
print("=" * 80)

# Sort Doppler samples by colored pixels count
doppler_samples.sort(key=lambda x: x[2], reverse=True)
print(f"\nTotal genuine Doppler frames found: {len(doppler_samples)}")

# Select 4 representative Doppler samples across UF and NUF
uf_dopplers = [s for s in doppler_samples if 'UF' in s[1] and 'NUF' not in s[1]]
nuf_dopplers = [s for s in doppler_samples if 'NUF' in s[1]]

print(f"  Doppler frames in UF : {len(uf_dopplers)}")
print(f"  Doppler frames in NUF: {len(nuf_dopplers)}")

selected_dopplers = []
if len(uf_dopplers) >= 2:
    selected_dopplers.extend(uf_dopplers[:2])
else:
    selected_dopplers.extend(uf_dopplers)

if len(nuf_dopplers) >= 2:
    selected_dopplers.extend(nuf_dopplers[:2])
else:
    selected_dopplers.extend(nuf_dopplers)

# Import refined preprocessing
import sys
sys.path.append("DCGAN&MobileNetV2/DCGAN&MobileNetV2")
from utils.preprocessing import preprocess_ultrasound

print("\nGenerating side-by-side comparison for 4 Doppler frames...")
for idx, (p, folder, count) in enumerate(selected_dopplers, 1):
    raw_bgr = cv2.imread(str(p))
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    
    prep_rgb = preprocess_ultrasound(raw_rgb, target_size=(224, 224))
    prep_bgr = cv2.cvtColor(prep_rgb, cv2.COLOR_RGB2BGR)
    
    raw_vis = cv2.resize(raw_bgr, (224, 224))
    
    label = folder.split('/')[-1]
    cv2.putText(raw_vis, f"DOPPLER RAW ({label})", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(prep_bgr, f"GRAY PREP ({label})", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    
    divider = np.full((224, 4, 3), 80, dtype=np.uint8)
    montage = np.hstack([raw_vis, divider, prep_bgr])
    
    out_name = f"doppler_sample_{idx:02d}_{label}_{p.stem}.png"
    cv2.imwrite(str(REPORTS_DIR / out_name), montage)
    cv2.imwrite(str(ART_DIR / out_name), montage)
    print(f"  Sample {idx}: {out_name} (colored pixels in sector: {count})")

print("\nDoppler quantification and samples complete.")
