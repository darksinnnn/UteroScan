import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import sys

BASE_DIR = Path("d:/Project/UteroScan")
SYS_PATH = BASE_DIR / "DCGAN&MobileNetV2" / "DCGAN&MobileNetV2"
sys.path.insert(0, str(SYS_PATH))

from utils.preprocessing import preprocess_ultrasound

REPORTS_DIR = BASE_DIR / "reports" / "augmentation_samples"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR = Path(r"C:\Users\ashis\.gemini\antigravity-ide\brain\af0e99e6-8ff9-4623-ad48-86540f6228c9")

def generate_check(sample_path, out_name):
    bgr = cv2.imread(str(sample_path))
    assert bgr is not None, f"Cannot read {sample_path}"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # 1. Base Preprocessed (224x224, median blur, CLAHE)
    base_rgb = preprocess_ultrasound(rgb, target_size=(224, 224))
    pil_base = Image.fromarray(base_rgb)
    
    # 2. Forced Maximum Rotation (+15.0 deg)
    rot_pil = TF.rotate(pil_base, angle=15.0, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
    rot_rgb = np.array(rot_pil)
    
    # 3. Forced Horizontal Flip (p=1.0)
    hflip_pil = TF.hflip(pil_base)
    hflip_rgb = np.array(hflip_pil)
    
    # 4. Forced Maximum Shear (+10.0 deg)
    shear_pil = TF.affine(pil_base, angle=0.0, translate=[0, 0], scale=1.0, shear=[10.0, 0.0],
                          interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
    shear_rgb = np.array(shear_pil)
    
    # 5. Combined Maximum Locked Augmentation (+15 deg rotation, H-Flip, 1.1x zoom, +10 deg shear)
    # Applied through TF with exact locked limits
    comb_pil = TF.hflip(pil_base)
    comb_pil = TF.affine(comb_pil, angle=15.0, translate=[0, 0], scale=1.1, shear=[10.0, 0.0],
                         interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
    comb_rgb = np.array(comb_pil)
    
    panels = [
        ("1. Base Preprocessed", base_rgb),
        ("2. Forced Rot (+15 deg)", rot_rgb),
        ("3. Forced H-Flip", hflip_rgb),
        ("4. Forced Shear (+10 deg)", shear_rgb),
        ("5. Max Combined (All)", comb_rgb)
    ]
    
    vis_panels = []
    for title, p_rgb in panels:
        p_bgr = cv2.cvtColor(p_rgb, cv2.COLOR_RGB2BGR)
        # Add high-contrast label banner at top
        banner = np.zeros((26, 224, 3), dtype=np.uint8)
        cv2.putText(banner, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        stacked = np.vstack([banner, p_bgr])
        vis_panels.append(stacked)
        
    divider = np.full((250, 4, 3), 120, dtype=np.uint8)
    montage = np.hstack([
        vis_panels[0], divider,
        vis_panels[1], divider,
        vis_panels[2], divider,
        vis_panels[3], divider,
        vis_panels[4]
    ])
    
    out_file = REPORTS_DIR / out_name
    art_file = ART_DIR / out_name
    cv2.imwrite(str(out_file), montage)
    cv2.imwrite(str(art_file), montage)
    print(f"Saved: {out_file} and {art_file}")
    
    # Print metrics
    diff_rot = np.mean(np.abs(rot_rgb.astype(float) - base_rgb.astype(float)))
    diff_flip = np.mean(np.abs(hflip_rgb.astype(float) - base_rgb.astype(float)))
    diff_shear = np.mean(np.abs(shear_rgb.astype(float) - base_rgb.astype(float)))
    diff_comb = np.mean(np.abs(comb_rgb.astype(float) - base_rgb.astype(float)))
    
    print(f"  Mean Pixel Diff vs Base:")
    print(f"    - Forced Rotation (+15 deg) : {diff_rot:.2f} / 255")
    print(f"    - Forced H-Flip             : {diff_flip:.2f} / 255")
    print(f"    - Forced Shear (+10 deg)    : {diff_shear:.2f} / 255")
    print(f"    - Max Combined (All)        : {diff_comb:.2f} / 255")

def main():
    print("=" * 75)
    print("GENERATING DETERMINISTIC EXAGGERATED AUGMENTATION DEBUG SAMPLES")
    print("=" * 75)
    
    # Highly asymmetric UF sample (left brightness 47.9 vs right 23.9)
    p1 = BASE_DIR / "datasets" / "datasets" / "train" / "UF" / "1.2.826.0.1.3680043.2.461.11224027.808463803.jpg"
    generate_check(p1, "deterministic_aug_check_UF_asymmetric.png")
    
    # Fibroid mass UF sample from earlier
    p2 = BASE_DIR / "datasets" / "datasets" / "train" / "UF" / "1.2.826.0.1.3680043.2.461.11221583.696933856.jpg"
    generate_check(p2, "deterministic_aug_check_UF_fibroid.png")

if __name__ == "__main__":
    main()
