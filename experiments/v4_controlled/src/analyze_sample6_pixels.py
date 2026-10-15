import cv2
import numpy as np
import pandas as pd
from pathlib import Path

proj = Path("d:/Project/UteroScan")
crops_dir = proj / "experiments/v4_controlled/reports/contact_sheet_crops"

# Inspect Sample 6: 11250322.2764469079.jpg
img6_path = proj / "datasets/datasets/train/NUF/1.2.826.0.1.3680043.2.461.11250322.2764469079.jpg"
img6 = cv2.imread(str(img6_path))

# Inspect Sample 5: 11364923.1619662809.jpg
img5_path = proj / "datasets/datasets/train/NUF/1.2.826.0.1.3680043.2.461.11364923.1619662809.jpg"
img5 = cv2.imread(str(img5_path))

# Inspect Sample 7: 11364923.2046609818.jpg
img7_path = proj / "datasets/datasets/train/NUF/1.2.826.0.1.3680043.2.461.11364923.2046609818.jpg"
img7 = cv2.imread(str(img7_path))

# Inspect Sample 4: 11368989.377816910.jpg
img4_path = proj / "datasets/datasets/train/NUF/1.2.826.0.1.3680043.2.461.11368989.377816910.jpg"
img4 = cv2.imread(str(img4_path))

print(f"Sample 6 shape: {img6.shape}")
# In Sample 6, find where the non-grayscale pixels are:
diff6 = np.maximum(
    np.abs(img6[:, :, 2].astype(int) - img6[:, :, 1].astype(int)),
    np.maximum(
        np.abs(img6[:, :, 2].astype(int) - img6[:, :, 0].astype(int)),
        np.abs(img6[:, :, 1].astype(int) - img6[:, :, 0].astype(int))
    )
)

# Coordinates where max channel diff > 50
ys, xs = np.where(diff6 > 50)
print(f"Sample 6 pixels with channel diff > 50: {len(ys)}")
print(f"  Y range: [{ys.min()}, {ys.max()}], X range: [{xs.min()}, {xs.max()}]")

# Check if there is a color bar:
# A color bar is typically a vertical rectangle with width 10-30 px and height > 100 px
for x_col in range(img6.shape[1]):
    col_colored = np.sum(diff6[:, x_col] > 50)
    if col_colored > 80:
        print(f"  Column {x_col} has {col_colored} colored pixels! (Vertical bar indicator)")

# Check sector region color pixels:
sector_ys, sector_xs = np.where(diff6[150:650, 150:850] > 50)
print(f"Sample 6 color pixels strictly inside ultrasound sector (150-650, 150-850): {len(sector_ys)}")

# Crop the color region in Sample 6 and save it:
if len(sector_ys) > 0:
    crop_color = img6[150:650, 150:850]
    cv2.imwrite(str(crops_dir / "sample_6_sector_color_crop.png"), crop_color)
    print(f"Saved sector color crop of Sample 6 to {crops_dir / 'sample_6_sector_color_crop.png'}")

# Check Colorbar on the left:
colorbar_crop = img6[100:400, 20:100]
cv2.imwrite(str(crops_dir / "sample_6_colorbar_crop.png"), colorbar_crop)
print(f"Saved colorbar crop of Sample 6 to {crops_dir / 'sample_6_colorbar_crop.png'}")
