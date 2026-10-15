"""
server.py
UteroScan Fast Inference API
Serves the locked ResNet50 V1 model (checkpoints/resnet50_stage2_best.pth)
Exposes:
  - GET /api/health
  - GET /api/metrics
  - GET /api/samples
  - POST /api/predict (supports image upload or sample selection)
"""

import io
import os
import sys
import time
import base64
import json
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
COMPANION_DIR = BASE_DIR / "DCGAN&MobileNetV2" / "DCGAN&MobileNetV2"
sys.path.insert(0, str(COMPANION_DIR))
sys.path.insert(0, str(BASE_DIR))

from nets import get_model_from_name
from utils.utils import preprocess_input

app = FastAPI(title="UteroScan Diagnostic Inference API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "resnet50_stage2_best.pth"
METRICS_JSON_PATH = BASE_DIR / "experiments" / "v2_contamination_cleaned" / "reports" / "v1_model_on_clean_testset.json"

# Global Model Cache
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global MODEL
    if MODEL is None:
        print(f"Loading ResNet50 checkpoint from: {CHECKPOINT_PATH} on {DEVICE}")
        arch_fn = get_model_from_name["resnet50"]
        model = arch_fn(num_classes=2, pretrained=False)
        ckpt = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        MODEL = model
        print("ResNet50 successfully loaded into memory.")
    return MODEL

def clinical_preprocess(pil_img: Image.Image):
    """
    Applies the Phase 1 clinical transform:
    1. Grayscale
    2. Resize to 224x224 (cv2.INTER_AREA)
    3. 3x3 Median Filter (despeckling)
    4. CLAHE (clipLimit=1.2, tileGridSize=(8,8))
    Returns:
      - preprocessed_tensor: (1, 3, 224, 224) normalized for ImageNet
      - preprocessed_pil: PIL.Image for visual display
    """
    # 1. Grayscale
    gray = np.array(pil_img.convert("L"))

    # 2. Resize to 224x224
    h, w = gray.shape[:2]
    if (w, h) != (224, 224):
        gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)

    # 3. 3x3 Median Filter
    median = cv2.medianBlur(gray, 3)

    # 4. CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(median)

    # 3-channel replicate
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    preprocessed_pil = Image.fromarray(rgb)

    # ImageNet normalization
    img_data = np.array(rgb, dtype=np.float32)
    img_data = preprocess_input(img_data)
    tensor = torch.from_numpy(np.transpose(img_data, (2, 0, 1))).unsqueeze(0).float()

    return tensor, preprocessed_pil

def image_to_base64(pil_img: Image.Image, format="JPEG") -> str:
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format, quality=90)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

# Pre-curated sample cases from held-out test cohort
CURATED_SAMPLES = [
    {
        "id": "sample_uf_1",
        "title": "Case A: Hypoechoic Myometrial Mass",
        "category": "Fibroid (UF)",
        "ground_truth": "UF",
        "path": "datasets/datasets/test/UF/1.2.826.0.1.3680043.2.461.11760531.1495225233.jpg",
        "description": "Circumscribed hypoechoic mass with posterior acoustic shadowing within the posterior uterine wall.",
        "difficulty": "Classic High-Confidence Finding"
    },
    {
        "id": "sample_uf_2",
        "title": "Case B: Well-Defined Fibroid Nodule",
        "category": "Fibroid (UF)",
        "ground_truth": "UF",
        "path": "datasets/datasets/test/UF/1.2.826.0.1.3680043.2.461.11762380.1358278817.jpg",
        "description": "Focal round lesion with architectural distortion of outer myometrial contour.",
        "difficulty": "Moderate Presentation"
    },
    {
        "id": "sample_nuf_1",
        "title": "Case C: Uniform Homogeneous Myometrium",
        "category": "Normal (NUF)",
        "ground_truth": "NUF",
        "path": "datasets/datasets/test/NUF/1.2.826.0.1.3680043.2.461.11221584.2869724010.jpg",
        "description": "Smooth, uniform intermediate echotexture across anterior and posterior walls without focal lesions.",
        "difficulty": "Clear Normal Control"
    },
    {
        "id": "sample_nuf_2",
        "title": "Case D: Normal Endometrial Cavity & Stroma",
        "category": "Normal (NUF)",
        "ground_truth": "NUF",
        "path": "datasets/datasets/test/NUF/1.2.826.0.1.3680043.2.461.11221584.3416006565.jpg",
        "description": "Intact central endometrial stripe with symmetric surrounding myometrial halo, no mass effect.",
        "difficulty": "Clear Normal Control"
    },
    {
        "id": "sample_borderline",
        "title": "Case E: Heterogeneous Acoustic Texture",
        "category": "Borderline / Complex",
        "ground_truth": "NUF",
        "path": "datasets/datasets/test/NUF/1.2.826.0.1.3680043.2.461.11221585.1778394553.jpg",
        "description": "Mild non-specific acoustic heterogeneity near fundus, requiring clinical threshold calibration.",
        "difficulty": "Challenging Diagnostic Boundary"
    }
]

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/api/health")
def get_health():
    return {
        "status": "online",
        "model": "ResNet50 Two-Stage Clinical Fine-Tuning",
        "checkpoint": "checkpoints/resnet50_stage2_best.pth",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "input_resolution": "224x224",
        "preprocessing_pipeline": "Grayscale -> 224x224 INTER_AREA -> 3x3 Median -> CLAHE(1.2) -> ImageNet Norm",
        "held_out_clean_test_metrics": {
            "accuracy_default_050": 0.8660,
            "accuracy_optimal_065": 0.8941,
            "sensitivity_default_050": 0.9509,
            "specificity_default_050": 0.7785,
            "roc_auc": 0.9533,
            "pr_auc": 0.9514
        }
    }

@app.get("/api/metrics")
def get_metrics():
    if METRICS_JSON_PATH.is_file():
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Metrics report not found"}

@app.get("/api/samples")
def get_samples():
    samples_with_data = []
    for s in CURATED_SAMPLES:
        full_path = BASE_DIR / s["path"]
        if full_path.is_file():
            with Image.open(full_path) as img:
                # generate small preview base64
                img_thumb = img.copy()
                img_thumb.thumbnail((250, 250))
                thumb_b64 = image_to_base64(img_thumb)
                samples_with_data.append({
                    **s,
                    "thumbnail_base64": thumb_b64,
                    "original_size": list(img.size)
                })
    return samples_with_data

class PredictRequest(BaseModel):
    sample_id: Optional[str] = None
    image_base64: Optional[str] = None

@app.post("/api/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None)
):
    pil_img = None
    source_name = "Uploaded Image"

    if sample_id:
        matched = next((s for s in CURATED_SAMPLES if s["id"] == sample_id), None)
        if matched:
            full_path = BASE_DIR / matched["path"]
            if full_path.is_file():
                pil_img = Image.open(full_path).convert("RGB")
                source_name = matched["title"]
        if pil_img is None:
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found on server.")
    elif file:
        try:
            contents = await file.read()
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            source_name = file.filename
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Must provide an image file upload or a valid sample_id.")

    model = load_model()

    # Preprocessing
    t_prep_start = time.perf_counter()
    raw_b64 = image_to_base64(pil_img)
    tensor, preprocessed_pil = clinical_preprocess(pil_img)
    prep_b64 = image_to_base64(preprocessed_pil)
    prep_time_ms = (time.perf_counter() - t_prep_start) * 1000.0

    # Inference
    t_inf_start = time.perf_counter()
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    inf_time_ms = (time.perf_counter() - t_inf_start) * 1000.0

    prob_uf = float(probs[0])
    prob_nuf = float(probs[1])

    # Classifications across key operating thresholds
    pred_050 = "UF" if prob_uf >= 0.50 else "NUF"
    pred_065 = "UF" if prob_uf >= 0.65 else "NUF"

    # Clinical interpretation string
    if prob_uf >= 0.65:
        risk_level = "High Confidence Fibroid Finding"
        clinical_call = "Uterine Fibroid Detected (Positive)"
        badge_type = "positive_confirmed"
    elif prob_uf >= 0.50:
        risk_level = "Borderline / Indeterminate Scan"
        clinical_call = "Uterine Fibroid Detected (Borderline at Standard Cutoff)"
        badge_type = "positive_borderline"
    else:
        risk_level = "Normal Non-Fibroid Finding"
        clinical_call = "Normal Myometrium / Non-Fibroid (Negative)"
        badge_type = "negative"

    return {
        "status": "success",
        "source": source_name,
        "classification": {
            "primary_label": pred_050,
            "optimal_threshold_label": pred_065,
            "clinical_call": clinical_call,
            "risk_level": risk_level,
            "badge_type": badge_type
        },
        "probabilities": {
            "UF": round(prob_uf, 4),
            "NUF": round(prob_nuf, 4),
            "UF_percentage": round(prob_uf * 100, 1),
            "NUF_percentage": round(prob_nuf * 100, 1)
        },
        "threshold_calibration": {
            "standard_threshold_050": {
                "threshold": 0.50,
                "prediction": pred_050,
                "is_positive": pred_050 == "UF"
            },
            "optimal_threshold_065": {
                "threshold": 0.65,
                "prediction": pred_065,
                "is_positive": pred_065 == "UF",
                "notes": "Optimal accuracy operating point (89.41% accuracy, 88.61% specificity)"
            }
        },
        "latency": {
            "preprocessing_ms": round(prep_time_ms, 2),
            "inference_ms": round(inf_time_ms, 2),
            "total_ms": round(prep_time_ms + inf_time_ms, 2)
        },
        "visualizations": {
            "raw_image": raw_b64,
            "preprocessed_image": prep_b64
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
