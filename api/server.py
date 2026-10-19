"""
server.py
UteroScan Fast Diagnostic Inference API
Serves the authoritative ResNet50 E0 production model (checkpoints/production/resnet50_e0_production.pth)

Endpoints:
  - GET  /api/health
  - GET  /api/metrics
  - GET  /api/samples
  - POST /api/predict
"""

import io
import os
import sys
import time
import base64
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from api.inference import (
    UteroScanInferenceEngine,
    clinical_preprocess_image,
    image_to_base64,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_METADATA_PATH,
    MAX_IMAGE_BYTES
)

app = FastAPI(
    title="UteroScan Diagnostic Inference API",
    version="1.0.0",
    description="Authoritative CADx platform serving the Recovered E0 ResNet50 baseline."
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Inference Engine (singleton)
ENGINE: Optional[UteroScanInferenceEngine] = None

# Curated sample cases strictly from the eligible development cohort (recovered_folds.csv)
CURATED_SAMPLES = [
    {
        "id": "sample_uf_1",
        "title": "Case A: Focal Hypoechoic Myometrial Mass",
        "category": "Fibroid (UF)",
        "ground_truth": "UF",
        "path": "datasets/datasets/train/UF/1.2.826.0.1.3680043.2.461.11219581.1358591605.jpg",
        "description": "Focal hypoechoic mass with characteristic acoustic attenuation within posterior uterine wall.",
        "difficulty": "Classic High-Confidence Finding"
    },
    {
        "id": "sample_uf_2",
        "title": "Case B: Well-Defined Leiomyoma Nodule",
        "category": "Fibroid (UF)",
        "ground_truth": "UF",
        "path": "datasets/datasets/train/UF/1.2.826.0.1.3680043.2.461.11221583.696933856.jpg",
        "description": "Circumscribed intramural nodule with contour distortion and altered myometrial acoustic architecture.",
        "difficulty": "Moderate Intramural Presentation"
    },
    {
        "id": "sample_nuf_1",
        "title": "Case C: Uniform Homogeneous Myometrium",
        "category": "Normal (NUF)",
        "ground_truth": "NUF",
        "path": "datasets/datasets/train/NUF/1.2.826.0.1.3680043.2.461.11229757.2783613826.jpg",
        "description": "Smooth, uniform intermediate echotexture across anterior and posterior walls without focal lesions.",
        "difficulty": "Clear Normal Control"
    },
    {
        "id": "sample_nuf_2",
        "title": "Case D: Normal Endometrial Cavity & Stroma",
        "category": "Normal (NUF)",
        "ground_truth": "NUF",
        "path": "datasets/datasets/train/NUF/1.2.826.0.1.3680043.2.461.11229758.1427511541.jpg",
        "description": "Intact central endometrial stripe with symmetric surrounding myometrial halo, no mass effect.",
        "difficulty": "Clear Normal Control"
    },
    {
        "id": "sample_borderline",
        "title": "Case E: Heterogeneous Acoustic Myometrial Texture",
        "category": "Borderline / Complex",
        "ground_truth": "NUF",
        "path": "datasets/datasets/train/NUF/1.2.826.0.1.3680043.2.461.11229758.2709081616.jpg",
        "description": "Subtle physiological echotextural variation near fundus, requiring clinical threshold calibration.",
        "difficulty": "Challenging Diagnostic Boundary"
    }
]


def get_engine() -> UteroScanInferenceEngine:
    global ENGINE
    if ENGINE is None:
        try:
            ENGINE = UteroScanInferenceEngine(DEFAULT_CHECKPOINT_PATH)
        except Exception as e:
            print(f"CRITICAL: Failed to load UteroScan E0 model: {e}")
            raise e
    return ENGINE


@app.on_event("startup")
async def startup_event():
    print(f"Initializing UteroScan API with authoritative E0 checkpoint: {DEFAULT_CHECKPOINT_PATH}")
    get_engine()


@app.get("/api/health")
def get_health():
    engine = get_engine()
    return {
        "status": "online",
        "model": "ResNet50 E0 Recovered Baseline (Standalone Production)",
        "model_version": engine.model_version,
        "checkpoint": str(engine.checkpoint_path.name),
        "device": str(engine.device),
        "cuda_available": engine.device.type == "cuda",
        "input_resolution": "224x224",
        "preprocessing_pipeline": "Grayscale -> 224x224 cv2.INTER_AREA -> 3x3 Median -> CLAHE(clipLimit=1.2, tileGridSize=(8,8)) -> 3ch RGB replicate -> ImageNet Norm",
        "published_development_benchmark": {
            "protocol": "5-Fold Nested Grouped Cross-Validation (917 study proxy groups)",
            "mean_outer_accuracy": 0.9412,
            "mean_outer_accuracy_std": 0.0064,
            "pooled_accuracy": 0.9410,
            "mean_sensitivity": 0.9475,
            "mean_specificity": 0.9357,
            "mean_f1": 0.9372,
            "mean_roc_auc": 0.9843
        },
        "disclaimer": "Development cross-validation results from recovered cohort. Investigational research prototype, not certified for clinical diagnostic use."
    }


@app.get("/api/metrics")
def get_metrics():
    report_path = BASE_DIR / "experiments" / "final_data_audit" / "e0_retrained" / "reports" / "recovered_e0_report.json"
    if report_path.is_file():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if DEFAULT_METADATA_PATH.is_file():
        with open(DEFAULT_METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Metrics report not found"}


@app.get("/api/samples")
def get_samples():
    samples_with_data = []
    for s in CURATED_SAMPLES:
        full_path = BASE_DIR / s["path"]
        if full_path.is_file():
            with Image.open(full_path) as img:
                img_thumb = img.copy()
                img_thumb.thumbnail((250, 250))
                thumb_b64 = image_to_base64(img_thumb)
                samples_with_data.append({
                    **s,
                    "thumbnail_base64": thumb_b64,
                    "original_size": list(img.size)
                })
    return samples_with_data


@app.post("/api/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    custom_threshold: Optional[float] = Form(0.65)
):
    engine = get_engine()

    source_name = "Uploaded Image"
    image_bytes = None

    if sample_id:
        matched = next((s for s in CURATED_SAMPLES if s["id"] == sample_id), None)
        if not matched:
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
        full_path = BASE_DIR / matched["path"]
        if not full_path.is_file():
            raise HTTPException(status_code=404, detail=f"Sample image file not found at {matched['path']}.")
        with open(full_path, "rb") as f:
            image_bytes = f.read()
        source_name = matched["title"]
    elif file:
        # Validate MIME type
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"Unsupported file type '{file.content_type}'. Must be an image.")
        try:
            image_bytes = await file.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail=f"File exceeds maximum allowed size of {MAX_IMAGE_BYTES // (1024*1024)}MB.")
        source_name = file.filename or "Uploaded Image"
    else:
        raise HTTPException(status_code=400, detail="Must provide an image file upload or a valid sample_id.")

    # Execute inference
    try:
        tau = float(custom_threshold) if custom_threshold is not None else 0.65
        result = engine.predict(image_bytes, source_name=source_name, custom_threshold=tau)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the ultrasound image.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
