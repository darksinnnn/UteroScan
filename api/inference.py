"""
api/inference.py
Authoritative Inference Engine for UteroScan Production Deployment

Model: E0 — ResNet50 (Full-Frame, ImageNet V2 Backbone)
Input Pipeline:
    1. Grayscale conversion
    2. cv2.INTER_AREA resize to 224x224
    3. 3x3 Median Blur despeckling
    4. CLAHE contrast standardization (clipLimit=1.2, tileGridSize=(8,8))
    5. 3-channel replicate
    6. ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Classes:
    0: "UF" (Uterine Fibroid)
    1: "NUF" (Normal / Non-Uterine Fibroid)

Strict Invariants:
    - No training-time augmentations during inference (no flip, no rotation).
    - torch.no_grad() and model.eval() enforced.
    - Fails loudly if production checkpoint is missing (no silent fallback).
    - Validates image dimensions, size, and decoding.
"""

import io
import os
import time
import json
import base64
from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "production" / "resnet50_e0_production.pth"
DEFAULT_METADATA_PATH = BASE_DIR / "checkpoints" / "production" / "model_metadata.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Maximum allowed file size for safety (20 MB)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MIN_RESOLUTION = (32, 32)
MAX_RESOLUTION = (8192, 8192)


def clinical_preprocess_image(
    image_input: Union[Image.Image, np.ndarray, bytes]
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
    """
    Applies the verified E0 clinical transformation pipeline:
        1. Decode to numpy array
        2. Validate dimensions
        3. Convert to grayscale
        4. Resize to 224x224 (cv2.INTER_AREA)
        5. 3x3 Median Filter
        6. CLAHE (clipLimit=1.2, tileGridSize=(8,8))
        7. 3-channel RGB replicate
        8. ImageNet normalization

    Returns:
        tensor: (1, 3, 224, 224) torch.FloatTensor
        preprocessed_pil: PIL.Image of preprocessed image for UI display
        original_shape: (height, width) of input
    """
    # 1. Decode/normalize to OpenCV BGR numpy array
    if isinstance(image_input, bytes):
        if len(image_input) > MAX_IMAGE_BYTES:
            raise ValueError(f"Uploaded file exceeds maximum limit of {MAX_IMAGE_BYTES // (1024*1024)}MB.")
        nparr = np.frombuffer(image_input, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Unable to decode image. Ensure file is a valid image format (PNG, JPEG, TIFF).")
    elif isinstance(image_input, Image.Image):
        rgb_arr = np.array(image_input.convert("RGB"))
        img_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.ndim == 3 and image_input.shape[2] == 3:
            img_bgr = image_input.copy()
        elif image_input.ndim == 3 and image_input.shape[2] == 4:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(f"Unsupported array shape for image input: {image_input.shape}")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    orig_h, orig_w = img_bgr.shape[:2]
    if orig_h < MIN_RESOLUTION[0] or orig_w < MIN_RESOLUTION[1]:
        raise ValueError(f"Image resolution ({orig_w}x{orig_h}) is below minimum allowed {MIN_RESOLUTION[1]}x{MIN_RESOLUTION[0]}.")
    if orig_h > MAX_RESOLUTION[0] or orig_w > MAX_RESOLUTION[1]:
        raise ValueError(f"Image resolution ({orig_w}x{orig_h}) exceeds maximum allowed {MAX_RESOLUTION[1]}x{MAX_RESOLUTION[0]}.")

    # 2. Grayscale conversion
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 3. Resize to 224x224 using cv2.INTER_AREA (preserves speckle distribution on downsampling)
    if (orig_w, orig_h) != (224, 224):
        resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
    else:
        resized = gray

    # 4. 3x3 Median Blur despeckling
    blurred = cv2.medianBlur(resized, 3)

    # 5. CLAHE (clipLimit=1.2, tileGridSize=(8, 8))
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # 6. 3-channel RGB replicate
    rgb = np.stack([enhanced, enhanced, enhanced], axis=-1)
    preprocessed_pil = Image.fromarray(rgb)

    # 7. ImageNet normalization
    norm_transform = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    tensor = norm_transform(tensor).unsqueeze(0)  # (1, 3, 224, 224)

    return tensor, preprocessed_pil, (orig_h, orig_w)


def image_to_base64(pil_img: Image.Image, format: str = "JPEG") -> str:
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format, quality=90)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"


class UteroScanInferenceEngine:
    """
    Production inference engine that loads and serves the E0 ResNet50 model.
    Instantiated once at application startup.
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.model_metadata: Dict[str, Any] = {}
        self.model_version = "E0_ResNet50_Production_v1"
        self._load_production_model()

    def _load_production_model(self):
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Production checkpoint missing at: {self.checkpoint_path}\n"
                "Please run `python scripts/train_production_e0.py` to produce the authoritative E0 production weights."
            )

        print(f"Loading UteroScan E0 production checkpoint from: {self.checkpoint_path} on {self.device}")
        
        # Build torchvision ResNet50 with 2-class linear classifier
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)

        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        # Freeze all parameters for strict inference invariance
        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.model_version = checkpoint.get("model_version", "E0_ResNet50_Production_v1")

        # Load metadata if present
        meta_path = self.checkpoint_path.parent / "model_metadata.json"
        if meta_path.is_file():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.model_metadata = json.load(f)

        print(f"UteroScan E0 model loaded successfully (version: {self.model_version}).")

    def predict(
        self,
        image_input: Union[Image.Image, np.ndarray, bytes],
        source_name: str = "Uploaded Image",
        custom_threshold: float = 0.65
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference on an ultrasound scan.
        """
        if self.model is None:
            raise RuntimeError("Inference engine is not initialized.")

        # 1. Preprocessing
        t_prep_start = time.perf_counter()
        tensor, prep_pil, orig_shape = clinical_preprocess_image(image_input)
        prep_time_ms = (time.perf_counter() - t_prep_start) * 1000.0

        # Raw image preview base64
        if isinstance(image_input, Image.Image):
            raw_pil = image_input
        elif isinstance(image_input, bytes):
            raw_pil = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            raw_pil = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        raw_b64 = image_to_base64(raw_pil)
        prep_b64 = image_to_base64(prep_pil)

        # 2. Model Inference
        t_inf_start = time.perf_counter()
        tensor = tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        inf_time_ms = (time.perf_counter() - t_inf_start) * 1000.0

        prob_uf = float(probs[0])
        prob_nuf = float(probs[1])

        # Dual threshold operating calls
        pred_050 = "UF" if prob_uf >= 0.50 else "NUF"
        pred_optimal = "UF" if prob_uf >= 0.65 else "NUF"
        pred_custom = "UF" if prob_uf >= custom_threshold else "NUF"

        # Documented confidence: probability assigned to the predicted class
        confidence = prob_uf if pred_050 == "UF" else prob_nuf

        # Clinical interpretation description
        if prob_uf >= 0.65:
            risk_level = "High-Confidence Fibroid Finding"
            clinical_call = "Uterine Fibroid Detected (Positive at Optimal Specificity)"
            badge_type = "positive_confirmed"
        elif prob_uf >= 0.50:
            risk_level = "Borderline / Indeterminate Scan"
            clinical_call = "Uterine Fibroid Detected (Positive at Standard Screening Cutoff)"
            badge_type = "positive_borderline"
        else:
            risk_level = "Physiological Non-Fibroid Finding"
            clinical_call = "Normal Myometrium / Non-Fibroid (Negative)"
            badge_type = "negative"

        return {
            "status": "success",
            "model_version": self.model_version,
            "architecture": "ResNet50 (Full-Frame E0 Baseline)",
            "source": source_name,
            "classification": {
                "primary_label": pred_050,
                "optimal_threshold_label": pred_optimal,
                "custom_threshold_label": pred_custom,
                "confidence": round(float(confidence), 4),
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
                    "is_positive": pred_050 == "UF",
                    "clinical_intent": "Maximum Screening Sensitivity (~94.75%)"
                },
                "optimal_threshold_065": {
                    "threshold": 0.65,
                    "prediction": pred_optimal,
                    "is_positive": pred_optimal == "UF",
                    "clinical_intent": "Optimal Diagnostic Accuracy (~94.12%, Specificity ~93.57%)"
                },
                "custom_threshold": {
                    "threshold": round(custom_threshold, 2),
                    "prediction": pred_custom,
                    "is_positive": pred_custom == "UF"
                }
            },
            "image_meta": {
                "original_dimensions": [orig_shape[1], orig_shape[0]],
                "processed_dimensions": [224, 224]
            },
            "latency": {
                "preprocessing_ms": round(prep_time_ms, 2),
                "inference_ms": round(inf_time_ms, 2),
                "total_ms": round(prep_time_ms + inf_time_ms, 2)
            },
            "visualizations": {
                "raw_image": raw_b64,
                "preprocessed_image": prep_b64
            },
            "disclaimer": "Research Prototype: Investigational use only. Not certified for standalone clinical diagnostic decisions."
        }
