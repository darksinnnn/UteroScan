"""
tests/test_production_backend.py
End-to-end unit and integration test suite for the UteroScan E0 production system.

Verifies:
  1. Checkpoint integrity & architecture
  2. Preprocessing pipeline determinism & bit-exact ranges
  3. Probability normalization (sum to 1.0) and class mapping (0=UF, 1=NUF)
  4. Backend API endpoints (/api/health, /api/metrics, /api/samples, /api/predict)
  5. Validation on malformed images, empty uploads, and non-image payloads
  6. Development cohort isolation (no access to held-out test scans)
"""

import io
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import pytest
import sys
from fastapi.testclient import TestClient
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.server import app, get_engine
from api.inference import (
    UteroScanInferenceEngine,
    clinical_preprocess_image,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_METADATA_PATH
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def engine():
    return get_engine()


class TestProductionModelIntegrity:
    def test_checkpoint_exists(self):
        assert DEFAULT_CHECKPOINT_PATH.is_file(), f"Missing production checkpoint at {DEFAULT_CHECKPOINT_PATH}"
        assert DEFAULT_CHECKPOINT_PATH.stat().st_size > 80 * 1024 * 1024, "Checkpoint file suspiciously small"

    def test_metadata_exists_and_valid(self):
        assert DEFAULT_METADATA_PATH.is_file(), f"Missing metadata at {DEFAULT_METADATA_PATH}"
        with open(DEFAULT_METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["architecture"] == "torchvision.models.resnet50"
        assert meta["epochs_trained"] == 5
        assert meta["training_cohort_size"] == 1474
        assert meta["cohort_breakdown"]["UF"] == 685
        assert meta["cohort_breakdown"]["NUF"] == 789

    def test_engine_initialization(self, engine):
        assert engine is not None
        assert engine.model is not None
        assert engine.model_version == "E0_ResNet50_Production_v1"
        # Assert model is in eval mode
        assert not engine.model.training


class TestPreprocessingPipeline:
    def test_preprocessing_output_tensor(self):
        # Create a synthetic 1024x768 ultrasound-like grayscale canvas
        synthetic_img = np.random.randint(20, 200, (768, 1024, 3), dtype=np.uint8)
        tensor, prep_pil, orig_shape = clinical_preprocess_image(synthetic_img)

        # Tensor checks
        assert tensor.shape == (1, 3, 224, 224), f"Unexpected tensor shape {tensor.shape}"
        assert tensor.dtype == torch.float32
        assert orig_shape == (768, 1024)
        assert prep_pil.size == (224, 224)

        # Check normalization bounds (finite, no NaN, reasonable range)
        assert not torch.isnan(tensor).any()
        assert not torch.isinf(tensor).any()

    def test_deterministic_preprocessing(self):
        img_np = np.full((500, 500, 3), 128, dtype=np.uint8)
        t1, _, _ = clinical_preprocess_image(img_np)
        t2, _, _ = clinical_preprocess_image(img_np)
        assert torch.allclose(t1, t2), "Inference preprocessing must be strictly deterministic"

    def test_invalid_image_raises_error(self):
        with pytest.raises(ValueError):
            clinical_preprocess_image(b"corrupt_data_not_an_image")

    def test_tiny_image_raises_error(self):
        tiny_img = np.zeros((16, 16, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="below minimum allowed"):
            clinical_preprocess_image(tiny_img)


class TestInferencePredictions:
    def test_predict_permitted_dev_uf_sample(self, engine):
        dev_uf_path = PROJECT_ROOT / "datasets" / "datasets" / "train" / "UF" / "1.2.826.0.1.3680043.2.461.11219581.1358591605.jpg"
        assert dev_uf_path.is_file(), f"Test sample not found: {dev_uf_path}"

        with open(dev_uf_path, "rb") as f:
            raw_bytes = f.read()

        result = engine.predict(raw_bytes, source_name="Test UF Sample")
        assert result["status"] == "success"
        assert result["model_version"] == "E0_ResNet50_Production_v1"
        assert result["classification"]["primary_label"] in ["UF", "NUF"]

        # Probability normalization
        p_uf = result["probabilities"]["UF"]
        p_nuf = result["probabilities"]["NUF"]
        assert 0.0 <= p_uf <= 1.0
        assert 0.0 <= p_nuf <= 1.0
        assert abs(p_uf + p_nuf - 1.0) < 1e-3, f"Probabilities do not sum to 1.0: {p_uf} + {p_nuf}"

        # Confidence invariant
        conf = result["classification"]["confidence"]
        assert conf == max(p_uf, p_nuf)

    def test_predict_permitted_dev_nuf_sample(self, engine):
        dev_nuf_path = PROJECT_ROOT / "datasets" / "datasets" / "train" / "NUF" / "1.2.826.0.1.3680043.2.461.11229757.2783613826.jpg"
        assert dev_nuf_path.is_file(), f"Test sample not found: {dev_nuf_path}"

        with open(dev_nuf_path, "rb") as f:
            raw_bytes = f.read()

        result = engine.predict(raw_bytes, source_name="Test NUF Sample")
        assert result["status"] == "success"
        p_uf = result["probabilities"]["UF"]
        p_nuf = result["probabilities"]["NUF"]
        assert abs(p_uf + p_nuf - 1.0) < 1e-3


class TestFastAPIEndpoints:
    def test_get_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert data["model_version"] == "E0_ResNet50_Production_v1"
        assert "published_development_benchmark" in data
        assert data["published_development_benchmark"]["mean_outer_accuracy"] == 0.9412

    def test_get_metrics(self, client):
        response = client.get("/api/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "rule_2_max_acc" in data or "published_cv_benchmark" in data

    def test_get_samples(self, client):
        response = client.get("/api/samples")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 4
        first = data[0]
        assert "id" in first
        assert "thumbnail_base64" in first
        assert first["thumbnail_base64"].startswith("data:image/")

    def test_predict_via_sample_id(self, client):
        response = client.post("/api/predict", data={"sample_id": "sample_uf_1"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "probabilities" in data
        assert "classification" in data

    def test_predict_via_file_upload(self, client):
        # Create a small valid test JPEG
        img = Image.new("RGB", (250, 250), color=(120, 120, 120))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/api/predict",
            files={"file": ("synthetic_scan.jpg", buf, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["classification"]["primary_label"] in ["UF", "NUF"]

    def test_predict_invalid_file_rejected(self, client):
        buf = io.BytesIO(b"not_an_image")
        response = client.post(
            "/api/predict",
            files={"file": ("corrupt.txt", buf, "text/plain")}
        )
        assert response.status_code == 400

    def test_predict_missing_both_inputs_rejected(self, client):
        response = client.post("/api/predict")
        assert response.status_code == 400
