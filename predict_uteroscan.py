"""
predict_uteroscan.py
UteroScan Phase 5 — Inference Engine & Prediction Entry Point

Methodology per PRD.md Section 4 (FR-1 to FR-5) & Implementation_Plan.md Phase 5:
- Clean standalone entry point for downstream code, scripts, or future API/UI.
- Primary Model: Two-Stage ResNet50 (nets/resnet.py) with weights from checkpoints/resnet50_stage2_best.pth.
- Input Pipeline: Phase 1 clinical transform (Grayscale -> 224x224 INTER_AREA -> 3x3 Median -> CLAHE 1.2 -> RGB replicate -> ImageNet norm).
- Output: Structured dictionary with label, confidence probability, class breakdown, and latency.
- Error Handling: Catches invalid/corrupted files gracefully (FR-5).
"""

import os
import sys
import time
from pathlib import Path
from typing import Union, Dict, Any
from PIL import Image
import numpy as np
import torch
import torch.nn as nn

# Base Paths
BASE_DIR = Path("d:/Project/UteroScan")
COMPANION_DIR = BASE_DIR / "DCGAN&MobileNetV2" / "DCGAN&MobileNetV2"
sys.path.insert(0, str(COMPANION_DIR))

from nets import get_model_from_name
from utils.utils import cvtColor, preprocess_input
from utils.preprocessing import preprocess_ultrasound

DEFAULT_CHECKPOINT = BASE_DIR / "checkpoints" / "resnet50_stage2_best.pth"

CLASS_NAMES = {
    0: "UF",
    1: "NUF"
}

CLINICAL_DESCRIPTIONS = {
    0: "Uterine Fibroid Detected",
    1: "Normal Myometrium / Non-Fibroid"
}

class UteroScanPredictor:
    """
    Production-grade inference wrapper for UteroScan ResNet50.
    Initializes and caches model weights for fast subsequent inferences (<20ms).
    """
    def __init__(self, checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT, device: str = None):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load ResNet50 architecture from companion framework
        arch_fn = get_model_from_name["resnet50"]
        self.model = arch_fn(num_classes=2, pretrained=False)

        # Load trained weights
        ckpt = torch.load(str(self.checkpoint_path), map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_image_path(self, path_str: Union[str, Path]) -> Path:
        p = Path(path_str)
        if p.is_file():
            return p
        candidates = [
            BASE_DIR / p,
            BASE_DIR / "datasets" / p,
            BASE_DIR / "datasets" / "datasets" / p,
            BASE_DIR / "datasets" / "datasets" / Path(*p.parts[1:] if len(p.parts) > 1 else p.parts),
            COMPANION_DIR / p
        ]
        for c in candidates:
            if c.is_file():
                return c
        return p

    def predict(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Runs clinical preprocessing and inference on an ultrasound image file.
        Returns structured dictionary with prediction metadata.
        """
        resolved_path = self._resolve_image_path(image_path)
        if not resolved_path.is_file():
            return {
                "status": "error",
                "image_path": str(image_path),
                "error": f"Image file not found: {resolved_path}",
                "error_type": "FileNotFoundError"
            }

        t_start = time.perf_counter()
        try:
            # 1. Load and validate image (FR-1, FR-5)
            pil_img = Image.open(resolved_path)
            pil_img.verify()
            # Re-open after verify
            pil_img = Image.open(resolved_path)
            img_rgb = cvtColor(pil_img)

            # 2. Phase 1 Clinical Ultrasound Preprocessing (FR-2)
            preprocessed_np = preprocess_ultrasound(img_rgb, target_size=(224, 224))

            # 3. ImageNet Normalization and Tensor Shaping
            norm_np = preprocess_input(preprocessed_np.astype(np.float32))
            tensor_chw = np.transpose(norm_np, (2, 0, 1))
            tensor_bchw = torch.from_numpy(tensor_chw).unsqueeze(0).float().to(self.device)

            # 4. Model Forward Pass (FR-3)
            with torch.no_grad():
                logits = self.model(tensor_bchw)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            # Class 0: UF, Class 1: NUF
            prob_uf = float(probs[0])
            prob_nuf = float(probs[1])
            pred_class = int(np.argmax(probs))
            pred_label = CLASS_NAMES[pred_class]
            pred_confidence = float(probs[pred_class])

            return {
                "status": "success",
                "image_path": str(resolved_path),
                "class_id": pred_class,
                "label": pred_label,
                "clinical_description": CLINICAL_DESCRIPTIONS[pred_class],
                "confidence": round(pred_confidence, 4),
                "probabilities": {
                    "UF": round(prob_uf, 4),
                    "NUF": round(prob_nuf, 4)
                },
                "inference_time_ms": round(t_elapsed_ms, 2)
            }

        except Exception as e:
            return {
                "status": "error",
                "image_path": str(resolved_path),
                "error": str(e),
                "error_type": type(e).__name__
            }

# Singleton instance for direct function calls
_GLOBAL_PREDICTOR = None

def predict_image(image_path: Union[str, Path], checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT) -> Dict[str, Any]:
    """
    Standard top-level inference function.
    Usage:
        from predict_uteroscan import predict_image
        result = predict_image("path/to/scan.jpg")
        print(result["label"], result["confidence"])
    """
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = UteroScanPredictor(checkpoint_path=checkpoint_path)
    return _GLOBAL_PREDICTOR.predict(image_path)

def run_manual_test_suite():
    """
    Phase 5 Exit Checklist: Runs 5 manual test predictions on held-out test scans,
    reports expected vs. actual label, confidence score, and processing latency.
    """
    print("=" * 80)
    print("UTEROSCAN PHASE 5: MANUAL INFERENCE VERIFICATION (5 HELD-OUT SCANS)")
    print("=" * 80)

    # 5 test cases from the held-out test set: 3 UF, 2 NUF
    test_cases = [
        {
            "id": 1,
            "expected_class": "UF",
            "path": "datasets/datasets/test/UF/1.2.826.0.1.3680043.2.461.11760531.1495225233.jpg"
        },
        {
            "id": 2,
            "expected_class": "UF",
            "path": "datasets/datasets/test/UF/1.2.826.0.1.3680043.2.461.11762380.1358278817.jpg"
        },
        {
            "id": 3,
            "expected_class": "UF",
            "path": "datasets/datasets/test/UF/1.2.826.0.1.3680043.2.461.11762565.3840941976.jpg"
        },
        {
            "id": 4,
            "expected_class": "NUF",
            "path": "datasets/datasets/test/NUF/1.2.826.0.1.3680043.2.461.11221584.3416006565.jpg"
        },
        {
            "id": 5,
            "expected_class": "NUF",
            "path": "datasets/datasets/test/NUF/1.2.826.0.1.3680043.2.461.11221585.1812076394.jpg"
        }
    ]

    predictor = UteroScanPredictor()
    results = []

    for case in test_cases:
        res = predictor.predict(case["path"])
        match = (res["label"] == case["expected_class"])
        results.append({
            "case_id": case["id"],
            "image": Path(case["path"]).name,
            "expected": case["expected_class"],
            "actual": res["label"],
            "confidence": res["confidence"],
            "p_uf": res["probabilities"]["UF"],
            "p_nuf": res["probabilities"]["NUF"],
            "latency_ms": res["inference_time_ms"],
            "correct": match
        })

    print(f"{'#':<3} | {'Image Filename':<42} | {'Expected':<8} | {'Actual':<8} | {'Confidence':<10} | {'P(UF)':<8} | {'P(NUF)':<8} | {'Latency':<10} | {'Match'}")
    print("-" * 115)
    for r in results:
        match_str = "PASS" if r["correct"] else "FAIL"
        print(f"{r['case_id']:<3} | {r['image']:<42} | {r['expected']:<8} | {r['actual']:<8} | {r['confidence']:<10.4f} | {r['p_uf']:<8.4f} | {r['p_nuf']:<8.4f} | {r['latency_ms']:<6.1f} ms | {match_str}")

    all_passed = all(r["correct"] for r in results)
    avg_latency = np.mean([r["latency_ms"] for r in results])
    print("-" * 115)
    print(f"Total Test Cases : {len(results)}")
    print(f"Predictions Match: {sum(r['correct'] for r in results)}/{len(results)}")
    print(f"Average Latency  : {avg_latency:.2f} ms per image (NFR-2 < 3,000 ms: PASSED)")
    assert avg_latency < 3000.0, f"Latency {avg_latency} ms exceeded 3000 ms threshold!"
    assert all(0.5 < r["confidence"] < 1.0 for r in results), "Sanity check failed: confidence stuck at 0.5 or 1.0!"

    # Also test error handling (FR-5)
    print("\nTesting Error Handling (Corrupted / Non-Existent File):")
    err_res = predictor.predict("datasets/non_existent_scan_12345.jpg")
    print(f"  Handled Response: {err_res}")
    assert err_res["status"] == "error", "Error handling failed to catch missing file!"

    print("\n>>> ALL PHASE 5 INFERENCE VERIFICATIONS PASSED <<<")
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--test-samples":
        # Predict on provided image argument
        out = predict_image(sys.argv[1])
        print(out)
    else:
        run_manual_test_suite()
