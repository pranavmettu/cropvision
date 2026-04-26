"""Load and run the CropVision disease classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.calibration import apply_confidence_threshold
from src.config import MODEL_DIR
from src.dataset import get_eval_transforms
from src.disease_label_normalizer import normalize_disease_label
from src.train_cv import build_model
from src.utils import get_device, load_json

DISEASE_MODEL_DIR = MODEL_DIR / "disease"
DISEASE_MODEL_PATH = DISEASE_MODEL_DIR / "cropvision_disease_model.pt"
DISEASE_CLASS_NAMES_PATH = DISEASE_MODEL_DIR / "disease_class_names.json"
DISEASE_METADATA_PATH = DISEASE_MODEL_DIR / "model_metadata.json"


def disease_model_status() -> dict[str, Any]:
    class_count = 0
    if DISEASE_CLASS_NAMES_PATH.exists():
        try:
            class_count = len(load_json(DISEASE_CLASS_NAMES_PATH))
        except Exception:
            class_count = 0
    metadata = {}
    if DISEASE_METADATA_PATH.exists():
        try:
            metadata = json.loads(DISEASE_METADATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    return {
        "model_exists": DISEASE_MODEL_PATH.exists(),
        "class_names_exists": DISEASE_CLASS_NAMES_PATH.exists(),
        "class_count": class_count,
        "model_version": metadata.get("model_version_name", "latest"),
        "architecture": metadata.get("model_name", metadata.get("architecture", "unknown")),
    }


def load_disease_model(device: torch.device | None = None) -> tuple[torch.nn.Module, list[str], dict[str, Any]]:
    if not DISEASE_MODEL_PATH.exists() or not DISEASE_CLASS_NAMES_PATH.exists():
        raise FileNotFoundError("Disease model not found. Train or install a disease model first.")
    device = device or get_device()
    class_names = load_json(DISEASE_CLASS_NAMES_PATH)
    checkpoint = torch.load(DISEASE_MODEL_PATH, map_location=device)
    architecture = checkpoint.get("architecture", "efficientnet_b0")
    model = build_model(len(class_names), architecture=architecture, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    metadata = {}
    if DISEASE_METADATA_PATH.exists():
        metadata = json.loads(DISEASE_METADATA_PATH.read_text(encoding="utf-8"))
    metadata.setdefault("architecture", architecture)
    return model, class_names, metadata


def predict_disease(image_path: str, top_k: int = 3, confidence_threshold: float = 0.55) -> dict[str, Any]:
    if not DISEASE_MODEL_PATH.exists() or not DISEASE_CLASS_NAMES_PATH.exists():
        return {
            "available": False,
            "message": "Disease model not found. Train or install a disease model first.",
            "predicted_disease_class": None,
            "confidence": None,
            "top_k_predictions": [],
            "is_uncertain": True,
            "uncertainty_reason": "Disease model files are missing.",
            "model_path": str(DISEASE_MODEL_PATH),
            "model_version": None,
            "broad_problem_category": "unknown_or_uncertain",
            "normalized_label_info": None,
        }
    device = get_device()
    model, class_names, metadata = load_disease_model(device)
    image = Image.open(image_path).convert("RGB")
    tensor = get_eval_transforms()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
    values, indices = torch.topk(probabilities, k=min(top_k, len(class_names)))
    top_predictions = [
        {"class_name": class_names[idx.item()], "confidence": float(value.item())}
        for value, idx in zip(values, indices)
    ]
    raw_label = top_predictions[0]["class_name"]
    thresholded = apply_confidence_threshold(raw_label, top_predictions[0]["confidence"], confidence_threshold)
    label_info = normalize_disease_label(raw_label)
    return {
        "available": True,
        "message": "Disease prediction completed.",
        "predicted_disease_class": thresholded.predicted_class,
        "raw_predicted_disease_class": raw_label,
        "confidence": top_predictions[0]["confidence"],
        "top_k_predictions": top_predictions,
        "is_uncertain": thresholded.is_uncertain,
        "uncertainty_reason": thresholded.uncertainty_reason,
        "model_path": str(DISEASE_MODEL_PATH),
        "model_version": metadata.get("model_version_name", "latest"),
        "broad_problem_category": "unknown_or_uncertain" if thresholded.is_uncertain else label_info["broad_problem_category"],
        "normalized_label_info": label_info,
    }
