"""Prediction helpers and CLI for the CropVision image model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.config import DEFAULT_CLASS_NAMES_PATH, DEFAULT_CV_MODEL_PATH
from src.calibration import apply_confidence_threshold
from src.dataset import get_eval_transforms
from src.train_cv import build_model
from src.utils import get_device, load_class_names


def load_cv_model(
    checkpoint_path: Path = DEFAULT_CV_MODEL_PATH,
    device: torch.device | None = None,
    class_names_path: Path = DEFAULT_CLASS_NAMES_PATH,
) -> tuple[torch.nn.Module, list[str], str]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_path}. Train first with python -m src.train_cv.")
    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names")
    if not class_names and class_names_path.exists():
        class_names = load_class_names(class_names_path)
    if not class_names:
        raise ValueError(
            "Checkpoint is missing class_names and models/class_names.json was not found. "
            "Train first with python -m src.train_cv."
        )
    architecture = checkpoint.get("architecture", "resnet18")
    model = build_model(len(class_names), architecture=architecture, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, architecture


def predict_image(
    image_path: Path,
    checkpoint_path: Path = DEFAULT_CV_MODEL_PATH,
    top_k: int = 3,
    confidence_threshold: float = 0.5,
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or get_device()
    model, class_names, _ = load_cv_model(checkpoint_path, device)
    image = Image.open(image_path).convert("RGB")
    tensor = get_eval_transforms()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)

    k = min(top_k, len(class_names))
    values, indices = torch.topk(probabilities, k=k)
    top_predictions = [
        {"class_name": class_names[idx.item()], "confidence": float(value.item())}
        for value, idx in zip(values, indices)
    ]
    thresholded = apply_confidence_threshold(
        top_predictions[0]["class_name"],
        top_predictions[0]["confidence"],
        confidence_threshold,
    )
    return {
        "predicted_class": thresholded.predicted_class,
        "raw_predicted_class": top_predictions[0]["class_name"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
        "top_3_predictions": top_predictions,
        "is_uncertain": thresholded.is_uncertain,
        "uncertainty_reason": thresholded.uncertainty_reason,
    }


def format_top_k_predictions(top_predictions: list[dict[str, float | str]]) -> list[str]:
    """Format top-k predictions for CLIs, tests, and simple UI rendering."""
    return [f"{item['class_name']}: {float(item['confidence']):.1%}" for item in top_predictions]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a plant disease class from one image.")
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CV_MODEL_PATH)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = predict_image(args.image_path, args.checkpoint, args.top_k, args.confidence_threshold, get_device(args.cpu))
    print(result)
