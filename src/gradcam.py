"""Grad-CAM explanations for CropVision image predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from src.config import DEFAULT_CV_MODEL_PATH, FIGURES_DIR, IMAGENET_MEAN, IMAGENET_STD, ensure_project_dirs
from src.calibration import apply_confidence_threshold
from src.dataset import get_eval_transforms
from src.predict_cv import load_cv_model
from src.utils import get_device


class GradCAM:
    """Small Grad-CAM implementation for ResNet/EfficientNet classifiers."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module: torch.nn.Module, _grad_input: tuple[torch.Tensor], grad_output: tuple[torch.Tensor]) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy()


def get_target_layer(model: torch.nn.Module, architecture: str) -> torch.nn.Module:
    if architecture == "efficientnet_b0":
        return model.features[-1]
    if architecture == "resnet18":
        return model.layer4[-1]
    raise ValueError(f"Unsupported architecture for Grad-CAM: {architecture}")


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = tensor.detach().cpu() * std + mean
    image = image.clamp(0, 1)
    return np.asarray(to_pil_image(image))


def make_overlay(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    cmap = plt.get_cmap("jet")
    cam_image = Image.fromarray(np.uint8(cam * 255)).resize((image_rgb.shape[1], image_rgb.shape[0]))
    heatmap = np.asarray(cmap(np.asarray(cam_image) / 255.0))[:, :, :3]
    overlay = (1 - alpha) * (image_rgb / 255.0) + alpha * heatmap
    return Image.fromarray(np.uint8(np.clip(overlay, 0, 1) * 255))


def gradcam_predict(
    image_path: Path,
    checkpoint_path: Path = DEFAULT_CV_MODEL_PATH,
    output_path: Path | None = None,
    confidence_threshold: float = 0.5,
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or get_device()
    model, class_names, architecture = load_cv_model(checkpoint_path, device)
    image = Image.open(image_path).convert("RGB")
    input_tensor = get_eval_transforms()(image).unsqueeze(0).to(device)

    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    values, indices = torch.topk(probabilities, k=min(3, len(class_names)))
    top_predictions = [
        {"class_name": class_names[idx.item()], "confidence": float(value.item())}
        for value, idx in zip(values, indices)
    ]
    thresholded = apply_confidence_threshold(
        top_predictions[0]["class_name"],
        top_predictions[0]["confidence"],
        confidence_threshold,
    )

    cam_runner = GradCAM(model, get_target_layer(model, architecture))
    try:
        cam = cam_runner(input_tensor, int(indices[0].item()))
    finally:
        cam_runner.remove_hooks()

    original = denormalize_image(input_tensor.squeeze(0))
    overlay = make_overlay(original, cam)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(output_path)

    return {
        "predicted_class": thresholded.predicted_class,
        "raw_predicted_class": top_predictions[0]["class_name"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
        "top_3_predictions": top_predictions,
        "is_uncertain": thresholded.is_uncertain,
        "uncertainty_reason": thresholded.uncertainty_reason,
        "overlay_image": overlay,
        "output_path": output_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM explanation for one image.")
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CV_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    ensure_project_dirs()
    args = parse_args()
    output = args.output or FIGURES_DIR / f"gradcam_{args.image_path.stem}.png"
    result = gradcam_predict(args.image_path, args.checkpoint, output, args.confidence_threshold, get_device(args.cpu))
    print({k: v for k, v in result.items() if k != "overlay_image"})
