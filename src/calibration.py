"""Calibration and uncertainty utilities for CropVision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim


def expected_calibration_error(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error from confidences and correctness."""
    if len(confidences) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        if not np.any(in_bin):
            continue
        bin_acc = np.mean(predictions[in_bin] == labels[in_bin])
        bin_conf = np.mean(confidences[in_bin])
        ece += np.mean(in_bin) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_curve(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers: list[float] = []
    accuracies: list[float] = []
    avg_confidences: list[float] = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        if not np.any(in_bin):
            continue
        bin_centers.append((lower + upper) / 2)
        accuracies.append(float(np.mean(predictions[in_bin] == labels[in_bin])))
        avg_confidences.append(float(np.mean(confidences[in_bin])))
    return np.array(bin_centers), np.array(accuracies), np.array(avg_confidences)


def plot_reliability_diagram(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
) -> float:
    ece = expected_calibration_error(confidences, predictions, labels, n_bins)
    _, accuracies, avg_confidences = reliability_curve(confidences, predictions, labels, n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    if len(avg_confidences):
        ax.plot(avg_confidences, accuracies, marker="o", label=f"Model (ECE={ece:.3f})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Calibration Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return ece


@dataclass
class ThresholdedPrediction:
    predicted_class: str
    confidence: float
    is_uncertain: bool
    uncertainty_reason: str | None


def apply_confidence_threshold(
    predicted_class: str,
    confidence: float,
    threshold: float = 0.5,
) -> ThresholdedPrediction:
    if confidence < threshold:
        return ThresholdedPrediction(
            predicted_class="uncertain",
            confidence=confidence,
            is_uncertain=True,
            uncertainty_reason=(
                f"Confidence {confidence:.1%} is below the threshold of {threshold:.1%}. "
                "The image may be out-of-distribution, low quality, or not represented in training data."
            ),
        )
    return ThresholdedPrediction(predicted_class=predicted_class, confidence=confidence, is_uncertain=False, uncertainty_reason=None)


class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaling for validation logits."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(initial_temperature))))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(min=1e-3, max=100.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> float:
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.log_temperature], lr=0.05, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach().cpu().item())
