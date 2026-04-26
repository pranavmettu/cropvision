"""Evaluate the CropVision image classifier and save reports/figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.calibration import plot_reliability_diagram
from src.config import (
    DEFAULT_CLASSIFICATION_REPORT_PATH,
    DEFAULT_CLASS_NAMES_PATH,
    DEFAULT_CV_MODEL_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_EVAL_METRICS_PATH,
    FIGURES_DIR,
    REPORTS_DIR,
    ensure_project_dirs,
)
from src.dataset import load_imagefolder
from src.predict_cv import load_cv_model
from src.utils import get_device, load_class_names


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path, normalized: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(class_names) * 0.45), max(6, len(class_names) * 0.4)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="CropVision Normalized Confusion Matrix" if normalized else "CropVision Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]:.2f}" if normalized else format(int(cm[i, j]), "d")
            ax.text(j, i, text, ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    if probabilities.size == 0:
        return 0.0
    k = min(k, probabilities.shape[1])
    top_k = np.argsort(probabilities, axis=1)[:, -k:]
    return float(np.mean([label in row for label, row in zip(labels, top_k)]))


def save_misclassified_examples(
    dataset,
    y_true: list[int],
    y_pred: list[int],
    confidences: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    max_examples: int = 24,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for idx, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label == pred_label:
            continue
        try:
            image_path, _ = dataset.samples[idx]
        except AttributeError:
            continue
        image = plt.imread(image_path)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"true: {class_names[true_label]}\npred: {class_names[pred_label]} ({confidences[idx]:.1%})",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"misclassified_{saved:03d}.png", dpi=150)
        plt.close(fig)
        saved += 1
        if saved >= max_examples:
            break


def write_model_card(metrics: dict[str, float], class_names: list[str], args: argparse.Namespace, output_path: Path) -> None:
    content = f"""# CropVision Model Card

## Model Name

CropVision plant disease image classifier

## Intended Use

Educational ML portfolio demonstration for plant leaf image classification and explainability.

## Not Intended Use

This model is not intended for professional crop diagnosis, agronomic decisions, pesticide decisions, or safety-critical agricultural recommendations.

## Dataset

Expected ImageFolder dataset: `{args.data_dir}`. Users should document the specific dataset source, license, collection conditions, and class names.

## Architecture

Architecture is loaded from `{args.checkpoint}`. Supported architectures are ResNet18 and EfficientNet-B0.

## Training Setup

Training uses PyTorch transfer learning with 224x224 ImageNet normalization, optional frozen-backbone training, optional weighted loss, early stopping, and CPU-friendly defaults.

## Metrics

- Accuracy: {metrics.get("accuracy", 0):.4f}
- Macro F1: {metrics.get("macro_f1", 0):.4f}
- Weighted F1: {metrics.get("weighted_f1", 0):.4f}
- Top-3 accuracy: {metrics.get("top_3_accuracy", 0):.4f}
- Expected Calibration Error: {metrics.get("expected_calibration_error", 0):.4f}

## Classes

{", ".join(class_names)}

## Known Limitations

- PlantVillage-style datasets are controlled and may not generalize to field images.
- Model confidence is not the same as clinical or agronomic certainty.
- Grad-CAM highlights correlated visual evidence, not guaranteed causal disease symptoms.
- Weather risk is a synthetic demo unless trained with real disease incidence labels.

## Ethical and Practical Risks

Incorrect predictions could lead to inappropriate crop management if misused. The app displays uncertainty warnings and educational disclaimers, but human expert review remains essential.

## Future Improvements

- Validate on field datasets such as PlantDoc.
- Add calibration with held-out validation data and temperature scaling artifacts.
- Add lesion segmentation or object detection as future work.
- Add model monitoring and drift checks for deployed demos.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    device = get_device(args.cpu)
    dataset = load_imagefolder(Path(args.data_dir), train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model, class_names, _ = load_cv_model(Path(args.checkpoint), device)
    class_names_path = Path(args.class_names_path)
    if class_names_path.exists():
        class_names = load_class_names(class_names_path)

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = probabilities.argmax(axis=1).tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())
            y_prob.extend(probabilities)

    prob_array = np.vstack(y_prob) if y_prob else np.empty((0, len(class_names)))
    label_array = np.array(y_true)
    pred_array = np.array(y_pred)
    confidences = prob_array.max(axis=1) if len(prob_array) else np.array([])
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    top3_acc = top_k_accuracy(prob_array, label_array, k=3)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    ece = plot_reliability_diagram(confidences, pred_array, label_array, FIGURES_DIR / "calibration_curve.png")
    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "top_3_accuracy": float(top3_acc),
        "expected_calibration_error": float(ece),
        "num_examples": int(len(y_true)),
        "num_classes": int(len(class_names)),
    }

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Top-3 accuracy: {top3_acc:.4f}")
    print(f"Expected Calibration Error: {ece:.4f}")
    print("\nClassification report:\n")
    print(report_df.round(4))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(args.report_csv)
    Path(args.metrics_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion_matrix(cm, class_names, FIGURES_DIR / "confusion_matrix.png")
    plot_confusion_matrix(cm_norm, class_names, FIGURES_DIR / "confusion_matrix_normalized.png", normalized=True)
    save_misclassified_examples(
        dataset,
        y_true,
        y_pred,
        confidences,
        class_names,
        FIGURES_DIR / "misclassified_examples",
        max_examples=args.max_misclassified,
    )
    write_model_card(metrics, class_names, args, REPORTS_DIR / "model_card.md")
    print(f"Saved evaluation reports to {REPORTS_DIR} and figures to {FIGURES_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CropVision computer vision model.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CV_MODEL_PATH)
    parser.add_argument("--class_names_path", type=Path, default=DEFAULT_CLASS_NAMES_PATH)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--report_csv", type=Path, default=DEFAULT_CLASSIFICATION_REPORT_PATH)
    parser.add_argument("--metrics_json", type=Path, default=DEFAULT_EVAL_METRICS_PATH)
    parser.add_argument("--max_misclassified", type=int, default=24)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
