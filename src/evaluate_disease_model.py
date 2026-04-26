"""Evaluate the dedicated CropVision disease classifier."""

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

from src.config import FIGURES_DIR, REPORTS_DIR
from src.dataset import load_imagefolder
from src.disease_model import DISEASE_CLASS_NAMES_PATH, DISEASE_MODEL_PATH, load_disease_model
from src.evaluate_cv import plot_confusion_matrix, save_misclassified_examples, top_k_accuracy
from src.utils import get_device, load_json


def evaluate(args: argparse.Namespace) -> dict[str, float]:
    if not DISEASE_MODEL_PATH.exists() or not DISEASE_CLASS_NAMES_PATH.exists():
        raise FileNotFoundError("Disease model not found. Train or install a disease model first.")
    dataset = load_imagefolder(args.data_dir, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = get_device(args.cpu)
    model, class_names, metadata = load_disease_model(device)

    if list(dataset.classes) != list(class_names):
        print("Warning: evaluation folder classes differ from disease_class_names.json. Metrics use model class order.")

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            y_true.extend(labels.tolist())
            y_pred.extend(probabilities.argmax(axis=1).tolist())
            y_prob.extend(probabilities)

    prob_array = np.vstack(y_prob) if y_prob else np.empty((0, len(class_names)))
    label_array = np.array(y_true)
    pred_array = np.array(y_pred)
    confidences = prob_array.max(axis=1) if len(prob_array) else np.array([])
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top_3_accuracy": float(top_k_accuracy(prob_array, label_array, k=3)),
        "num_examples": int(len(y_true)),
        "num_classes": int(len(class_names)),
        "model_version": metadata.get("model_version_name", "latest"),
    }
    report = pd.DataFrame(
        classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0)
    ).transpose()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize="true")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORTS_DIR / "disease_classification_report.csv")
    (REPORTS_DIR / "disease_eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion_matrix(cm, class_names, FIGURES_DIR / "disease_confusion_matrix.png")
    plot_confusion_matrix(cm_norm, class_names, FIGURES_DIR / "disease_confusion_matrix_normalized.png", normalized=True)
    save_misclassified_examples(dataset, y_true, y_pred, confidences, class_names, FIGURES_DIR / "disease_misclassified_examples")

    print("Disease model evaluation")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Top-3 accuracy: {metrics['top_3_accuracy']:.4f}")
    print(f"Saved reports to {REPORTS_DIR}")
    write_disease_model_card(metrics, class_names, args.data_dir, metadata)
    return metrics


def write_disease_model_card(metrics: dict, class_names: list[str], data_dir: Path, metadata: dict) -> None:
    content = f"""# CropVision Disease Model Card

## Model Architecture

{metadata.get("model_name", metadata.get("architecture", "unknown"))}

## Training Dataset

Expected ImageFolder data from `{data_dir}` with {len(class_names)} disease classes.

## Intended Use

Educational plant health screening demo: classify visible leaf/crop disease patterns from images.

## Not Intended Use

Not professional crop diagnosis, treatment advice, pesticide advice, or a replacement for an agronomist.

## Metrics

- Accuracy: {metrics.get("accuracy", 0):.4f}
- Macro F1: {metrics.get("macro_f1", 0):.4f}
- Weighted F1: {metrics.get("weighted_f1", 0):.4f}
- Top-3 accuracy: {metrics.get("top_3_accuracy", 0):.4f}

## Known Limitations

- Controlled datasets may not generalize to field images.
- Unknown plants or diseases can produce confident but wrong predictions.
- Confidence thresholding matters because low-confidence images should be treated as uncertain.
- The app does not self-train automatically; only human-verified feedback should be used for retraining.

## Retraining Process

Prepare an ImageFolder disease dataset, train with `python -m src.train_disease_model`, evaluate with
`python -m src.evaluate_disease_model`, then optionally retrain with verified feedback.
"""
    (REPORTS_DIR / "disease_model_card.md").write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the dedicated CropVision disease model.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
