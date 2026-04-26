"""External validation for ImageFolder-style field datasets.

This script evaluates a trained CropVision checkpoint on a separate dataset such
as PlantDoc. It only scores classes whose folder names overlap with the trained
class names and reports unknown classes separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from src.config import DEFAULT_CLASS_NAMES_PATH, DEFAULT_CV_MODEL_PATH, DEFAULT_DATA_DIR, FIGURES_DIR, REPORTS_DIR, ensure_project_dirs
from src.dataset import get_eval_transforms
from src.evaluate_cv import plot_confusion_matrix, top_k_accuracy
from src.label_normalizer import normalize_label
from src.predict_cv import load_cv_model
from src.utils import get_device, load_class_names


class OverlapImageFolder(Dataset):
    """ImageFolder wrapper that skips classes not present in the trained model."""

    def __init__(self, data_dir: Path, trained_class_names: list[str]) -> None:
        self.base = datasets.ImageFolder(str(data_dir), transform=None)
        self.transform = get_eval_transforms()
        self.trained_class_names = trained_class_names
        self.trained_class_to_idx = {name: idx for idx, name in enumerate(trained_class_names)}
        self.normalized_to_trained = {normalize_label(name)["normalized_class"]: idx for idx, name in enumerate(trained_class_names)}
        self.samples: list[tuple[str, int, str]] = []
        self.normalized_samples: list[tuple[str, int, str]] = []
        self.unknown_class_counts: dict[str, int] = {}

        for path, external_idx in self.base.samples:
            external_name = self.base.classes[external_idx]
            if external_name in self.trained_class_to_idx:
                self.samples.append((path, self.trained_class_to_idx[external_name], external_name))
            elif normalize_label(external_name)["normalized_class"] in self.normalized_to_trained:
                trained_idx = self.normalized_to_trained[normalize_label(external_name)["normalized_class"]]
                self.normalized_samples.append((path, trained_idx, external_name))
            else:
                self.unknown_class_counts[external_name] = self.unknown_class_counts.get(external_name, 0) + 1

        if not self.samples and self.normalized_samples:
            self.samples = self.normalized_samples

        sample_label_indices = sorted({label for _, label, _ in self.samples})
        self.overlap_label_indices = sample_label_indices
        self.overlap_class_names = [trained_class_names[idx] for idx in sample_label_indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label, _ = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def write_external_report(
    metrics: dict,
    report_df: pd.DataFrame,
    unknown_class_counts: dict[str, int],
    output_path: Path,
) -> None:
    unknown_lines = "\n".join(f"- `{name}`: {count} images" for name, count in unknown_class_counts.items()) or "- None"
    content = f"""# External Validation Report

## Summary

- External examples evaluated: {metrics["num_evaluated_examples"]}
- Unknown examples skipped: {metrics["num_unknown_examples"]}
- Overlapping classes: {metrics["num_overlap_classes"]}
- Accuracy: {metrics["accuracy"]:.4f}
- Macro F1: {metrics["macro_f1"]:.4f}
- Weighted F1: {metrics["weighted_f1"]:.4f}
- Top-3 accuracy: {metrics["top_3_accuracy"]:.4f}
- Exact class match accuracy: {metrics.get("exact_match_accuracy", 0):.4f}
- Normalized class match accuracy: {metrics.get("normalized_match_accuracy", 0):.4f}
- Broad problem category accuracy: {metrics.get("broad_problem_category_accuracy", 0):.4f}

## Unknown Classes Skipped

{unknown_lines}

## Domain Shift Notes

External datasets such as PlantDoc often contain real-world field images with variable lighting, complex backgrounds, blur, occlusion, different camera distances, and mixed symptom presentation. A model trained primarily on clean PlantVillage-style images can perform worse on these images because the validation distribution is no longer controlled. This report is meant to expose that domain shift explicitly rather than hide it behind in-distribution metrics.

## Per-Class Report

```text
{report_df.round(4).to_string()}
```

## Interpretation

Use this report as a robustness check. Strong in-distribution performance with weak external validation suggests the next highest-value improvement is collecting field-like training images, applying stronger augmentation, or calibrating confidence on a validation set that better matches deployment conditions.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def external_validate(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    device = get_device(args.cpu)
    if args.model_version:
        version_dir = Path("models") / "versions" / args.model_version
        args.checkpoint = version_dir / "cropvision_cv.pt"
        args.class_names_path = version_dir / "class_names.json"
    class_names = load_class_names(Path(args.class_names_path))
    model, _, _ = load_cv_model(Path(args.checkpoint), device, Path(args.class_names_path))
    dataset = OverlapImageFolder(Path(args.data_dir), class_names)
    if len(dataset) == 0:
        unknown = ", ".join(dataset.unknown_class_counts) or "none found"
        raise ValueError(
            "No external validation images matched trained class names. "
            f"Unknown classes found: {unknown}. Rename folders or train with overlapping classes."
        )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            y_prob.extend(probabilities)
            y_pred.extend(probabilities.argmax(axis=1).tolist())
            y_true.extend(labels.tolist())

    prob_array = np.vstack(y_prob)
    label_array = np.array(y_true)
    pred_array = np.array(y_pred)
    overlap_indices = dataset.overlap_label_indices
    overlap_names = dataset.overlap_class_names
    report = classification_report(
        label_array,
        pred_array,
        labels=overlap_indices,
        target_names=overlap_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    metrics = {
        "accuracy": float(accuracy_score(label_array, pred_array)),
        "macro_f1": float(f1_score(label_array, pred_array, labels=overlap_indices, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(label_array, pred_array, labels=overlap_indices, average="weighted", zero_division=0)),
        "top_3_accuracy": float(top_k_accuracy(prob_array, label_array, k=3)),
        "num_evaluated_examples": int(len(dataset)),
        "num_unknown_examples": int(sum(dataset.unknown_class_counts.values())),
        "num_overlap_classes": int(len(overlap_names)),
        "unknown_class_counts": dataset.unknown_class_counts,
    }
    true_raw = [item[2] for item in dataset.samples]
    pred_raw = [class_names[idx] for idx in pred_array]
    true_norm = [normalize_label(name)["normalized_class"] for name in true_raw]
    pred_norm = [normalize_label(name)["normalized_class"] for name in pred_raw]
    true_broad = [normalize_label(name)["broad_problem_category"] for name in true_raw]
    pred_broad = [normalize_label(name)["broad_problem_category"] for name in pred_raw]
    metrics["exact_match_accuracy"] = float(np.mean([t == p for t, p in zip(true_raw, pred_raw)]))
    metrics["normalized_match_accuracy"] = float(np.mean([t == p for t, p in zip(true_norm, pred_norm)]))
    metrics["broad_problem_category_accuracy"] = float(np.mean([t == p for t, p in zip(true_broad, pred_broad)]))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_df.to_csv(args.report_csv)
    cm = confusion_matrix(label_array, pred_array, labels=overlap_indices)
    cm_norm = confusion_matrix(label_array, pred_array, labels=overlap_indices, normalize="true")
    plot_confusion_matrix(cm, overlap_names, FIGURES_DIR / "external_confusion_matrix.png")
    plot_confusion_matrix(cm_norm, overlap_names, FIGURES_DIR / "external_confusion_matrix_normalized.png", normalized=True)
    write_external_report(metrics, report_df, dataset.unknown_class_counts, Path(args.markdown_report))

    print("External validation summary")
    print(f"Evaluated examples: {metrics['num_evaluated_examples']}")
    print(f"Skipped unknown examples: {metrics['num_unknown_examples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Top-3 accuracy: {metrics['top_3_accuracy']:.4f}")
    print(f"Normalized match accuracy: {metrics['normalized_match_accuracy']:.4f}")
    print(f"Broad problem category accuracy: {metrics['broad_problem_category_accuracy']:.4f}")
    print(f"Saved external validation reports to {REPORTS_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CropVision on an external ImageFolder dataset.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CV_MODEL_PATH)
    parser.add_argument("--class_names_path", type=Path, default=DEFAULT_CLASS_NAMES_PATH)
    parser.add_argument("--model_version", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--metrics_json", type=Path, default=REPORTS_DIR / "external_validation_metrics.json")
    parser.add_argument("--report_csv", type=Path, default=REPORTS_DIR / "external_validation_report.csv")
    parser.add_argument("--markdown_report", type=Path, default=REPORTS_DIR / "external_validation_report.md")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    external_validate(parse_args())
