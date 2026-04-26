"""Inspect ImageFolder-style datasets before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import DEFAULT_DATA_DIR, FIGURES_DIR, IMAGE_EXTENSIONS, REPORTS_DIR, ensure_project_dirs


def inspect_imagefolder(data_dir: Path) -> dict:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}. TODO: place ImageFolder class folders under data/raw/plantvillage/."
        )

    class_counts: dict[str, int] = {}
    empty_folders: list[str] = []
    non_image_files: list[str] = []
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        image_count = 0
        for file_path in class_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_count += 1
            else:
                non_image_files.append(str(file_path.relative_to(data_dir)))
        class_counts[class_dir.name] = image_count
        if image_count == 0:
            empty_folders.append(class_dir.name)

    total_images = sum(class_counts.values())
    min_count = min(class_counts.values()) if class_counts else 0
    max_count = max(class_counts.values()) if class_counts else 0
    imbalance_ratio = float(max_count / min_count) if min_count else None
    warnings: list[str] = []
    if len(class_counts) < 2:
        warnings.append("Expected at least two class folders.")
    if empty_folders:
        warnings.append(f"Empty class folders found: {', '.join(empty_folders)}")
    if min_count and min_count < 10:
        warnings.append("At least one class has fewer than 10 images; validation metrics may be unstable.")
    if imbalance_ratio and imbalance_ratio > 3:
        warnings.append(f"Severe imbalance detected: largest class is {imbalance_ratio:.1f}x the smallest class.")
    if non_image_files:
        warnings.append(f"Found {len(non_image_files)} non-image files.")

    return {
        "data_dir": str(data_dir),
        "num_classes": len(class_counts),
        "total_images": total_images,
        "class_counts": class_counts,
        "empty_folders": empty_folders,
        "non_image_files": non_image_files,
        "imbalance_ratio": imbalance_ratio,
        "warnings": warnings,
    }


def save_class_distribution_plot(class_counts: dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, len(class_counts) * 0.45), 5))
    ax.bar(class_counts.keys(), class_counts.values(), color="#3f7f93")
    ax.set_title("Dataset Class Distribution")
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    report = inspect_imagefolder(Path(args.data_dir))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_class_distribution_plot(report["class_counts"], Path(args.output_plot))

    print(f"Classes: {report['num_classes']}")
    print(f"Images: {report['total_images']}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Saved dataset report to {args.output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an ImageFolder plant disease dataset.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_json", type=Path, default=REPORTS_DIR / "dataset_report.json")
    parser.add_argument("--output_plot", type=Path, default=FIGURES_DIR / "class_distribution.png")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
