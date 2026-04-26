"""Prepare ImageFolder datasets for the dedicated disease model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.config import FIGURES_DIR, IMAGE_EXTENSIONS, REPORTS_DIR
from src.dataset_manager import _iter_class_dirs
from src.disease_label_normalizer import normalize_disease_label


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_disease_dataset(dataset: str, source_dir: Path, output_dir: Path) -> dict:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset folder not found: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    skipped_files: list[dict[str, str]] = []
    class_distribution: Counter[str] = Counter()
    raw_to_normalized: dict[str, str] = {}

    for class_dir in _iter_class_dirs(source_dir):
        raw_label = class_dir.name
        label_info = normalize_disease_label(raw_label)
        normalized_class = label_info["normalized_class"]
        raw_to_normalized[raw_label] = normalized_class
        target_dir = output_dir / normalized_class
        target_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                skipped_files.append({"path": str(image_path), "reason": "non_image_extension"})
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
                digest = file_sha256(image_path)
                if digest in seen_hashes:
                    skipped_files.append({"path": str(image_path), "reason": "duplicate_hash"})
                    continue
                seen_hashes.add(digest)
                target_name = f"{image_path.stem}_{digest[:10]}{image_path.suffix.lower()}"
                shutil.copy2(image_path, target_dir / target_name)
                class_distribution[normalized_class] += 1
            except Exception as exc:
                skipped_files.append({"path": str(image_path), "reason": f"corrupt_or_unreadable: {exc}"})

    distribution_path = REPORTS_DIR / "disease_dataset_class_distribution.csv"
    with distribution_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "image_count"])
        writer.writeheader()
        for class_name, count in sorted(class_distribution.items()):
            writer.writerow({"class_name": class_name, "image_count": count})

    plot_disease_distribution(class_distribution, FIGURES_DIR / "disease_dataset_class_distribution.png")
    report = {
        "dataset": dataset,
        "source_path": str(source_dir),
        "output_path": str(output_dir),
        "num_classes": len(class_distribution),
        "num_images": int(sum(class_distribution.values())),
        "skipped_files": skipped_files,
        "class_distribution": dict(sorted(class_distribution.items())),
        "raw_to_normalized_label_map": raw_to_normalized,
    }
    (REPORTS_DIR / "disease_dataset_import_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Imported {report['num_images']} images across {report['num_classes']} classes into {output_dir}")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files. See reports/disease_dataset_import_report.json")
    return report


def plot_disease_distribution(class_distribution: Counter[str], output_path: Path) -> None:
    if not class_distribution:
        return
    names = list(sorted(class_distribution))
    counts = [class_distribution[name] for name in names]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.35), 5))
    ax.bar(names, counts, color="#3d7c5f")
    ax.set_title("Disease Dataset Class Distribution")
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import plant disease datasets into normalized ImageFolder format.")
    parser.add_argument("--import-local", action="store_true")
    parser.add_argument("--dataset", choices=["plantvillage", "kaggle_new_plant_diseases", "imagefolder"], default="imagefolder")
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.import_local:
        raise SystemExit("Use --import-local with --source_dir and --output_dir.")
    import_disease_dataset(args.dataset, args.source_dir, args.output_dir)
