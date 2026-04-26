"""Build a combined normalized reference ImageFolder dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, IMAGE_EXTENSIONS, REFERENCE_DATASETS_DIR, REFERENCE_TRAIN_DIR, REPORTS_DIR
from src.label_normalizer import normalize_label


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_or_link(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "symlink":
        if dst.exists():
            return
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def build_reference_dataset(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    class_counts: dict[str, int] = {}
    skipped_classes: list[str] = []
    duplicate_count = 0

    selected = [name.strip() for name in args.datasets.split(",") if name.strip()]
    for dataset_name in selected:
        dataset_dir = source_root / dataset_name
        if not dataset_dir.exists():
            skipped_classes.append(f"{dataset_name}:missing_dataset_dir")
            continue
        for class_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            normalized = normalize_label(class_dir.name) if args.normalize_labels else {"normalized_class": class_dir.name}
            target_class = normalized["normalized_class"]
            images = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            random.shuffle(images)
            if args.min_images_per_class and len(images) < args.min_images_per_class:
                skipped_classes.append(f"{dataset_name}/{class_dir.name}:too_few_images")
                continue
            if args.max_images_per_class:
                images = images[: args.max_images_per_class]
            for image_path in images:
                digest = file_sha256(image_path)
                if digest in seen_hashes:
                    duplicate_count += 1
                    continue
                seen_hashes.add(digest)
                current_count = class_counts.get(target_class, 0)
                suffix = image_path.suffix.lower()
                target_path = output_dir / target_class / f"{dataset_name}_{current_count:07d}{suffix}"
                _copy_or_link(image_path, target_path, args.copy_mode)
                class_counts[target_class] = current_count + 1

    total_images = sum(class_counts.values())
    sorted_counts = sorted(class_counts.items(), key=lambda item: item[1])
    report = {
        "datasets": selected,
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "total_images": total_images,
        "total_classes": len(class_counts),
        "duplicate_count": duplicate_count,
        "skipped_classes": skipped_classes,
        "largest_classes": sorted_counts[-10:],
        "smallest_classes": sorted_counts[:10],
        "class_distribution": class_counts,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "reference_dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    dist_df = pd.DataFrame([{"class_name": k, "image_count": v} for k, v in sorted_counts])
    dist_df.to_csv(REPORTS_DIR / "reference_dataset_class_distribution.csv", index=False)
    if not dist_df.empty:
        fig, ax = plt.subplots(figsize=(max(8, len(dist_df) * 0.25), 5))
        ax.bar(dist_df["class_name"], dist_df["image_count"], color="#3f7f93")
        ax.set_title("Reference Dataset Class Distribution")
        ax.set_ylabel("Images")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "reference_dataset_class_distribution.png", dpi=180)
        plt.close(fig)

    print(f"Total images: {total_images}")
    print(f"Total classes: {len(class_counts)}")
    print(f"Duplicates skipped: {duplicate_count}")
    print(f"Skipped classes: {len(skipped_classes)}")
    print(f"Largest classes: {report['largest_classes'][-5:]}")
    print(f"Smallest classes: {report['smallest_classes'][:5]}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined CropVision reference dataset.")
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--source_root", type=Path, default=REFERENCE_DATASETS_DIR)
    parser.add_argument("--output_dir", type=Path, default=REFERENCE_TRAIN_DIR)
    parser.add_argument("--normalize_labels", action="store_true")
    parser.add_argument("--max_images_per_class", type=int, default=None)
    parser.add_argument("--min_images_per_class", type=int, default=None)
    parser.add_argument("--copy_mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    build_reference_dataset(parse_args())
