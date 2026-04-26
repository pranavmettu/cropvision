"""Download/import public plant disease datasets into clean ImageFolder form."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from src.config import IMAGE_EXTENSIONS, REPORTS_DIR
from src.dataset_registry import DATASET_REGISTRY, get_dataset_info, print_registry

KAGGLE_DATASETS = {"new_plant_diseases_kaggle": "vipoooool/new-plant-diseases-dataset"}


def _iter_class_dirs(source_dir: Path) -> list[Path]:
    direct = [path for path in source_dir.iterdir() if path.is_dir() and any(p.suffix.lower() in IMAGE_EXTENSIONS for p in path.rglob("*"))]
    if direct:
        return sorted(direct)
    nested: list[Path] = []
    for root in source_dir.rglob("*"):
        if root.is_dir() and any(p.suffix.lower() in IMAGE_EXTENSIONS for p in root.iterdir() if p.is_file()):
            nested.append(root)
    return sorted(nested)


def import_local_dataset(dataset: str, source_dir: Path, output_dir: Path) -> dict:
    get_dataset_info(dataset)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset folder not found: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    class_distribution: dict[str, int] = {}
    skipped_files: list[str] = []
    total = 0
    for class_dir in _iter_class_dirs(source_dir):
        class_name = class_dir.name
        target_dir = output_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for file_path in class_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                skipped_files.append(str(file_path))
                continue
            try:
                with Image.open(file_path) as img:
                    img.verify()
                target_path = target_dir / f"{file_path.stem}_{count:06d}{file_path.suffix.lower()}"
                shutil.copy2(file_path, target_path)
                count += 1
                total += 1
            except Exception:
                skipped_files.append(str(file_path))
        if count:
            class_distribution[class_name] = count

    report = {
        "dataset": dataset,
        "source_path": str(source_dir),
        "output_path": str(output_dir),
        "num_classes": len(class_distribution),
        "num_images": total,
        "skipped_files": skipped_files,
        "class_distribution": class_distribution,
    }
    report_dir = REPORTS_DIR / "dataset_imports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{dataset}_import_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def download_kaggle_dataset(dataset: str, output_dir: Path) -> None:
    if dataset not in KAGGLE_DATASETS:
        raise ValueError(f"No Kaggle dataset configured for {dataset}")
    try:
        import kaggle  # type: ignore
    except ImportError:
        print("Kaggle API is not installed. To use this command:")
        print("1. pip install kaggle")
        print("2. Create a Kaggle API token from your Kaggle account settings")
        print("3. Place kaggle.json in ~/.kaggle/")
        print("4. Rerun this command")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_files(KAGGLE_DATASETS[dataset], path=str(output_dir), unzip=True)
    print(f"Downloaded {dataset} to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage CropVision reference datasets.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--import-local", action="store_true")
    parser.add_argument("--download-kaggle", action="store_true")
    parser.add_argument("--dataset", choices=sorted(DATASET_REGISTRY), default=None)
    parser.add_argument("--source_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list:
        print_registry()
    elif args.import_local:
        if not args.dataset or not args.source_dir or not args.output_dir:
            raise SystemExit("--import-local requires --dataset, --source_dir, and --output_dir")
        report = import_local_dataset(args.dataset, args.source_dir, args.output_dir)
        print(json.dumps({k: v for k, v in report.items() if k != "skipped_files"}, indent=2))
    elif args.download_kaggle:
        if not args.dataset or not args.output_dir:
            raise SystemExit("--download-kaggle requires --dataset and --output_dir")
        download_kaggle_dataset(args.dataset, args.output_dir)
    else:
        raise SystemExit("Use --list, --import-local, or --download-kaggle.")
