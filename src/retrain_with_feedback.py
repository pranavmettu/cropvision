"""Explicit retraining workflow for reference data plus verified feedback."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.config import REFERENCE_TRAIN_DIR, USER_FEEDBACK_DIR
from src.train_cv import parse_args as parse_train_args
from src.train_cv import train


def combine_reference_and_feedback(base_data_dir: Path, feedback_dir: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(base_data_dir, output_dir)
    if feedback_dir.exists():
        for class_dir in sorted(path for path in feedback_dir.iterdir() if path.is_dir()):
            target_dir = output_dir / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    shutil.copy2(image_path, target_dir / f"feedback_{image_path.name}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain CropVision with reference data plus human-verified feedback.")
    parser.add_argument("--base_data_dir", type=Path, default=REFERENCE_TRAIN_DIR)
    parser.add_argument("--feedback_dir", type=Path, default=USER_FEEDBACK_DIR)
    parser.add_argument("--work_dir", type=Path, default=Path("data/processed/reference_plus_feedback_train"))
    parser.add_argument("--model_version_name", type=str, default="reference_plus_feedback_v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Combining reference data with verified feedback. Blind self-training is intentionally not supported.")
    combined_dir = combine_reference_and_feedback(args.base_data_dir, args.feedback_dir, args.work_dir)
    train_args = parse_train_args(
        [
            "--data_dir",
            str(combined_dir),
            "--model_version_name",
            args.model_version_name,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
        ]
    )
    train(train_args)
