"""Retrain the disease model with reference data plus human-verified feedback."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.config import IMAGE_EXTENSIONS, REFERENCE_TRAIN_DIR, REPORTS_DIR, USER_FEEDBACK_DIR
from src.train_disease_model import main as train_disease_main


def combine_disease_data_with_feedback(
    base_data_dir: Path,
    feedback_dir: Path,
    output_dir: Path,
    allow_new_classes: bool = False,
) -> dict:
    if not base_data_dir.exists():
        raise FileNotFoundError(f"Base disease dataset not found: {base_data_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(base_data_dir, output_dir)
    existing_classes = {path.name for path in output_dir.iterdir() if path.is_dir()}
    added = 0
    skipped_new_classes: list[str] = []
    if feedback_dir.exists():
        for class_dir in sorted(path for path in feedback_dir.iterdir() if path.is_dir()):
            if class_dir.name not in existing_classes and not allow_new_classes:
                skipped_new_classes.append(class_dir.name)
                continue
            target_dir = output_dir / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    shutil.copy2(image_path, target_dir / f"verified_feedback_{image_path.name}")
                    added += 1
    return {"output_dir": str(output_dir), "added_feedback_images": added, "skipped_new_classes": skipped_new_classes}


def write_retraining_report(summary: dict, args: argparse.Namespace) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    skipped = ", ".join(summary["skipped_new_classes"]) or "None"
    content = f"""# Disease Feedback Retraining Report

Base dataset: `{args.base_data_dir}`

Feedback directory: `{args.feedback_dir}`

Combined training directory: `{summary['output_dir']}`

Added verified feedback images: {summary['added_feedback_images']}

Skipped new classes: {skipped}

Blind self-training is intentionally not supported. Uploaded images are included only after human confirmation.
"""
    (REPORTS_DIR / "disease_feedback_retraining_report.md").write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain disease model with verified feedback.")
    parser.add_argument("--base_data_dir", type=Path, default=REFERENCE_TRAIN_DIR)
    parser.add_argument("--feedback_dir", type=Path, default=USER_FEEDBACK_DIR)
    parser.add_argument("--work_dir", type=Path, default=Path("data/processed/disease_training_with_feedback"))
    parser.add_argument("--model_version_name", type=str, default="disease_feedback_v1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--allow_new_classes", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = combine_disease_data_with_feedback(args.base_data_dir, args.feedback_dir, args.work_dir, args.allow_new_classes)
    write_retraining_report(summary, args)
    train_args = [
        "--data_dir",
        str(args.work_dir),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--model_version_name",
        args.model_version_name,
        "--freeze_backbone",
        "--weighted_loss",
    ]
    train_disease_main(train_args)
