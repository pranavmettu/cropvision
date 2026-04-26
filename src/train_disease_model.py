"""Train the dedicated CropVision disease classifier."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.config import FIGURES_DIR
from src.disease_model import DISEASE_CLASS_NAMES_PATH, DISEASE_METADATA_PATH, DISEASE_MODEL_DIR, DISEASE_MODEL_PATH
from src.train_cv import parse_args as parse_train_args
from src.train_cv import train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train CropVision disease model.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed/cropvision_reference_train"))
    parser.add_argument("--model_name", choices=["resnet18", "efficientnet_b0", "mobilenet_v3_small"], default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--weighted_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--max_images_per_class", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--model_version_name", type=str, default="disease_v1")
    args = parser.parse_args(argv)

    DISEASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_argv = [
        "--data_dir",
        str(args.data_dir),
        "--model_name",
        args.model_name,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--val_split",
        str(args.val_split),
        "--label_smoothing",
        str(args.label_smoothing),
        "--model_version_name",
        args.model_version_name,
        "--output",
        str(DISEASE_MODEL_PATH),
        "--class_names_path",
        str(DISEASE_CLASS_NAMES_PATH),
        "--history_path",
        "reports/disease_training_history.csv",
    ]
    train_argv.append("--freeze_backbone" if args.freeze_backbone and not args.fine_tune else "--no-freeze_backbone")
    train_argv.append("--weighted_loss" if args.weighted_loss else "--no-weighted_loss")
    if args.max_images_per_class:
        train_argv.extend(["--max_images_per_class", str(args.max_images_per_class)])
    train(parse_train_args(train_argv))

    version_dir = Path("models/versions") / args.model_version_name
    if version_dir.exists():
        if (version_dir / "cropvision_cv.pt").exists():
            shutil.copy2(version_dir / "cropvision_cv.pt", version_dir / "disease_model.pt")
        if (version_dir / "class_names.json").exists():
            shutil.copy2(version_dir / "class_names.json", version_dir / "disease_class_names.json")
    if (FIGURES_DIR / "loss_curve.png").exists() or (FIGURES_DIR / "accuracy_curve.png").exists():
        # Keep disease-specific aliases for portfolio reports without changing the shared trainer.
        if (FIGURES_DIR / "loss_curve.png").exists():
            shutil.copy2(FIGURES_DIR / "loss_curve.png", FIGURES_DIR / "disease_loss_curve.png")
            shutil.copy2(FIGURES_DIR / "loss_curve.png", FIGURES_DIR / "disease_training_curves.png")
        if (FIGURES_DIR / "accuracy_curve.png").exists():
            shutil.copy2(FIGURES_DIR / "accuracy_curve.png", FIGURES_DIR / "disease_accuracy_curve.png")
    if (FIGURES_DIR / "class_distribution.png").exists():
        shutil.copy2(FIGURES_DIR / "class_distribution.png", FIGURES_DIR / "disease_class_distribution.png")
    metadata = {
        "model_version_name": args.model_version_name,
        "model_name": args.model_name,
        "data_dir": str(args.data_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weighted_loss": args.weighted_loss,
        "label_smoothing": args.label_smoothing,
    }
    DISEASE_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved disease model to {DISEASE_MODEL_PATH}")


if __name__ == "__main__":
    main()
