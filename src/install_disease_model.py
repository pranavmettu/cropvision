"""Install, register, or train a CropVision disease model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.disease_model import DISEASE_CLASS_NAMES_PATH, DISEASE_METADATA_PATH, DISEASE_MODEL_DIR, DISEASE_MODEL_PATH


def install_local_checkpoint(checkpoint_path: Path, class_names_path: Path) -> None:
    if not checkpoint_path.exists() or not class_names_path.exists():
        raise FileNotFoundError("Checkpoint and class names paths must both exist.")
    DISEASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, DISEASE_MODEL_PATH)
    shutil.copy2(class_names_path, DISEASE_CLASS_NAMES_PATH)
    DISEASE_METADATA_PATH.write_text(
        json.dumps({"install_mode": "local_checkpoint", "source_checkpoint": str(checkpoint_path)}, indent=2),
        encoding="utf-8",
    )
    print(f"Installed disease model to {DISEASE_MODEL_PATH}")


def install_huggingface_model(model_id: str) -> None:
    try:
        import transformers  # type: ignore  # noqa: F401
    except ImportError:
        print("Install transformers to use Hugging Face disease models: pip install transformers")
        return
    DISEASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DISEASE_METADATA_PATH.write_text(
        json.dumps(
            {
                "install_mode": "huggingface",
                "model_id": model_id,
                "note": "Hugging Face adapter metadata saved. Use a compatible PyTorch checkpoint or add adapter code for this model format.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved Hugging Face disease model metadata. Compatible checkpoint adapter may still be required.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install or prepare a CropVision disease model.")
    parser.add_argument("--mode", choices=["local_checkpoint", "huggingface", "train_from_dataset"], required=True)
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    parser.add_argument("--class_names_path", type=Path, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed/cropvision_reference_train"))
    parser.add_argument("--model_name", type=str, default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "local_checkpoint":
        install_local_checkpoint(args.checkpoint_path, args.class_names_path)
    elif args.mode == "huggingface":
        install_huggingface_model(args.model_id)
    else:
        from src.train_disease_model import main as train_disease_main

        train_disease_main(
            [
                "--data_dir",
                str(args.data_dir),
                "--model_name",
                args.model_name,
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
            ]
        )
