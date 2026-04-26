"""Train a transfer-learning image classifier for plant disease diagnosis."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn, optim
from tqdm import tqdm
from torchvision import models

from src.config import (
    DEFAULT_CLASS_NAMES_PATH,
    DEFAULT_CV_MODEL_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_TRAIN_HISTORY_PATH,
    FIGURES_DIR,
    MODEL_DIR,
    ensure_project_dirs,
)
from src.dataset import create_dataloaders
from src.utils import get_device, save_json, set_seed


def maybe_start_mlflow(args: argparse.Namespace):
    """Start an MLflow run only when requested and installed."""
    if not getattr(args, "use_mlflow", False):
        return None
    try:
        import mlflow  # type: ignore
    except ImportError:
        print("MLflow requested but not installed. Install with `pip install mlflow`; continuing without MLflow logging.")
        return None

    mlflow.set_experiment(args.mlflow_experiment)
    run = mlflow.start_run()
    mlflow.log_params(
        {
            "data_dir": str(args.data_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "model_name": args.model_name,
            "freeze_backbone": args.freeze_backbone,
            "weighted_loss": args.weighted_loss,
            "early_stopping_patience": args.early_stopping_patience,
            "max_images_per_class": args.max_images_per_class,
            "confidence_threshold": args.confidence_threshold,
            "seed": args.seed,
        }
    )
    return mlflow, run


def build_model(num_classes: int, architecture: str = "resnet18", pretrained: bool = True) -> nn.Module:
    def _weights_or_none(weights_enum):
        if not pretrained:
            return None
        return weights_enum.DEFAULT

    if architecture == "efficientnet_b0":
        try:
            model = models.efficientnet_b0(weights=_weights_or_none(models.EfficientNet_B0_Weights))
        except Exception as exc:
            print(f"Could not load pretrained EfficientNet-B0 weights ({exc}). Falling back to random initialization.")
            model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    if architecture == "mobilenet_v3_small":
        try:
            model = models.mobilenet_v3_small(weights=_weights_or_none(models.MobileNet_V3_Small_Weights))
        except Exception as exc:
            print(f"Could not load pretrained MobileNetV3 weights ({exc}). Falling back to random initialization.")
            model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if architecture == "convnext_tiny":
        try:
            model = models.convnext_tiny(weights=_weights_or_none(models.ConvNeXt_Tiny_Weights))
        except Exception as exc:
            print(f"Could not load pretrained ConvNeXt-Tiny weights ({exc}). Falling back to random initialization.")
            model = models.convnext_tiny(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if architecture == "resnet18":
        try:
            model = models.resnet18(weights=_weights_or_none(models.ResNet18_Weights))
        except Exception as exc:
            print(f"Could not load pretrained ResNet18 weights ({exc}). Falling back to random initialization.")
            model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError("architecture must be resnet18, efficientnet_b0, mobilenet_v3_small, or convnext_tiny.")


def freeze_backbone_layers(model: nn.Module, architecture: str) -> None:
    if architecture == "resnet18":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")
    elif architecture == "efficientnet_b0":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("classifier.")
    elif architecture in {"mobilenet_v3_small", "convnext_tiny"}:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("classifier.")


def _labels_from_dataset(dataset: torch.utils.data.Dataset) -> list[int]:
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base_labels = _labels_from_dataset(dataset.dataset)
        return [base_labels[int(idx)] for idx in dataset.indices]
    if hasattr(dataset, "targets"):
        return [int(label) for label in dataset.targets]
    if hasattr(dataset, "samples"):
        return [int(label) for _, label in dataset.samples]
    labels: list[int] = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        labels.append(int(label))
    return labels


def class_weights_from_loader(loader: torch.utils.data.DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = _labels_from_dataset(loader.dataset)
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def plot_class_distribution(class_names: list[str], labels: list[int], output_path: Path) -> None:
    counts = np.bincount(labels, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(max(7, len(class_names) * 0.45), 5))
    ax.bar(class_names, counts, color="#3f7f93")
    ax.set_title("Class Distribution")
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_training_curves(history: list[dict[str, float]], output_dir: Path) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["epoch"], df["train_loss"], label="Train loss")
    ax.plot(df["epoch"], df["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["epoch"], df["train_acc"], label="Train accuracy")
    ax.plot(df["epoch"], df["val_acc"], label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Training and Validation Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_curve.png", dpi=180)
    plt.close(fig)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


def train(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    set_seed(args.seed)
    device = get_device(args.cpu)
    data_dir = Path(args.data_dir)
    version_dir = MODEL_DIR / "versions" / args.model_version_name if args.model_version_name else None
    output_path = version_dir / "cropvision_cv.pt" if version_dir else Path(args.output)
    class_names_output = version_dir / "class_names.json" if version_dir else Path(args.class_names_path)
    mlflow_context = maybe_start_mlflow(args)
    mlflow = mlflow_context[0] if mlflow_context else None

    try:
        train_loader, val_loader, class_names = create_dataloaders(
            data_dir=data_dir,
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers,
            seed=args.seed,
            max_images_per_class=args.max_images_per_class,
        )
        model_name = args.model_name or args.architecture
        model = build_model(len(class_names), model_name, pretrained=not args.no_pretrained).to(device)
        if args.freeze_backbone:
            freeze_backbone_layers(model, model_name)
            print("Frozen backbone enabled: training classifier head only.")

        train_labels = _labels_from_dataset(train_loader.dataset)
        plot_class_distribution(class_names, train_labels, FIGURES_DIR / "class_distribution.png")

        if args.weighted_loss:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights_from_loader(train_loader, len(class_names), device),
                label_smoothing=args.label_smoothing,
            )
            print("Weighted loss enabled for class imbalance.")
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

        best_f1 = -1.0
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        history: list[dict[str, float]] = []
        checkpoint_path = output_path
        class_names_path = class_names_output

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, device)
            scheduler.step(val_f1)
            current_lr = optimizer.param_groups[0]["lr"]
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_macro_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "lr": current_lr,
            }
            history.append(metrics)
            if mlflow is not None:
                mlflow.log_metrics({key: value for key, value in metrics.items() if key != "epoch"}, step=epoch)
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
                f"val_macro_f1={val_f1:.4f} | lr={current_lr:.2e}"
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_val_loss = val_loss
                epochs_without_improvement = 0
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "architecture": model_name,
                        "num_classes": len(class_names),
                        "class_names": class_names,
                        "best_val_macro_f1": best_f1,
                        "best_val_loss": best_val_loss,
                        "confidence_threshold": args.confidence_threshold,
                    },
                    checkpoint_path,
                )
                save_json(class_names, class_names_path)
                print(f"Saved best model to {checkpoint_path} with macro F1={best_f1:.4f}")
                if mlflow is not None:
                    mlflow.log_metric("best_val_macro_f1", best_f1)
                    mlflow.log_metric("best_val_loss", best_val_loss)
                    mlflow.log_param("best_model_path", str(checkpoint_path))
            else:
                epochs_without_improvement += 1
                if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                    print(f"Early stopping after {epoch} epochs without macro F1 improvement.")
                    break

        history_path = Path(args.history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(history_path, index=False)
        plot_training_curves(history, FIGURES_DIR)
        config_payload = vars(args).copy()
        config_payload["data_dir"] = str(args.data_dir)
        config_payload["output"] = str(args.output)
        config_payload["class_names_path"] = str(args.class_names_path)
        config_payload["history_path"] = str(args.history_path)
        metrics_payload = {
            "best_val_macro_f1": float(best_f1),
            "best_val_loss": float(best_val_loss),
            "epochs_trained": len(history),
        }
        if version_dir:
            version_dir.mkdir(parents=True, exist_ok=True)
            (version_dir / "training_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
            (version_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
            label_map_path = MODEL_DIR / "label_map.json"
            if label_map_path.exists():
                shutil.copy2(label_map_path, version_dir / "label_map.json")
            if checkpoint_path.exists():
                shutil.copy2(checkpoint_path, DEFAULT_CV_MODEL_PATH)
            if class_names_path.exists():
                shutil.copy2(class_names_path, DEFAULT_CLASS_NAMES_PATH)
            print(f"Saved versioned model artifacts to {version_dir}")
        print(f"Saved training history to {history_path}")
        if mlflow is not None:
            mlflow.log_artifact(str(history_path))
            for artifact in (FIGURES_DIR / "loss_curve.png", FIGURES_DIR / "accuracy_curve.png", FIGURES_DIR / "class_distribution.png"):
                if artifact.exists():
                    mlflow.log_artifact(str(artifact))
            if checkpoint_path.exists():
                mlflow.log_artifact(str(checkpoint_path))
    finally:
        if mlflow is not None:
            mlflow.end_run()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CropVision computer vision model.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--model_name", choices=["resnet18", "efficientnet_b0", "mobilenet_v3_small", "convnext_tiny"], default="efficientnet_b0")
    parser.add_argument("--architecture", choices=["resnet18", "efficientnet_b0", "mobilenet_v3_small", "convnext_tiny"], default=None, help="Backward-compatible alias for --model_name.")
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fine_tune", action="store_true", help="Alias for --no-freeze_backbone.")
    parser.add_argument("--weighted_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--model_version_name", type=str, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--max_images_per_class", type=int, default=None)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--no_pretrained", action="store_true", help="Do not load ImageNet weights.")
    parser.add_argument("--output", type=Path, default=DEFAULT_CV_MODEL_PATH)
    parser.add_argument("--class_names_path", type=Path, default=DEFAULT_CLASS_NAMES_PATH)
    parser.add_argument("--history_path", type=Path, default=DEFAULT_TRAIN_HISTORY_PATH)
    parser.add_argument("--use_mlflow", action="store_true", help="Log optional training metadata to MLflow if installed.")
    parser.add_argument("--mlflow_experiment", type=str, default="CropVision")
    args = parser.parse_args(argv)
    if args.architecture is not None:
        args.model_name = args.architecture
    if args.fine_tune:
        args.freeze_backbone = False
    return args


if __name__ == "__main__":
    train(parse_args())
