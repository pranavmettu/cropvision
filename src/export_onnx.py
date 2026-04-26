"""Export a trained CropVision PyTorch checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import DEFAULT_CV_MODEL_PATH, IMAGE_SIZE, MODEL_DIR, ensure_project_dirs
from src.predict_cv import load_cv_model
from src.utils import get_device


def export_onnx(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    device = get_device(force_cpu=True)
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    model, class_names, architecture = load_cv_model(checkpoint, device)
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=args.opset,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"ONNX export requires an optional package that is not installed: {exc}")
        print("Install optional export tools with `pip install onnx onnxruntime` and rerun this command.")
        return
    if not output.exists():
        raise RuntimeError(f"ONNX export did not create expected file: {output}")

    print(f"Exported {architecture} model with {len(class_names)} classes to {output}")
    print(f"ONNX file size: {output.stat().st_size / (1024 * 1024):.2f} MB")

    try:
        import onnx  # type: ignore

        onnx_model = onnx.load(output)
        onnx.checker.check_model(onnx_model)
        print("ONNX structural check passed.")
    except ImportError:
        print("Optional package 'onnx' is not installed. Install with `pip install onnx` to run structural validation.")

    try:
        import onnxruntime  # noqa: F401

        print("ONNX Runtime is installed; you can benchmark with python -m src.benchmark_inference.")
    except ImportError:
        print("Optional package 'onnxruntime' is not installed. Install with `pip install onnxruntime` to benchmark ONNX inference.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CropVision checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CV_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "cropvision_cv.onnx")
    parser.add_argument("--opset", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    export_onnx(parse_args())
