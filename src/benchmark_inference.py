"""CPU inference benchmarking for PyTorch and optional ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.config import DEFAULT_CV_MODEL_PATH, IMAGE_SIZE, MODEL_DIR, REPORTS_DIR, ensure_project_dirs
from src.predict_cv import load_cv_model


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    values = np.array(latencies_ms, dtype=float)
    return {
        "avg_latency_ms": float(values.mean()),
        "p50_latency_ms": float(np.percentile(values, 50)),
        "p95_latency_ms": float(np.percentile(values, 95)),
    }


def benchmark_pytorch(checkpoint: Path, warmup: int, iterations: int) -> dict:
    device = torch.device("cpu")
    model, _, _ = load_cv_model(checkpoint, device)
    model.eval()
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
        latencies: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            latencies.append((time.perf_counter() - start) * 1000)
    return {
        "backend": "pytorch_cpu",
        "iterations": iterations,
        "model_file_size_mb": checkpoint.stat().st_size / (1024 * 1024) if checkpoint.exists() else None,
        **summarize_latencies(latencies),
    }


def benchmark_onnx(onnx_path: Path, warmup: int, iterations: int) -> dict | None:
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}; skipping ONNX benchmark.")
        return None
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        print("Optional package 'onnxruntime' is not installed; skipping ONNX benchmark.")
        return None

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})
    latencies: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - start) * 1000)
    return {
        "backend": "onnxruntime_cpu",
        "iterations": iterations,
        "model_file_size_mb": onnx_path.stat().st_size / (1024 * 1024),
        **summarize_latencies(latencies),
    }


def benchmark(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    results = {
        "pytorch": benchmark_pytorch(Path(args.checkpoint), args.warmup, args.iterations),
        "onnxruntime": benchmark_onnx(Path(args.onnx_model), args.warmup, args.iterations),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Inference benchmark summary")
    for name, result in results.items():
        if result is None:
            continue
        print(
            f"{name}: avg={result['avg_latency_ms']:.2f} ms | "
            f"p50={result['p50_latency_ms']:.2f} ms | p95={result['p95_latency_ms']:.2f} ms | "
            f"size={result['model_file_size_mb']:.2f} MB | n={result['iterations']}"
        )
    print(f"Saved benchmark results to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CropVision inference on CPU.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CV_MODEL_PATH)
    parser.add_argument("--onnx_model", type=Path, default=MODEL_DIR / "cropvision_cv.onnx")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "inference_benchmark.json")
    return parser.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())
