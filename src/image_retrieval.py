"""Visual similarity retrieval for CropVision training examples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torchvision import datasets

from src.config import DEFAULT_CV_MODEL_PATH, DEFAULT_DATA_DIR, DEFAULT_RETRIEVAL_ARTIFACT_PATH, IMAGE_EXTENSIONS, ensure_project_dirs
from src.dataset import get_eval_transforms
from src.predict_cv import load_cv_model
from src.utils import get_device


def _feature_model_from_classifier(model: nn.Module, architecture: str) -> nn.Module:
    if architecture == "resnet18":
        return nn.Sequential(*list(model.children())[:-1])
    if architecture == "efficientnet_b0":
        return nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)))
    if architecture == "mobilenet_v3_small":
        return nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)))
    if architecture == "convnext_tiny":
        return nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)))
    raise ValueError(f"Unsupported architecture for retrieval: {architecture}")


def extract_embedding(image_path: str | Path, feature_model: nn.Module, device: torch.device) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    tensor = get_eval_transforms()(image).unsqueeze(0).to(device)
    feature_model.eval()
    with torch.no_grad():
        embedding = feature_model(tensor).flatten(1).cpu().numpy()[0]
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.astype(np.float32)


def build_retrieval_index(data_dir: Path, checkpoint_path: Path, output_path: Path, max_images: int | None = None) -> dict[str, Any]:
    ensure_project_dirs()
    device = get_device(force_cpu=True)
    model, class_names, architecture = load_cv_model(checkpoint_path, device)
    feature_model = _feature_model_from_classifier(model, architecture).to(device)
    dataset = datasets.ImageFolder(str(data_dir))
    samples = [
        (Path(path), dataset.classes[label])
        for path, label in dataset.samples
        if Path(path).suffix.lower() in IMAGE_EXTENSIONS
    ]
    if max_images is not None:
        samples = samples[:max_images]
    if not samples:
        raise ValueError(f"No images found for retrieval index in {data_dir}")

    embeddings = np.vstack([extract_embedding(path, feature_model, device) for path, _ in samples])
    paths = [str(path) for path, _ in samples]
    labels = [label for _, label in samples]
    artifact = {
        "backend": "sklearn_nearest_neighbors",
        "embeddings": embeddings,
        "paths": paths,
        "labels": labels,
        "architecture": architecture,
        "checkpoint_path": str(checkpoint_path),
    }

    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        artifact["backend"] = "faiss_flat_ip"
        artifact["faiss_index"] = faiss.serialize_index(index)
    except ImportError:
        index = NearestNeighbors(metric="cosine")
        index.fit(embeddings)
        artifact["sklearn_index"] = index

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return {"num_images": len(paths), "backend": artifact["backend"], "output_path": str(output_path)}


def find_similar_images(
    image_path: str,
    top_k: int = 5,
    artifact_path: Path = DEFAULT_RETRIEVAL_ARTIFACT_PATH,
    checkpoint_path: Path = DEFAULT_CV_MODEL_PATH,
) -> list[dict[str, Any]]:
    if not artifact_path.exists():
        return []
    artifact = joblib.load(artifact_path)
    device = get_device(force_cpu=True)
    model, _, architecture = load_cv_model(checkpoint_path, device)
    feature_model = _feature_model_from_classifier(model, architecture).to(device)
    query = extract_embedding(image_path, feature_model, device).reshape(1, -1)

    if artifact.get("backend") == "faiss_flat_ip":
        try:
            import faiss  # type: ignore

            index = faiss.deserialize_index(artifact["faiss_index"])
            scores, indices = index.search(query.astype(np.float32), top_k)
            pairs = zip(indices[0].tolist(), scores[0].tolist())
        except ImportError:
            pairs = []
    else:
        index = artifact["sklearn_index"]
        distances, indices = index.kneighbors(query, n_neighbors=min(top_k, len(artifact["paths"])))
        pairs = zip(indices[0].tolist(), (1.0 - distances[0]).tolist())

    results: list[dict[str, Any]] = []
    for idx, score in pairs:
        if idx < 0:
            continue
        results.append({"image_path": artifact["paths"][idx], "label": artifact["labels"][idx], "similarity": float(score)})
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or query CropVision visual retrieval artifacts.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CV_MODEL_PATH)
    parser.add_argument("--artifact_path", type=Path, default=DEFAULT_RETRIEVAL_ARTIFACT_PATH)
    parser.add_argument("--build_index", action="store_true")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.build_index:
        summary = build_retrieval_index(args.data_dir, args.checkpoint, args.artifact_path, args.max_images)
        print(summary)
    elif args.image_path:
        print(find_similar_images(args.image_path, args.top_k, args.artifact_path, args.checkpoint))
    else:
        raise SystemExit("Use --build_index or provide --image_path.")
