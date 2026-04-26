"""Reference image retrieval database for trained disease examples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from src.config import DEFAULT_CV_MODEL_PATH, IMAGE_EXTENSIONS, REFERENCE_INDEX_DIR, REFERENCE_TRAIN_DIR
from src.image_retrieval import _feature_model_from_classifier, extract_embedding
from src.label_normalizer import normalize_label
from src.predict_cv import load_cv_model
from src.utils import get_device


def format_retrieval_metadata(image_path: str, class_label: str, similarity: float) -> dict[str, Any]:
    normalized = normalize_label(class_label)
    return {
        "image_path": image_path,
        "class_label": class_label,
        "similarity_score": float(similarity),
        "plant_species": normalized["plant_species"],
        "disease_name": normalized["disease_name"],
        "broad_problem_category": normalized["broad_problem_category"],
    }


def build_reference_index(data_dir: Path, output_dir: Path, checkpoint_path: Path = DEFAULT_CV_MODEL_PATH, max_images: int | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(force_cpu=True)
    model, _, architecture = load_cv_model(checkpoint_path, device)
    feature_model = _feature_model_from_classifier(model, architecture).to(device)

    samples: list[tuple[Path, str]] = []
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, class_dir.name))
    if max_images:
        samples = samples[:max_images]
    if not samples:
        raise ValueError(f"No reference images found in {data_dir}")

    embeddings = np.vstack([extract_embedding(path, feature_model, device) for path, _ in samples])
    metadata = pd.DataFrame([format_retrieval_metadata(str(path), label, 1.0) for path, label in samples])
    np.savez_compressed(output_dir / "embeddings.npz", embeddings=embeddings)
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        artifact = {"backend": "faiss", "index": faiss.serialize_index(index)}
    except ImportError:
        index = NearestNeighbors(metric="cosine")
        index.fit(embeddings)
        artifact = {"backend": "sklearn", "index": index}
    joblib.dump(artifact, output_dir / "index.joblib")
    summary = {"num_images": len(samples), "embedding_dim": int(embeddings.shape[1]), "backend": artifact["backend"]}
    print(summary)
    return summary


def find_reference_matches(
    image_path: str | Path,
    top_k: int = 5,
    index_dir: Path = REFERENCE_INDEX_DIR,
    checkpoint_path: Path = DEFAULT_CV_MODEL_PATH,
) -> list[dict[str, Any]]:
    if not (index_dir / "index.joblib").exists() or not (index_dir / "metadata.csv").exists() or not (index_dir / "embeddings.npz").exists():
        return []
    artifact = joblib.load(index_dir / "index.joblib")
    metadata = pd.read_csv(index_dir / "metadata.csv")
    embeddings = np.load(index_dir / "embeddings.npz")["embeddings"]
    device = get_device(force_cpu=True)
    model, _, architecture = load_cv_model(checkpoint_path, device)
    feature_model = _feature_model_from_classifier(model, architecture).to(device)
    query = extract_embedding(image_path, feature_model, device).reshape(1, -1)

    if artifact["backend"] == "faiss":
        try:
            import faiss  # type: ignore

            index = faiss.deserialize_index(artifact["index"])
            scores, indices = index.search(query.astype(np.float32), min(top_k, len(metadata)))
            pairs = zip(indices[0], scores[0])
        except ImportError:
            pairs = []
    else:
        distances, indices = artifact["index"].kneighbors(query, n_neighbors=min(top_k, len(embeddings)))
        pairs = zip(indices[0], 1.0 - distances[0])
    results = []
    for idx, score in pairs:
        row = metadata.iloc[int(idx)].to_dict()
        row["similarity_score"] = float(score)
        results.append(row)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/query CropVision reference retrieval index.")
    parser.add_argument("--build_index", action="store_true")
    parser.add_argument("--data_dir", type=Path, default=REFERENCE_TRAIN_DIR)
    parser.add_argument("--output_dir", type=Path, default=REFERENCE_INDEX_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CV_MODEL_PATH)
    parser.add_argument("--image_path", type=Path, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.build_index:
        build_reference_index(args.data_dir, args.output_dir, args.checkpoint, args.max_images)
    elif args.image_path:
        print(find_reference_matches(args.image_path, args.top_k, args.output_dir, args.checkpoint))
    else:
        raise SystemExit("Use --build_index or --image_path.")
