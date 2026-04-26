"""Reference image retrieval for disease examples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.config import IMAGE_EXTENSIONS, REFERENCE_TRAIN_DIR
from src.disease_label_normalizer import normalize_disease_label
from src.disease_model import DISEASE_MODEL_PATH, load_disease_model
from src.image_retrieval import _feature_model_from_classifier, extract_embedding
from src.utils import get_device

DISEASE_REFERENCE_INDEX_DIR = Path("models/disease_reference_index")


def format_disease_retrieval_metadata(image_path: str, class_label: str, similarity_score: float) -> dict[str, Any]:
    label_info = normalize_disease_label(class_label)
    return {
        "image_path": image_path,
        "class_label": class_label,
        "plant_species": label_info["plant_species"],
        "disease_name": label_info["disease_name"],
        "broad_problem_category": label_info["broad_problem_category"],
        "similarity_score": float(similarity_score),
    }


def build_disease_reference_index(data_dir: Path, output_dir: Path = DISEASE_REFERENCE_INDEX_DIR, max_images: int | None = None) -> dict[str, Any]:
    if not DISEASE_MODEL_PATH.exists():
        raise FileNotFoundError("Disease model not found. Train or install a disease model before building the disease reference index.")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(force_cpu=True)
    model, _, metadata = load_disease_model(device)
    architecture = metadata.get("architecture", metadata.get("model_name", "efficientnet_b0"))
    feature_model = _feature_model_from_classifier(model, architecture).to(device)

    samples: list[tuple[Path, str]] = []
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, class_dir.name))
    if max_images is not None:
        samples = samples[:max_images]
    if not samples:
        raise ValueError(f"No disease reference images found in {data_dir}")

    embeddings = np.vstack([extract_embedding(path, feature_model, device) for path, _ in samples])
    metadata_rows = [format_disease_retrieval_metadata(str(path), label, 1.0) for path, label in samples]
    np.savez_compressed(output_dir / "embeddings.npz", embeddings=embeddings)
    pd.DataFrame(metadata_rows).to_csv(output_dir / "metadata.csv", index=False)

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
    summary = {"num_images": len(samples), "embedding_dim": int(embeddings.shape[1]), "backend": artifact["backend"], "output_dir": str(output_dir)}
    print(summary)
    return summary


def find_similar_disease_examples(
    image_path: str | Path,
    top_k: int = 5,
    index_dir: Path = DISEASE_REFERENCE_INDEX_DIR,
) -> list[dict[str, Any]]:
    required = [index_dir / "index.joblib", index_dir / "metadata.csv", index_dir / "embeddings.npz"]
    if not all(path.exists() for path in required):
        return []
    device = get_device(force_cpu=True)
    model, _, metadata = load_disease_model(device)
    architecture = metadata.get("architecture", metadata.get("model_name", "efficientnet_b0"))
    feature_model = _feature_model_from_classifier(model, architecture).to(device)
    query = extract_embedding(image_path, feature_model, device).reshape(1, -1)
    artifact = joblib.load(index_dir / "index.joblib")
    rows = pd.read_csv(index_dir / "metadata.csv")
    embeddings = np.load(index_dir / "embeddings.npz")["embeddings"]

    if artifact["backend"] == "faiss":
        try:
            import faiss  # type: ignore

            index = faiss.deserialize_index(artifact["index"])
            scores, indices = index.search(query.astype(np.float32), min(top_k, len(rows)))
            pairs = zip(indices[0], scores[0])
        except ImportError:
            pairs = []
    else:
        distances, indices = artifact["index"].kneighbors(query, n_neighbors=min(top_k, len(embeddings)))
        pairs = zip(indices[0], 1.0 - distances[0])
    results = []
    for idx, score in pairs:
        row = rows.iloc[int(idx)].to_dict()
        row["similarity_score"] = float(score)
        results.append(row)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/query the disease reference image retrieval index.")
    parser.add_argument("--build_index", action="store_true")
    parser.add_argument("--data_dir", type=Path, default=REFERENCE_TRAIN_DIR)
    parser.add_argument("--output_dir", type=Path, default=DISEASE_REFERENCE_INDEX_DIR)
    parser.add_argument("--image_path", type=Path, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.build_index:
        build_disease_reference_index(args.data_dir, args.output_dir, args.max_images)
    elif args.image_path:
        print(find_similar_disease_examples(args.image_path, args.top_k, args.output_dir))
    else:
        raise SystemExit("Use --build_index or --image_path.")
