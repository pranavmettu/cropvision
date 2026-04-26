"""Normalize plant disease dataset labels into a shared schema."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd

from src.config import IMAGE_EXTENSIONS, MODEL_DIR, REPORTS_DIR
from src.problem_taxonomy import map_disease_class_to_problem_category


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("(", "_").replace(")", "_").replace(",", "_")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def normalize_label(raw_label: str) -> dict:
    original = raw_label.strip()
    if "___" in original:
        plant, disease = original.split("___", 1)
    elif "__" in original:
        plant, disease = original.split("__", 1)
    else:
        parts = re.split(r"[_ -]+", original, maxsplit=1)
        plant = parts[0] if parts else "unknown"
        disease = parts[1] if len(parts) > 1 else "unknown"

    plant_species = slugify(plant) or "unknown"
    disease_name = slugify(disease) or "unknown"
    health_status = "healthy" if disease_name == "healthy" or "healthy" in disease_name else "diseased"
    normalized_class = f"{plant_species}__{disease_name}"
    broad_problem_category = map_disease_class_to_problem_category(original)
    if plant_species == "unknown" or disease_name == "unknown":
        broad_problem_category = "unknown_or_uncertain"

    return {
        "original_label": original,
        "plant_species": plant_species,
        "disease_name": disease_name,
        "health_status": health_status,
        "normalized_class": normalized_class,
        "broad_problem_category": broad_problem_category,
    }


def normalize_dataset_labels(input_dir: str | Path, output_dir: str | Path) -> list[dict]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    label_map: dict[str, dict] = {}
    for class_dir in sorted(path for path in input_path.iterdir() if path.is_dir()):
        normalized = normalize_label(class_dir.name)
        label_map[class_dir.name] = normalized
        target_dir = output_path / normalized["normalized_class"]
        target_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                shutil.copy2(image_path, target_dir / f"{class_dir.name}_{count:06d}{image_path.suffix.lower()}")
                count += 1
        rows.append({**normalized, "image_count": count})

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "label_normalization_report.csv", index=False)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize ImageFolder class labels.")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = normalize_dataset_labels(args.input_dir, args.output_dir)
    print(f"Normalized {len(rows)} classes into {args.output_dir}")
