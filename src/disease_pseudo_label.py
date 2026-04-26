"""Suggest pseudo-labels for review without moving or training on images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.config import IMAGE_EXTENSIONS, REPORTS_DIR
from src.disease_model import predict_disease


def should_suggest_pseudo_label(confidence: float | None, threshold: float) -> bool:
    return confidence is not None and confidence >= threshold


def generate_pseudo_label_suggestions(input_dir: Path, threshold: float, output_csv: Path) -> list[dict]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    rows: list[dict] = []
    for image_path in sorted(input_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        result = predict_disease(str(image_path), confidence_threshold=threshold)
        confidence = result.get("confidence")
        if result.get("available") and should_suggest_pseudo_label(confidence, threshold):
            rows.append(
                {
                    "image_path": str(image_path),
                    "suggested_label": result.get("raw_predicted_disease_class") or result.get("predicted_disease_class"),
                    "confidence": confidence,
                    "broad_problem_category": result.get("broad_problem_category"),
                    "review_required": True,
                }
            )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "suggested_label", "confidence", "broad_problem_category", "review_required"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} pseudo-label suggestions to {output_csv}. Review them before training.")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suggest disease pseudo-labels for human review.")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--output_csv", type=Path, default=REPORTS_DIR / "disease_pseudo_label_suggestions.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_pseudo_label_suggestions(args.input_dir, args.threshold, args.output_csv)
