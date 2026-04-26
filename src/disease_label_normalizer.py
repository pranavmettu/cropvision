"""Normalize plant disease labels into structured disease-ID fields."""

from __future__ import annotations

from src.label_normalizer import normalize_label


def normalize_disease_label(raw_label: str) -> dict:
    info = normalize_label(raw_label)
    disease = info["disease_name"]
    category = info["broad_problem_category"]
    if any(token in disease for token in ("rot", "decay", "black_rot")):
        category = "rot_or_decay"
    if "black_rot" in disease:
        category = "rot_or_decay"
    return {
        "raw_label": raw_label,
        "plant_species": info["plant_species"],
        "disease_name": disease,
        "health_status": info["health_status"],
        "normalized_class": info["normalized_class"],
        "broad_problem_category": category,
    }
