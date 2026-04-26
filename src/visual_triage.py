"""Lightweight visual triage fallback when no trained disease model exists.

This is not a replacement for the PyTorch disease classifier. It provides broad,
educational observations from simple image statistics so the app can still give
useful feedback before a user trains a local model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _rgb_array(image_path: str | Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((384, 384))
    return np.asarray(image).astype(np.float32) / 255.0


def _edge_density(gray: np.ndarray) -> float:
    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx**2 + gy**2)
    return float((magnitude > 0.08).mean())


def analyze_leaf_visual_triage(image_path: str | Path) -> dict[str, Any]:
    """Return broad symptom observations using simple color/texture heuristics."""
    arr = _rgb_array(image_path)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    brightness = arr.mean(axis=2)
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    green_mask = (g > r * 1.05) & (g > b * 1.05) & (g > 0.18)
    yellow_mask = (r > 0.42) & (g > 0.38) & (b < 0.32) & (np.abs(r - g) < 0.22)
    brown_mask = (r > 0.25) & (g > 0.15) & (b < 0.22) & (r > g * 1.05)
    pale_mask = green_mask & (brightness > 0.48) & ((g - np.maximum(r, b)) < 0.18)
    dark_spot_mask = green_mask & (brightness < 0.22)

    green_ratio = float(green_mask.mean())
    yellow_ratio = float(yellow_mask.mean())
    brown_ratio = float(brown_mask.mean())
    pale_ratio = float(pale_mask.mean())
    dark_spot_ratio = float(dark_spot_mask.mean())
    edge_ratio = _edge_density(gray)

    observations: list[str] = []
    category = "unknown_or_uncertain"
    confidence = 0.35

    if brown_ratio > 0.08 or dark_spot_ratio > 0.05:
        category = "fungal_leaf_spot"
        confidence = min(0.72, 0.45 + brown_ratio + dark_spot_ratio)
        observations.append("Brown/dark regions may indicate necrosis, spotting, old damage, or dead tissue.")
    if yellow_ratio > 0.10 or pale_ratio > 0.16:
        category = "nutrient_deficiency_like_yellowing" if category == "unknown_or_uncertain" else category
        confidence = max(confidence, min(0.70, 0.42 + yellow_ratio + pale_ratio))
        observations.append("Yellowing or pale green tissue is visible, which can be associated with nutrient stress, water stress, or disease.")
    if green_ratio > 0.20 and brown_ratio < 0.08 and yellow_ratio < 0.12 and edge_ratio > 0.22:
        category = "abiotic_stress"
        confidence = max(confidence, 0.55)
        observations.append("The leaf appears green but textured, curled, or puckered; this can fit abiotic stress, edema, herbicide drift, viral-like curl, or growth stress.")
    if green_ratio > 0.25 and brown_ratio < 0.03 and yellow_ratio < 0.05 and edge_ratio < 0.18:
        category = "healthy"
        confidence = max(confidence, 0.50)
        observations.append("The visible leaf area is mostly green without strong yellowing or necrotic spotting.")

    if not observations:
        observations.append("The image does not match a strong rule-based pattern. A trained disease model is needed for better classification.")

    summary = (
        f"No trained disease model is installed, so CropVision is using broad visual triage only. "
        f"The image most closely maps to {category.replace('_', ' ')} with low-to-moderate heuristic confidence. "
        "This is not a professional diagnosis or treatment recommendation."
    )
    return {
        "mode": "rule_based_visual_triage",
        "problem_category": category,
        "confidence": float(confidence),
        "observations": observations,
        "metrics": {
            "green_ratio": green_ratio,
            "yellow_ratio": yellow_ratio,
            "brown_ratio": brown_ratio,
            "pale_ratio": pale_ratio,
            "dark_spot_ratio": dark_spot_ratio,
            "edge_density": edge_ratio,
        },
        "final_summary": summary,
        "educational_disclaimer": "Educational ML demo only. Not professional crop diagnosis or treatment advice.",
    }
