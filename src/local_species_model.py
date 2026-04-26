"""Optional local species recognition adapter.

This module is intentionally conservative. Large species models can be heavy, so
CropVision only attempts local species recognition when a model name is
explicitly configured and optional dependencies/checkpoints are available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import LOCAL_SPECIES_MODEL_NAME


def identify_species_local(image_path: str | Path, model_name: str | None = LOCAL_SPECIES_MODEL_NAME) -> dict[str, Any]:
    if model_name is None:
        return {
            "available": False,
            "provider": "local_species_model",
            "message": "Local species model not configured.",
            "top_suggestions": [],
        }
    try:
        import timm  # type: ignore
    except ImportError:
        return {
            "available": False,
            "provider": "local_species_model",
            "message": "Optional dependency 'timm' is not installed. Install it only if you want local species recognition.",
            "top_suggestions": [],
        }

    return {
        "available": False,
        "provider": "local_species_model",
        "message": (
            f"Model name '{model_name}' is configured and timm {timm.__version__} is installed, "
            "but no local species checkpoint/class mapping has been provided."
        ),
        "top_suggestions": [],
        "image_path": str(image_path),
    }
