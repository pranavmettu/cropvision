"""Optional plant/species identification for CropVision."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from src.config import PLANT_ID_EXAMPLES_DIR
from src.local_species_model import identify_species_local

PLANTNET_ENDPOINT = "https://my-api.plantnet.org/v2/identify/all"


def _parse_plantnet_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    for item in payload.get("results", [])[:5]:
        species = item.get("species", {})
        common_names = species.get("commonNames") or []
        suggestions.append(
            {
                "scientific_name": species.get("scientificNameWithoutAuthor") or species.get("scientificName"),
                "common_name": common_names[0] if common_names else None,
                "confidence": float(item.get("score", 0.0)),
                "family": (species.get("family") or {}).get("scientificNameWithoutAuthor"),
                "genus": (species.get("genus") or {}).get("scientificNameWithoutAuthor"),
            }
        )
    return suggestions


def identify_plant_plantnet(
    image_path: str,
    organs: list[str] | None = None,
    save_raw_response: bool = False,
    timeout: int = 30,
) -> dict[str, Any]:
    """Identify plant species with Pl@ntNet when PLANTNET_API_KEY is configured."""
    api_key = os.getenv("PLANTNET_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        return {
            "available": False,
            "provider": "plantnet",
            "message": "Pl@ntNet API key not found. Plant ID unavailable unless local model is enabled.",
            "top_suggestions": [],
        }

    path = Path(image_path)
    if not path.exists():
        return {
            "available": False,
            "provider": "plantnet",
            "message": f"Image path not found: {path}",
            "top_suggestions": [],
        }

    organs = organs or ["leaf"]
    try:
        with path.open("rb") as image_file:
            files = [("images", (path.name, image_file, "image/jpeg"))]
            data = [("organs", organ) for organ in organs]
            response = requests.post(
                PLANTNET_ENDPOINT,
                params={"api-key": api_key},
                files=files,
                data=data,
                timeout=timeout,
            )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return {
            "available": False,
            "provider": "plantnet",
            "message": f"Pl@ntNet request failed: {exc}",
            "top_suggestions": [],
        }
    except ValueError as exc:
        return {
            "available": False,
            "provider": "plantnet",
            "message": f"Could not parse Pl@ntNet response: {exc}",
            "top_suggestions": [],
        }

    if save_raw_response:
        PLANT_ID_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PLANT_ID_EXAMPLES_DIR / f"{path.stem}_plantnet_response.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    suggestions = _parse_plantnet_results(payload)
    top = suggestions[0] if suggestions else {}
    return {
        "available": bool(suggestions),
        "provider": "plantnet",
        "scientific_name": top.get("scientific_name"),
        "common_name": top.get("common_name"),
        "confidence": top.get("confidence"),
        "top_suggestions": suggestions,
        "message": "Plant ID completed with Pl@ntNet." if suggestions else "Pl@ntNet returned no species suggestions.",
    }


def identify_plant_local(image_path: str) -> dict[str, Any]:
    return identify_species_local(image_path)
