"""Store human-verified user feedback separately from training data."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from src.config import USER_FEEDBACK_DIR
from src.label_normalizer import normalize_label


def image_characteristics(image_path: str | Path) -> dict[str, Any]:
    with Image.open(image_path) as image:
        return {"width": image.width, "height": image.height, "mode": image.mode}


def save_verified_feedback(
    image_path: str | Path,
    correct_label: str,
    original_prediction: str | None = None,
    reference_matches: list[dict] | None = None,
    model_version: str | None = None,
    feedback_root: Path = USER_FEEDBACK_DIR,
) -> dict[str, Any]:
    normalized = normalize_label(correct_label)
    target_dir = feedback_root / normalized["normalized_class"]
    target_dir.mkdir(parents=True, exist_ok=True)
    source = Path(image_path)
    feedback_id = uuid.uuid4().hex
    target_image = target_dir / f"{feedback_id}{source.suffix.lower() or '.jpg'}"
    shutil.copy2(source, target_image)
    metadata = {
        "feedback_id": feedback_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "image_path": str(target_image),
        "original_prediction": original_prediction,
        "correct_label": correct_label,
        "normalized_label": normalized,
        "image_characteristics": image_characteristics(target_image),
        "reference_retrieval_matches": reference_matches or [],
        "model_version": model_version or "latest",
        "warning": "Human-confirmed data only. Do not add blind model predictions as training labels.",
    }
    (target_dir / f"{feedback_id}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
