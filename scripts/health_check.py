"""Lightweight repository health check for CropVision."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_CLASS_NAMES_PATH, DEFAULT_CV_MODEL_PATH, DEFAULT_WEATHER_MODEL_PATH, FIGURES_DIR, MODEL_DIR, REPORTS_DIR


REQUIRED_FOLDERS = [
    ROOT / "data" / "raw",
    ROOT / "data" / "processed",
    MODEL_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    ROOT / "src",
    ROOT / "app",
]

IMPORTS = [
    "src.config",
    "src.dataset",
    "src.train_cv",
    "src.evaluate_cv",
    "src.predict_cv",
    "src.gradcam",
    "src.calibration",
    "src.weather_features",
    "src.train_weather_model",
    "src.multimodal_predict",
]


def main() -> int:
    ok = True
    for folder in REQUIRED_FOLDERS:
        if folder.exists():
            print(f"OK folder: {folder.relative_to(ROOT)}")
        else:
            print(f"MISSING folder: {folder.relative_to(ROOT)}")
            ok = False

    for module_name in IMPORTS:
        try:
            importlib.import_module(module_name)
            print(f"OK import: {module_name}")
        except Exception as exc:
            print(f"FAILED import: {module_name} -> {exc}")
            ok = False

    for artifact in (DEFAULT_CV_MODEL_PATH, DEFAULT_CLASS_NAMES_PATH, DEFAULT_WEATHER_MODEL_PATH):
        status = "OK" if artifact.exists() else "OPTIONAL/MISSING"
        print(f"{status} artifact: {artifact.relative_to(ROOT)}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
