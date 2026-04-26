"""Lightweight repository health check for CropVision."""

from __future__ import annotations

import importlib
import py_compile
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
    "src.external_validate",
    "src.export_onnx",
    "src.benchmark_inference",
    "src.plant_id",
    "src.local_species_model",
    "src.problem_taxonomy",
    "src.image_retrieval",
]

KEY_SOURCE_FILES = [
    ROOT / "app.py",
    ROOT / "app" / "streamlit_app.py",
    ROOT / "src" / "external_validate.py",
    ROOT / "src" / "export_onnx.py",
    ROOT / "src" / "benchmark_inference.py",
    ROOT / "src" / "plant_id.py",
    ROOT / "src" / "problem_taxonomy.py",
    ROOT / "src" / "image_retrieval.py",
    ROOT / "scripts" / "run_advanced_demo_check.py",
    ROOT / "Makefile",
    ROOT / "README.md",
    ROOT / ".env.example",
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

    for source_file in KEY_SOURCE_FILES:
        if source_file.exists():
            print(f"OK source file: {source_file.relative_to(ROOT)}")
        else:
            print(f"MISSING source file: {source_file.relative_to(ROOT)}")
            ok = False

    try:
        py_compile.compile(str(ROOT / "app" / "streamlit_app.py"), doraise=True)
        py_compile.compile(str(ROOT / "app.py"), doraise=True)
        print("OK Streamlit app syntax/import entrypoints")
    except Exception as exc:
        print(f"FAILED Streamlit app syntax/import entrypoints -> {exc}")
        ok = False

    for artifact in (DEFAULT_CV_MODEL_PATH, DEFAULT_CLASS_NAMES_PATH, DEFAULT_WEATHER_MODEL_PATH):
        status = "OK" if artifact.exists() else "OPTIONAL/MISSING"
        print(f"{status} artifact: {artifact.relative_to(ROOT)}")

    onnx_model = MODEL_DIR / "cropvision_cv.onnx"
    status = "OK" if onnx_model.exists() else "OPTIONAL/MISSING"
    print(f"{status} artifact: {onnx_model.relative_to(ROOT)}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
