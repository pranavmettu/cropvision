"""Check optional advanced CropVision demo components."""

from __future__ import annotations

import importlib
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_CLASS_NAMES_PATH, DEFAULT_CV_MODEL_PATH, DEFAULT_RETRIEVAL_ARTIFACT_PATH
from src.problem_taxonomy import map_disease_class_to_problem_category


def check_path(path: Path, label: str, required: bool = False) -> bool:
    if path.exists():
        print(f"OK {label}: {path.relative_to(ROOT)}")
        return True
    status = "MISSING" if required else "OPTIONAL/MISSING"
    print(f"{status} {label}: {path.relative_to(ROOT)}")
    return not required


def main() -> int:
    ok = True
    ok &= check_path(DEFAULT_CV_MODEL_PATH, "disease model checkpoint")
    ok &= check_path(DEFAULT_CLASS_NAMES_PATH, "class_names.json")
    ok &= check_path(DEFAULT_RETRIEVAL_ARTIFACT_PATH, "retrieval index")
    ok &= check_path(ROOT / ".env.example", ".env.example", required=True)

    for module_name in ("src.plant_id", "src.local_species_model", "src.problem_taxonomy", "src.image_retrieval"):
        try:
            importlib.import_module(module_name)
            print(f"OK import: {module_name}")
        except Exception as exc:
            print(f"FAILED import: {module_name} -> {exc}")
            ok = False

    mapped = map_disease_class_to_problem_category("Tomato___Early_blight")
    if mapped == "blight_like_symptoms":
        print("OK taxonomy mapping: Tomato___Early_blight -> blight_like_symptoms")
    else:
        print(f"FAILED taxonomy mapping: got {mapped}")
        ok = False

    try:
        py_compile.compile(str(ROOT / "app" / "streamlit_app.py"), doraise=True)
        print("OK Streamlit app syntax")
    except Exception as exc:
        print(f"FAILED Streamlit app syntax -> {exc}")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
