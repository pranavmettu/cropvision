"""Shared configuration for CropVision."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "sample_images"

DEFAULT_DATA_DIR = RAW_DATA_DIR / "plantvillage"
DEFAULT_CV_MODEL_PATH = MODEL_DIR / "cropvision_cv.pt"
DEFAULT_CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
DEFAULT_WEATHER_MODEL_PATH = MODEL_DIR / "weather_risk_model.joblib"
DEFAULT_TRAIN_HISTORY_PATH = REPORTS_DIR / "train_history.csv"
DEFAULT_EVAL_METRICS_PATH = REPORTS_DIR / "eval_metrics.json"
DEFAULT_CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"
RETRIEVAL_DIR = MODEL_DIR / "retrieval"
DEFAULT_RETRIEVAL_ARTIFACT_PATH = RETRIEVAL_DIR / "retrieval_artifacts.joblib"
PLANT_ID_EXAMPLES_DIR = REPORTS_DIR / "plant_id_examples"
LOCAL_SPECIES_MODEL_NAME = None

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
WEATHER_FEATURE_COLUMNS = [
    "rainfall_7d",
    "humidity_avg_7d",
    "temp_avg",
    "temp_max",
    "heat_stress_days",
    "wet_days",
]


def ensure_project_dirs() -> None:
    """Create local artifact folders expected by scripts."""
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, REPORTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
