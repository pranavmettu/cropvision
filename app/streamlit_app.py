"""Streamlit app for interactive CropVision predictions."""

from __future__ import annotations

import sys
import tempfile
import json
import os
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from src.config import DEFAULT_CLASS_NAMES_PATH, DEFAULT_CV_MODEL_PATH, DEFAULT_RETRIEVAL_ARTIFACT_PATH, DEFAULT_WEATHER_MODEL_PATH, REFERENCE_INDEX_DIR, SAMPLE_IMAGES_DIR  # noqa: E402
from src.feedback_store import save_verified_feedback  # noqa: E402
from src.gradcam import gradcam_predict  # noqa: E402
from src.multimodal_predict import build_advanced_summary, predict_weather_risk  # noqa: E402
from src.plant_id import identify_plant_local, identify_plant_plantnet  # noqa: E402
from src.problem_taxonomy import map_disease_class_to_problem_category  # noqa: E402
from src.reference_retrieval import find_reference_matches  # noqa: E402
from src.visual_triage import analyze_leaf_visual_triage  # noqa: E402
from src.weather_features import fetch_weather_features  # noqa: E402


def _class_names_count() -> int | None:
    if not DEFAULT_CLASS_NAMES_PATH.exists():
        return None
    try:
        return len(json.loads(DEFAULT_CLASS_NAMES_PATH.read_text(encoding="utf-8")))
    except Exception:
        return None


def _latest_model_version() -> str:
    versions_dir = Path("models/versions")
    if versions_dir.exists():
        versions = sorted([path.name for path in versions_dir.iterdir() if path.is_dir()])
        if versions:
            return versions[-1]
    return "latest"


st.set_page_config(page_title="CropVision", layout="wide")
st.title("CropVision")
st.warning("Educational ML demo only. Not professional crop diagnosis or treatment advice.")

st.markdown(
    """
    <style>
    .interpretation-card {
        border: 1px solid #d5dadd;
        border-radius: 8px;
        padding: 1rem;
        background: #f8faf9;
        color: #17201a;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Mode")
    mode = st.radio("Recognition mode", ["Local disease model only", "Advanced plant ID + disease model"])
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.50, 0.05)

    st.header("Optional modules")
    plantnet_key = os.getenv("PLANTNET_API_KEY")
    plantnet_available = bool(plantnet_key and plantnet_key != "your_api_key_here")
    enable_plantnet = st.checkbox(
        "Enable Pl@ntNet API",
        value=False,
        disabled=mode == "Local disease model only" or not plantnet_available,
    )
    if mode == "Advanced plant ID + disease model" and not plantnet_available:
        st.caption("Pl@ntNet is optional. Add a real PLANTNET_API_KEY in .env to enable species ID.")
    enable_local_species = st.checkbox("Enable local species fallback", value=False, disabled=mode == "Local disease model only")
    enable_weather = st.checkbox("Enable weather risk", value=False)
    enable_retrieval = st.checkbox("Enable visual similarity", value=False)

    st.header("Model status")
    if DEFAULT_CV_MODEL_PATH.exists():
        st.success("Disease model found")
        class_count = _class_names_count()
        st.caption(f"Model version: {_latest_model_version()}")
        if class_count is not None:
            st.caption(f"Trained disease classes: {class_count}")
    else:
        st.error("Disease model missing")
    if DEFAULT_WEATHER_MODEL_PATH.exists():
        st.success("Weather model found")
    else:
        st.info("Weather model optional")
    if DEFAULT_RETRIEVAL_ARTIFACT_PATH.exists():
        st.success("Retrieval index found")
    elif (REFERENCE_INDEX_DIR / "index.joblib").exists():
        st.success("Reference retrieval index found")
    else:
        st.info("Retrieval index optional")

    if enable_weather:
        st.header("Weather inputs")
        latitude = st.number_input("Latitude", value=42.45, format="%.6f")
        longitude = st.number_input("Longitude", value=-76.48, format="%.6f")
        default_end = date.today()
        default_start = default_end - timedelta(days=7)
        start_date = st.date_input("Start date", value=default_start)
        end_date = st.date_input("End date", value=default_end)
    else:
        latitude = longitude = start_date = end_date = None

    st.header("About")
    st.caption(
        "Advanced mode separates plant/species identification from visible health-problem recognition, "
        "then adds uncertainty, weather context, and similar-image retrieval when available."
    )

uploaded_file = st.file_uploader("Upload a plant/crop image", type=["jpg", "jpeg", "png"])
sample_options = []
if SAMPLE_IMAGES_DIR.exists():
    sample_options = sorted(path for path in SAMPLE_IMAGES_DIR.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
sample_name = st.selectbox("Or choose a sample image", ["None"] + [path.name for path in sample_options]) if sample_options else "None"

if uploaded_file is None and sample_name == "None":
    st.caption("Upload an image to run plant health recognition.")
    st.stop()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    source_name = uploaded_file.name
else:
    selected_path = next(path for path in sample_options if path.name == sample_name)
    image = Image.open(selected_path).convert("RGB")
    source_name = selected_path.name

with tempfile.NamedTemporaryFile(delete=False, suffix=Path(source_name).suffix or ".jpg") as tmp:
    image.save(tmp.name)
    tmp_path = Path(tmp.name)

left, right = st.columns([1, 1])
with left:
    st.subheader("Input image")
    st.image(image, caption=source_name, use_container_width=True)

plant_id_result = None
if mode == "Advanced plant ID + disease model":
    st.subheader("Plant ID")
    if enable_plantnet:
        with st.spinner("Checking Pl@ntNet..."):
            plant_id_result = identify_plant_plantnet(str(tmp_path))
    elif enable_local_species:
        plant_id_result = identify_plant_local(str(tmp_path))
    else:
        plant_id_result = {"available": False, "message": "Plant ID disabled.", "top_suggestions": []}

    if plant_id_result.get("available"):
        st.write(f"Scientific name: **{plant_id_result.get('scientific_name') or 'Unknown'}**")
        st.write(f"Common name: **{plant_id_result.get('common_name') or 'Not provided'}**")
        confidence = plant_id_result.get("confidence")
        if confidence is not None:
            st.write(f"Plant ID confidence: **{confidence:.1%}**")
        with st.expander("Top plant suggestions"):
            st.json(plant_id_result.get("top_suggestions", []))
    else:
        st.info(plant_id_result.get("message", "Plant ID unavailable."))

if not DEFAULT_CV_MODEL_PATH.exists():
    st.subheader("Broad visual triage")
    triage = analyze_leaf_visual_triage(tmp_path)
    st.info(
        "No trained disease model found, so this is a rule-based visual triage fallback. "
        "Train the PyTorch disease model for real model predictions."
    )
    st.metric(triage["problem_category"], f"{triage['confidence']:.0%}")
    st.write("Observed signals:")
    for observation in triage["observations"]:
        st.write(f"- {observation}")
    with st.expander("Heuristic image metrics"):
        st.json(triage["metrics"])
    st.subheader("Final conservative interpretation")
    st.markdown(f"<div class='interpretation-card'>{triage['final_summary']}</div>", unsafe_allow_html=True)
    st.stop()

try:
    disease_result = gradcam_predict(
        tmp_path,
        checkpoint_path=DEFAULT_CV_MODEL_PATH,
        confidence_threshold=confidence_threshold,
    )
except Exception as exc:
    st.error(f"Disease/problem prediction failed: {exc}")
    st.stop()

with right:
    st.subheader("Grad-CAM")
    st.image(disease_result["overlay_image"], caption="Regions most associated with the disease/problem prediction.", use_container_width=True)

raw_disease = disease_result["raw_predicted_class"]
problem_category = "unknown_or_uncertain" if disease_result["is_uncertain"] else map_disease_class_to_problem_category(raw_disease)
normalized_prediction = __import__("src.label_normalizer", fromlist=["normalize_label"]).normalize_label(raw_disease)

pred_col, top_col = st.columns([1, 1])
with pred_col:
    st.subheader("Disease/problem prediction")
    display_label = "uncertain" if disease_result["is_uncertain"] else raw_disease
    st.metric(display_label, f"{disease_result['confidence']:.1%}")
    st.write(f"Plant species: **{normalized_prediction['plant_species']}**")
    st.write(f"Disease name: **{normalized_prediction['disease_name']}**")
    st.write(f"Broad issue category: **{problem_category}**")
    if disease_result["is_uncertain"]:
        st.warning(disease_result["uncertainty_reason"])

with top_col:
    st.subheader("Top 3 disease predictions")
    for item in disease_result["top_predictions"]:
        st.write(f"{item['class_name']}: {item['confidence']:.1%}")

similar_examples = []
if enable_retrieval:
    st.subheader("Reference examples from training database")
    if (REFERENCE_INDEX_DIR / "index.joblib").exists():
        try:
            similar_examples = find_reference_matches(str(tmp_path), top_k=3)
            sim_cols = st.columns(max(1, len(similar_examples)))
            for col, item in zip(sim_cols, similar_examples):
                with col:
                    caption = f"{item['class_label']} ({item['similarity_score']:.2f})"
                    st.image(item["image_path"], caption=caption, use_container_width=True)
        except Exception as exc:
            st.info(f"Reference retrieval unavailable: {exc}")
    else:
        st.info("Reference index missing. Build it with `python -m src.reference_retrieval --build_index --data_dir data/processed/cropvision_reference_train --output_dir models/reference_index`.")

weather_result = None
if enable_weather:
    st.subheader("Weather risk")
    if start_date and end_date and start_date > end_date:
        st.error("Start date must be before or equal to end date.")
    else:
        with st.spinner("Fetching NASA POWER weather data..."):
            try:
                features = fetch_weather_features(float(latitude), float(longitude), start_date, end_date)
                weather_result = predict_weather_risk(features, DEFAULT_WEATHER_MODEL_PATH)
                st.write(f"Risk level: **{weather_result['weather_risk_level']}**")
                if weather_result["weather_risk_confidence"] is not None:
                    st.write(f"Risk model confidence: **{weather_result['weather_risk_confidence']:.1%}**")
                st.json(weather_result["weather_features"])
            except Exception as exc:
                st.error(f"Weather risk failed: {exc}")

weather_level = weather_result["weather_risk_level"] if weather_result else None
summary = build_advanced_summary(
    plant_id_result,
    raw_disease,
    problem_category,
    disease_result["confidence"],
    disease_result["is_uncertain"],
    weather_level,
)
st.subheader("Final conservative interpretation")
st.markdown(f"<div class='interpretation-card'>{summary}</div>", unsafe_allow_html=True)
st.warning("The model can only recognize diseases represented in its training database. Unknown plants or diseases may produce uncertain predictions.")

st.subheader("Verified feedback")
st.caption("Optional: save this uploaded image only if you can confirm the correct label. The app will not retrain automatically.")
confirm_prediction = st.checkbox("I confirm/correct this label for future explicit retraining")
correct_label = st.text_input("Correct label", value=raw_disease)
if st.button("Save verified feedback", disabled=not confirm_prediction):
    try:
        metadata = save_verified_feedback(
            tmp_path,
            correct_label,
            original_prediction=raw_disease,
            reference_matches=similar_examples,
            model_version=_latest_model_version(),
        )
        st.success(f"Saved verified feedback: {metadata['normalized_label']['normalized_class']}")
    except Exception as exc:
        st.error(f"Could not save feedback: {exc}")
