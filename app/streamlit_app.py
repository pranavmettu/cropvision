"""Streamlit app for interactive CropVision predictions."""

from __future__ import annotations

import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CV_MODEL_PATH, DEFAULT_WEATHER_MODEL_PATH, SAMPLE_IMAGES_DIR  # noqa: E402
from src.gradcam import gradcam_predict  # noqa: E402
from src.multimodal_predict import build_combined_summary, predict_weather_risk  # noqa: E402
from src.weather_features import fetch_weather_features  # noqa: E402


st.set_page_config(page_title="CropVision", layout="wide")
st.title("CropVision")
st.warning("This tool is for educational ML demonstration only, not professional crop diagnosis.")

st.markdown(
    """
    <style>
    .interpretation-card {
        border: 1px solid #d5dadd;
        border-radius: 8px;
        padding: 1rem;
        background: #f8faf9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model status")
    if DEFAULT_CV_MODEL_PATH.exists():
        st.success("Image model found")
    else:
        st.error("Image model missing")
        st.caption("Train it with `python -m src.train_cv --data_dir data/raw/plantvillage --epochs 3`.")
    if DEFAULT_WEATHER_MODEL_PATH.exists():
        st.success("Weather model found")
    else:
        st.info("Weather model optional")

    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.50, 0.05)

    st.header("Optional weather risk")
    include_weather = st.checkbox("Include NASA POWER weather features", value=False)
    latitude = st.number_input("Latitude", value=42.45, format="%.6f")
    longitude = st.number_input("Longitude", value=-76.48, format="%.6f")
    default_end = date.today()
    default_start = default_end - timedelta(days=7)
    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=default_end)

    st.header("About this project")
    st.caption(
        "CropVision combines transfer learning, Grad-CAM, confidence-aware predictions, "
        "and optional NASA POWER weather features for an agtech ML portfolio demo."
    )

uploaded_file = st.file_uploader("Upload a crop or leaf image", type=["jpg", "jpeg", "png"])

sample_options = []
if SAMPLE_IMAGES_DIR.exists():
    sample_options = sorted(
        path for path in SAMPLE_IMAGES_DIR.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
if sample_options:
    sample_name = st.selectbox("Or choose a sample image", ["None"] + [path.name for path in sample_options])
else:
    sample_name = "None"

if not DEFAULT_CV_MODEL_PATH.exists():
    st.info(
        "No trained image model found. Train it first with: "
        "`python -m src.train_cv --data_dir data/raw/plantvillage --epochs 3`"
    )

if uploaded_file is None and sample_name == "None":
    st.caption("Upload an image to run diagnosis.")
    st.stop()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    source_name = uploaded_file.name
else:
    selected_path = next(path for path in sample_options if path.name == sample_name)
    image = Image.open(selected_path).convert("RGB")
    source_name = selected_path.name

if not DEFAULT_CV_MODEL_PATH.exists():
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=Path(source_name).suffix or ".jpg") as tmp:
    image.save(tmp.name)
    tmp_path = Path(tmp.name)

try:
    result = gradcam_predict(
        tmp_path,
        checkpoint_path=DEFAULT_CV_MODEL_PATH,
        confidence_threshold=confidence_threshold,
    )
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

left, right = st.columns([1, 1])
with left:
    st.subheader("Input image")
    st.image(image, caption=source_name, use_container_width=True)

with right:
    st.subheader("Grad-CAM explanation")
    st.image(result["overlay_image"], caption="Highlighted regions most associated with the prediction.", use_container_width=True)

summary_left, summary_right = st.columns([1, 1])
with summary_left:
    st.subheader("Top prediction")
    display_label = result["predicted_class"] if result["is_uncertain"] else result["raw_predicted_class"]
    st.metric(display_label, f"{result['confidence']:.1%}")
    if result["is_uncertain"]:
        st.warning(result["uncertainty_reason"])
        st.caption(
            "Low confidence may mean the image is out-of-distribution, low quality, or not represented in the training data."
        )

with summary_right:
    st.subheader("Top 3 predictions")
    for item in result["top_predictions"]:
        st.write(f"{item['class_name']}: {item['confidence']:.1%}")

weather_result = None
if include_weather:
    if start_date > end_date:
        st.error("Start date must be before or equal to end date.")
    else:
        with st.spinner("Fetching NASA POWER weather data..."):
            try:
                features = fetch_weather_features(latitude, longitude, start_date, end_date)
                weather_result = predict_weather_risk(features, DEFAULT_WEATHER_MODEL_PATH)
            except Exception as exc:
                st.error(f"Weather risk failed: {exc}")

if weather_result:
    st.subheader("Weather risk")
    risk_level = weather_result["weather_risk_level"]
    confidence = weather_result["weather_risk_confidence"]
    st.write(f"Risk level: **{risk_level}**")
    if confidence is not None:
        st.write(f"Risk model confidence: **{confidence:.1%}**")
    st.write(weather_result["message"])
    st.json(weather_result["weather_features"])

weather_level = weather_result["weather_risk_level"] if weather_result else None
summary_class = result["raw_predicted_class"] if result["is_uncertain"] else result["predicted_class"]
summary = build_combined_summary(summary_class, result["confidence"], weather_level)
st.subheader("Combined interpretation")
st.markdown(f"<div class='interpretation-card'>{summary}</div>", unsafe_allow_html=True)
