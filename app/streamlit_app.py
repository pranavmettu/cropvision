"""Streamlit app for interactive CropVision plant and disease recognition."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(PROJECT_ROOT / ".env", override=True)
except ImportError:
    pass

from src.config import DEFAULT_CV_MODEL_PATH, DEFAULT_WEATHER_MODEL_PATH, SAMPLE_IMAGES_DIR  # noqa: E402
from src.disease_model import DISEASE_MODEL_PATH, disease_model_status, predict_disease  # noqa: E402
from src.disease_reference_retrieval import DISEASE_REFERENCE_INDEX_DIR, find_similar_disease_examples  # noqa: E402
from src.feedback_store import save_verified_feedback  # noqa: E402
from src.gradcam import gradcam_predict  # noqa: E402
from src.multimodal_predict import build_advanced_summary, predict_weather_risk  # noqa: E402
from src.plant_id import identify_plant_local, identify_plant_plantnet  # noqa: E402
from src.visual_triage import analyze_leaf_visual_triage  # noqa: E402
from src.weather_features import fetch_weather_features  # noqa: E402


def _load_disease_classes() -> list[str]:
    path = Path("models/disease/disease_class_names.json")
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _sample_images() -> list[Path]:
    if not SAMPLE_IMAGES_DIR.exists():
        return []
    return sorted(path for path in SAMPLE_IMAGES_DIR.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})


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

disease_status = disease_model_status()
disease_classes = _load_disease_classes()
plantnet_key = os.getenv("PLANTNET_API_KEY")
plantnet_available = bool(plantnet_key and plantnet_key != "your_api_key_here")

with st.sidebar:
    st.header("Mode")
    mode = st.radio("Recognition mode", ["Local disease model only", "Advanced plant ID + disease model"])
    confidence_threshold = st.slider("Disease confidence threshold", 0.0, 1.0, 0.55, 0.05)

    st.header("Optional modules")
    enable_plantnet = st.checkbox(
        "Enable Pl@ntNet API",
        value=False,
        disabled=mode == "Local disease model only" or not plantnet_available,
    )
    if mode == "Advanced plant ID + disease model" and not plantnet_available:
        st.caption("Pl@ntNet is optional. Add a real PLANTNET_API_KEY in .env and restart Streamlit to enable it.")
    enable_local_species = st.checkbox("Enable local species fallback", value=False, disabled=mode == "Local disease model only")
    enable_weather = st.checkbox("Enable weather risk", value=False)
    enable_disease_retrieval = st.checkbox("Show similar disease reference images", value=True)
    enable_feedback = st.checkbox("Enable verified feedback saving", value=False)

    st.header("Model status")
    st.info("Plant ID: optional API/local layer")
    if disease_status["model_exists"] and disease_status["class_names_exists"]:
        st.success("Disease model found")
        st.caption(f"Version: {disease_status['model_version']}")
        st.caption(f"Architecture: {disease_status['architecture']}")
        st.caption(f"Disease classes: {disease_status['class_count']}")
    else:
        st.error("Disease model missing")
        st.caption("Train it with `python -m src.train_disease_model --data_dir data/processed/cropvision_reference_train`.")
    if (DISEASE_REFERENCE_INDEX_DIR / "index.joblib").exists():
        st.success("Disease reference index found")
    else:
        st.info("Disease reference index optional")
    if DEFAULT_WEATHER_MODEL_PATH.exists():
        st.success("Weather model found")
    else:
        st.info("Weather model optional")

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
        "CropVision first identifies the plant when enabled, then predicts visible disease/problem patterns "
        "with a separate disease model. Verified feedback is saved only when you choose it."
    )

uploaded_file = st.file_uploader("Upload a plant/crop image", type=["jpg", "jpeg", "png"])
samples = _sample_images()
sample_name = st.selectbox("Or choose a sample image", ["None"] + [path.name for path in samples]) if samples else "None"

if uploaded_file is None and sample_name == "None":
    st.caption("Upload an image to run plant health recognition.")
    st.stop()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    source_name = uploaded_file.name
else:
    selected_path = next(path for path in samples if path.name == sample_name)
    image = Image.open(selected_path).convert("RGB")
    source_name = selected_path.name

with tempfile.NamedTemporaryFile(delete=False, suffix=Path(source_name).suffix or ".jpg") as tmp:
    image.save(tmp.name)
    tmp_path = Path(tmp.name)

left, right = st.columns([1, 1])
with left:
    st.subheader("Uploaded image")
    st.image(image, caption=source_name, use_container_width=True)

plant_id_result = None
if mode == "Advanced plant ID + disease model":
    st.subheader("Plant identification")
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

disease_result = predict_disease(str(tmp_path), top_k=3, confidence_threshold=confidence_threshold)
similar_examples: list[dict] = []
weather_result = None

if not disease_result.get("available"):
    st.subheader("Broad visual triage")
    triage = analyze_leaf_visual_triage(tmp_path)
    st.info(
        "No trained disease model found, so this is a rule-based visual triage fallback. "
        "Train the disease model for real disease predictions."
    )
    st.metric(triage["problem_category"], f"{triage['confidence']:.0%}")
    st.write("Observed signals:")
    for observation in triage["observations"]:
        st.write(f"- {observation}")
    with st.expander("Heuristic image metrics"):
        st.json(triage["metrics"])
    st.subheader("Final conservative interpretation")
    st.markdown(f"<div class='interpretation-card'>{triage['final_summary']}</div>", unsafe_allow_html=True)
    st.warning("The model can only recognize diseases represented in its training database. Unknown plants or diseases may produce uncertain predictions.")
    st.stop()

label_info = disease_result["normalized_label_info"]
with right:
    st.subheader("Disease Grad-CAM")
    try:
        gradcam_result = gradcam_predict(tmp_path, checkpoint_path=DISEASE_MODEL_PATH, confidence_threshold=confidence_threshold)
        st.image(gradcam_result["overlay_image"], caption="Disease-model visual explanation", use_container_width=True)
    except Exception as exc:
        st.info(f"Grad-CAM unavailable for this disease model: {exc}")

pred_col, top_col = st.columns([1, 1])
with pred_col:
    st.subheader("Disease identification")
    display_label = "uncertain" if disease_result["is_uncertain"] else disease_result["raw_predicted_disease_class"]
    st.metric(display_label, f"{disease_result['confidence']:.1%}")
    st.write(f"Plant species from disease label: **{label_info['plant_species']}**")
    st.write(f"Disease name: **{label_info['disease_name']}**")
    st.write(f"Broad problem category: **{disease_result['broad_problem_category']}**")
    if disease_result["is_uncertain"]:
        st.warning(
            f"{disease_result['uncertainty_reason']} The image may be low quality, out-of-distribution, "
            "or the disease may not be represented in the training database."
        )

with top_col:
    st.subheader("Top 3 disease predictions")
    for item in disease_result["top_k_predictions"]:
        st.write(f"{item['class_name']}: {item['confidence']:.1%}")

if enable_disease_retrieval:
    st.subheader("Similar disease reference examples")
    if (DISEASE_REFERENCE_INDEX_DIR / "index.joblib").exists():
        try:
            similar_examples = find_similar_disease_examples(tmp_path, top_k=3)
            if similar_examples:
                cols = st.columns(len(similar_examples))
                for col, item in zip(cols, similar_examples):
                    with col:
                        st.image(item["image_path"], caption=f"{item['class_label']} ({item['similarity_score']:.2f})", use_container_width=True)
            else:
                st.info("No similar disease examples were returned.")
        except Exception as exc:
            st.info(f"Disease reference retrieval unavailable: {exc}")
    else:
        st.info("Disease reference index missing. Build it after training with `python -m src.disease_reference_retrieval --build_index --data_dir data/processed/cropvision_reference_train --output_dir models/disease_reference_index`.")

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
    disease_result["raw_predicted_disease_class"],
    disease_result["broad_problem_category"],
    disease_result["confidence"],
    disease_result["is_uncertain"],
    weather_level,
)
st.subheader("Final conservative interpretation")
st.markdown(f"<div class='interpretation-card'>{summary}</div>", unsafe_allow_html=True)
st.warning("The model can only recognize diseases represented in its training database. Unknown plants or diseases may produce uncertain predictions.")

if enable_feedback:
    st.subheader("Help improve disease identification")
    st.caption("Feedback is saved only when you choose it. The app does not retrain automatically.")
    feedback_choice = st.radio("Feedback", ["Skip feedback", "Prediction is correct", "Prediction is wrong"])
    correct_label = disease_result["raw_predicted_disease_class"]
    if feedback_choice == "Prediction is wrong":
        options = ["Custom label"] + disease_classes
        selected_label = st.selectbox("Correct disease class", options)
        custom_label = st.text_input("Custom label", value="")
        correct_label = custom_label.strip() if selected_label == "Custom label" else selected_label
        st.info("New labels require explicit retraining before the model can recognize them.")
    if feedback_choice != "Skip feedback":
        if st.button("Save verified feedback", disabled=not correct_label):
            try:
                metadata = save_verified_feedback(
                    tmp_path,
                    correct_label,
                    original_prediction=disease_result["raw_predicted_disease_class"],
                    reference_matches=similar_examples,
                    model_version=disease_result.get("model_version"),
                )
                st.success(f"Saved verified feedback: {metadata['normalized_label']['normalized_class']}")
            except Exception as exc:
                st.error(f"Could not save feedback: {exc}")
