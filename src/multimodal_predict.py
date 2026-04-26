"""Combine image disease prediction with optional weather-risk features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config import DEFAULT_CV_MODEL_PATH, DEFAULT_RETRIEVAL_ARTIFACT_PATH, DEFAULT_WEATHER_MODEL_PATH, WEATHER_FEATURE_COLUMNS
from src.image_retrieval import find_similar_images
from src.plant_id import identify_plant_local, identify_plant_plantnet
from src.problem_taxonomy import map_disease_class_to_problem_category
from src.predict_cv import predict_image
from src.weather_features import features_to_frame, fetch_weather_features


def predict_weather_risk(features: dict[str, float], model_path: Path = DEFAULT_WEATHER_MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        return {
            "weather_risk_level": "not_available",
            "weather_risk_confidence": None,
            "weather_features": features,
            "message": f"Weather model not found at {model_path}. Run python -m src.train_weather_model first.",
        }
    payload = joblib.load(model_path)
    model = payload["model"]
    columns = payload.get("feature_columns", WEATHER_FEATURE_COLUMNS)
    frame = features_to_frame(features)[columns]
    label = str(model.predict(frame)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        confidence = float(np.max(probabilities))
    return {
        "weather_risk_level": label,
        "weather_risk_confidence": confidence,
        "weather_features": features,
        "message": "Weather risk estimated from synthetic demo model.",
    }


def build_combined_summary(predicted_disease: str, image_confidence: float, weather_risk_level: str | None) -> str:
    confidence_text = f"{image_confidence:.0%}"
    if weather_risk_level in (None, "not_available"):
        return f"Image model predicts {predicted_disease} with {confidence_text} confidence. Weather risk was not included."
    if weather_risk_level == "high":
        return f"Image model predicts {predicted_disease} with {confidence_text} confidence, and recent weather indicates high stress/disease risk."
    if weather_risk_level == "medium":
        return f"Image model predicts {predicted_disease} with {confidence_text} confidence, with moderate weather-based risk."
    return f"Image model predicts {predicted_disease} with {confidence_text} confidence, while weather-based risk appears low."


def build_advanced_summary(
    plant_id: dict[str, Any] | None,
    raw_disease_class: str,
    problem_category: str,
    confidence: float,
    is_uncertain: bool,
    weather_risk_level: str | None = None,
) -> str:
    plant_text = ""
    if plant_id and plant_id.get("available"):
        plant_name = plant_id.get("common_name") or plant_id.get("scientific_name")
        if plant_name:
            plant_text = f"Plant ID suggests {plant_name}. "
    if is_uncertain:
        return (
            f"{plant_text}The health issue is uncertain. The image may be out-of-distribution, "
            "low quality, or the problem may not be represented in the training data."
        )
    weather_text = ""
    if weather_risk_level == "high":
        weather_text = " Recent wet or stressful conditions may increase disease/stress risk."
    elif weather_risk_level == "medium":
        weather_text = " Recent weather indicates moderate contextual crop stress risk."
    elif weather_risk_level == "low":
        weather_text = " Recent weather risk appears low."
    return (
        f"{plant_text}The disease model predicts {raw_disease_class} with {confidence:.0%} confidence, "
        f"which maps to {problem_category.replace('_', ' ')}.{weather_text} "
        "This is an educational ML prediction, not a professional diagnosis."
    )


def multimodal_predict(
    image_path: Path,
    checkpoint_path: Path = DEFAULT_CV_MODEL_PATH,
    weather_model_path: Path = DEFAULT_WEATHER_MODEL_PATH,
    latitude: float | None = None,
    longitude: float | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    confidence_threshold: float = 0.5,
    use_plantnet: bool = False,
    use_local_species_model: bool = False,
    use_retrieval: bool = False,
    retrieval_top_k: int = 3,
) -> dict[str, Any]:
    image_result = predict_image(image_path, checkpoint_path, confidence_threshold=confidence_threshold)
    plant_id = None
    if use_plantnet:
        plant_id = identify_plant_plantnet(str(image_path))
    elif use_local_species_model:
        plant_id = identify_plant_local(str(image_path))

    weather_result = None
    if all(value is not None for value in (latitude, longitude, start_date, end_date)):
        features = fetch_weather_features(float(latitude), float(longitude), str(start_date), str(end_date))
        weather_result = predict_weather_risk(features, weather_model_path)

    weather_level = weather_result["weather_risk_level"] if weather_result else None
    raw_disease = image_result["raw_predicted_class"]
    problem_category = "unknown_or_uncertain" if image_result["is_uncertain"] else map_disease_class_to_problem_category(raw_disease)
    similar_examples = []
    if use_retrieval and DEFAULT_RETRIEVAL_ARTIFACT_PATH.exists():
        similar_examples = find_similar_images(str(image_path), top_k=retrieval_top_k)
    final_summary = build_advanced_summary(
        plant_id,
        raw_disease,
        problem_category,
        image_result["confidence"],
        image_result["is_uncertain"],
        weather_level,
    )
    return {
        "plant_id": plant_id,
        "disease_prediction": {
            "predicted_class": image_result["predicted_class"],
            "raw_predicted_class": raw_disease,
            "confidence": image_result["confidence"],
        },
        "problem_category": problem_category,
        "top_3_predictions": image_result["top_3_predictions"],
        "uncertainty": {
            "is_uncertain": image_result["is_uncertain"],
            "reason": image_result["uncertainty_reason"],
        },
        "weather_risk": weather_result,
        "similar_examples": similar_examples,
        "final_summary": final_summary,
        "educational_disclaimer": "Educational ML demo only. Not professional crop diagnosis or treatment advice.",
        # Backward-compatible keys for older scripts.
        "predicted_disease": image_result["predicted_class"],
        "raw_predicted_disease": raw_disease,
        "image_confidence": image_result["confidence"],
        "top_predictions": image_result["top_predictions"],
        "weather": weather_result,
        "combined_risk_summary": final_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CropVision multimodal prediction.")
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CV_MODEL_PATH)
    parser.add_argument("--weather_model", type=Path, default=DEFAULT_WEATHER_MODEL_PATH)
    parser.add_argument("--latitude", type=float, default=None)
    parser.add_argument("--longitude", type=float, default=None)
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--use_plantnet", action="store_true")
    parser.add_argument("--use_local_species_model", action="store_true")
    parser.add_argument("--use_retrieval", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        multimodal_predict(
            args.image_path,
            args.checkpoint,
            args.weather_model,
            args.latitude,
            args.longitude,
            args.start_date,
            args.end_date,
            args.confidence_threshold,
            args.use_plantnet,
            args.use_local_species_model,
            args.use_retrieval,
        )
    )
