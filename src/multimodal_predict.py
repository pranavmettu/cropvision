"""Combine image disease prediction with optional weather-risk features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config import DEFAULT_CV_MODEL_PATH, DEFAULT_WEATHER_MODEL_PATH, WEATHER_FEATURE_COLUMNS
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


def multimodal_predict(
    image_path: Path,
    checkpoint_path: Path = DEFAULT_CV_MODEL_PATH,
    weather_model_path: Path = DEFAULT_WEATHER_MODEL_PATH,
    latitude: float | None = None,
    longitude: float | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    image_result = predict_image(image_path, checkpoint_path, confidence_threshold=confidence_threshold)
    weather_result = None
    if all(value is not None for value in (latitude, longitude, start_date, end_date)):
        features = fetch_weather_features(float(latitude), float(longitude), str(start_date), str(end_date))
        weather_result = predict_weather_risk(features, weather_model_path)

    weather_level = weather_result["weather_risk_level"] if weather_result else None
    return {
        "predicted_disease": image_result["predicted_class"],
        "raw_predicted_disease": image_result["raw_predicted_class"],
        "image_confidence": image_result["confidence"],
        "top_predictions": image_result["top_predictions"],
        "is_uncertain": image_result["is_uncertain"],
        "uncertainty_reason": image_result["uncertainty_reason"],
        "weather": weather_result,
        "combined_risk_summary": build_combined_summary(
            image_result["predicted_class"], image_result["confidence"], weather_level
        ),
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
        )
    )
