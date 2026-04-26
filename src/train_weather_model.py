"""Train a demo weather-risk model on synthetic labels.

This model is for portfolio/demo use only. Replace the simulated labels with
real disease incidence data before treating outputs as agronomic evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_WEATHER_MODEL_PATH, WEATHER_FEATURE_COLUMNS, ensure_project_dirs


def make_synthetic_weather_dataset(n_samples: int = 1200, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    rainfall_7d = rng.gamma(shape=2.0, scale=12.0, size=n_samples)
    humidity_avg_7d = rng.normal(loc=65.0, scale=18.0, size=n_samples).clip(20, 100)
    temp_avg = rng.normal(loc=24.0, scale=6.0, size=n_samples).clip(5, 42)
    temp_max = (temp_avg + rng.normal(loc=6.0, scale=3.0, size=n_samples)).clip(8, 48)
    heat_stress_days = rng.binomial(7, np.clip((temp_max - 25) / 18, 0, 1))
    wet_days = rng.binomial(7, np.clip(rainfall_7d / 70, 0, 1))

    X = pd.DataFrame(
        {
            "rainfall_7d": rainfall_7d,
            "humidity_avg_7d": humidity_avg_7d,
            "temp_avg": temp_avg,
            "temp_max": temp_max,
            "heat_stress_days": heat_stress_days,
            "wet_days": wet_days,
        }
    )[WEATHER_FEATURE_COLUMNS]

    # Transparent synthetic rule: wet/humid conditions and heat stress increase risk.
    risk_score = (
        0.035 * rainfall_7d
        + 0.025 * humidity_avg_7d
        + 0.22 * heat_stress_days
        + 0.18 * wet_days
        + 0.04 * np.maximum(temp_max - 30, 0)
        + rng.normal(0, 0.45, n_samples)
    )
    y = np.where(risk_score < 2.6, "low", np.where(risk_score < 4.1, "medium", "high"))
    return X, y


def train_weather_model(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    X, y = make_synthetic_weather_dataset(args.samples, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed, class_weight="balanced")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Synthetic weather-risk model report:")
    print(classification_report(y_test, preds, zero_division=0))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": WEATHER_FEATURE_COLUMNS, "labels": ["low", "medium", "high"]}, output)
    print(f"Saved weather risk model to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train synthetic RandomForest weather risk model.")
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--n_estimators", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_WEATHER_MODEL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    train_weather_model(parse_args())
