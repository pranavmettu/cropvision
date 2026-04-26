"""Fetch NASA POWER weather data and engineer disease-risk features."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.config import WEATHER_FEATURE_COLUMNS

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMETERS = ["PRECTOTCORR", "T2M", "T2M_MAX", "RH2M"]


def fetch_nasa_power_daily(
    latitude: float,
    longitude: float,
    start_date: str | date,
    end_date: str | date,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch daily NASA POWER weather data for one point.

    Dates should be provided as YYYY-MM-DD strings or datetime.date objects.
    """
    start = pd.to_datetime(start_date).strftime("%Y%m%d")
    end = pd.to_datetime(end_date).strftime("%Y%m%d")
    params = {
        "parameters": ",".join(POWER_PARAMETERS),
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    response = requests.get(NASA_POWER_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    parameter_data: dict[str, dict[str, Any]] = payload.get("properties", {}).get("parameter", {})
    if not parameter_data:
        raise ValueError("NASA POWER response did not include daily parameter data.")

    df = pd.DataFrame(parameter_data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(-999, np.nan).sort_index()
    return df


def calculate_weather_features(weather_df: pd.DataFrame) -> dict[str, float]:
    if weather_df.empty:
        raise ValueError("Weather dataframe is empty.")

    recent_7 = weather_df.tail(7)
    rainfall = recent_7.get("PRECTOTCORR", pd.Series(dtype=float))
    humidity = recent_7.get("RH2M", pd.Series(dtype=float))
    temp = weather_df.get("T2M", pd.Series(dtype=float))
    temp_max = weather_df.get("T2M_MAX", temp)

    features = {
        "rainfall_7d": float(rainfall.sum(skipna=True)),
        "humidity_avg_7d": float(humidity.mean(skipna=True)) if not humidity.dropna().empty else 50.0,
        "temp_avg": float(temp.mean(skipna=True)),
        "temp_max": float(temp_max.max(skipna=True)),
        "heat_stress_days": float((temp_max > 30.0).sum()),
        "wet_days": float((rainfall > 1.0).sum()),
    }
    return {name: features[name] for name in WEATHER_FEATURE_COLUMNS}


def fetch_weather_features(latitude: float, longitude: float, start_date: str | date, end_date: str | date) -> dict[str, float]:
    weather_df = fetch_nasa_power_daily(latitude, longitude, start_date, end_date)
    return calculate_weather_features(weather_df)


def features_to_frame(features: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([[features.get(col, 0.0) for col in WEATHER_FEATURE_COLUMNS]], columns=WEATHER_FEATURE_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch NASA POWER weather features.")
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--start_date", type=str, required=True)
    parser.add_argument("--end_date", type=str, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    features = fetch_weather_features(args.latitude, args.longitude, args.start_date, args.end_date)
    print(features)
    if args.output_csv:
        features_to_frame(features).to_csv(args.output_csv, index=False)
