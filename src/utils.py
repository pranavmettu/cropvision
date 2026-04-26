"""Utility helpers used across CropVision scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set reproducible defaults without requiring deterministic GPU kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_class_names(path: Path) -> list[str]:
    data = load_json(path)
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"Expected {path} to contain a JSON list of class-name strings.")
    return data


def require_path(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{message}\nExpected path: {path}")
