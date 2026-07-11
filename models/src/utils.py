"""
Utility functions for the CashYar AI prediction engine.

This module centralizes:
- Logging configuration
- File/path helpers
- Serialization helpers
- Common date helpers
- Safe numeric conversions
- Shared constants
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

USERS_FILE = DATA_DIR / "users.csv"
TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"
BEHAVIOR_FILE = DATA_DIR / "behavioral_summary.csv"

SPENDING_MODEL_FILE = MODELS_DIR / "spending_model.pkl"
OVERSPEND_MODEL_FILE = MODELS_DIR / "overspend_model.pkl"
SAVINGS_MODEL_FILE = MODELS_DIR / "savings_model.pkl"
TRAINING_ARTIFACTS_FILE = MODELS_DIR / "training_artifacts.pkl"

DEFAULT_RANDOM_STATE = 42


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a project logger.
    """
    logger = logging.getLogger("cashyar_ai")
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logging()


def ensure_directories() -> None:
    """
    Ensure required directories exist.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_joblib(obj: Any, path: Path) -> None:
    """
    Save an object with joblib.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Saved artifact to %s", path)


def load_joblib(path: Path) -> Any:
    """
    Load an object with joblib.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return joblib.load(path)


def to_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    """
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to int.
    """
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Divide safely and avoid division-by-zero errors.
    """
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def clip_non_negative(value: float) -> float:
    """
    Keep values non-negative.
    """
    return max(0.0, to_float(value))


def days_in_month_from_timestamp(ts: pd.Timestamp) -> int:
    """
    Return the number of days in the timestamp's month.
    """
    if pd.isna(ts):
        return 30
    return int(ts.days_in_month)


def month_key(ts: pd.Timestamp) -> str:
    """
    Convert timestamp to YYYY-MM month key.
    """
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m")


def normalize_text(value: Any) -> str:
    """
    Normalize text values for categories and labels.
    """
    if pd.isna(value):
        return "unknown"
    return str(value).strip().lower().replace(" ", "_")


def top_n_feature_importance(
    feature_names: List[str],
    importances: Iterable[float],
    top_n: int = 8
) -> List[Dict[str, float]]:
    """
    Convert raw feature importances into a sorted explainability structure.
    """
    pairs = [
        {"feature": str(name), "importance": float(score)}
        for name, score in zip(feature_names, importances)
    ]
    pairs = sorted(pairs, key=lambda x: x["importance"], reverse=True)
    return pairs[:top_n]


def confidence_from_probability(probability: float) -> str:
    """
    Convert a probability value to a qualitative confidence level.
    """
    p = abs(probability - 0.5) * 2
    if p >= 0.75:
        return "high"
    if p >= 0.45:
        return "medium"
    return "low"


def risk_level_from_probability(probability: float) -> str:
    """
    Convert overspending probability to a risk label.
    """
    if probability >= 0.8:
        return "high"
    if probability >= 0.55:
        return "medium"
    return "low"


def json_ready(value: Any) -> Any:
    """
    Convert numpy/pandas objects into JSON-serializable objects.
    """
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid helper.
    """
    x = float(x)
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def month_progress_ratio(current_day: int, days_in_month: int) -> float:
    """
    Ratio of current day progress in month.
    """
    if days_in_month <= 0:
        return 0.0
    return min(max(current_day / days_in_month, 0.0), 1.0)


def validate_required_columns(df: pd.DataFrame, required_columns: List[str], df_name: str) -> None:
    """
    Raise an informative error if required columns are missing.
    """
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")