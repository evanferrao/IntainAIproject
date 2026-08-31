"""Artifact serialization and loading utilities."""

import json
from pathlib import Path
from typing import Any, Dict, Union
import joblib
import pandas as pd
import numpy as np


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy / Pandas types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, np.datetime64)):
            return str(obj)
        return super().default(obj)


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save dictionary to formatted JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NpEncoder)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model(model: Any, filepath: Union[str, Path]) -> None:
    """Save model artifact with joblib."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(filepath: Union[str, Path]) -> Any:
    """Load model artifact with joblib."""
    path = Path(filepath)
    return joblib.load(path)
