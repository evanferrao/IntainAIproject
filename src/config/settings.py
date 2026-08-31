"""Configuration management and project constants."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external" / "lending_club"
RAW_DATA_DIR = DATA_DIR / "raw"
CANONICAL_DATA_DIR = DATA_DIR / "canonical"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
LOGS_DIR = ARTIFACTS_DIR / "logs"

# Ensure runtime directories exist
for directory in [
    EXTERNAL_DATA_DIR, RAW_DATA_DIR, CANONICAL_DATA_DIR, PROCESSED_DATA_DIR,
    MODELS_DIR, METRICS_DIR, PREDICTIONS_DIR, PLOTS_DIR, REPORTS_DIR, LOGS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# Random Seed
RANDOM_SEED = 42

# Allowed Loan Statuses
VALID_STATUSES = [
    "CURRENT",
    "DELINQUENT_30",
    "DELINQUENT_60",
    "DELINQUENT_90",
    "DEFAULT",
    "PREPAID",
    "CLOSED",
]

TERMINAL_STATUSES = ["DEFAULT", "PREPAID", "CLOSED"]

# Credit score bins & labels
CREDIT_SCORE_BINS = [0, 580, 670, 740, 800, 850]
CREDIT_SCORE_LABELS = ["Deep Subprime", "Subprime", "Near Prime", "Prime", "Super Prime"]

# DTI bins & labels
DTI_BINS = [-1.0, 0.20, 0.35, 0.50, 1.00, 10.0]
DTI_LABELS = ["<20%", "20-35%", "35-50%", "50-100%", ">100%"]

# Default Pipeline Hyperparameters
@dataclass
class PipelineConfig:
    random_seed: int = RANDOM_SEED
    max_sample_records: int = 50_000
    monthly_panel_months: int = 24
    test_size_months: int = 6
    n_isolation_forest_estimators: int = 100
    contamination_rate: float = 0.03
    calibrated_method: str = "isotonic"
    lightgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1
    })

DEFAULT_CONFIG = PipelineConfig()
