"""Tests for Data Profiling and Drift Detection."""

import numpy as np
import pandas as pd
from src.profiling.profiler import DataProfiler
from src.profiling.drift import DriftDetector
from src.quality.quality_scorer import DataQualityScorer


def test_data_profiler():
    df = pd.DataFrame({
        "credit_score": [650, 700, 720, 800, np.nan, 680],
        "original_balance": [10000.0, 15000.0, 20000.0, 25000.0, 30000.0, 50000.0],
        "state": ["CA", "NY", "TX", "CA", "FL", "CA"]
    })
    profiler = DataProfiler()
    summary = profiler.profile_dataset(df, dataset_name="TestSet")

    assert summary["row_count"] == 6
    assert summary["column_count"] == 3
    assert summary["missingness"]["column_missing_counts"]["credit_score"] == 1
    assert "mean" in summary["columns"]["original_balance"]


def test_drift_detector_psi():
    np.random.seed(42)
    train_vals = np.random.normal(700, 30, 1000)
    # Similar distribution
    test_stable = np.random.normal(700, 30, 500)
    # Shifted distribution
    test_drifted = np.random.normal(630, 40, 500)

    detector = DriftDetector()
    psi_stable = detector.calculate_psi(train_vals, test_stable)
    psi_drifted = detector.calculate_psi(train_vals, test_drifted)

    assert psi_stable < 0.10, "Expected stable distribution to have PSI < 0.10"
    assert psi_drifted > 0.25, "Expected shifted distribution to have PSI > 0.25"


def test_quality_scorer():
    df = pd.DataFrame({
        "loan_id": ["L1", "L2", "L3"],
        "credit_score": [720.0, np.nan, 310.0],
        "dti": [15.0, np.nan, 95.0],
        "original_balance": [15000.0, 10000.0, 5000.0]
    })
    scorer = DataQualityScorer()
    scored = scorer.score_records(df)
    batch_summary = scorer.score_dataset(scored)

    assert "record_quality_score" in scored.columns
    assert scored.loc[0, "record_quality_score"] > scored.loc[1, "record_quality_score"]
    assert "batch_quality_score" in batch_summary
