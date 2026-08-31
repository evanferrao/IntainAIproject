"""Tests for Feature Engineering and Leakage Prevention."""

import pandas as pd
import numpy as np
from src.features.feature_pipeline import FeatureEngineeringPipeline
from src.features.time_splitter import TimeAwareSplitter


def test_time_aware_splitter():
    df = pd.DataFrame({
        "loan_id": [f"L_{i}" for i in range(12)],
        "reporting_month": [
            "2018-01", "2018-02", "2018-03", "2018-04",
            "2018-05", "2018-06", "2018-07", "2018-08",
            "2018-09", "2018-10", "2018-11", "2018-12"
        ],
        "current_balance": [1000.0 * i for i in range(12)]
    })

    splitter = TimeAwareSplitter(test_months=3)
    train_df, test_df, meta = splitter.split(df)

    assert len(train_df) == 9
    assert len(test_df) == 3
    assert train_df["reporting_month"].max() < test_df["reporting_month"].min()


def test_feature_pipeline_no_leakage():
    train_df = pd.DataFrame({
        "loan_id": ["L1", "L2", "L3"],
        "reporting_month": ["2018-01", "2018-02", "2018-03"],
        "loan_age_months": [1, 5, 10],
        "original_term": [36, 36, 60],
        "current_balance": [9500.0, 8000.0, 15000.0],
        "original_balance": [10000.0, 10000.0, 20000.0],
        "installment": [320.0, 320.0, 450.0],
        "annual_income": [60000.0, 75000.0, 90000.0],
        "interest_rate": [11.5, 14.2, 9.8],
        "credit_score": [700, 660, 750],
        "dti": [16.0, 24.0, 12.0],
        "current_status": ["CURRENT", "DELINQUENT_30", "CURRENT"],
        "home_ownership": ["MORTGAGE", "RENT", "OWN"],
        "loan_purpose": ["debt_consolidation", "credit_card", "home_improvement"]
    })

    pipe = FeatureEngineeringPipeline()
    pipe.fit(train_df)
    X_train = pipe.transform(train_df)

    assert X_train.shape[0] == 3
    assert "balance_ratio" in X_train.columns
    assert "installment_to_income" in X_train.columns
    assert "seasonality_sin" in X_train.columns
