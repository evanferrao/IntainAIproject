"""Tests for Submission Generation and Validation."""

import pandas as pd
from src.config.schema import SubmissionRecord


def test_submission_schema_conformance():
    sub_df = pd.DataFrame({
        "loan_id": ["LC_001", "LC_002"],
        "delinquency_probability": [0.05, 0.42],
        "default_probability": [0.015, 0.28],
        "prepayment_probability": [0.08, 0.03],
        "next_state": ["CURRENT", "DELINQUENT_30"],
        "exception_probability": [0.01, 0.65],
        "exception_type": ["NONE", "SOURCE_RECONCILIATION_CONFLICT"],
        "anomaly_score": [10.5, 78.4],
        "top_drivers": ["standard_performing", "status_mismatch"],
        "reviewer_action": ["AUTO_APPROVE", "RECONCILE_SOURCE_CONFLICT"],
        "confidence": ["HIGH", "MODERATE"]
    })

    for row in sub_df.to_dict(orient="records"):
        rec = SubmissionRecord(**row)
        assert rec.loan_id in ["LC_001", "LC_002"]
        assert 0.0 <= rec.default_probability <= 1.0
        assert 0.0 <= rec.anomaly_score <= 100.0
