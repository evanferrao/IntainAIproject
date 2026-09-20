"""Tests for Evidence-Driven Reviewer Triage Policy Engine."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.config.reviewer_policy import ReviewerPolicyConfig
from src.anomaly.reviewer_policy import ReviewerPolicyEngine


def test_reviewer_policy_dispositions():
    engine = ReviewerPolicyEngine()

    # 1. Clean, low-risk loan -> AUTO_APPROVE
    res_clean = engine.evaluate_loan(
        loan_id="L_CLEAN",
        default_prob=0.002,
        delinquency_prob=0.001,
        anomaly_score=12.0,
        is_anomaly=False,
        data_quality_score=95.0,
        has_deterministic_violation=False,
        has_reconciliation_conflict=False
    )
    assert res_clean["disposition"] == "AUTO_APPROVE"
    assert any("Clean data profile" in r for r in res_clean["reasons"])

    # 2. High default risk loan -> HIGH_RISK_REVIEW
    res_high_risk = engine.evaluate_loan(
        loan_id="L_HIGH_RISK",
        default_prob=0.085,
        delinquency_prob=0.005,
        anomaly_score=20.0,
        is_anomaly=False,
        data_quality_score=92.0
    )
    assert res_high_risk["disposition"] == "HIGH_RISK_REVIEW"
    assert any("High predicted default risk" in r for r in res_high_risk["reasons"])

    # 3. Statistical anomaly -> FLAG_FOR_REVIEW
    res_anomaly = engine.evaluate_loan(
        loan_id="L_ANOMALY",
        default_prob=0.003,
        delinquency_prob=0.002,
        anomaly_score=55.0,
        is_anomaly=True,
        data_quality_score=90.0
    )
    assert res_anomaly["disposition"] == "FLAG_FOR_REVIEW"
    assert any("statistical outlier" in r.lower() for r in res_anomaly["reasons"])

    # 4. Source reconciliation conflict -> RECONCILE_SOURCE_CONFLICT
    res_recon = engine.evaluate_loan(
        loan_id="L_RECON",
        default_prob=0.002,
        delinquency_prob=0.001,
        has_reconciliation_conflict=True,
        reconciliation_conflicts=["Status mismatch: master CURRENT vs servicer DELINQUENT_30"]
    )
    assert res_recon["disposition"] == "RECONCILE_SOURCE_CONFLICT"
    assert any("Source tape conflict" in r for r in res_recon["reasons"])

    # 5. Deterministic rule violation -> MANUAL_AUDIT_REQUIRED
    res_audit = engine.evaluate_loan(
        loan_id="L_AUDIT",
        default_prob=0.001,
        has_deterministic_violation=True,
        deterministic_violations=["VR-001: Negative balance"]
    )
    assert res_audit["disposition"] == "MANUAL_AUDIT_REQUIRED"
    assert any("Deterministic rule violation" in r for r in res_audit["reasons"])


def test_reviewer_policy_batch_evaluation():
    engine = ReviewerPolicyEngine()
    df = pd.DataFrame({"loan_id": ["L1", "L2", "L3", "L4"]})

    p_def = np.array([0.001, 0.09, 0.002, 0.002])
    p_del = np.array([0.001, 0.01, 0.001, 0.001])
    ano_scores = np.array([10.0, 15.0, 60.0, 10.0])
    is_ano = np.array([False, False, True, False])
    rec_flags = pd.Series([False, False, False, True])

    triage_df = engine.evaluate_batch(
        df=df,
        default_probs=p_def,
        delinquency_probs=p_del,
        anomaly_scores=ano_scores,
        is_anomaly_flags=is_ano,
        reconciliation_flags=rec_flags
    )

    actions = triage_df["reviewer_action"].tolist()
    assert actions[0] == "AUTO_APPROVE"
    assert actions[1] == "HIGH_RISK_REVIEW"
    assert actions[2] == "FLAG_FOR_REVIEW"
    assert actions[3] == "RECONCILE_SOURCE_CONFLICT"


def test_reviewer_policy_does_not_call_groq():
    """Proves that reviewer disposition logic runs locally without invoking Groq."""
    engine = ReviewerPolicyEngine()

    with patch("groq.Groq") as mock_groq:
        res = engine.evaluate_loan(
            loan_id="L_TEST",
            default_prob=0.06,
            delinquency_prob=0.01,
            anomaly_score=50.0,
            is_anomaly=True
        )
        assert res["disposition"] in ["HIGH_RISK_REVIEW", "FLAG_FOR_REVIEW"]
        mock_groq.assert_not_called()
