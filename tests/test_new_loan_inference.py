"""Tests for User-Entered / New Loan Inference and Explanation."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.config.settings import MODELS_DIR
from src.utils.serialization import load_model
from src.anomaly.rule_engine import DeterministicRuleEngine
from src.anomaly.reviewer_policy import ReviewerPolicyEngine
from src.explainability.shap_explainer import ShapExplainer


@pytest.fixture(scope="module")
def pipeline_models():
    return {
        "feature_pipeline": load_model(MODELS_DIR / "feature_pipeline.joblib"),
        "default": load_model(MODELS_DIR / "default_model.joblib"),
        "delinquency": load_model(MODELS_DIR / "delinquency_model.joblib"),
        "prepayment": load_model(MODELS_DIR / "prepayment_model.joblib"),
        "next_state": load_model(MODELS_DIR / "next_state_model.joblib"),
        "isolation_forest": load_model(MODELS_DIR / "isolation_forest.joblib")
    }


def test_user_entered_new_loan_inference(pipeline_models):
    """Verifies that a newly entered user loan executes end-to-end inference and explanation without Groq."""
    models = pipeline_models

    user_loan = {
        "loan_id": "NEW_USER_LOAN_01",
        "origination_month": "2026-01",
        "original_balance": 18000.0,
        "funded_balance": 18000.0,
        "interest_rate": 16.5,
        "original_term": 36,
        "installment": 637.2,
        "credit_score": 645.0,
        "credit_score_band": "Subprime",
        "dti": 28.5,
        "dti_band": "20-35%",
        "annual_income": 65000.0,
        "employment_length": 4,
        "home_ownership": "RENT",
        "loan_purpose": "debt_consolidation",
        "state": "IL",
        "revolving_utilization": 65.0,
        "delinquencies_2yrs": 0,
        "inquiries_6m": 1,
        "total_accounts": 18,
        "servicer_name": "Beacon_Credit_Ops",
        "document_status": "VERIFIED",
        "reporting_month": "2026-01",
        "month_index": 1,
        "loan_age_months": 1,
        "remaining_term_months": 35,
        "current_balance": 18000.0,
        "current_status": "CURRENT",
        "days_past_due": 0
    }

    df_user = pd.DataFrame([user_loan])

    with patch("groq.Groq") as mock_groq:
        # 1. Feature transform
        X_user = models["feature_pipeline"].transform(df_user)
        assert X_user.shape[0] == 1

        # 2. Predictions
        p_def = float(models["default"].predict_proba(X_user)[0])
        p_del = float(models["delinquency"].predict_proba(X_user)[0])
        p_prep = float(models["prepayment"].predict_proba(X_user)[0])
        pred_next = str(models["next_state"].predict(X_user)[0])

        assert 0.0 <= p_def <= 1.0
        assert 0.0 <= p_del <= 1.0
        assert 0.0 <= p_prep <= 1.0

        # 3. Anomaly scoring
        ml_scores, is_ano = models["isolation_forest"].score_samples(X_user)
        ano_score = float(ml_scores[0])

        # 4. Reviewer Policy Triage
        policy_eng = ReviewerPolicyEngine()
        triage = policy_eng.evaluate_loan(
            loan_id="NEW_USER_LOAN_01",
            default_prob=p_def,
            delinquency_prob=p_del,
            prepayment_prob=p_prep,
            anomaly_score=ano_score,
            is_anomaly=bool(is_ano[0]),
            data_quality_score=95.0
        )
        assert triage["disposition"] in ["AUTO_APPROVE", "FLAG_FOR_REVIEW", "HIGH_RISK_REVIEW", "MANUAL_AUDIT_REQUIRED"]
        assert len(triage["reasons"]) > 0

        # 5. Local SHAP explanation
        shap_exp = ShapExplainer()
        shap_res = shap_exp.explain_instance("default", models["default"], X_user, top_k=6)
        assert len(shap_res["top_drivers"]) == 6

        mock_groq.assert_not_called()
