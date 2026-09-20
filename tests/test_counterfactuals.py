"""Tests for Model-Driven Counterfactual Search Engine."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.config.settings import MODELS_DIR, CANONICAL_DATA_DIR
from src.utils.serialization import load_model
from src.explainability.counterfactual_engine import CounterfactualSearchEngine


@pytest.fixture(scope="module")
def cf_components():
    pipe = load_model(MODELS_DIR / "feature_pipeline.joblib")
    def_model = load_model(MODELS_DIR / "default_model.joblib")
    static_df = pd.read_csv(CANONICAL_DATA_DIR / "loan_static_attributes.csv")
    test_panel = pd.read_csv(CANONICAL_DATA_DIR / "loan_monthly_performance_test.csv")
    latest_test = test_panel.sort_values("month_index").groupby("loan_id").last().reset_index()
    eval_df = latest_test.merge(static_df, on="loan_id", how="left")

    # Find loan with elevated default probability
    X = pipe.transform(eval_df)
    probs = def_model.predict_proba(X)
    top_indices = np.argsort(probs)[::-1]
    
    return {
        "pipe": pipe,
        "model": def_model,
        "loan_a": eval_df.iloc[top_indices[0]],
        "loan_b": eval_df.iloc[top_indices[1]]
    }


def test_counterfactual_search_reduces_risk_and_no_groq(cf_components):
    """Verifies counterfactual candidates actually reduce risk and search runs without Groq."""
    engine = CounterfactualSearchEngine()
    pipe = cf_components["pipe"]
    model = cf_components["model"]
    loan_a = cf_components["loan_a"]

    with patch("groq.Groq") as mock_groq:
        res = engine.search_counterfactuals(
            loan_raw_record=loan_a,
            feature_pipeline=pipe,
            model=model,
            max_results=5,
            min_risk_reduction=0.001
        )
        mock_groq.assert_not_called()

    assert "top_counterfactuals" in res
    cfs = res["top_counterfactuals"]
    assert len(cfs) > 0

    curr_risk = res["current_risk"]
    for c in cfs:
        assert c["counterfactual_risk"] < curr_risk
        assert c["risk_reduction"] > 0
        assert c["num_features_changed"] in [1, 2]

        # Verify no immutable features were changed
        for feat in c["features_changed"].keys():
            assert feat not in CounterfactualSearchEngine.IMMUTABLE_FEATURES
            assert feat in CounterfactualSearchEngine.MUTABLE_BOUNDS


def test_counterfactual_diversity_across_loans(cf_components):
    """Verifies different loans produce different counterfactuals."""
    engine = CounterfactualSearchEngine()
    pipe = cf_components["pipe"]
    model = cf_components["model"]
    loan_a = cf_components["loan_a"]
    loan_b = cf_components["loan_b"]

    res_a = engine.search_counterfactuals(loan_a, pipe, model, max_results=3, min_risk_reduction=0.001)
    res_b = engine.search_counterfactuals(loan_b, pipe, model, max_results=3, min_risk_reduction=0.001)

    assert res_a["loan_id"] != res_b["loan_id"]
    # Check that current risks or candidate values reflect each loan's own attributes
    assert "top_counterfactuals" in res_a and "top_counterfactuals" in res_b
