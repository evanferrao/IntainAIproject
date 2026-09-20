"""Tests for Local and Global Explainability Suite (SHAP, LIME, PDP, Ceteris Paribus)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.config.settings import MODELS_DIR, CANONICAL_DATA_DIR
from src.utils.serialization import load_model
from src.explainability.shap_explainer import ShapExplainer
from src.explainability.lime_explainer import LimeExplainer
from src.explainability.pdp_explainer import PartialDependenceExplainer
from src.explainability.ceteris_paribus import CeterisParibusExplainer


@pytest.fixture(scope="module")
def loaded_components():
    pipe = load_model(MODELS_DIR / "feature_pipeline.joblib")
    def_model = load_model(MODELS_DIR / "default_model.joblib")
    static_df = pd.read_csv(CANONICAL_DATA_DIR / "loan_static_attributes.csv")
    test_panel = pd.read_csv(CANONICAL_DATA_DIR / "loan_monthly_performance_test.csv")
    latest_test = test_panel.sort_values("month_index").groupby("loan_id").last().reset_index()
    eval_df = latest_test.merge(static_df, on="loan_id", how="left")
    X_sample = pipe.transform(eval_df.head(50))
    return {
        "pipe": pipe,
        "model": def_model,
        "eval_df": eval_df,
        "X_sample": X_sample
    }


def test_shap_explanation_no_groq(loaded_components):
    """Verifies SHAP explanation computes locally against fitted model and does not call Groq."""
    shap_exp = ShapExplainer()
    model = loaded_components["model"]
    X_sample = loaded_components["X_sample"]
    row = X_sample.iloc[[0]]

    with patch("groq.Groq") as mock_groq:
        res = shap_exp.explain_instance(
            model_key="default",
            model=model,
            transformed_row=row,
            top_k=8
        )
        mock_groq.assert_not_called()

    assert "top_drivers" in res
    assert len(res["top_drivers"]) > 0
    assert "shap_value" in res["top_drivers"][0]
    assert "direction" in res["top_drivers"][0]


def test_lime_explanation_no_groq(loaded_components):
    """Verifies LIME explanation computes locally and does not call Groq."""
    X_sample = loaded_components["X_sample"]
    model = loaded_components["model"]
    lime_exp = LimeExplainer(background_data=X_sample)

    def predict_fn(arr):
        target_est = model.improved_model if hasattr(model, "improved_model") else model
        return target_est.predict_proba(arr)

    with patch("groq.Groq") as mock_groq:
        res = lime_exp.explain_instance(
            predict_fn=predict_fn,
            transformed_row=X_sample.iloc[0],
            num_features=6
        )
        mock_groq.assert_not_called()

    assert "attributions" in res
    assert len(res["attributions"]) > 0
    assert "weight" in res["attributions"][0]


def test_pdp_generation_no_groq(loaded_components):
    """Verifies PDP computes directly from fitted model and does not call Groq."""
    pdp_exp = PartialDependenceExplainer()
    model = loaded_components["model"]
    X_sample = loaded_components["X_sample"]

    with patch("groq.Groq") as mock_groq:
        res = pdp_exp.compute_pdp(model=model, X_sample=X_sample, feature_name="credit_score", grid_resolution=15)
        mock_groq.assert_not_called()

    assert res["is_appropriate"] == True
    assert len(res["grid_values"]) > 0
    assert len(res["average_predictions"]) == len(res["grid_values"])


def test_ceteris_paribus_generation_no_groq(loaded_components):
    """Verifies Ceteris Paribus evaluates model variations locally without Groq."""
    cp_exp = CeterisParibusExplainer()
    eval_df = loaded_components["eval_df"]
    pipe = loaded_components["pipe"]
    model = loaded_components["model"]
    loan_row = eval_df.iloc[0]

    with patch("groq.Groq") as mock_groq:
        res = cp_exp.compute_profile(
            loan_raw_record=loan_row,
            feature_pipeline=pipe,
            model=model,
            feature_name="credit_score",
            n_points=11
        )
        mock_groq.assert_not_called()

    assert res["success"] == True
    assert len(res["grid_values"]) >= 11
    assert len(res["predicted_probabilities"]) == len(res["grid_values"])
    assert "not a causal estimate" in res["disclaimer"]
