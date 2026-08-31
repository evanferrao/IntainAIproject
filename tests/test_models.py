"""Tests for ML Predictive Models and Calibration."""

import numpy as np
import pandas as pd
from src.models.delinquency_model import DelinquencyModel
from src.models.default_model import DefaultModel
from src.models.next_state_model import NextStateModel
from src.evaluation.metrics import evaluate_binary_classifier, evaluate_multiclass_classifier


def test_delinquency_model_training_and_calibration():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "credit_score": np.random.normal(700, 40, n),
        "dti": np.random.uniform(5, 35, n),
        "interest_rate": np.random.uniform(6, 25, n),
        "balance_ratio": np.random.uniform(0.2, 1.0, n)
    })
    # Target: higher default if low credit score, high dti
    prob = 1.0 / (1.0 + np.exp(-((700 - X["credit_score"]) / 50.0 + (X["dti"] / 20.0))))
    y = (np.random.rand(n) < prob).astype(int)

    model = DelinquencyModel(use_calibration=True, seed=42)
    model.train_baseline(X, pd.Series(y))
    model.train_improved(X, pd.Series(y))

    p_base = model.predict_proba(X, model_type="baseline")
    p_imp = model.predict_proba(X, model_type="improved")

    assert len(p_base) == n
    assert len(p_imp) == n
    assert (p_imp >= 0.0).all() and (p_imp <= 1.0).all()

    base_m, imp_m = model.evaluate(X, pd.Series(y))
    assert "roc_auc" in imp_m
    assert "brier_score" in imp_m


def test_next_state_model_multiclass():
    np.random.seed(42)
    n = 150
    X = pd.DataFrame({
        "credit_score": np.random.normal(700, 40, n),
        "dti": np.random.uniform(5, 35, n),
        "days_past_due": np.random.choice([0, 30, 60], size=n)
    })
    y = pd.Series(np.random.choice(["CURRENT", "DELINQUENT_30", "DELINQUENT_60", "DEFAULT", "PREPAID"], size=n))

    ns_model = NextStateModel(seed=42)
    ns_model.train_baseline(X, y)
    ns_model.train_improved(X, y)

    preds = ns_model.predict(X)
    probs = ns_model.predict_proba(X)

    assert len(preds) == n
    assert probs.shape == (n, len(ns_model.class_names))
