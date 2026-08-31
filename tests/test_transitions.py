"""Tests for Markov Transition Matrix and Projections."""

import pandas as pd
import numpy as np
from src.transitions.markov_model import MarkovTransitionModel


def test_markov_transition_model():
    panel_data = pd.DataFrame({
        "current_status": ["CURRENT", "CURRENT", "DELINQUENT_30", "DELINQUENT_30", "DEFAULT", "PREPAID"] * 20,
        "next_state": ["CURRENT", "DELINQUENT_30", "CURRENT", "DELINQUENT_60", "DEFAULT", "PREPAID"] * 20,
        "credit_score_band": ["Prime"] * 120
    })

    markov = MarkovTransitionModel()
    markov.fit(panel_data, segment_cols=["credit_score_band"])

    # Verify global transition matrix properties (row stochastic: sums to 1.0)
    mat = markov.global_transition_matrix
    assert mat is not None
    row_sums = mat.sum(axis=1)
    np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-3)

    # Verify terminal absorption
    assert mat.loc["DEFAULT", "DEFAULT"] == 1.0
    assert mat.loc["PREPAID", "PREPAID"] == 1.0

    # Multi-period projections
    proj = markov.project_multi_period(horizon_months=12)
    assert len(proj) == 13
    assert "cumulative_default" in proj.columns
    assert "cumulative_prepayment" in proj.columns
    assert proj.loc[12, "cumulative_default"] >= proj.loc[0, "cumulative_default"]
