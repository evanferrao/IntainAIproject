"""Tests for Scenario and Stress Simulation."""

import pandas as pd
from src.scenarios.scenario_engine import ScenarioSimulationEngine


def test_scenario_simulation_engine():
    portfolio_df = pd.DataFrame({
        "loan_id": [f"L_{i}" for i in range(50)],
        "credit_score": [700] * 50,
        "interest_rate": [12.0] * 50,
        "current_balance": [10000.0] * 50,
        "credit_score_band": ["Prime"] * 50,
        "loan_purpose": ["debt_consolidation"] * 50
    })

    engine = ScenarioSimulationEngine()
    results = engine.run_all_scenarios(portfolio_df)

    assert "BASE" in results
    assert "ADVERSE_CREDIT" in results
    assert "HIGH_PREPAYMENT" in results

    base_def = results["BASE"]["portfolio_metrics"]["projected_default_rate"]
    adverse_def = results["ADVERSE_CREDIT"]["portfolio_metrics"]["projected_default_rate"]
    assert adverse_def > base_def, "Adverse credit scenario must project higher default rate than base."
