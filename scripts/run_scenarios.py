"""Scenario stress testing execution script."""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import CANONICAL_DATA_DIR, MODELS_DIR, METRICS_DIR
from src.scenarios.scenario_engine import ScenarioSimulationEngine
from src.utils.serialization import load_model, save_json
from src.utils.logger import logger


def run_all_stress_scenarios():
    logger.info("================ STARTING SCENARIO STRESS SIMULATIONS ================")
    
    # 1. Load data
    static_df = pd.read_csv(CANONICAL_DATA_DIR / "loan_static_attributes.csv")
    test_panel = pd.read_csv(CANONICAL_DATA_DIR / "loan_monthly_performance_test.csv")
    portfolio_df = test_panel.merge(static_df, on="loan_id", how="left")

    # 2. Load models
    feature_pipeline = load_model(MODELS_DIR / "feature_pipeline.joblib")
    delinq_model = load_model(MODELS_DIR / "delinquency_model.joblib")
    def_model = load_model(MODELS_DIR / "default_model.joblib")
    prep_model = load_model(MODELS_DIR / "prepayment_model.joblib")

    # 3. Simulate
    engine = ScenarioSimulationEngine()
    scenario_results = engine.run_all_scenarios(
        portfolio_df=portfolio_df,
        delinquency_model=delinq_model,
        default_model=def_model,
        prepayment_model=prep_model,
        feature_pipeline=feature_pipeline
    )

    # 4. Save results
    out_path = METRICS_DIR / "scenario_simulation_results.json"
    save_json(scenario_results, out_path)
    logger.info(f"Saved scenario simulation results to {out_path}")
    logger.info("================ SCENARIO SIMULATIONS COMPLETE ================")
    return scenario_results


if __name__ == "__main__":
    run_all_stress_scenarios()
