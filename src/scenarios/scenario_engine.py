"""Macroeconomic Scenario and Portfolio Stress Simulation Engine."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from src.config.settings import CANONICAL_DATA_DIR
from src.utils.logger import logger


class ScenarioSimulationEngine:
    """Simulates portfolio credit risk, transition shifts, and loss projections under macroeconomic stress scenarios."""

    def __init__(self, scenarios_path: Optional[Path] = None):
        self.scenarios_path = scenarios_path or (CANONICAL_DATA_DIR / "macro_scenarios.csv")
        self.scenarios_df = self._load_scenarios()

    def _load_scenarios(self) -> pd.DataFrame:
        """Loads documented scenario parameters from macro_scenarios.csv."""
        if self.scenarios_path.exists():
            return pd.read_csv(self.scenarios_path)
        else:
            logger.warning(f"Scenario definitions file not found at {self.scenarios_path}. Using fallback.")
            return pd.DataFrame([
                {"scenario_id": 1, "scenario_name": "BASE", "unemployment_shock_bps": 0, "interest_rate_shift_bps": 0, "hpi_shift_pct": 2.0, "credit_score_shift": 0, "prepayment_multiplier": 1.0, "default_multiplier": 1.0, "description": "Baseline"},
                {"scenario_id": 2, "scenario_name": "ADVERSE_CREDIT", "unemployment_shock_bps": 300, "interest_rate_shift_bps": 100, "hpi_shift_pct": -10.0, "credit_score_shift": -40, "prepayment_multiplier": 0.45, "default_multiplier": 2.40, "description": "Severe recession"},
                {"scenario_id": 3, "scenario_name": "HIGH_PREPAYMENT", "unemployment_shock_bps": -50, "interest_rate_shift_bps": -150, "hpi_shift_pct": 5.0, "credit_score_shift": 10, "prepayment_multiplier": 1.85, "default_multiplier": 0.75, "description": "Rate easing"}
            ])

    def simulate_scenario(
        self,
        portfolio_df: pd.DataFrame,
        scenario_name: str,
        delinquency_model: Optional[Any] = None,
        default_model: Optional[Any] = None,
        prepayment_model: Optional[Any] = None,
        feature_pipeline: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Applies scenario stress transformations to portfolio features and recalculates risk distributions."""
        scenario_row = self.scenarios_df[self.scenarios_df["scenario_name"] == scenario_name]
        if len(scenario_row) == 0:
            raise ValueError(f"Scenario '{scenario_name}' not defined in {self.scenarios_path}")

        params = scenario_row.iloc[0].to_dict()
        logger.info(f"Simulating scenario '{scenario_name}': {params.get('description', '')}...")

        stressed_df = portfolio_df.copy()
        
        # Apply macroeconomic parameter shifts
        fico_shift = float(params.get("credit_score_shift", 0.0))
        rate_shift_pct = float(params.get("interest_rate_shift_bps", 0.0)) / 100.0
        default_mult = float(params.get("default_multiplier", 1.0))
        prepay_mult = float(params.get("prepayment_multiplier", 1.0))

        if "credit_score" in stressed_df.columns:
            stressed_df["credit_score"] = (stressed_df["credit_score"] + fico_shift).clip(300, 850)
            
        if "interest_rate" in stressed_df.columns:
            stressed_df["interest_rate"] = (stressed_df["interest_rate"] + rate_shift_pct).clip(1.0, 40.0)

        # Predict with ML models or apply calibrated multipliers
        if feature_pipeline is not None and default_model is not None and prepayment_model is not None:
            try:
                X_stressed = feature_pipeline.transform(stressed_df)
                p_delinq = delinquency_model.predict_proba(X_stressed) if delinquency_model else np.full(len(stressed_df), 0.05 * default_mult)
                p_default = default_model.predict_proba(X_stressed) * default_mult
                p_prepay = prepayment_model.predict_proba(X_stressed) * prepay_mult
            except Exception as e:
                logger.warning(f"Feature transformation failed during scenario: {e}. Applying parametric shifts.")
                p_delinq = np.clip(np.full(len(stressed_df), 0.06) * default_mult, 0.0, 1.0)
                p_default = np.clip(np.full(len(stressed_df), 0.025) * default_mult, 0.0, 1.0)
                p_prepay = np.clip(np.full(len(stressed_df), 0.08) * prepay_mult, 0.0, 1.0)
        else:
            # Baseline portfolio averages scaled by multipliers
            p_delinq = np.clip(np.full(len(stressed_df), 0.06) * default_mult, 0.0, 1.0)
            p_default = np.clip(np.full(len(stressed_df), 0.025) * default_mult, 0.0, 1.0)
            p_prepay = np.clip(np.full(len(stressed_df), 0.08) * prepay_mult, 0.0, 1.0)

        p_delinq = np.clip(p_delinq, 0.0, 1.0)
        p_default = np.clip(p_default, 0.0, 1.0)
        p_prepay = np.clip(p_prepay, 0.0, 1.0)

        stressed_df["sim_delinquency_prob"] = p_delinq
        stressed_df["sim_default_prob"] = p_default
        stressed_df["sim_prepayment_prob"] = p_prepay

        # Portfolio aggregate metrics
        total_balance = float(stressed_df["current_balance"].sum()) if "current_balance" in stressed_df else float(len(stressed_df) * 12000.0)
        expected_default_balance = float((stressed_df["current_balance"] * p_default).sum()) if "current_balance" in stressed_df else float(total_balance * p_default.mean())
        expected_prepay_balance = float((stressed_df["current_balance"] * p_prepay).sum()) if "current_balance" in stressed_df else float(total_balance * p_prepay.mean())

        # Assuming standard 50% Loss Given Default (LGD)
        expected_loss_balance = expected_default_balance * 0.50

        # Segment breakdowns (by credit_score_band and loan_purpose)
        segment_results = {}
        for seg_col in ["credit_score_band", "loan_purpose", "state"]:
            if seg_col in stressed_df.columns:
                seg_grp = stressed_df.groupby(seg_col).agg(
                    record_count=("loan_id", "count"),
                    avg_default_rate=("sim_default_prob", "mean"),
                    avg_prepay_rate=("sim_prepayment_prob", "mean"),
                    avg_delinq_rate=("sim_delinquency_prob", "mean")
                ).round(4).to_dict(orient="index")
                segment_results[seg_col] = seg_grp

        scenario_summary = {
            "scenario_name": scenario_name,
            "description": params.get("description", ""),
            "macro_assumptions": params,
            "portfolio_metrics": {
                "total_records": len(stressed_df),
                "total_portfolio_balance": round(total_balance, 2),
                "projected_delinquency_rate": round(float(np.mean(p_delinq)), 4),
                "projected_default_rate": round(float(np.mean(p_default)), 4),
                "projected_prepayment_rate": round(float(np.mean(p_prepay)), 4),
                "expected_default_balance": round(expected_default_balance, 2),
                "expected_prepay_balance": round(expected_prepay_balance, 2),
                "expected_loss_dollars": round(expected_loss_balance, 2),
                "expected_loss_rate": round(float(expected_loss_balance / max(1.0, total_balance)), 4)
            },
            "segment_breakdowns": segment_results,
            "disclaimer": "Scenario outputs are model-based stress projections, not guaranteed causal forecasts."
        }

        return scenario_summary

    def run_all_scenarios(
        self,
        portfolio_df: pd.DataFrame,
        delinquency_model: Optional[Any] = None,
        default_model: Optional[Any] = None,
        prepayment_model: Optional[Any] = None,
        feature_pipeline: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Runs simulations across all defined scenarios (BASE, ADVERSE_CREDIT, HIGH_PREPAYMENT)."""
        logger.info("Executing comprehensive multi-scenario portfolio stress simulation...")
        all_results = {}
        for scenario_name in self.scenarios_df["scenario_name"].unique():
            res = self.simulate_scenario(
                portfolio_df,
                scenario_name=scenario_name,
                delinquency_model=delinquency_model,
                default_model=default_model,
                prepayment_model=prepayment_model,
                feature_pipeline=feature_pipeline
            )
            all_results[scenario_name] = res

        return all_results
