"""Real Model-Driven Counterfactual Search and Risk-Reduction Optimization Engine."""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from src.utils.logger import logger


class CounterfactualSearchEngine:
    """Performs genuine model-driven search for feasible, sparse feature perturbations that reduce predicted risk.
    Executed strictly locally against the fitted model pipeline. No Groq or LLM calls."""

    DISCLAIMER = (
        "Counterfactuals represent model-implied changes that reduce predicted risk under controlled "
        "feature modifications. They are not causal estimates and do not guarantee approval."
    )

    # Immutable features that must never be altered in counterfactual search
    IMMUTABLE_FEATURES = {
        "loan_id", "origination_month", "reporting_month", "month_index",
        "loan_age_months", "remaining_term_months", "current_status",
        "days_past_due", "modification_flag", "prepayment_flag", "default_flag",
        "next_state", "next_3m_delinquency_flag", "next_12m_default_flag",
        "next_12m_prepayment_flag", "servicer_name", "state"
    }

    # Mutable features and plausible perturbation steps
    MUTABLE_FEATURE_STEPS = {
        "credit_score": [20.0, 40.0, 60.0, 80.0, 100.0],     # Upward improvement
        "dti": [-3.0, -6.0, -10.0, -15.0],                    # Downward debt reduction
        "interest_rate": [-2.0, -4.0, -6.0, -8.0],             # Refinancing / lower coupon
        "revolving_utilization": [-10.0, -20.0, -30.0, -40.0], # Balance pay-down
        "annual_income": [0.10, 0.25, 0.50]                    # Income increase (percentage)
    }

    # Bounds for mutable features
    MUTABLE_BOUNDS = {
        "credit_score": (300.0, 850.0),
        "dti": (0.0, 100.0),
        "interest_rate": (3.0, 40.0),
        "revolving_utilization": (0.0, 100.0),
        "annual_income": (1000.0, 2000000.0)
    }

    def __init__(self):
        pass

    def search_counterfactuals(
        self,
        loan_raw_record: pd.Series,
        feature_pipeline: Any,
        model: Any,
        model_name: str = "Default_Model",
        max_results: int = 5,
        min_risk_reduction: float = 0.005
    ) -> Dict[str, Any]:
        """Searches candidate feature modifications and re-runs the actual model to find sparse risk-reducing counterfactuals."""
        base_df = pd.DataFrame([loan_raw_record.to_dict()])
        X_base = feature_pipeline.transform(base_df)

        target_estimator = model.improved_model if hasattr(model, "improved_model") else model
        if hasattr(target_estimator, "predict_proba"):
            curr_risk = float(target_estimator.predict_proba(X_base)[:, 1][0])
        else:
            curr_risk = float(model.predict_proba(X_base)[0])

        candidates = []

        # 1. Single-Feature Perturbations (Sparsity = 1)
        for feat, steps in self.MUTABLE_FEATURE_STEPS.items():
            if feat not in loan_raw_record.index or pd.isna(loan_raw_record[feat]):
                continue
            curr_val = float(loan_raw_record[feat])
            bounds = self.MUTABLE_BOUNDS[feat]

            for step in steps:
                if feat == "annual_income":
                    new_val = curr_val * (1.0 + step)
                else:
                    new_val = curr_val + step

                new_val = np.clip(new_val, bounds[0], bounds[1])
                if abs(new_val - curr_val) < 1e-4:
                    continue

                pert_df = base_df.copy()
                pert_df.loc[0, feat] = new_val

                # Re-run the actual model
                X_pert = feature_pipeline.transform(pert_df)
                if hasattr(target_estimator, "predict_proba"):
                    pert_risk = float(target_estimator.predict_proba(X_pert)[:, 1][0])
                else:
                    pert_risk = float(model.predict_proba(X_pert)[0])

                risk_reduction = curr_risk - pert_risk
                if risk_reduction >= min_risk_reduction:
                    cost = abs(new_val - curr_val) / max(1.0, curr_val)
                    candidates.append({
                        "features_changed": {
                            feat: {
                                "current": round(curr_val, 2),
                                "counterfactual": round(new_val, 2),
                                "delta": round(new_val - curr_val, 2)
                            }
                        },
                        "num_features_changed": 1,
                        "counterfactual_risk": round(pert_risk, 4),
                        "risk_reduction": round(risk_reduction, 4),
                        "relative_reduction": round(risk_reduction / max(1e-6, curr_risk), 4),
                        "distance_cost": round(cost, 3)
                    })

        # 2. Dual-Feature Perturbations (Sparsity = 2)
        top_single_feats = list(set([list(c["features_changed"].keys())[0] for c in candidates]))
        if len(top_single_feats) >= 2:
            pairs = [
                ("credit_score", "dti"),
                ("credit_score", "interest_rate"),
                ("dti", "revolving_utilization"),
                ("credit_score", "revolving_utilization")
            ]
            for f1, f2 in pairs:
                if f1 not in loan_raw_record.index or f2 not in loan_raw_record.index:
                    continue
                v1 = float(loan_raw_record[f1])
                v2 = float(loan_raw_record[f2])

                # Use moderate steps for each
                s1 = 40.0 if f1 == "credit_score" else -5.0
                s2 = -5.0 if f2 == "dti" else -2.0 if f2 == "interest_rate" else -15.0

                nv1 = np.clip(v1 + s1, self.MUTABLE_BOUNDS[f1][0], self.MUTABLE_BOUNDS[f1][1])
                nv2 = np.clip(v2 + s2, self.MUTABLE_BOUNDS[f2][0], self.MUTABLE_BOUNDS[f2][1])

                pert_df = base_df.copy()
                pert_df.loc[0, f1] = nv1
                pert_df.loc[0, f2] = nv2

                X_pert = feature_pipeline.transform(pert_df)
                if hasattr(target_estimator, "predict_proba"):
                    pert_risk = float(target_estimator.predict_proba(X_pert)[:, 1][0])
                else:
                    pert_risk = float(model.predict_proba(X_pert)[0])

                risk_reduction = curr_risk - pert_risk
                if risk_reduction >= min_risk_reduction:
                    cost = (abs(nv1 - v1) / max(1.0, v1)) + (abs(nv2 - v2) / max(1.0, v2))
                    candidates.append({
                        "features_changed": {
                            f1: {"current": round(v1, 2), "counterfactual": round(nv1, 2), "delta": round(nv1 - v1, 2)},
                            f2: {"current": round(v2, 2), "counterfactual": round(nv2, 2), "delta": round(nv2 - v2, 2)}
                        },
                        "num_features_changed": 2,
                        "counterfactual_risk": round(pert_risk, 4),
                        "risk_reduction": round(risk_reduction, 4),
                        "relative_reduction": round(risk_reduction / max(1e-6, curr_risk), 4),
                        "distance_cost": round(cost, 3)
                    })

        # Rank candidates: primary = sparsity (ascending), secondary = risk_reduction (descending)
        candidates = sorted(
            candidates,
            key=lambda c: (c["num_features_changed"], -c["risk_reduction"], c["distance_cost"])
        )

        return {
            "loan_id": str(loan_raw_record.get("loan_id", "CUSTOM_LOAN")),
            "model_name": model_name,
            "current_risk": round(curr_risk, 4),
            "total_candidates_evaluated": len(candidates),
            "top_counterfactuals": candidates[:max_results],
            "disclaimer": self.DISCLAIMER,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
