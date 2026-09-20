"""Genuine Partial Dependence Plot (PDP) Explainability Engine."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

from src.utils.logger import logger


class PartialDependenceExplainer:
    """Computes genuine Partial Dependence Plots directly from fitted ML models. Local computation only, no Groq calls."""

    # Features appropriate for continuous PDP curves
    RECOMMENDED_PDP_FEATURES = [
        "credit_score", "dti", "interest_rate", "original_balance",
        "installment", "revolving_utilization", "annual_income",
        "installment_to_income", "credit_burden_index", "interest_rate_spread"
    ]

    def __init__(self):
        pass

    def compute_pdp(
        self,
        model: Any,
        X_sample: pd.DataFrame,
        feature_name: str,
        grid_resolution: int = 25
    ) -> Dict[str, Any]:
        """Computes partial dependence for a selected feature across a representative sample."""
        if feature_name not in X_sample.columns:
            return {
                "feature": feature_name,
                "is_appropriate": False,
                "reason": f"Feature '{feature_name}' not found in transformed model feature space.",
                "grid_values": [],
                "average_predictions": []
            }

        col_vals = X_sample[feature_name]
        n_unique = col_vals.nunique()

        if n_unique <= 1:
            return {
                "feature": feature_name,
                "is_appropriate": False,
                "reason": f"Feature '{feature_name}' is constant across the sample (only 1 unique value).",
                "grid_values": [],
                "average_predictions": []
            }

        try:
            # Handle CalibratedClassifierCV or improved_model wrapper
            target_estimator = model.improved_model if hasattr(model, "improved_model") else model

            pdp_res = partial_dependence(
                target_estimator,
                X=X_sample,
                features=[feature_name],
                grid_resolution=grid_resolution,
                kind="average"
            )

            grid_vals = [round(float(v), 4) for v in pdp_res["grid_values"][0]]
            avg_preds = [round(float(p), 4) for p in pdp_res["average"][0]]

            return {
                "feature": feature_name,
                "is_appropriate": True,
                "grid_values": grid_vals,
                "average_predictions": avg_preds,
                "min_response": min(avg_preds),
                "max_response": max(avg_preds),
                "spread": round(max(avg_preds) - min(avg_preds), 4)
            }
        except Exception as e:
            logger.error(f"Error computing PDP for {feature_name}: {e}")
            return {
                "feature": feature_name,
                "is_appropriate": False,
                "reason": f"Computation error: {str(e)}",
                "grid_values": [],
                "average_predictions": []
            }
