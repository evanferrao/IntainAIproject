"""Genuine Ceteris Paribus / Individual Conditional Expectation (ICE) Local Profiles."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import logger


class CeterisParibusExplainer:
    """Computes genuine Ceteris Paribus local profiles by repeatedly evaluating the actual model under controlled feature variation.
    Local computation only, no Groq calls."""

    DISCLAIMER = "Model response under controlled feature variation — not a causal estimate."

    DEFAULT_RANGES = {
        "credit_score": (580.0, 850.0),
        "dti": (0.0, 50.0),
        "interest_rate": (5.0, 35.0),
        "original_balance": (1000.0, 40000.0),
        "installment": (50.0, 1500.0),
        "revolving_utilization": (0.0, 100.0),
        "annual_income": (15000.0, 250000.0)
    }

    def __init__(self):
        pass

    def compute_profile(
        self,
        loan_raw_record: pd.Series,
        feature_pipeline: Any,
        model: Any,
        feature_name: str,
        n_points: int = 25,
        custom_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """Evaluates model prediction curves for a single loan by varying feature_name while holding all else fixed."""
        if feature_name not in loan_raw_record.index:
            return {
                "feature": feature_name,
                "success": False,
                "error": f"Feature '{feature_name}' not present in loan record."
            }

        curr_val = float(loan_raw_record[feature_name]) if pd.notna(loan_raw_record[feature_name]) else 0.0

        # Determine feature evaluation grid
        val_range = custom_range or self.DEFAULT_RANGES.get(feature_name)
        if val_range is None:
            min_v = max(0.0, curr_val * 0.5)
            max_v = max(curr_val * 1.5, 10.0)
            val_range = (min_v, max_v)

        grid = np.linspace(val_range[0], val_range[1], n_points)
        # Ensure current value is in grid for reference
        if curr_val not in grid:
            grid = np.sort(np.append(grid, curr_val))

        # Construct batch of records holding everything else constant
        base_df = pd.DataFrame([loan_raw_record.to_dict()])
        perturbed_records = pd.concat([base_df.assign(**{feature_name: v}) for v in grid], ignore_index=True)

        # Transform and predict using actual model
        X_perturbed = feature_pipeline.transform(perturbed_records)
        target_estimator = model.improved_model if hasattr(model, "improved_model") else model

        if hasattr(target_estimator, "predict_proba"):
            probs = target_estimator.predict_proba(X_perturbed)[:, 1]
        else:
            probs = model.predict_proba(X_perturbed)

        # Find prediction at current value
        curr_idx = np.where(grid == curr_val)[0]
        curr_pred = float(probs[curr_idx[0]]) if len(curr_idx) > 0 else float(probs[0])

        return {
            "feature": feature_name,
            "current_value": round(curr_val, 2),
            "current_prediction": round(curr_pred, 4),
            "grid_values": [round(float(v), 2) for v in grid],
            "predicted_probabilities": [round(float(p), 4) for p in probs],
            "disclaimer": self.DISCLAIMER,
            "success": True
        }
