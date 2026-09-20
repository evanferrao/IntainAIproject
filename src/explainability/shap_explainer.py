"""Genuine Local and Global SHAP Model Explainability Engine."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import shap

from src.utils.logger import logger


class ShapExplainer:
    """Computes genuine SHAP attributions using TreeExplainer on trained LightGBM models."""

    def __init__(self):
        self._explainers: Dict[str, Any] = {}

    def _get_underlying_tree_estimator(self, model: Any) -> Any:
        """Extracts the underlying tree estimator from CalibratedClassifierCV if wrapped."""
        if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
            return model.calibrated_classifiers_[0].estimator
        if hasattr(model, "improved_model"):
            return self._get_underlying_tree_estimator(model.improved_model)
        return model

    def _get_or_create_explainer(self, model_key: str, model: Any) -> shap.TreeExplainer:
        """Retrieves or creates a cached TreeExplainer for the specified model."""
        if model_key not in self._explainers:
            underlying = self._get_underlying_tree_estimator(model)
            logger.info(f"Initializing SHAP TreeExplainer for model: {model_key} ({type(underlying).__name__})")
            self._explainers[model_key] = shap.TreeExplainer(underlying)
        return self._explainers[model_key]

    def explain_instance(
        self,
        model_key: str,
        model: Any,
        transformed_row: pd.DataFrame,
        raw_row: Optional[pd.Series] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Computes local SHAP attributions for a single loan instance. Local computation only, no Groq calls."""
        explainer = self._get_or_create_explainer(model_key, model)
        feature_names = list(transformed_row.columns)

        # Compute SHAP values
        raw_shap = explainer.shap_values(transformed_row)
        
        # Handle binary classification output format (list of 2 arrays, or single array)
        if isinstance(raw_shap, list) and len(raw_shap) == 2:
            shap_vals = raw_shap[1][0]  # Positive class (e.g. Default / Delinquent)
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 2:
            shap_vals = raw_shap[0]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            shap_vals = raw_shap[0, :, 1]
        else:
            shap_vals = np.asarray(raw_shap).flatten()

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) == 2:
            base_val = float(expected_value[1])
        else:
            base_val = float(expected_value) if np.isscalar(expected_value) else float(expected_value[0])

        # Assemble feature attribution details
        rows = []
        for feat, val, s_val in zip(feature_names, transformed_row.iloc[0].values, shap_vals):
            rows.append({
                "feature": feat,
                "feature_value": round(float(val), 4),
                "shap_value": round(float(s_val), 5),
                "abs_shap": abs(float(s_val)),
                "direction": "INCREASES_RISK" if s_val > 0 else "DECREASES_RISK"
            })

        df_attributions = pd.DataFrame(rows).sort_values("abs_shap", ascending=False).reset_index(drop=True)

        top_drivers = df_attributions.head(top_k).to_dict(orient="records")
        top_positive = df_attributions[df_attributions["shap_value"] > 0].head(5).to_dict(orient="records")
        top_negative = df_attributions[df_attributions["shap_value"] < 0].head(5).to_dict(orient="records")

        return {
            "model_key": model_key,
            "base_value": round(base_val, 4),
            "top_drivers": top_drivers,
            "top_positive_risk_drivers": top_positive,
            "top_negative_risk_drivers": top_negative,
            "all_attributions": df_attributions.to_dict(orient="records")
        }

    def compute_global_summary(
        self,
        model_key: str,
        model: Any,
        X_sample: pd.DataFrame,
        max_features: int = 15
    ) -> pd.DataFrame:
        """Computes mean absolute SHAP values across a representative sample."""
        explainer = self._get_or_create_explainer(model_key, model)
        raw_shap = explainer.shap_values(X_sample)

        if isinstance(raw_shap, list) and len(raw_shap) == 2:
            shap_matrix = raw_shap[1]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 2:
            shap_matrix = raw_shap
        else:
            shap_matrix = np.asarray(raw_shap)

        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        summary_df = pd.DataFrame({
            "feature": X_sample.columns,
            "mean_abs_shap": np.round(mean_abs_shap, 5)
        }).sort_values("mean_abs_shap", ascending=False).head(max_features).reset_index(drop=True)

        return summary_df
