"""Genuine Local LIME Explainability Engine."""

from typing import Dict, Any, List, Optional, Callable
import numpy as np
import pandas as pd
from lime import lime_tabular

from src.utils.logger import logger


class LimeExplainer:
    """Generates genuine local surrogate explanations using LimeTabularExplainer. Local computation only, no Groq calls."""

    def __init__(self, background_data: pd.DataFrame, class_names: Optional[List[str]] = None):
        self.feature_names = list(background_data.columns)
        self.class_names = class_names or ["Performing", "Adverse_Outcome"]
        
        logger.info(f"Initializing LimeTabularExplainer on {len(background_data)} background samples...")
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.asarray(background_data),
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode="classification",
            random_state=42
        )

    def explain_instance(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        transformed_row: pd.Series,
        num_features: int = 8,
        label: int = 1
    ) -> Dict[str, Any]:
        """Computes local LIME linear surrogate explanation for a single loan instance."""
        row_vals = transformed_row.to_numpy()

        exp = self.explainer.explain_instance(
            data_row=row_vals,
            predict_fn=predict_fn,
            num_features=num_features,
            labels=(label,)
        )

        raw_list = exp.as_list(label=label)
        
        attributions = []
        positive_contributors = []
        negative_contributors = []

        for rule_str, weight in raw_list:
            item = {
                "rule": rule_str,
                "weight": round(float(weight), 5),
                "abs_weight": abs(float(weight)),
                "direction": "INCREASES_RISK" if weight > 0 else "DECREASES_RISK"
            }
            attributions.append(item)
            if weight > 0:
                positive_contributors.append(item)
            else:
                negative_contributors.append(item)

        return {
            "num_features": num_features,
            "intercept": round(float(exp.intercept[label]), 5) if hasattr(exp, "intercept") and label in exp.intercept else 0.0,
            "attributions": attributions,
            "positive_contributors": positive_contributors,
            "negative_contributors": negative_contributors
        }
