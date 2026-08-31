"""Global Model Explainability Engine (Feature Importance & Permutation Importance)."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.utils.logger import logger


class GlobalExplainer:
    """Calculates model feature importances, permutation importances, and global risk attribution."""

    def __init__(self):
        pass

    def get_tree_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """Extracts native feature importance from tree-based estimators (LightGBM, XGBoost, etc.)."""
        # Handle CalibratedClassifierCV wrapper if present
        underlying = model
        if hasattr(model, "calibrated_classifiers_"):
            underlying = model.calibrated_classifiers_[0].estimator

        importances = None
        if hasattr(underlying, "feature_importances_"):
            importances = underlying.feature_importances_
        elif hasattr(underlying, "coef_"):
            importances = np.abs(underlying.coef_[0])

        if importances is None or len(importances) != len(feature_names):
            logger.warning(f"Could not extract native importances directly. Creating uniform fallback.")
            importances = np.ones(len(feature_names))

        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        })

        # Normalize to sum to 100%
        imp_df["importance_pct"] = np.round((imp_df["importance"] / max(1e-8, imp_df["importance"].sum())) * 100.0, 2)
        imp_df = imp_df.sort_values("importance_pct", ascending=False).reset_index(drop=True)
        return imp_df

    def compute_permutation_importance(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_repeats: int = 5,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Computes model-agnostic permutation feature importances."""
        logger.info(f"Computing permutation importance over {len(X_val):,} validation records...")
        r = permutation_importance(
            model, X_val, y_val,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="roc_auc" if len(np.unique(y_val)) == 2 else "f1_macro",
            n_jobs=-1
        )

        perm_df = pd.DataFrame({
            "feature": X_val.columns,
            "mean_importance_drop": np.round(r.importances_mean, 4),
            "std_importance_drop": np.round(r.importances_std, 4)
        }).sort_values("mean_importance_drop", ascending=False).reset_index(drop=True)

        return perm_df
