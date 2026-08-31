"""Train / Test Drift Detection Engine (PSI and Kolmogorov-Smirnov Tests)."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import logger


class DriftDetector:
    """Calculates Population Stability Index (PSI), Kolmogorov-Smirnov (KS) statistic, and categorical divergence."""

    def __init__(self, psi_bins: int = 10, epsilon: float = 1e-4):
        self.psi_bins = psi_bins
        self.epsilon = epsilon

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray) -> float:
        """Calculates Population Stability Index (PSI) between reference and evaluation arrays."""
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]

        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Create quantile bins from expected baseline
        quantiles = np.linspace(0, 100, self.psi_bins + 1)
        try:
            bins = np.percentile(expected, quantiles)
            bins[0] = -np.inf
            bins[-1] = np.inf
            bins = np.unique(bins)
            if len(bins) < 3:
                # If too few unique bins, fallback to linear spacing
                bins = np.linspace(expected.min() - 1e-3, expected.max() + 1e-3, self.psi_bins + 1)
        except Exception:
            bins = np.linspace(-1e3, 1e3, self.psi_bins + 1)

        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)

        expected_pct = np.clip(expected_counts / len(expected), self.epsilon, 1.0)
        actual_pct = np.clip(actual_counts / len(actual), self.epsilon, 1.0)

        # Normalize back
        expected_pct = expected_pct / expected_pct.sum()
        actual_pct = actual_pct / actual_pct.sum()

        psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(max(0.0, psi_val))

    def evaluate_drift(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluates feature distribution drift between training and test sets."""
        cols = features or [c for c in train_df.columns if c in test_df.columns and c not in ["loan_id", "reporting_month"]]
        logger.info(f"Evaluating train/test drift across {len(cols)} candidate features...")

        drift_results: Dict[str, Any] = {}
        high_drift_features = []

        for col in cols:
            train_series = train_df[col]
            test_series = test_df[col]

            if pd.api.types.is_numeric_dtype(train_series) and pd.api.types.is_numeric_dtype(test_series):
                train_vals = train_series.dropna().to_numpy()
                test_vals = test_series.dropna().to_numpy()

                if len(train_vals) > 5 and len(test_vals) > 5:
                    psi_val = round(self.calculate_psi(train_vals, test_vals), 4)
                    ks_stat, ks_pval = stats.ks_2samp(train_vals, test_vals)
                    ks_stat = round(float(ks_stat), 4)
                    ks_pval = round(float(ks_pval), 6)

                    # PSI thresholds: <0.10: Stable, 0.10-0.25: Moderate Drift, >0.25: Significant Drift
                    drift_status = "STABLE" if psi_val < 0.10 else "MODERATE_DRIFT" if psi_val < 0.25 else "SIGNIFICANT_DRIFT"

                    drift_info = {
                        "type": "numeric",
                        "psi": psi_val,
                        "ks_statistic": ks_stat,
                        "ks_p_value": ks_pval,
                        "drift_status": drift_status,
                        "train_mean": round(float(train_vals.mean()), 4),
                        "test_mean": round(float(test_vals.mean()), 4),
                    }
                    drift_results[col] = drift_info
                    if psi_val >= 0.25:
                        high_drift_features.append(col)
            else:
                # Categorical drift: compute Total Variation Distance (TVD)
                train_p = train_series.value_counts(normalize=True)
                test_p = test_series.value_counts(normalize=True)
                all_cats = list(set(train_p.index).union(set(test_p.index)))
                
                tvd = 0.5 * sum(abs(train_p.get(cat, 0.0) - test_p.get(cat, 0.0)) for cat in all_cats)
                tvd = round(float(tvd), 4)
                drift_status = "STABLE" if tvd < 0.10 else "MODERATE_DRIFT" if tvd < 0.20 else "SIGNIFICANT_DRIFT"

                drift_info = {
                    "type": "categorical",
                    "total_variation_distance": tvd,
                    "drift_status": drift_status,
                    "train_top_cat": str(train_p.index[0]) if len(train_p) > 0 else None,
                    "test_top_cat": str(test_p.index[0]) if len(test_p) > 0 else None,
                }
                drift_results[col] = drift_info
                if tvd >= 0.20:
                    high_drift_features.append(col)

        summary = {
            "total_features_evaluated": len(cols),
            "high_drift_features_count": len(high_drift_features),
            "high_drift_features": high_drift_features,
            "overall_drift_status": "STABLE" if len(high_drift_features) == 0 else "WARNING" if len(high_drift_features) <= 2 else "ALERT",
            "feature_metrics": drift_results
        }
        return summary
