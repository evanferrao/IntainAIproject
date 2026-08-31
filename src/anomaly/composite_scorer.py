"""Composite Anomaly Scoring and Reviewer Exception Triage."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.utils.logger import logger


class CompositeAnomalyScorer:
    """Combines deterministic rule violations, ML Isolation Forest scores, statistical outliers,
    and source reconciliation conflicts into a unified calibrated anomaly index (0 to 100)."""

    def __init__(
        self,
        weight_deterministic: float = 0.40,
        weight_ml: float = 0.30,
        weight_reconciliation: float = 0.20,
        weight_outlier: float = 0.10
    ):
        self.w_det = weight_deterministic
        self.w_ml = weight_ml
        self.w_rec = weight_reconciliation
        self.w_out = weight_outlier

    def score(
        self,
        df: pd.DataFrame,
        ml_scores: Optional[np.ndarray] = None,
        deterministic_flags: Optional[pd.Series] = None,
        reconciliation_flags: Optional[pd.Series] = None,
        z_outlier_flags: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Calculates normalized composite anomaly scores and assigns reviewer triage actions."""
        scored_df = df.copy()
        n = len(scored_df)

        # 1. Deterministic component (0 or 100)
        det_score = np.where(deterministic_flags, 100.0, 0.0) if deterministic_flags is not None else np.zeros(n)

        # 2. ML anomaly score (0 to 100)
        ml_score = ml_scores if ml_scores is not None else np.zeros(n)

        # 3. Reconciliation conflict component (0 or 100)
        rec_score = np.where(reconciliation_flags, 100.0, 0.0) if reconciliation_flags is not None else np.zeros(n)

        # 4. Outlier component (0 or 100)
        out_score = np.where(z_outlier_flags, 80.0, 0.0) if z_outlier_flags is not None else np.zeros(n)

        # Weighted combination
        raw_composite = (
            self.w_det * det_score +
            self.w_ml * ml_score +
            self.w_rec * rec_score +
            self.w_out * out_score
        )

        composite_score = np.clip(raw_composite, 0.0, 100.0)
        scored_df["anomaly_score"] = np.round(composite_score, 1)

        # Assign Severity: LOW (<25), MEDIUM (25-50), HIGH (50-75), CRITICAL (>=75)
        scored_df["anomaly_severity"] = pd.cut(
            scored_df["anomaly_score"],
            bins=[-1.0, 25.0, 50.0, 75.0, 100.0],
            labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        ).astype(str)

        # Generate Actionable Reviewer Recommendations and Top Drivers
        actions = []
        top_drivers = []
        exception_types = []
        exception_probs = []

        for idx in range(n):
            score_val = composite_score[idx]
            is_det = det_score[idx] > 0
            is_rec = rec_score[idx] > 0
            is_ml = ml_score[idx] >= 65.0
            
            drivers = []
            if is_det:
                drivers.append("deterministic_rule_violation")
            if is_rec:
                drivers.append("source_reconciliation_conflict")
            if is_ml:
                drivers.append("multivariate_statistical_outlier")
            if not drivers:
                drivers.append("standard_performing_profile")

            top_drivers.append("; ".join(drivers))

            # Exception classification
            if is_rec and is_det:
                etype = "RULE_AND_RECON_CONFLICT"
                action = "MANUAL_AUDIT_REQUIRED"
            elif is_rec:
                etype = "SOURCE_RECONCILIATION_CONFLICT"
                action = "RECONCILE_SOURCE_CONFLICT"
            elif is_det:
                etype = "BUSINESS_RULE_VIOLATION"
                action = "MANUAL_AUDIT_REQUIRED"
            elif is_ml:
                etype = "STATISTICAL_ANOMALY"
                action = "FLAG_FOR_REVIEW"
            else:
                etype = "NONE"
                action = "AUTO_APPROVE"

            exception_types.append(etype)
            actions.append(action)
            exception_probs.append(round(score_val / 100.0, 4))

        scored_df["exception_type"] = exception_types
        scored_df["exception_probability"] = exception_probs
        scored_df["top_drivers"] = top_drivers
        scored_df["reviewer_action"] = actions

        logger.info(f"Composite anomaly scoring completed. Severity breakdown:\n{scored_df['anomaly_severity'].value_counts().to_dict()}")
        return scored_df

    def get_reviewer_dossiers(self, scored_df: pd.DataFrame, n_samples: int = 25) -> pd.DataFrame:
        """Extracts priority flagged exception dossiers for human loan reviewer audit."""
        flagged = scored_df[scored_df["anomaly_severity"].isin(["CRITICAL", "HIGH", "MEDIUM"])].copy()
        if len(flagged) == 0:
            return scored_df.head(n_samples)
        
        # Sort by anomaly score descending
        flagged = flagged.sort_values("anomaly_score", ascending=False)
        return flagged.head(n_samples).reset_index(drop=True)
