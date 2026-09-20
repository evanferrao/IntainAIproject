"""Composite Anomaly Scoring and Reviewer Exception Triage."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.config.reviewer_policy import ReviewerPolicyConfig, DEFAULT_REVIEWER_POLICY_CONFIG
from src.anomaly.reviewer_policy import ReviewerPolicyEngine
from src.utils.logger import logger


class CompositeAnomalyScorer:
    """Combines deterministic rule violations, ML Isolation Forest scores, statistical outliers,
    and source reconciliation conflicts into a unified calibrated anomaly index (0 to 100)."""

    def __init__(
        self,
        weight_deterministic: float = 0.40,
        weight_ml: float = 0.30,
        weight_reconciliation: float = 0.20,
        weight_outlier: float = 0.10,
        policy_config: Optional[ReviewerPolicyConfig] = None
    ):
        self.w_det = weight_deterministic
        self.w_ml = weight_ml
        self.w_rec = weight_reconciliation
        self.w_out = weight_outlier
        self.policy_engine = ReviewerPolicyEngine(config=policy_config)

    def score(
        self,
        df: pd.DataFrame,
        ml_scores: Optional[np.ndarray] = None,
        deterministic_flags: Optional[pd.Series] = None,
        reconciliation_flags: Optional[pd.Series] = None,
        z_outlier_flags: Optional[pd.Series] = None,
        default_probs: Optional[np.ndarray] = None,
        delinquency_probs: Optional[np.ndarray] = None,
        prepayment_probs: Optional[np.ndarray] = None,
        is_anomaly_flags: Optional[np.ndarray] = None,
        data_quality_scores: Optional[np.ndarray] = None,
        confidences: Optional[pd.Series] = None,
        deterministic_details: Optional[List[List[str]]] = None,
        reconciliation_details: Optional[List[List[str]]] = None
    ) -> pd.DataFrame:
        """Calculates normalized composite anomaly scores and assigns evidence-driven reviewer triage actions."""
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

        # Use ReviewerPolicyEngine for evidence-driven disposition assignment
        p_def = default_probs if default_probs is not None else np.zeros(n)
        p_del = delinquency_probs if delinquency_probs is not None else np.zeros(n)
        p_prep = prepayment_probs if prepayment_probs is not None else np.zeros(n)
        is_ano = is_anomaly_flags if is_anomaly_flags is not None else (ml_score >= self.policy_engine.config.anomaly_score_flag_threshold)
        dq_scores = data_quality_scores if data_quality_scores is not None else np.full(n, 95.0)

        triage_df = self.policy_engine.evaluate_batch(
            df=scored_df,
            default_probs=p_def,
            delinquency_probs=p_del,
            prepayment_probs=p_prep,
            anomaly_scores=composite_score,
            is_anomaly_flags=is_ano,
            data_quality_scores=dq_scores,
            deterministic_flags=deterministic_flags,
            deterministic_details=deterministic_details,
            reconciliation_flags=reconciliation_flags,
            reconciliation_details=reconciliation_details,
            confidences=confidences
        )

        # Classify exception types and drivers
        top_drivers = []
        exception_types = []
        exception_probs = []

        for idx in range(n):
            score_val = composite_score[idx]
            is_det = det_score[idx] > 0
            is_rec = rec_score[idx] > 0
            is_ml = bool(is_ano[idx]) or ml_score[idx] >= self.policy_engine.config.anomaly_score_flag_threshold
            is_high_risk = float(p_def[idx]) >= self.policy_engine.config.high_risk_default_threshold

            drivers = []
            if is_det:
                drivers.append("deterministic_rule_violation")
            if is_rec:
                drivers.append("source_reconciliation_conflict")
            if is_high_risk:
                drivers.append("elevated_default_hazard")
            if is_ml:
                drivers.append("multivariate_statistical_outlier")
            if not drivers:
                drivers.append("standard_performing_profile")

            top_drivers.append("; ".join(drivers))

            # Exception classification
            if is_rec and is_det:
                etype = "RULE_AND_RECON_CONFLICT"
            elif is_rec:
                etype = "SOURCE_RECONCILIATION_CONFLICT"
            elif is_det:
                etype = "BUSINESS_RULE_VIOLATION"
            elif is_high_risk:
                etype = "HIGH_CREDIT_RISK"
            elif is_ml:
                etype = "STATISTICAL_ANOMALY"
            else:
                etype = "NONE"

            exception_types.append(etype)
            exception_probs.append(round(score_val / 100.0, 4))

        scored_df["exception_type"] = exception_types
        scored_df["exception_probability"] = exception_probs
        scored_df["top_drivers"] = top_drivers
        scored_df["reviewer_action"] = triage_df["reviewer_action"]
        scored_df["reviewer_reasons"] = triage_df["reviewer_reasons"]
        scored_df["reviewer_primary_trigger"] = triage_df["reviewer_primary_trigger"]

        logger.info(f"Composite anomaly scoring completed. Reviewer action breakdown:\n{scored_df['reviewer_action'].value_counts().to_dict()}")
        return scored_df

    def get_reviewer_dossiers(self, scored_df: pd.DataFrame, n_samples: int = 35) -> pd.DataFrame:
        """Extracts priority flagged exception dossiers for human loan reviewer audit."""
        # Include non-AUTO_APPROVE loans, or HIGH/CRITICAL anomaly severity
        flagged = scored_df[
            (scored_df["reviewer_action"] != "AUTO_APPROVE") |
            (scored_df["anomaly_severity"].isin(["CRITICAL", "HIGH"]))
        ].copy()
        if len(flagged) == 0:
            return scored_df.head(n_samples)
        
        # Sort by anomaly score descending, then default_probability if available
        sort_cols = ["anomaly_score"]
        if "default_probability" in flagged.columns:
            sort_cols.append("default_probability")
        flagged = flagged.sort_values(sort_cols, ascending=False)
        return flagged.head(n_samples).reset_index(drop=True)
