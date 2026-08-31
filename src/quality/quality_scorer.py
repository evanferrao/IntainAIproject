"""Data Quality Scoring Engine for Record-Level and Batch-Level Integrity."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.utils.logger import logger


class DataQualityScorer:
    """Computes transparent record-level (0-100) and dataset-level (0-100) quality scores."""

    def __init__(
        self,
        missingness_weight: float = 0.30,
        outlier_weight: float = 0.20,
        rule_violation_weight: float = 0.35,
        reconciliation_conflict_weight: float = 0.15
    ):
        self.w_missing = missingness_weight
        self.w_outlier = outlier_weight
        self.w_rule = rule_violation_weight
        self.w_recon = reconciliation_conflict_weight

    def score_records(
        self,
        df: pd.DataFrame,
        rule_violations_mask: Optional[pd.Series] = None,
        reconciliation_conflicts_mask: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Calculates record-level quality score between 0 and 100 for each row."""
        scored_df = df.copy()
        n_cols = len(df.columns)

        # 1. Missingness penalty (0 to 100 penalty)
        missing_count = scored_df.isna().sum(axis=1)
        missing_pct = missing_count / max(1, n_cols)
        missing_penalty = np.clip(missing_pct * 100.0 * 2.5, 0.0, 100.0)

        # 2. Outlier penalty based on numerical z-scores
        num_cols = scored_df.select_dtypes(include=[np.number]).columns
        outlier_penalty = np.zeros(len(scored_df))
        if len(num_cols) > 0:
            z_scores = np.abs((scored_df[num_cols] - scored_df[num_cols].mean()) / scored_df[num_cols].std().replace(0, 1))
            outlier_counts = (z_scores > 3.5).sum(axis=1)
            outlier_penalty = np.clip(outlier_counts * 15.0, 0.0, 100.0)

        # 3. Rule violation penalty
        rule_penalty = np.zeros(len(scored_df))
        if rule_violations_mask is not None:
            rule_penalty = np.where(rule_violations_mask, 50.0, 0.0)

        # 4. Reconciliation conflict penalty
        recon_penalty = np.zeros(len(scored_df))
        if reconciliation_conflicts_mask is not None:
            recon_penalty = np.where(reconciliation_conflicts_mask, 40.0, 0.0)

        # Combined penalty calculation
        total_penalty = (
            self.w_missing * missing_penalty +
            self.w_outlier * outlier_penalty +
            self.w_rule * rule_penalty +
            self.w_recon * recon_penalty
        )

        quality_score = np.clip(100.0 - total_penalty, 0.0, 100.0)
        scored_df["record_quality_score"] = np.round(quality_score, 1)

        # Quality Grade classification
        scored_df["quality_grade"] = pd.cut(
            scored_df["record_quality_score"],
            bins=[-1, 50, 75, 90, 100],
            labels=["POOR", "FAIR", "GOOD", "EXCELLENT"]
        ).astype(str)

        return scored_df

    def score_dataset(self, scored_df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregates record scores into a comprehensive dataset/batch quality report."""
        if "record_quality_score" not in scored_df.columns:
            scored_df = self.score_records(scored_df)

        scores = scored_df["record_quality_score"]
        mean_score = round(float(scores.mean()), 2)
        median_score = round(float(scores.median()), 2)
        p10 = round(float(np.percentile(scores, 10)), 2)
        p90 = round(float(np.percentile(scores, 90)), 2)

        grade_distribution = scored_df["quality_grade"].value_counts(normalize=True).round(4).to_dict()

        batch_quality_status = (
            "EXCELLENT" if mean_score >= 90.0
            else "GOOD" if mean_score >= 80.0
            else "NEEDS_REVIEW" if mean_score >= 65.0
            else "CRITICAL_ISSUES"
        )

        summary = {
            "batch_quality_score": mean_score,
            "median_quality_score": median_score,
            "p10_quality_score": p10,
            "p90_quality_score": p90,
            "quality_status": batch_quality_status,
            "grade_breakdown_pct": {k: round(v * 100.0, 2) for k, v in grade_distribution.items()},
            "records_evaluated": len(scored_df),
            "unfit_records_count": int((scores < 60.0).sum())
        }
        
        logger.info(f"Dataset Quality Evaluation: Batch Score {mean_score}/100 ({batch_quality_status})")
        return summary
