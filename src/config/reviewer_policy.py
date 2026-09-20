"""Configuration and threshold definitions for Reviewer Triage Policy."""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ReviewerPolicyConfig:
    """Documented deterministic threshold configuration for loan reviewer triage."""

    # Default Risk Thresholds (12-Month Horizon)
    # Portfolio median is ~0.0%, 90th percentile is ~0.03%, top risks reach 12%
    high_risk_default_threshold: float = 0.05        # >= 5.0% triggers HIGH_RISK_REVIEW
    moderate_risk_default_threshold: float = 0.01    # >= 1.0% triggers FLAG_FOR_REVIEW

    # Delinquency Hazard Thresholds (3-Month Horizon)
    high_risk_delinquency_threshold: float = 0.02     # >= 2.0% triggers HIGH_RISK_REVIEW
    moderate_risk_delinquency_threshold: float = 0.005 # >= 0.5% triggers FLAG_FOR_REVIEW

    # Anomaly Index Thresholds (0 to 100)
    anomaly_score_critical_threshold: float = 75.0   # >= 75 triggers MANUAL_AUDIT_REQUIRED
    anomaly_score_flag_threshold: float = 45.0       # >= 45 triggers FLAG_FOR_REVIEW

    # Data Quality Score Thresholds (0 to 100)
    data_quality_audit_threshold: float = 70.0       # < 70 triggers MANUAL_AUDIT_REQUIRED
    data_quality_flag_threshold: float = 85.0        # < 85 triggers FLAG_FOR_REVIEW

    # Model Confidence / Borderline Margin Threshold
    low_confidence_margin_threshold: float = 0.15    # Margin |p - 0.5| < 0.15 triggers FLAG_FOR_REVIEW

    def to_dict(self) -> Dict[str, Any]:
        return {
            "high_risk_default_threshold": self.high_risk_default_threshold,
            "moderate_risk_default_threshold": self.moderate_risk_default_threshold,
            "high_risk_delinquency_threshold": self.high_risk_delinquency_threshold,
            "moderate_risk_delinquency_threshold": self.moderate_risk_delinquency_threshold,
            "anomaly_score_critical_threshold": self.anomaly_score_critical_threshold,
            "anomaly_score_flag_threshold": self.anomaly_score_flag_threshold,
            "data_quality_audit_threshold": self.data_quality_audit_threshold,
            "data_quality_flag_threshold": self.data_quality_flag_threshold,
            "low_confidence_margin_threshold": self.low_confidence_margin_threshold,
        }


DEFAULT_REVIEWER_POLICY_CONFIG = ReviewerPolicyConfig()
