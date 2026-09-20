"""Deterministic Evidence-Driven Reviewer Triage Policy Engine."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.config.reviewer_policy import ReviewerPolicyConfig, DEFAULT_REVIEWER_POLICY_CONFIG
from src.utils.logger import logger


class ReviewerPolicyEngine:
    """Evaluates multi-dimensional model and data evidence to deterministically assign reviewer dispositions."""

    def __init__(self, config: Optional[ReviewerPolicyConfig] = None):
        self.config = config or DEFAULT_REVIEWER_POLICY_CONFIG

    def evaluate_loan(
        self,
        loan_id: str,
        default_prob: float = 0.0,
        delinquency_prob: float = 0.0,
        prepayment_prob: float = 0.0,
        anomaly_score: float = 0.0,
        is_anomaly: bool = False,
        data_quality_score: float = 100.0,
        has_deterministic_violation: bool = False,
        deterministic_violations: Optional[List[str]] = None,
        has_reconciliation_conflict: bool = False,
        reconciliation_conflicts: Optional[List[str]] = None,
        confidence: str = "HIGH",
        decision_margin: Optional[float] = None
    ) -> Dict[str, Any]:
        """Evaluates a single loan record and assigns an evidence-driven reviewer disposition."""
        reasons: List[str] = []
        det_violations = deterministic_violations or []
        recon_conflicts = reconciliation_conflicts or []

        # 1. Check Source Reconciliation Conflicts
        if has_reconciliation_conflict:
            conflict_desc = "; ".join(recon_conflicts) if recon_conflicts else "Servicer tape mismatch detected"
            reasons.append(f"Source tape conflict: {conflict_desc}")
            if has_deterministic_violation:
                reasons.append(f"Deterministic rule violation: {', '.join(det_violations)}")
                return {
                    "disposition": "MANUAL_AUDIT_REQUIRED",
                    "reasons": reasons,
                    "primary_trigger": "RECONCILIATION_AND_RULE_VIOLATION",
                    "evidence": self._build_evidence(
                        loan_id, default_prob, delinquency_prob, prepayment_prob,
                        anomaly_score, is_anomaly, data_quality_score,
                        has_deterministic_violation, has_reconciliation_conflict, confidence
                    )
                }
            return {
                "disposition": "RECONCILE_SOURCE_CONFLICT",
                "reasons": reasons,
                "primary_trigger": "SOURCE_RECONCILIATION_CONFLICT",
                "evidence": self._build_evidence(
                    loan_id, default_prob, delinquency_prob, prepayment_prob,
                    anomaly_score, is_anomaly, data_quality_score,
                    has_deterministic_violation, has_reconciliation_conflict, confidence
                )
            }

        # 2. Check Deterministic Rule Violations & Severe Quality / Anomaly
        if has_deterministic_violation:
            reasons.append(f"Deterministic rule violation: {', '.join(det_violations) if det_violations else 'Validation rule violated'}")
        if data_quality_score < self.config.data_quality_audit_threshold:
            reasons.append(f"Severe data quality deficiency: score {data_quality_score:.1f}/100 (<{self.config.data_quality_audit_threshold})")
        if anomaly_score >= self.config.anomaly_score_critical_threshold:
            reasons.append(f"Critical composite anomaly score: {anomaly_score:.1f}/100 (>={self.config.anomaly_score_critical_threshold})")

        if reasons:
            return {
                "disposition": "MANUAL_AUDIT_REQUIRED",
                "reasons": reasons,
                "primary_trigger": "DATA_INTEGRITY_OR_CRITICAL_ANOMALY",
                "evidence": self._build_evidence(
                    loan_id, default_prob, delinquency_prob, prepayment_prob,
                    anomaly_score, is_anomaly, data_quality_score,
                    has_deterministic_violation, has_reconciliation_conflict, confidence
                )
            }

        # 3. Check High Predicted Risk
        if default_prob >= self.config.high_risk_default_threshold:
            reasons.append(f"High predicted default risk: {default_prob:.1%} (>={self.config.high_risk_default_threshold:.1%})")
        if delinquency_prob >= self.config.high_risk_delinquency_threshold:
            reasons.append(f"High predicted delinquency hazard: {delinquency_prob:.1%} (>={self.config.high_risk_delinquency_threshold:.1%})")

        if reasons:
            return {
                "disposition": "HIGH_RISK_REVIEW",
                "reasons": reasons,
                "primary_trigger": "HIGH_PREDICTED_RISK",
                "evidence": self._build_evidence(
                    loan_id, default_prob, delinquency_prob, prepayment_prob,
                    anomaly_score, is_anomaly, data_quality_score,
                    has_deterministic_violation, has_reconciliation_conflict, confidence
                )
            }

        # 4. Check Moderate Risk, Statistical Anomalies, Borderline Quality / Confidence
        if default_prob >= self.config.moderate_risk_default_threshold:
            reasons.append(f"Moderate default risk: {default_prob:.1%} (>={self.config.moderate_risk_default_threshold:.1%})")
        if delinquency_prob >= self.config.moderate_risk_delinquency_threshold:
            reasons.append(f"Moderate delinquency hazard: {delinquency_prob:.1%} (>={self.config.moderate_risk_delinquency_threshold:.1%})")
        if is_anomaly or anomaly_score >= self.config.anomaly_score_flag_threshold:
            reasons.append(f"Multivariate statistical outlier: anomaly score {anomaly_score:.1f}/100")
        if data_quality_score < self.config.data_quality_flag_threshold:
            reasons.append(f"Moderate data quality score: {data_quality_score:.1f}/100 (<{self.config.data_quality_flag_threshold})")
        if confidence == "LOW":
            margin_str = f" (margin: {decision_margin:.2f})" if decision_margin is not None else ""
            reasons.append(f"Low model prediction confidence{margin_str}")

        if reasons:
            return {
                "disposition": "FLAG_FOR_REVIEW",
                "reasons": reasons,
                "primary_trigger": "MODERATE_RISK_OR_STATISTICAL_ANOMALY",
                "evidence": self._build_evidence(
                    loan_id, default_prob, delinquency_prob, prepayment_prob,
                    anomaly_score, is_anomaly, data_quality_score,
                    has_deterministic_violation, has_reconciliation_conflict, confidence
                )
            }

        # 5. Clean / Low-Risk Profile
        reasons.append(f"Clean data profile: quality score {data_quality_score:.1f}/100")
        reasons.append(f"Low default risk: {default_prob:.1%}")
        reasons.append("No material anomalies or servicer reconciliation conflicts detected")
        return {
            "disposition": "AUTO_APPROVE",
            "reasons": reasons,
            "primary_trigger": "CLEAN_LOW_RISK_PROFILE",
            "evidence": self._build_evidence(
                loan_id, default_prob, delinquency_prob, prepayment_prob,
                anomaly_score, is_anomaly, data_quality_score,
                has_deterministic_violation, has_reconciliation_conflict, confidence
            )
        }

    def _build_evidence(
        self,
        loan_id: str,
        default_prob: float,
        delinquency_prob: float,
        prepayment_prob: float,
        anomaly_score: float,
        is_anomaly: bool,
        data_quality_score: float,
        has_deterministic_violation: bool,
        has_reconciliation_conflict: bool,
        confidence: str
    ) -> Dict[str, Any]:
        return {
            "loan_id": loan_id,
            "default_probability": round(default_prob, 4),
            "delinquency_probability": round(delinquency_prob, 4),
            "prepayment_probability": round(prepayment_prob, 4),
            "anomaly_score": round(anomaly_score, 1),
            "is_statistical_anomaly": bool(is_anomaly),
            "data_quality_score": round(data_quality_score, 1),
            "has_deterministic_violation": bool(has_deterministic_violation),
            "has_reconciliation_conflict": bool(has_reconciliation_conflict),
            "model_confidence": confidence
        }

    def evaluate_batch(
        self,
        df: pd.DataFrame,
        default_probs: np.ndarray,
        delinquency_probs: np.ndarray,
        prepayment_probs: Optional[np.ndarray] = None,
        anomaly_scores: Optional[np.ndarray] = None,
        is_anomaly_flags: Optional[np.ndarray] = None,
        data_quality_scores: Optional[np.ndarray] = None,
        deterministic_flags: Optional[pd.Series] = None,
        deterministic_details: Optional[List[List[str]]] = None,
        reconciliation_flags: Optional[pd.Series] = None,
        reconciliation_details: Optional[List[List[str]]] = None,
        confidences: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Evaluates an entire batch of loans and assigns dispositions and structured reasons."""
        n = len(df)
        p_def = np.asarray(default_probs)
        p_del = np.asarray(delinquency_probs)
        p_prep = np.asarray(prepayment_probs) if prepayment_probs is not None else np.zeros(n)
        ano_scores = np.asarray(anomaly_scores) if anomaly_scores is not None else np.zeros(n)
        ano_flags = np.asarray(is_anomaly_flags) if is_anomaly_flags is not None else np.zeros(n, dtype=bool)
        dq_scores = np.asarray(data_quality_scores) if data_quality_scores is not None else np.full(n, 100.0)
        
        det_flags = deterministic_flags.to_numpy() if deterministic_flags is not None else np.zeros(n, dtype=bool)
        rec_flags = reconciliation_flags.to_numpy() if reconciliation_flags is not None else np.zeros(n, dtype=bool)
        confs = confidences.to_numpy() if confidences is not None else np.array(["HIGH"] * n)

        dispositions = []
        all_reasons = []
        primary_triggers = []

        loan_ids = df["loan_id"].astype(str).tolist() if "loan_id" in df.columns else [f"LOAN_{i}" for i in range(n)]

        for i in range(n):
            det_v = deterministic_details[i] if deterministic_details is not None and i < len(deterministic_details) else []
            rec_c = reconciliation_details[i] if reconciliation_details is not None and i < len(reconciliation_details) else []

            res = self.evaluate_loan(
                loan_id=loan_ids[i],
                default_prob=float(p_def[i]),
                delinquency_prob=float(p_del[i]),
                prepayment_prob=float(p_prep[i]),
                anomaly_score=float(ano_scores[i]),
                is_anomaly=bool(ano_flags[i]),
                data_quality_score=float(dq_scores[i]),
                has_deterministic_violation=bool(det_flags[i]),
                deterministic_violations=det_v,
                has_reconciliation_conflict=bool(rec_flags[i]),
                reconciliation_conflicts=rec_c,
                confidence=str(confs[i])
            )
            dispositions.append(res["disposition"])
            all_reasons.append("; ".join(res["reasons"]))
            primary_triggers.append(res["primary_trigger"])

        result_df = pd.DataFrame({
            "reviewer_action": dispositions,
            "reviewer_reasons": all_reasons,
            "reviewer_primary_trigger": primary_triggers
        }, index=df.index)

        logger.info(f"Evaluated reviewer dispositions across {n:,} loans:\n{result_df['reviewer_action'].value_counts().to_dict()}")
        return result_df
