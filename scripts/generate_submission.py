"""Submission generation script with strict validation."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import (
    CANONICAL_DATA_DIR, MODELS_DIR, PREDICTIONS_DIR,
    VALID_STATUSES
)
from src.config.schema import SubmissionRecord
from src.anomaly.rule_engine import DeterministicRuleEngine
from src.anomaly.reconciliation import SourceReconciliationEngine
from src.anomaly.composite_scorer import CompositeAnomalyScorer
from src.explainability.uncertainty import ModelUncertaintyEstimator
from src.utils.serialization import load_model
from src.utils.logger import logger


def generate_submission():
    logger.info("================ STARTING SUBMISSION GENERATION ================")

    # 1. Load test data & master records
    static_df = pd.read_csv(CANONICAL_DATA_DIR / "loan_static_attributes.csv")
    test_panel = pd.read_csv(CANONICAL_DATA_DIR / "loan_monthly_performance_test.csv")
    servicer_df = pd.read_csv(CANONICAL_DATA_DIR / "servicer_updates.csv")

    # Take unique test loans (latest state per loan)
    latest_test = test_panel.sort_values("month_index").groupby("loan_id").last().reset_index()
    eval_df = latest_test.merge(static_df, on="loan_id", how="left")

    logger.info(f"Generating submission for {len(eval_df):,} out-of-time evaluation loans...")

    # 2. Load trained models
    feature_pipeline = load_model(MODELS_DIR / "feature_pipeline.joblib")
    delinq_model = load_model(MODELS_DIR / "delinquency_model.joblib")
    def_model = load_model(MODELS_DIR / "default_model.joblib")
    prep_model = load_model(MODELS_DIR / "prepayment_model.joblib")
    ns_model = load_model(MODELS_DIR / "next_state_model.joblib")
    iso_model = load_model(MODELS_DIR / "isolation_forest.joblib")

    # 3. Model Predictions
    X_eval = feature_pipeline.transform(eval_df)
    
    p_delinq = delinq_model.predict_proba(X_eval)
    p_def = def_model.predict_proba(X_eval)
    p_prep = prep_model.predict_proba(X_eval)
    pred_next_state = ns_model.predict(X_eval)

    # 4. Uncertainty & Confidence Assignment
    uncertainty_est = ModelUncertaintyEstimator()
    conf_df = uncertainty_est.assign_confidence(p_def)

    # 5. Data Quality Scoring
    from src.quality.quality_scorer import DataQualityScorer
    quality_scorer = DataQualityScorer()
    dq_records = quality_scorer.score_records(eval_df)
    dq_scores = dq_records["record_quality_score"].to_numpy()

    # 6. Anomaly Evaluation & Evidence-Driven Reviewer Triage
    rule_engine = DeterministicRuleEngine()
    static_evaluated = rule_engine.evaluate_static_records(eval_df)
    det_flags = static_evaluated["has_deterministic_violation"]

    recon_engine = SourceReconciliationEngine()
    recon_evaluated = recon_engine.reconcile(eval_df, servicer_df)
    recon_flags = recon_evaluated["has_reconciliation_conflict"]

    ml_scores, is_anomaly = iso_model.score_samples(X_eval)

    scorer = CompositeAnomalyScorer()
    scored_anomalies = scorer.score(
        eval_df,
        ml_scores=ml_scores,
        deterministic_flags=det_flags,
        reconciliation_flags=recon_flags,
        default_probs=p_def,
        delinquency_probs=p_delinq,
        prepayment_probs=p_prep,
        is_anomaly_flags=is_anomaly,
        data_quality_scores=dq_scores,
        confidences=conf_df["confidence"]
    )

    # 6. Assemble Submission DataFrame
    sub_df = pd.DataFrame({
        "loan_id": eval_df["loan_id"].astype(str),
        "delinquency_probability": np.round(p_delinq, 4),
        "default_probability": np.round(p_def, 4),
        "prepayment_probability": np.round(p_prep, 4),
        "next_state": pred_next_state,
        "exception_probability": scored_anomalies["exception_probability"],
        "exception_type": scored_anomalies["exception_type"],
        "anomaly_score": scored_anomalies["anomaly_score"],
        "top_drivers": scored_anomalies["top_drivers"],
        "reviewer_action": scored_anomalies["reviewer_action"],
        "confidence": conf_df["confidence"]
    })

    # 7. Strict Schema and Range Validation
    logger.info("Validating submission records against Pydantic schema...")
    required_cols = [
        "loan_id", "delinquency_probability", "default_probability",
        "prepayment_probability", "next_state", "exception_probability",
        "exception_type", "anomaly_score", "top_drivers", "reviewer_action", "confidence"
    ]
    assert list(sub_df.columns) == required_cols, "Submission column mismatch!"
    assert not sub_df["loan_id"].isna().any(), "Found missing loan_id values in submission!"
    assert (sub_df["delinquency_probability"] >= 0.0).all() and (sub_df["delinquency_probability"] <= 1.0).all()
    assert (sub_df["default_probability"] >= 0.0).all() and (sub_df["default_probability"] <= 1.0).all()
    assert (sub_df["prepayment_probability"] >= 0.0).all() and (sub_df["prepayment_probability"] <= 1.0).all()
    assert (sub_df["anomaly_score"] >= 0.0).all() and (sub_df["anomaly_score"] <= 100.0).all()

    # Validate each row via Pydantic model for first 100 records
    for row in sub_df.head(100).to_dict(orient="records"):
        SubmissionRecord(**row)

    out_file = PREDICTIONS_DIR / "submission.csv"
    sub_df.to_csv(out_file, index=False)
    logger.info(f"Successfully validated and saved {len(sub_df):,} submission records to {out_file}")
    logger.info("================ SUBMISSION GENERATION COMPLETE ================")
    return sub_df


if __name__ == "__main__":
    generate_submission()
