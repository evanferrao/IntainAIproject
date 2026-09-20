"""Unified End-to-End Orchestrator Pipeline for Loan Performance Intelligence Engine."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import (
    CANONICAL_DATA_DIR, MODELS_DIR, METRICS_DIR,
    PLOTS_DIR, REPORTS_DIR, PREDICTIONS_DIR, RANDOM_SEED
)
from src.ingestion.download import download_lending_club_data
from scripts.prepare_data import prepare_all_data
from scripts.train_models import train_all_models
from scripts.run_scenarios import run_all_stress_scenarios
from scripts.generate_submission import generate_submission
from src.profiling.profiler import DataProfiler
from src.profiling.drift import DriftDetector
from src.quality.quality_scorer import DataQualityScorer
from src.anomaly.rule_engine import DeterministicRuleEngine
from src.anomaly.reconciliation import SourceReconciliationEngine
from src.anomaly.composite_scorer import CompositeAnomalyScorer
from src.explainability.global_explainer import GlobalExplainer
from src.reporting.report_generator import ExecutiveReportGenerator
from src.utils.serialization import save_json, load_model
from src.utils.logger import logger


def run_full_pipeline():
    logger.info("######################################################################")
    logger.info("### LOAN PERFORMANCE INTELLIGENCE ENGINE — END-TO-END PIPELINE     ###")
    logger.info("######################################################################")

    # Step 1: Data Download & Acquisition
    logger.info("\n>>> STAGE 1: DATA ACQUISITION & CHECK")
    download_lending_club_data()

    # Step 2: Ingestion & Canonical Transformation
    logger.info("\n>>> STAGE 2: DATA PREPARATION & CANONICAL ADAPTATION")
    prepare_all_data()

    # Load canonical datasets
    static_df = pd.read_csv(CANONICAL_DATA_DIR / "loan_static_attributes.csv")
    train_panel = pd.read_csv(CANONICAL_DATA_DIR / "loan_monthly_performance_train.csv")
    test_panel = pd.read_csv(CANONICAL_DATA_DIR / "loan_monthly_performance_test.csv")
    servicer_df = pd.read_csv(CANONICAL_DATA_DIR / "servicer_updates.csv")

    # Step 3: Data Profiling & Quality Scoring
    logger.info("\n>>> STAGE 3: DATA PROFILING & QUALITY INTELLIGENCE")
    profiler = DataProfiler(seed=RANDOM_SEED)
    static_profile = profiler.profile_dataset(static_df, dataset_name="Canonical_Static_Loans")
    save_json(static_profile, METRICS_DIR / "data_profiling_summary.json")

    # Integrity check
    integrity_check = profiler.validate_dates_and_integrity(static_df, train_panel)
    save_json(integrity_check, METRICS_DIR / "date_and_integrity_audit.json")

    # Drift Analysis
    logger.info("\n>>> STAGE 4: TRAIN / TEST DRIFT DETECTION (PSI & KS-TEST)")
    drift_detector = DriftDetector()
    drift_summary = drift_detector.evaluate_drift(train_panel, test_panel)
    save_json(drift_summary, METRICS_DIR / "drift_metrics.json")

    # Quality Scoring
    scorer = DataQualityScorer()
    scored_records = scorer.score_records(static_df)
    batch_quality_summary = scorer.score_dataset(scored_records)
    save_json(batch_quality_summary, METRICS_DIR / "data_quality_score.json")

    # Step 4: Model Training & Calibration
    logger.info("\n>>> STAGE 5: MACHINE LEARNING MODEL TRAINING & CALIBRATION")
    model_metrics = train_all_models()

    # Step 5: Anomaly & Exception Detection
    logger.info("\n>>> STAGE 6: ANOMALY & EXCEPTION DETECTION TRIAGE")
    rule_engine = DeterministicRuleEngine()
    
    recon_engine = SourceReconciliationEngine()

    iso_model = load_model(MODELS_DIR / "isolation_forest.joblib")
    feature_pipeline = load_model(MODELS_DIR / "feature_pipeline.joblib")
    
    # Extract latest observations
    latest_panel = test_panel.sort_values("month_index").groupby("loan_id").last().reset_index()
    merged_eval = latest_panel.merge(static_df, on="loan_id", how="left")
    X_eval = feature_pipeline.transform(merged_eval)
    
    static_rules = rule_engine.evaluate_static_records(merged_eval)
    recon_df = recon_engine.reconcile(merged_eval, servicer_df)
    ml_scores, is_anomaly = iso_model.score_samples(X_eval)

    def_model = load_model(MODELS_DIR / "default_model.joblib")
    delinq_model = load_model(MODELS_DIR / "delinquency_model.joblib")
    prep_model = load_model(MODELS_DIR / "prepayment_model.joblib")
    p_def = def_model.predict_proba(X_eval)
    p_delinq = delinq_model.predict_proba(X_eval)
    p_prep = prep_model.predict_proba(X_eval)

    from src.explainability.uncertainty import ModelUncertaintyEstimator
    conf_df = ModelUncertaintyEstimator().assign_confidence(p_def)
    from src.quality.quality_scorer import DataQualityScorer
    dq_scores = DataQualityScorer().score_records(merged_eval)["record_quality_score"].to_numpy()

    comp_scorer = CompositeAnomalyScorer()
    scored_anomalies = comp_scorer.score(
        merged_eval,
        ml_scores=ml_scores,
        deterministic_flags=static_rules["has_deterministic_violation"],
        reconciliation_flags=recon_df["has_reconciliation_conflict"],
        default_probs=p_def,
        delinquency_probs=p_delinq,
        prepayment_probs=p_prep,
        is_anomaly_flags=is_anomaly,
        data_quality_scores=dq_scores,
        confidences=conf_df["confidence"]
    )
    
    dossiers = comp_scorer.get_reviewer_dossiers(scored_anomalies, n_samples=35)
    dossiers_out = PREDICTIONS_DIR / "reviewer_exception_dossiers.csv"
    dossiers.to_csv(dossiers_out, index=False)
    logger.info(f"Saved {len(dossiers)} reviewer exception dossiers to {dossiers_out}")

    anomaly_summary = {
        "rule_violations_count": int(static_rules["has_deterministic_violation"].sum()),
        "reconciliation_conflicts_count": int(recon_df["has_reconciliation_conflict"].sum()),
        "ml_outliers_count": int(is_anomaly.sum()),
        "critical_or_high_exceptions_count": int((scored_anomalies["anomaly_severity"].isin(["CRITICAL", "HIGH"])).sum())
    }
    save_json(anomaly_summary, METRICS_DIR / "anomaly_detection_summary.json")

    # Step 6: Macroeconomic Scenarios
    logger.info("\n>>> STAGE 7: MACROECONOMIC STRESS TESTING")
    scenario_results = run_all_stress_scenarios()

    # Step 7: Global & Local Explainability
    logger.info("\n>>> STAGE 8: GLOBAL & LOCAL EXPLAINABILITY")
    explainer = GlobalExplainer()
    def_model = load_model(MODELS_DIR / "default_model.joblib")
    feat_imp = explainer.get_tree_feature_importance(def_model.improved_model, feature_pipeline.get_feature_names())
    save_json(feat_imp.to_dict(orient="records"), METRICS_DIR / "feature_importances.json")

    # Step 8: Submission Generation
    logger.info("\n>>> STAGE 9: SUBMISSION FILE GENERATION")
    generate_submission()

    # Step 9: Executive Reporting
    logger.info("\n>>> STAGE 10: AUTOMATED EXECUTIVE REPORT GENERATION")
    report_gen = ExecutiveReportGenerator()
    report_file = report_gen.generate_full_report(
        profiling_metrics=static_profile,
        quality_summary=batch_quality_summary,
        drift_summary=drift_summary,
        model_metrics=model_metrics,
        scenario_results=scenario_results,
        anomaly_summary=anomaly_summary
    )

    logger.info("\n######################################################################")
    logger.info("### PIPELINE EXECUTION FINISHED SUCCESSFULLY!                     ###")
    logger.info(f"### Report generated: {report_file}                               ###")
    logger.info("######################################################################")


if __name__ == "__main__":
    run_full_pipeline()
