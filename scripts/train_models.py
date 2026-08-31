"""Training script for all non-LLM machine learning models and transition matrices."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import (
    CANONICAL_DATA_DIR, MODELS_DIR, METRICS_DIR,
    DEFAULT_CONFIG, RANDOM_SEED
)
from src.features.feature_pipeline import FeatureEngineeringPipeline
from src.models.delinquency_model import DelinquencyModel
from src.models.default_model import DefaultModel
from src.models.prepayment_model import PrepaymentModel
from src.models.next_state_model import NextStateModel
from src.transitions.markov_model import MarkovTransitionModel
from src.anomaly.ml_anomaly import MLAnomalyDetector
from src.utils.serialization import save_model, save_json
from src.utils.logger import logger


def train_all_models():
    logger.info("================ STARTING MODEL TRAINING PIPELINE ================")

    # 1. Load canonical datasets
    static_df = pd.read_csv(CANONICAL_DATA_DIR / "loan_static_attributes.csv")
    train_panel = pd.read_csv(CANONICAL_DATA_DIR / "loan_monthly_performance_train.csv")
    test_panel = pd.read_csv(CANONICAL_DATA_DIR / "loan_monthly_performance_test.csv")

    logger.info(f"Loaded {len(static_df):,} static loans, {len(train_panel):,} train records, {len(test_panel):,} test records.")

    # 2. Merge static attributes with monthly panels
    train_full = train_panel.merge(static_df, on="loan_id", how="left")
    test_full = test_panel.merge(static_df, on="loan_id", how="left")

    # 3. Fit Feature Engineering Pipeline
    pipeline = FeatureEngineeringPipeline()
    pipeline.fit(train_full)
    save_model(pipeline, MODELS_DIR / "feature_pipeline.joblib")

    X_train = pipeline.transform(train_full)
    X_test = pipeline.transform(test_full)

    logger.info(f"Feature matrix shape: Train {X_train.shape}, Test {X_test.shape}")

    metrics_payload = {}

    # 4. Delinquency Model
    logger.info("--- Training Delinquency Prediction Models ---")
    y_delinq_train = train_full["next_3m_delinquency_flag"]
    y_delinq_test = test_full["next_3m_delinquency_flag"]
    
    delinq_model = DelinquencyModel(use_calibration=True, seed=RANDOM_SEED)
    delinq_model.train_baseline(X_train, y_delinq_train)
    delinq_model.train_improved(X_train, y_delinq_train)
    
    base_delinq_m, imp_delinq_m = delinq_model.evaluate(X_test, y_delinq_test)
    metrics_payload["delinquency"] = imp_delinq_m
    metrics_payload["delinquency_baseline"] = base_delinq_m
    save_model(delinq_model, MODELS_DIR / "delinquency_model.joblib")

    # 5. Default Model
    logger.info("--- Training Default Prediction Models ---")
    y_def_train = train_full["next_12m_default_flag"]
    y_def_test = test_full["next_12m_default_flag"]

    def_model = DefaultModel(use_calibration=True, seed=RANDOM_SEED)
    def_model.train_baseline(X_train, y_def_train)
    def_model.train_improved(X_train, y_def_train)

    base_def_m, imp_def_m = def_model.evaluate(X_test, y_def_test)
    metrics_payload["default"] = imp_def_m
    metrics_payload["default_baseline"] = base_def_m
    save_model(def_model, MODELS_DIR / "default_model.joblib")

    # 6. Prepayment Model
    logger.info("--- Training Prepayment Prediction Models ---")
    y_prep_train = train_full["next_12m_prepayment_flag"]
    y_prep_test = test_full["next_12m_prepayment_flag"]

    prep_model = PrepaymentModel(use_calibration=True, seed=RANDOM_SEED)
    prep_model.train_baseline(X_train, y_prep_train)
    prep_model.train_improved(X_train, y_prep_train)

    base_prep_m, imp_prep_m = prep_model.evaluate(X_test, y_prep_test)
    metrics_payload["prepayment"] = imp_prep_m
    metrics_payload["prepayment_baseline"] = base_prep_m
    save_model(prep_model, MODELS_DIR / "prepayment_model.joblib")

    # 7. Next-State Model
    logger.info("--- Training Next-State Multi-Class Models ---")
    y_ns_train = train_full["next_state"].fillna("CURRENT")
    y_ns_test = test_full["next_state"].fillna("CURRENT")

    ns_model = NextStateModel(seed=RANDOM_SEED)
    ns_model.train_baseline(X_train, y_ns_train)
    ns_model.train_improved(X_train, y_ns_train)

    base_ns_m, imp_ns_m = ns_model.evaluate(X_test, y_ns_test)
    metrics_payload["next_state"] = imp_ns_m
    metrics_payload["next_state_baseline"] = base_ns_m
    save_model(ns_model, MODELS_DIR / "next_state_model.joblib")

    # 8. Markov Transition Model
    logger.info("--- Fitting Markov Transition Model ---")
    markov = MarkovTransitionModel(seed=RANDOM_SEED)
    full_panel = pd.concat([train_full, test_full], ignore_index=True)
    markov.fit(full_panel, segment_cols=["credit_score_band", "dti_band"])
    save_model(markov, MODELS_DIR / "markov_transition_model.joblib")
    metrics_payload["transition_model"] = markov.to_dict()

    # 9. ML Anomaly Detector
    logger.info("--- Fitting ML Anomaly Detector (Isolation Forest) ---")
    iso_detector = MLAnomalyDetector(seed=RANDOM_SEED)
    iso_detector.fit(X_train)
    save_model(iso_detector, MODELS_DIR / "isolation_forest.joblib")

    # Persist all metrics to JSON
    metrics_out = METRICS_DIR / "model_metrics.json"
    save_json(metrics_payload, metrics_out)
    logger.info(f"Saved complete model metrics to {metrics_out}")

    logger.info("================ MODEL TRAINING COMPLETE ================")
    return metrics_payload


if __name__ == "__main__":
    train_all_models()
