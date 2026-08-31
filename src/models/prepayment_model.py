"""Prepayment Prediction Modeling (Baseline & Improved Calibrated LightGBM)."""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

from src.config.settings import DEFAULT_CONFIG, RANDOM_SEED
from src.evaluation.metrics import evaluate_binary_classifier
from src.utils.logger import logger


class PrepaymentModel:
    """Predicts next 12-month early loan payoff / voluntary prepayment probability."""

    def __init__(self, use_calibration: bool = True, seed: int = RANDOM_SEED):
        self.use_calibration = use_calibration
        self.seed = seed
        self.baseline_model: Optional[LogisticRegression] = None
        self.improved_model: Optional[Any] = None
        self.feature_names: list = []

    def train_baseline(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        """Trains baseline regularized Logistic Regression."""
        logger.info(f"Training Baseline Prepayment Model (LogisticRegression) on {len(X_train):,} samples...")
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.baseline_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=200,
                random_state=self.seed,
                solver="lbfgs"
            )
        )
        self.baseline_model.fit(X_train, y_train)
        return self.baseline_model

    def train_improved(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Trains high-performance LightGBM classifier with probability calibration."""
        logger.info(f"Training Improved Prepayment Model (LightGBM) on {len(X_train):,} samples...")
        self.feature_names = list(X_train.columns)

        train_params = params or DEFAULT_CONFIG.lightgbm_params.copy()
        train_params["random_state"] = self.seed

        base_lgb = lgb.LGBMClassifier(**train_params)

        if self.use_calibration:
            logger.info("Applying Isotonic Probability Calibration to Prepayment Model...")
            self.improved_model = CalibratedClassifierCV(
                estimator=base_lgb,
                method=DEFAULT_CONFIG.calibrated_method,
                cv=3
            )
            self.improved_model.fit(X_train, y_train)
        else:
            base_lgb.fit(X_train, y_train)
            self.improved_model = base_lgb

        return self.improved_model

    def predict_proba(self, X: pd.DataFrame, model_type: str = "improved") -> np.ndarray:
        """Predicts calibrated prepayment probabilities."""
        model = self.improved_model if model_type == "improved" else self.baseline_model
        if model is None:
            raise RuntimeError(f"{model_type} model has not been trained yet.")
        probs = model.predict_proba(X)
        return probs[:, 1]

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.50
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Evaluates both baseline and improved models on out-of-time test set."""
        baseline_metrics = {}
        if self.baseline_model is not None:
            base_prob = self.predict_proba(X_test, model_type="baseline")
            baseline_metrics = evaluate_binary_classifier(
                y_test.to_numpy(), base_prob, threshold=threshold, model_name="Baseline_Prepayment_LogReg"
            )

        improved_metrics = {}
        if self.improved_model is not None:
            imp_prob = self.predict_proba(X_test, model_type="improved")
            improved_metrics = evaluate_binary_classifier(
                y_test.to_numpy(), imp_prob, threshold=threshold, model_name="Improved_Prepayment_LightGBM"
            )

        return baseline_metrics, improved_metrics
