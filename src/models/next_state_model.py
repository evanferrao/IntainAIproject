"""Next-State Multi-Class Prediction Modeling."""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from src.config.settings import VALID_STATUSES, RANDOM_SEED
from src.evaluation.metrics import evaluate_multiclass_classifier
from src.utils.logger import logger


class NextStateModel:
    """Predicts next monthly state (CURRENT, DELINQUENT_30/60/90, DEFAULT, PREPAID, CLOSED)."""

    def __init__(self, class_names: Optional[List[str]] = None, seed: int = RANDOM_SEED):
        self.class_names = class_names or VALID_STATUSES
        self.seed = seed
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.idx_to_class = {i: c for i, c in enumerate(self.class_names)}
        self.baseline_model: Optional[LogisticRegression] = None
        self.improved_model: Optional[lgb.LGBMClassifier] = None

    def _encode_target(self, y: pd.Series) -> np.ndarray:
        """Encodes target strings to zero-based class integers."""
        return np.array([self.class_to_idx.get(str(val), 0) for val in y])

    def train_baseline(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        """Trains baseline Multinomial Logistic Regression."""
        logger.info(f"Training Baseline Next-State Model (Multinomial LogReg) on {len(X_train):,} samples...")
        y_enc = self._encode_target(y_train)
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
        self.baseline_model.fit(X_train, y_enc)
        return self.baseline_model

    def train_improved(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None
    ) -> lgb.LGBMClassifier:
        """Trains high-performance Multiclass LightGBM classifier."""
        logger.info(f"Training Improved Next-State Model (Multiclass LightGBM) on {len(X_train):,} samples...")
        y_enc = self._encode_target(y_train)

        train_params = params or {
            "objective": "multiclass",
            "num_class": len(self.class_names),
            "n_estimators": 120,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "class_weight": "balanced",
            "random_state": self.seed,
            "n_jobs": -1,
            "verbose": -1
        }

        self.improved_model = lgb.LGBMClassifier(**train_params)
        self.improved_model.fit(X_train, y_enc)
        return self.improved_model

    def predict_proba(self, X: pd.DataFrame, model_type: str = "improved") -> np.ndarray:
        """Predicts probability distributions across all state classes."""
        model = self.improved_model if model_type == "improved" else self.baseline_model
        if model is None:
            raise RuntimeError(f"{model_type} model has not been trained yet.")
        raw_probs = model.predict_proba(X)
        
        # Ensure output has full number of classes
        full_probs = np.zeros((len(X), len(self.class_names)))
        classes = getattr(model, "classes_", np.arange(raw_probs.shape[1]))
        for idx, cls_idx in enumerate(classes):
            if cls_idx < len(self.class_names):
                full_probs[:, cls_idx] = raw_probs[:, idx]
        return full_probs

    def predict(self, X: pd.DataFrame, model_type: str = "improved") -> List[str]:
        """Predicts most likely next state class string."""
        probs = self.predict_proba(X, model_type=model_type)
        pred_indices = np.argmax(probs, axis=1)
        return [self.idx_to_class.get(idx, "CURRENT") for idx in pred_indices]

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Evaluates baseline and improved multiclass models on test set."""
        baseline_metrics = {}
        if self.baseline_model is not None:
            base_prob = self.predict_proba(X_test, model_type="baseline")
            baseline_metrics = evaluate_multiclass_classifier(
                y_test.to_numpy(), base_prob, self.class_names, model_name="Baseline_NextState_LogReg"
            )

        improved_metrics = {}
        if self.improved_model is not None:
            imp_prob = self.predict_proba(X_test, model_type="improved")
            improved_metrics = evaluate_multiclass_classifier(
                y_test.to_numpy(), imp_prob, self.class_names, model_name="Improved_NextState_LightGBM"
            )

        return baseline_metrics, improved_metrics
