"""Unsupervised ML Anomaly Detection (Isolation Forest & Local Outlier Factor)."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from src.config.settings import DEFAULT_CONFIG, RANDOM_SEED
from src.utils.logger import logger


class MLAnomalyDetector:
    """Detects multi-variate statistical anomalies and atypical loan profiles using Isolation Forest."""

    def __init__(
        self,
        contamination: float = DEFAULT_CONFIG.contamination_rate,
        n_estimators: int = DEFAULT_CONFIG.n_isolation_forest_estimators,
        seed: int = RANDOM_SEED
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.seed = seed
        self.scaler = RobustScaler()
        self.iso_forest: Optional[IsolationForest] = None
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame) -> "MLAnomalyDetector":
        """Fits RobustScaler and IsolationForest on numerical feature space."""
        logger.info(f"Fitting IsolationForest on {len(X):,} samples across {len(X.columns)} features...")
        self.feature_names = list(X.columns)
        
        X_scaled = self.scaler.fit_transform(X)
        self.iso_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=-1
        )
        self.iso_forest.fit(X_scaled)
        logger.info("Fitted Isolation Forest anomaly detector.")
        return self

    def score_samples(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates normalized ML anomaly scores (0 to 100) and binary anomaly predictions (1 for anomaly, 0 normal)."""
        if self.iso_forest is None:
            raise RuntimeError("Anomaly detector must be fitted before scoring samples.")

        X_scaled = self.scaler.transform(X)
        
        # Raw decision function: lower / more negative means more anomalous
        raw_scores = self.iso_forest.decision_function(X_scaled)
        preds = self.iso_forest.predict(X_scaled)  # -1 for anomaly, 1 for normal
        is_anomaly = (preds == -1).astype(int)

        # Normalize score to 0..100 where 100 is highest anomaly risk
        # Map raw scores typically in range [-0.3, 0.2] to 0..100
        min_s, max_s = -0.25, 0.20
        norm_scores = np.clip((max_s - raw_scores) / (max_s - min_s) * 100.0, 0.0, 100.0)
        norm_scores = np.round(norm_scores, 1)

        return norm_scores, is_anomaly
