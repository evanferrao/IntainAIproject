"""Model Uncertainty and Confidence Quantification Engine."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class ModelUncertaintyEstimator:
    """Estimates prediction confidence tiers, Shannon entropy, and out-of-distribution uncertainty."""

    def __init__(self, entropy_threshold: float = 0.55):
        self.entropy_threshold = entropy_threshold

    def calculate_binary_entropy(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculates normalized Shannon entropy for binary predictions: H(p) = -p log2(p) - (1-p) log2(1-p)."""
        p = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        entropy = - (p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
        return np.round(entropy, 4)

    def assign_confidence(self, probabilities: np.ndarray) -> pd.DataFrame:
        """Assigns confidence ratings: HIGH (p < 0.15 or p > 0.85), MODERATE, or LOW (borderline 0.35 to 0.65)."""
        probs = np.asarray(probabilities)
        entropy = self.calculate_binary_entropy(probs)
        
        # Distance from decision boundary (0.50)
        margin = np.abs(probs - 0.50)

        confidence_labels = []
        for p, m in zip(probs, margin):
            if m >= 0.30:  # p <= 0.20 or p >= 0.80
                confidence_labels.append("HIGH")
            elif m >= 0.15:  # 0.20 < p < 0.35 or 0.65 < p < 0.80
                confidence_labels.append("MODERATE")
            else:  # 0.35 <= p <= 0.65 (High uncertainty zone)
                confidence_labels.append("LOW")

        res_df = pd.DataFrame({
            "predicted_probability": np.round(probs, 4),
            "prediction_entropy": entropy,
            "decision_margin": np.round(margin, 4),
            "confidence": confidence_labels
        })
        return res_df
