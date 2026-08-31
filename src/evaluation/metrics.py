"""Evaluation metrics calculation for binary and multiclass models."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, brier_score_loss, confusion_matrix,
    classification_report, precision_recall_curve
)
from sklearn.calibration import calibration_curve

from src.utils.logger import logger


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    model_name: str = "BinaryModel"
) -> Dict[str, Any]:
    """Calculates full suite of classification and calibration metrics for binary models."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # Check for single-class edge cases
    if len(np.unique(y_true)) < 2:
        logger.warning(f"Single class present in y_true for {model_name}. Returning basic metrics.")
        return {
            "model_name": model_name,
            "accuracy": float((y_true == y_pred).mean()),
            "brier_score": float(brier_score_loss(y_true, y_prob))
        }

    # Discrimination metrics
    roc_auc = round(float(roc_auc_score(y_true, y_prob)), 4)
    pr_auc = round(float(average_precision_score(y_true, y_prob)), 4)
    prec = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
    rec = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
    brier = round(float(brier_score_loss(y_true, y_prob)), 4)

    # Recall at fixed precision (e.g. >= 0.80 or >= 0.90)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    idx_80 = np.where(precisions >= 0.80)[0]
    recall_at_80_precision = round(float(recalls[idx_80[0]]), 4) if len(idx_80) > 0 else 0.0

    # Calibration Curve (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    # Expected Calibration Error (ECE)
    ece = round(float(np.mean(np.abs(prob_true - prob_pred))), 4) if len(prob_true) > 0 else 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "model_name": model_name,
        "n_samples": len(y_true),
        "positive_class_ratio": round(float(y_true.mean()), 4),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "recall_at_80_precision": recall_at_80_precision,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "confusion_matrix": cm,
        "decision_threshold": threshold
    }
    
    logger.info(f"Evaluated {model_name} -> ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, F1: {f1:.4f}, Brier: {brier:.4f}")
    return metrics


def evaluate_multiclass_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    model_name: str = "MulticlassModel"
) -> Dict[str, Any]:
    """Calculates multiclass evaluation metrics, macro F1, and per-class performance."""
    y_pred_idx = np.argmax(y_prob, axis=1)
    
    # Map classes if string
    if isinstance(y_true[0], str):
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        y_true_idx = np.array([class_to_idx.get(s, 0) for s in y_true])
    else:
        y_true_idx = np.asarray(y_true).astype(int)

    macro_f1 = round(float(f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)), 4)
    weighted_f1 = round(float(f1_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0)), 4)
    macro_precision = round(float(precision_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)), 4)
    macro_recall = round(float(recall_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)), 4)

    # Per-class metrics
    per_class_f1 = f1_score(y_true_idx, y_pred_idx, average=None, zero_division=0)
    per_class_prec = precision_score(y_true_idx, y_pred_idx, average=None, zero_division=0)
    per_class_rec = recall_score(y_true_idx, y_pred_idx, average=None, zero_division=0)

    class_metrics = {}
    for idx, cname in enumerate(class_names):
        if idx < len(per_class_f1):
            class_metrics[cname] = {
                "f1": round(float(per_class_f1[idx]), 4),
                "precision": round(float(per_class_prec[idx]), 4),
                "recall": round(float(per_class_rec[idx]), 4)
            }

    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(class_names)))).tolist()

    metrics = {
        "model_name": model_name,
        "n_samples": len(y_true_idx),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "class_breakdown": class_metrics,
        "confusion_matrix": cm,
        "class_names": class_names
    }

    logger.info(f"Evaluated {model_name} -> Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
    return metrics
