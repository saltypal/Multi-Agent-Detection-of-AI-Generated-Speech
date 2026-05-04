"""
fusion/evaluation.py
Evaluation metrics: EER, AUC, Accuracy, F1, confusion matrix, classification report.
TODO: Full implementation after all agent models are trained.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc as sklearn_auc


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Equal Error Rate — the threshold where FAR == FRR."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    try:
        eer = brentq(lambda x: interp1d(fpr, fnr - fpr)(x), 0, 1)
    except Exception:
        eer = float(np.mean(np.abs(fnr - fpr)))
    return float(eer)


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area Under the ROC Curve."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return float(sklearn_auc(fpr, tpr))
