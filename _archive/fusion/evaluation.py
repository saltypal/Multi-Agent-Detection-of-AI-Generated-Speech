"""
fusion/evaluation.py
Evaluation metrics: EER, AUC, per-agent and system-level reporting.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc as sklearn_auc


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Equal Error Rate — the threshold where FAR == FRR.
    Labels: 0 = bonafide (genuine), 1 = spoof (fake).
    Scores: higher score = more likely spoof.
    Returns EER in [0, 1].
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    # Find EER via linear interpolation
    try:
        eer = brentq(lambda x: interp1d(fpr, fnr - fpr)(x), 0, 1)
    except Exception:
        eer = float(np.mean(np.abs(fnr - fpr)))
    return float(eer)


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area Under the ROC Curve."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return float(sklearn_auc(fpr, tpr))


def evaluate_agent(
    agent_name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Compute EER + AUC for a single agent.
    Returns dict with keys: agent, eer, auc.
    """
    eer = compute_eer(labels, scores)
    auc = compute_auc(labels, scores)
    if verbose:
        print(f"  [{agent_name:12s}]  EER: {eer*100:.2f}%   AUC: {auc:.4f}")
    return {"agent": agent_name, "eer": float(eer), "auc": float(auc)}


def evaluate_system(
    scores_dict: dict[str, np.ndarray],   # {agent_name: score_array}
    labels: np.ndarray,
    fusion_scores: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate all agents and optionally the fused score.

    Parameters
    ----------
    scores_dict    : {agent_name: (N,) score array}
    labels         : (N,) ground-truth labels (0=bonafide, 1=spoof)
    fusion_scores  : (N,) fused final score (optional)

    Returns dict of per-agent + system metrics.
    """
    results = {}
    if verbose:
        print("\n" + "═" * 50)
        print("  System Evaluation")
        print("═" * 50)

    for name, scores in scores_dict.items():
        results[name] = evaluate_agent(name, labels, scores, verbose=verbose)

    if fusion_scores is not None:
        results["fusion"] = evaluate_agent("Fusion", labels, fusion_scores, verbose=verbose)

    if verbose:
        print("═" * 50)

    return results
