"""
prosodic/prosodic_explainability.py
Human-readable explanations for the Prosodic Agent.
"""

from __future__ import annotations
import numpy as np
from preprocessor.feature_extraction import compute_jitter, compute_shimmer, compute_f0


def explain_prosodic(
    score: float,
    chunk: np.ndarray,
    sr: int = 16_000,
) -> dict:
    """
    Parameters
    ----------
    score : p_pros from ProsodicModel
    chunk : raw audio chunk (1-D float32)
    sr    : sample rate

    Returns dict with jitter, shimmer, F0 stats, and a text explanation.
    """
    f0 = compute_f0(chunk, sr=sr)
    voiced = f0[f0 > 0]

    jitter = compute_jitter(f0)
    shimmer = compute_shimmer(chunk, sr=sr)
    f0_mean = float(voiced.mean()) if len(voiced) > 0 else 0.0
    f0_std  = float(voiced.std())  if len(voiced) > 0 else 0.0
    f0_cv   = f0_std / (f0_mean + 1e-9)   # coefficient of variation

    voiced_ratio = float(len(voiced) / (len(f0) + 1e-9))

    parts = []
    if jitter < 0.01:
        parts.append(f"very low jitter ({jitter:.4f}) — unnatural pitch regularity")
    if shimmer < 0.01:
        parts.append(f"very low shimmer ({shimmer:.4f}) — unnatural amplitude regularity")
    if f0_cv < 0.05:
        parts.append(f"low F0 coefficient of variation ({f0_cv:.3f})")
    if voiced_ratio < 0.3:
        parts.append(f"low voiced ratio ({voiced_ratio:.2f}) — possible synthesis artifact")

    if score >= 0.5:
        verdict = "Suspicious: " + ("; ".join(parts) if parts else "model score is high")
    else:
        verdict = "Appears natural prosodic behaviour"

    return {
        "score": round(score, 4),
        "jitter": round(jitter, 6),
        "shimmer": round(shimmer, 6),
        "f0_mean_hz": round(f0_mean, 2),
        "f0_cv": round(f0_cv, 4),
        "voiced_ratio": round(voiced_ratio, 4),
        "explanation": verdict,
    }
