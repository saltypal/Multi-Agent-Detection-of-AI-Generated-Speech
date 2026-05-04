"""
spectral/spectral_explainability.py
Generates human-readable explanations for the Spectral Agent's decisions.
"""

from __future__ import annotations
import numpy as np


def explain_spectral(
    score: float,
    feature_matrix: np.ndarray,
    sr: int = 16_000,
    n_mels: int = 40,
) -> dict:
    """
    Analyse a stacked spectral feature matrix and return an explanation dict.

    Parameters
    ----------
    score          : p_spec output from SpectralModel (0=real, 1=fake)
    feature_matrix : (F, T) stacked feature array (Mel + MFCC + LFCC + CQCC)
    sr             : sample rate
    n_mels         : number of mel bands (used to isolate mel rows)

    Returns
    -------
    dict with keys: score, high_freq_energy_ratio, spectral_flatness,
                    harmonic_deviation, explanation (str)
    """
    mel = feature_matrix[:n_mels, :]         # mel rows

    # High-frequency energy ratio (top 25 % of mel bins vs total)
    hf_cutoff = int(n_mels * 0.75)
    total_energy = np.abs(mel).sum() + 1e-9
    hf_energy = np.abs(mel[hf_cutoff:, :]).sum()
    hf_ratio = float(hf_energy / total_energy)

    # Spectral flatness (Wiener entropy) — flatter → more noise-like
    power = mel ** 2 + 1e-9
    geo_mean = np.exp(np.mean(np.log(power)))
    arith_mean = np.mean(power)
    flatness = float(geo_mean / arith_mean)

    # Harmonic deviation — variance across time of each mel band
    harmonic_deviation = float(np.mean(np.std(mel, axis=1)))

    # Build human-readable explanation
    parts = []
    if hf_ratio > 0.35:
        parts.append(f"elevated high-frequency energy (ratio={hf_ratio:.2f})")
    if flatness > 0.5:
        parts.append(f"high spectral flatness ({flatness:.2f}) suggesting noise artifacts")
    if harmonic_deviation < 1.0:
        parts.append(f"unusually low harmonic variation ({harmonic_deviation:.2f})")
    if score >= 0.5:
        verdict = "Suspicious: " + ("; ".join(parts) if parts else "model score is high")
    else:
        verdict = "Appears natural: no strong spectral anomalies detected"

    return {
        "score": round(score, 4),
        "high_freq_energy_ratio": round(hf_ratio, 4),
        "spectral_flatness": round(flatness, 4),
        "harmonic_deviation": round(harmonic_deviation, 4),
        "explanation": verdict,
    }
