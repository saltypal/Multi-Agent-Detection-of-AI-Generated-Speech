"""
linguistic/linguistic_explainability.py
Human-readable explanations for the Linguistic Agent.
"""

from __future__ import annotations
import numpy as np
from linguistic.linguistic_model import extract_text_features


def explain_linguistic(
    score: float,
    transcript: str,
    hand_features: np.ndarray | None = None,
) -> dict:
    """
    Parameters
    ----------
    score          : p_ling from LinguisticClassifier
    transcript     : ASR transcript string
    hand_features  : (5,) array from extract_text_features (optional)

    Returns a dict with detailed text-level metrics and an explanation string.
    """
    if hand_features is None:
        hand_features = extract_text_features(transcript)

    ppl_norm, rep_rate, disfluency_rate, inv_ttr, sent_len_norm = hand_features.tolist()
    ttr = 1.0 - inv_ttr

    parts = []
    if ppl_norm > 0.5:
        parts.append(f"high language model perplexity (score={ppl_norm:.2f})")
    if rep_rate > 0.3:
        parts.append(f"elevated bigram repetition ({rep_rate:.2f})")
    if disfluency_rate < 0.01:
        parts.append(f"unusually low disfluency rate ({disfluency_rate:.4f})")
    if ttr < 0.5:
        parts.append(f"low lexical diversity / type-token ratio ({ttr:.2f})")

    words = transcript.split()
    if score >= 0.5:
        verdict = "Suspicious: " + ("; ".join(parts) if parts else "model score is high")
    else:
        verdict = "Appears natural: no strong linguistic anomalies detected"

    return {
        "score": round(score, 4),
        "transcript_words": len(words),
        "perplexity_norm": round(ppl_norm, 4),
        "repetition_rate": round(rep_rate, 4),
        "disfluency_rate": round(disfluency_rate, 4),
        "type_token_ratio": round(ttr, 4),
        "explanation": verdict,
    }
