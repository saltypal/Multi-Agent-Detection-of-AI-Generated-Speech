"""
fusion/explainability.py
Combines per-agent explanations into a final structured output.
"""

from __future__ import annotations
import json
from typing import Any


def generate_explanation(
    p_final: float,
    decision: str,
    agent_scores: dict[str, float],
    agent_explanations: dict[str, dict[str, Any]],
    threshold: float = 0.5,
) -> dict:
    """
    Build a comprehensive explanation dict.

    Parameters
    ----------
    p_final            : fused probability (0=bonafide, 1=spoof)
    decision           : "spoof" | "bonafide"
    agent_scores       : {agent: probability}
    agent_explanations : {agent: explanation_dict from each agent's explain_*()}

    Returns
    -------
    A structured dict suitable for JSON serialisation.

    Example output
    --------------
    {
      "decision": "spoof",
      "confidence": 0.82,
      "confidence_label": "High",
      "agent_scores": { "spectral": 0.91, ... },
      "explanation": {
        "spectral": {"score": 0.91, "explanation": "...", ...},
        ...
      },
      "summary": "The audio is likely SYNTHETIC. ..."
    }
    """
    confidence = p_final if decision == "spoof" else 1.0 - p_final

    if confidence >= 0.80:
        conf_label = "High"
    elif confidence >= 0.60:
        conf_label = "Moderate"
    else:
        conf_label = "Low"

    # Build human-readable summary
    contributing = []
    for agent, expl in agent_explanations.items():
        short = expl.get("explanation", "")
        score = agent_scores.get(agent, 0.5)
        if score >= 0.6:
            contributing.append(f"{agent.capitalize()} ({score:.2f}): {short}")

    if decision == "spoof":
        summary = (
            f"The audio is likely SYNTHETIC with {conf_label.lower()} confidence "
            f"(p={p_final:.2f}). "
        )
        if contributing:
            summary += "Contributing factors: " + " | ".join(contributing) + "."
    else:
        summary = (
            f"The audio appears GENUINE with {conf_label.lower()} confidence "
            f"(p={1-p_final:.2f}). No major deepfake indicators detected."
        )

    return {
        "decision":         decision,
        "confidence":       round(p_final, 4),
        "confidence_label": conf_label,
        "agent_scores":     {k: round(v, 4) for k, v in agent_scores.items()},
        "explanation":      agent_explanations,
        "summary":          summary,
    }


def print_explanation(result: dict):
    """Pretty-print the explanation dict to stdout."""
    print("\n" + "═" * 60)
    dec = result["decision"].upper()
    conf = result["confidence"]
    lbl = result["confidence_label"]
    print(f"  Decision   : {dec}  ({lbl} confidence,  p={conf:.4f})")
    print("─" * 60)
    for agent, score in result["agent_scores"].items():
        expl = result["explanation"].get(agent, {})
        text = expl.get("explanation", "N/A")
        print(f"  {agent:12s}: {score:.4f}  → {text}")
    print("─" * 60)
    print(f"  Summary: {result['summary']}")
    print("═" * 60)


def save_explanation(result: dict, path: str):
    """Save explanation to a JSON file."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Explain] Saved → {path}")
