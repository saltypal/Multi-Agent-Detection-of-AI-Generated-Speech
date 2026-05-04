"""
fusion/decision_agent.py
Fusion of per-agent probabilities into a final decision.
Supports weighted average and logistic regression meta-classifier.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Literal

import numpy as np


class DecisionAgent:
    """
    Fuses scores from individual agents into a single probability.

    Parameters
    ----------
    mode    : "weighted" | "meta_classifier"
    weights : dict of agent_name -> weight (used in weighted mode)
    threshold : decision boundary for binary label (default 0.5)
    """

    AGENTS = ("spectral", "prosodic", "linguistic")

    DEFAULT_WEIGHTS = {
        "spectral":   0.35,
        "prosodic":   0.20,
        "linguistic": 0.45,
    }

    def __init__(
        self,
        mode: Literal["weighted", "meta_classifier"] = "weighted",
        weights: dict | None = None,
        threshold: float = 0.5,
    ):
        self.mode = mode
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.threshold = threshold
        self._clf = None   # scikit-learn logistic regression (meta_classifier mode)

    # ── Public API ────────────────────────────────────────────────────────────

    def fuse(self, scores: dict[str, float]) -> float:
        """
        Parameters
        ----------
        scores : {agent_name: probability}
        Returns p_final ∈ [0, 1]
        """
        if self.mode == "weighted":
            return self._weighted_fuse(scores)
        elif self.mode == "meta_classifier":
            if self._clf is None:
                raise RuntimeError("Meta-classifier not trained. Call train() first.")
            return self._meta_fuse(scores)
        else:
            raise ValueError(f"Unknown fusion mode: {self.mode}")

    def decide(self, scores: dict[str, float]) -> dict:
        """
        Returns a decision dict:
          {p_final, decision ("spoof"/"bonafide"), agent_scores}
        """
        p_final = self.fuse(scores)
        return {
            "p_final":      round(p_final, 4),
            "decision":     "spoof" if p_final >= self.threshold else "bonafide",
            "agent_scores": {k: round(v, 4) for k, v in scores.items()},
        }

    # ── Weighted fusion ───────────────────────────────────────────────────────

    def _weighted_fuse(self, scores: dict[str, float]) -> float:
        total_w, weighted_sum = 0.0, 0.0
        for agent, score in scores.items():
            w = self.weights.get(agent, 1.0 / len(self.AGENTS))
            weighted_sum += w * score
            total_w += w
        return float(weighted_sum / (total_w + 1e-9))

    # ── Meta-classifier fusion ────────────────────────────────────────────────

    def _meta_fuse(self, scores: dict[str, float]) -> float:
        vec = np.array([scores.get(a, 0.5) for a in self.AGENTS], dtype=np.float32)
        return float(self._clf.predict_proba(vec.reshape(1, -1))[0, 1])

    def train(
        self,
        scores_matrix: np.ndarray,   # (N, n_agents)
        labels: np.ndarray,          # (N,)
    ):
        """Train logistic regression meta-classifier."""
        from sklearn.linear_model import LogisticRegression
        self._clf = LogisticRegression(max_iter=500, C=1.0)
        self._clf.fit(scores_matrix, labels)
        print(f"[Fusion] Meta-classifier trained on {len(labels)} samples.")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        import pickle
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "mode": self.mode,
            "weights": self.weights,
            "threshold": self.threshold,
            "clf": self._clf,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[Fusion] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> "DecisionAgent":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(mode=data["mode"], weights=data["weights"], threshold=data["threshold"])
        agent._clf = data.get("clf")
        print(f"[Fusion] Loaded ← {path}")
        return agent
