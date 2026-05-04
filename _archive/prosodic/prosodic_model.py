"""
prosodic/prosodic_model.py
CNN + Bi-LSTM + Attention model for prosodic deepfake detection.
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv1d → BatchNorm1d → ReLU → MaxPool1d."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        )

    def forward(self, x):
        return self.net(x)


class AttentionPooling(nn.Module):
    """Soft attention over time steps → fixed-size context vector."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.attn(x), dim=1)
        return (w * x).sum(dim=1)


class ProsodicModel(nn.Module):
    """
    Prosodic Agent — CNN + Bi-LSTM + Attention.

    Input  : (B, n_features, T)  — prosodic feature sequence (5 channels)
    Output : (B,)                — probability of spoof ∈ [0, 1]

    Architecture
    ------------
    Conv1D blocks extract local patterns from 5-channel prosodic features,
    then Bi-LSTM captures temporal dynamics, attention pools across time.
    """

    def __init__(
        self,
        n_features: int = 5,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # CNN encoder (1-D convolutions over time)
        self.cnn = nn.Sequential(
            ConvBlock(n_features, 32, kernel=5, pool=2),
            ConvBlock(32, 64, kernel=3, pool=2),
            ConvBlock(64, 128, kernel=3, pool=2),
        )

        # Bi-LSTM
        self.lstm = nn.LSTM(
            128, lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out = lstm_hidden * 2

        # Attention + classifier head
        self.attn = AttentionPooling(lstm_out)
        self.head = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T)
        x = self.cnn(x)                    # (B, 128, T')
        x = x.permute(0, 2, 1)             # (B, T', 128)
        x, _ = self.lstm(x)                # (B, T', 2*H)
        x = self.attn(x)                   # (B, 2*H)
        x = self.head(x).squeeze(-1)       # (B,)
        return torch.sigmoid(x)

    @torch.no_grad()
    def predict(self, feature_chunks: list) -> float:
        self.eval()
        probs = []
        for chunk in feature_chunks:
            if not isinstance(chunk, torch.Tensor):
                import numpy as np
                chunk = torch.from_numpy(chunk.astype("float32"))
            chunk = chunk.unsqueeze(0).to(next(self.parameters()).device)
            probs.append(self.forward(chunk).item())
        return float(sum(probs) / len(probs)) if probs else 0.5


def load_trained_model(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    **model_kwargs,
) -> ProsodicModel:
    state = torch.load(checkpoint_path, map_location=device)
    model = ProsodicModel(**model_kwargs)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[ProsodicModel] Loaded from {checkpoint_path}  (epoch {state.get('epoch', '?')})")
    return model
