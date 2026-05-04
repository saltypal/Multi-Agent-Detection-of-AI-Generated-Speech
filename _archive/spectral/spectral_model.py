"""
spectral/spectral_model.py
CNN + Bi-LSTM + Attention architecture for spectral deepfake detection.
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU → MaxPool2d."""

    def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int] = (2, 2)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x):
        return self.net(x)


class AttentionPooling(nn.Module):
    """Soft attention over time steps → fixed-size context vector."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        weights = torch.softmax(self.attention(x), dim=1)   # (B, T, 1)
        return (weights * x).sum(dim=1)                      # (B, H)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class SpectralModel(nn.Module):
    """
    Spectral Agent model.

    Input  : (B, n_features, T)  — stacked spectral feature map
    Output : (B,)                — probability of spoof ∈ [0, 1]

    Architecture
    ------------
    Treat (n_features, T) as a 1-channel 2-D image → 3× ConvBlock
    → reshape → Bi-LSTM × 2 → AttentionPooling → FC head.
    """

    def __init__(
        self,
        n_features: int = 100,   # Mel(40) + MFCC(20) + LFCC(20) + CQCC(20)
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_features = n_features

        # CNN encoder
        self.cnn = nn.Sequential(
            ConvBlock(1, 32, pool=(2, 2)),
            ConvBlock(32, 64, pool=(2, 2)),
            ConvBlock(64, 128, pool=(2, 2)),
        )
        # After 3× MaxPool(2,2) the feature dim is reduced by 8
        cnn_feat_dim = max(1, n_features // 8) * 128

        # Project CNN output → LSTM input size
        self.proj = nn.Linear(cnn_feat_dim, 128)

        # Bi-LSTM
        self.lstm = nn.LSTM(
            128, lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out = lstm_hidden * 2  # bidirectional

        # Attention + classifier
        self.attn = AttentionPooling(lstm_out)
        self.head = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, F, T)
        returns: (B,) logit-sigmoid probabilities
        """
        B, F, T = x.shape
        x = x.unsqueeze(1)                    # (B, 1, F, T)
        x = self.cnn(x)                        # (B, 128, F', T')
        B2, C, F2, T2 = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B2, T2, C * F2)  # (B, T', C*F')
        x = F.relu(self.proj(x))              # (B, T', 128)
        x, _ = self.lstm(x)                   # (B, T', 2*H)
        x = self.attn(x)                      # (B, 2*H)
        x = self.head(x).squeeze(-1)          # (B,)
        return torch.sigmoid(x)

    # ── inference helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, feature_chunks: list) -> float:
        """
        Average probability over a list of chunk feature tensors.
        Each chunk: np.ndarray (F, T) or torch.Tensor (F, T).
        """
        self.eval()
        probs = []
        for chunk in feature_chunks:
            if not isinstance(chunk, torch.Tensor):
                chunk = torch.from_numpy(chunk)
            chunk = chunk.unsqueeze(0).to(next(self.parameters()).device)
            probs.append(self.forward(chunk).item())
        return float(sum(probs) / len(probs)) if probs else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_trained_model(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    **model_kwargs,
) -> SpectralModel:
    """Load SpectralModel from a checkpoint file."""
    state = torch.load(checkpoint_path, map_location=device)
    model = SpectralModel(**model_kwargs)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[SpectralModel] Loaded from {checkpoint_path}  (epoch {state.get('epoch', '?')})")
    return model
