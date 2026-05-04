"""
spectral/spectral_feature_extractor.py

Extracts a minimal, strong set of tabular spectral features (~35 features).
Optimized for XGBoost deepfake detection by removing noise and redundant features.

Features extracted:
  - MFCC (13 coeffs)       : mean, std  → 26
  - Spectral descriptors   : centroid, rolloff, flatness (mean, std) → 6
  - RMS energy             : mean, std  → 2
  - Zero crossing rate     : mean  → 1

Total: 35 features per audio file.
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import librosa

warnings.filterwarnings("ignore", category=UserWarning)

SR = 16_000  # default sample rate


def extract_spectral_row(
    path_or_waveform,
    sr: int = SR,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 160,
) -> dict:
    """
    Extract minimal tabular spectral features.

    Parameters
    ----------
    path_or_waveform : str, Path, or np.ndarray
        Audio file path or pre-loaded waveform (1-D float32).
    sr : int
        Target sample rate.

    Returns
    -------
    dict : feature_name → float  (~35 entries)
    """
    if isinstance(path_or_waveform, (str, Path)):
        y, _ = librosa.load(str(path_or_waveform), sr=sr, mono=True)
    else:
        y = np.asarray(path_or_waveform, dtype=np.float32)

    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak

    row = {}

    # Pre-compute STFT to speed up librosa spectral features massively
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # ── 1. MFCC (13 coefficients) — mean, std ────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    for i in range(n_mfcc):
        row[f"mfcc{i}_mean"] = float(np.nan_to_num(mfcc[i].mean()))
        row[f"mfcc{i}_std"]  = float(np.nan_to_num(mfcc[i].std()))

    # ── 2. Spectral descriptors ──────────────────────────────────────────────
    for name, fn in [
        ("centroid",  librosa.feature.spectral_centroid),
        ("rolloff",   librosa.feature.spectral_rolloff),
    ]:
        feat = fn(S=S, sr=sr)[0]
        row[f"{name}_mean"] = float(np.nan_to_num(feat.mean()))
        row[f"{name}_std"]  = float(np.nan_to_num(feat.std()))

    # Flatness doesn't take 'sr'
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    row["flatness_mean"] = float(np.nan_to_num(flatness.mean()))
    row["flatness_std"]  = float(np.nan_to_num(flatness.std()))

    # ── 3. Zero crossing rate ────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    row["zcr_mean"] = float(np.nan_to_num(zcr.mean()))

    # ── 4. RMS energy ────────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y)[0]
    row["rms_mean"] = float(np.nan_to_num(rms.mean()))
    row["rms_std"]  = float(np.nan_to_num(rms.std()))

    return row


def get_feature_names() -> list[str]:
    """Return ordered list of feature column names."""
    names = []
    for i in range(13):
        names += [f"mfcc{i}_mean", f"mfcc{i}_std"]
    for desc in ["centroid", "rolloff", "flatness"]:
        names += [f"{desc}_mean", f"{desc}_std"]
    names += ["zcr_mean", "rms_mean", "rms_std"]
    return names
