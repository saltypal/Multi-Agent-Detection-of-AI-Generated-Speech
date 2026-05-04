"""
spectral/spectral_feature_extractor.py

Extracts ~127 tabular spectral features from a single audio file or waveform.
Returns a flat dictionary suitable for DataFrame construction and XGBoost training.

Features extracted:
  - MFCC (20 coeffs)       : mean, std, delta-mean, delta-std  → 80
  - Spectral descriptors   : centroid, bandwidth, rolloff, flatness  → 12
  - Spectral contrast (7)  : mean, std  → 14
  - Zero crossing rate     : mean, std  → 2
  - RMS energy             : mean, std, max  → 3
  - Chroma (12 bins)       : mean  → 12
  - Mel spectrogram stats  : mean, std, max, min  → 4

Total: ~127 features per audio file.

Designed for upgradability — can later be replaced with LSTM/CNN learned features.
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
    n_mfcc: int = 20,
    n_mels: int = 40,
    n_fft: int = 512,
    hop_length: int = 160,
) -> dict:
    """
    Extract tabular spectral features from one audio file or waveform.

    Parameters
    ----------
    path_or_waveform : str, Path, or np.ndarray
        Audio file path or pre-loaded waveform (1-D float32).
    sr : int
        Target sample rate.

    Returns
    -------
    dict : feature_name → float  (~127 entries)
    """
    # Load audio if path
    if isinstance(path_or_waveform, (str, Path)):
        y, _ = librosa.load(str(path_or_waveform), sr=sr, mono=True)
    else:
        y = np.asarray(path_or_waveform, dtype=np.float32)

    # Peak normalise
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak

    row = {}

    # ── 1. MFCC (20 coefficients) — mean, std ────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    for i in range(n_mfcc):
        row[f"mfcc{i}_mean"] = float(np.nan_to_num(mfcc[i].mean()))
        row[f"mfcc{i}_std"]  = float(np.nan_to_num(mfcc[i].std()))

    # Delta MFCC — mean, std
    dmfcc = librosa.feature.delta(mfcc)
    for i in range(n_mfcc):
        row[f"dmfcc{i}_mean"] = float(np.nan_to_num(dmfcc[i].mean()))
        row[f"dmfcc{i}_std"]  = float(np.nan_to_num(dmfcc[i].std()))

    # ── 2. Spectral descriptors ──────────────────────────────────────────────
    for name, fn in [
        ("centroid",  librosa.feature.spectral_centroid),
        ("bandwidth", librosa.feature.spectral_bandwidth),
        ("rolloff",   librosa.feature.spectral_rolloff),
    ]:
        feat = fn(y=y, sr=sr)[0]
        row[f"{name}_mean"] = float(np.nan_to_num(feat.mean()))
        row[f"{name}_std"]  = float(np.nan_to_num(feat.std()))
        row[f"{name}_max"]  = float(np.nan_to_num(feat.max()))

    # Flatness doesn't take 'sr'
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    row["flatness_mean"] = float(np.nan_to_num(flatness.mean()))
    row["flatness_std"]  = float(np.nan_to_num(flatness.std()))
    row["flatness_max"]  = float(np.nan_to_num(flatness.max()))

    # ── 3. Spectral contrast (7 bands) ──────────────────────────────────────
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(contrast.shape[0]):
        row[f"contrast{i}_mean"] = float(np.nan_to_num(contrast[i].mean()))
        row[f"contrast{i}_std"]  = float(np.nan_to_num(contrast[i].std()))

    # ── 4. Zero crossing rate ────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    row["zcr_mean"] = float(np.nan_to_num(zcr.mean()))
    row["zcr_std"]  = float(np.nan_to_num(zcr.std()))

    # ── 5. RMS energy ────────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y)[0]
    row["rms_mean"] = float(np.nan_to_num(rms.mean()))
    row["rms_std"]  = float(np.nan_to_num(rms.std()))
    row["rms_max"]  = float(np.nan_to_num(rms.max()))

    # ── 6. Chroma (12 bins) ──────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        row[f"chroma{i}_mean"] = float(np.nan_to_num(chroma[i].mean()))

    # ── 7. Mel spectrogram global stats ──────────────────────────────────────
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    row["mel_mean"] = float(np.nan_to_num(mel_db.mean()))
    row["mel_std"]  = float(np.nan_to_num(mel_db.std()))
    row["mel_max"]  = float(np.nan_to_num(mel_db.max()))
    row["mel_min"]  = float(np.nan_to_num(mel_db.min()))

    return row


def get_feature_names() -> list[str]:
    """Return ordered list of feature column names (matches extract_spectral_row keys)."""
    names = []
    for i in range(20):
        names += [f"mfcc{i}_mean", f"mfcc{i}_std"]
    for i in range(20):
        names += [f"dmfcc{i}_mean", f"dmfcc{i}_std"]
    for desc in ["centroid", "bandwidth", "rolloff", "flatness"]:
        names += [f"{desc}_mean", f"{desc}_std", f"{desc}_max"]
    for i in range(7):
        names += [f"contrast{i}_mean", f"contrast{i}_std"]
    names += ["zcr_mean", "zcr_std"]
    names += ["rms_mean", "rms_std", "rms_max"]
    for i in range(12):
        names.append(f"chroma{i}_mean")
    names += ["mel_mean", "mel_std", "mel_max", "mel_min"]
    return names
