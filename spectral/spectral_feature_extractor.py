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
# pyrefly: ignore [missing-import]
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

try:
    import torch
    import torchaudio
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def extract_spectral_batch(waveforms: np.ndarray | torch.Tensor, sr: int = SR, n_mfcc: int = 13, n_fft: int = 512, hop_length: int = 160, device: str = 'cuda') -> list[dict]:
    """
    GPU-accelerated extraction of spectral features for a batch of audio.
    waveforms: shape (batch, frames). Padded to the same length.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch and torchaudio are required for batch extraction.")
        
    if not isinstance(waveforms, torch.Tensor):
        waveforms = torch.tensor(waveforms, dtype=torch.float32)
    waveforms = waveforms.to(device)
    batch_size = waveforms.size(0)
    
    # Normalize peak per waveform
    peaks = torch.abs(waveforms).max(dim=1, keepdim=True).values
    peaks[peaks == 0] = 1.0
    waveforms = waveforms / peaks
    
    # ── Pre-compute STFT ──
    # PyTorch stft: return_complex=True -> abs
    window = torch.hann_window(n_fft).to(device)
    stft_complex = torch.stft(waveforms, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, center=True)
    S = torch.abs(stft_complex) + 1e-9 # Shape: (batch, freq_bins, time_frames)
    
    # ── 1. MFCC ──
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr, n_mfcc=n_mfcc,
        melkwargs={'n_fft': n_fft, 'hop_length': hop_length, 'center': True}
    ).to(device)
    mfcc = mfcc_transform(waveforms) # (batch, n_mfcc, time)
    mfcc_mean = mfcc.mean(dim=-1).cpu().numpy()
    mfcc_std = mfcc.std(dim=-1).cpu().numpy()
    
    # ── 2. Spectral Descriptors ──
    freqs = torch.linspace(0, sr / 2, S.size(1), device=device).view(1, -1, 1)
    # Centroid
    centroid = torch.sum(freqs * S, dim=1) / torch.sum(S, dim=1)
    centroid_mean = centroid.mean(dim=-1).cpu().numpy()
    centroid_std = centroid.std(dim=-1).cpu().numpy()
    
    # Rolloff (85%)
    cumsum = torch.cumsum(S, dim=1)
    target = 0.85 * cumsum[:, -1:, :]
    rolloff_idx = (cumsum >= target).long().argmax(dim=1)
    rolloff = rolloff_idx.float() * (sr / 2 / (n_fft / 2))
    rolloff_mean = rolloff.mean(dim=-1).cpu().numpy()
    rolloff_std = rolloff.std(dim=-1).cpu().numpy()
    
    # Flatness
    geom_mean = torch.exp(torch.mean(torch.log(S), dim=1))
    arith_mean = torch.mean(S, dim=1)
    flatness = geom_mean / arith_mean
    flatness_mean = flatness.mean(dim=-1).cpu().numpy()
    flatness_std = flatness.std(dim=-1).cpu().numpy()
    
    # ── 3. Zero Crossing Rate ──
    # Simple per-frame ZCR approximation via unfold
    unfolded = waveforms.unfold(-1, n_fft, hop_length) # (batch, time, n_fft)
    zcr_frames = (unfolded[:, :, 1:] * unfolded[:, :, :-1] < 0).float().mean(dim=-1)
    zcr_mean = zcr_frames.mean(dim=-1).cpu().numpy()
    
    # ── 4. RMS Energy ──
    rms_frames = torch.sqrt(torch.mean(unfolded**2, dim=-1))
    rms_mean = rms_frames.mean(dim=-1).cpu().numpy()
    rms_std = rms_frames.std(dim=-1).cpu().numpy()
    
    # ── Assembly ──
    results = []
    for b in range(batch_size):
        row = {}
        for i in range(n_mfcc):
            row[f"mfcc{i}_mean"] = float(mfcc_mean[b, i])
            row[f"mfcc{i}_std"]  = float(mfcc_std[b, i])
            
        row["centroid_mean"] = float(centroid_mean[b])
        row["centroid_std"]  = float(centroid_std[b])
        row["rolloff_mean"]  = float(rolloff_mean[b])
        row["rolloff_std"]   = float(rolloff_std[b])
        row["flatness_mean"] = float(flatness_mean[b])
        row["flatness_std"]  = float(flatness_std[b])
        row["zcr_mean"]      = float(zcr_mean[b])
        row["rms_mean"]      = float(rms_mean[b])
        row["rms_std"]       = float(rms_std[b])
        
        # NaN Safety
        for k, v in row.items():
            if np.isnan(v) or np.isinf(v):
                row[k] = 0.0
        results.append(row)
        
    return results
