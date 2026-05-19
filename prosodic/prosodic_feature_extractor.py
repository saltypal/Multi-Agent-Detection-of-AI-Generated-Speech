"""
prosodic/prosodic_feature_extractor.py

Extracts a minimal, strong set of tabular prosodic features (~10 features).
Optimized for XGBoost deepfake detection by removing noise and redundant features.

Features extracted:
  - F0 (pitch)           : mean, std, range  → 3
  - Jitter               : local  → 1
  - Shimmer              : local  → 1
  - Energy/RMS           : mean, std  → 2
  - Speaking rate         : syllable_rate  → 1
  - Voiced/unvoiced      : voiced_ratio  → 1

Total: ~9 features per audio file.

Uses librosa + parselmouth (Praat) for robust prosodic analysis.
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
# pyrefly: ignore [missing-import]
import librosa
from scipy.signal import find_peaks

warnings.filterwarnings("ignore", category=UserWarning)

try:
    # pyrefly: ignore [missing-import]
    import parselmouth
    # pyrefly: ignore [missing-import]
    from parselmouth.praat import call
    HAS_PRAAT = True
except ImportError:
    HAS_PRAAT = False

SR = 16_000


def extract_prosodic_row(
    path_or_waveform,
    sr: int = SR,
) -> dict:
    """
    Extract minimal tabular prosodic features.

    Parameters
    ----------
    path_or_waveform : str, Path, or np.ndarray
        Audio file path or pre-loaded waveform (1-D float32).
    sr : int
        Target sample rate.

    Returns
    -------
    dict : feature_name → float  (~9 entries)
    """
    if isinstance(path_or_waveform, (str, Path)):
        y, _ = librosa.load(str(path_or_waveform), sr=sr, mono=True)
    else:
        y = np.asarray(path_or_waveform, dtype=np.float32)

    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak

    row = {}
    duration = len(y) / sr

    if HAS_PRAAT:
        _extract_praat_features(y, sr, row, duration)
    else:
        _extract_librosa_features(y, sr, row, duration)

    # ── Common features (both paths) ─────────────────────────────────────────

    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=160)[0]
    row["energy_mean"] = float(np.nan_to_num(rms.mean()))
    row["energy_std"]  = float(np.nan_to_num(rms.std()))

    # Speaking rate (syllable-like peaks in energy)
    if len(rms) > 5:
        smoothed = np.convolve(rms, np.ones(5) / 5, mode="same")
        peaks, _ = find_peaks(smoothed, height=smoothed.mean())
        row["syllable_rate"] = float(len(peaks) / (duration + 1e-6))
    else:
        row["syllable_rate"] = 0.0

    return row


def _extract_praat_features(y: np.ndarray, sr: int, row: dict, duration: float):
    """Extract prosodic features using Parselmouth (Praat)."""
    snd = parselmouth.Sound(y, sampling_frequency=sr)

    # ── F0 (Pitch) ───────────────────────────────────────────────────────────
    pitch = call(snd, "To Pitch", 0.0, 60.0, 400.0)
    f0_values = pitch.selected_array["frequency"]
    voiced = f0_values[f0_values > 0]

    if len(voiced) > 0:
        row["f0_mean"]  = float(voiced.mean())
        row["f0_std"]   = float(voiced.std())
        row["f0_range"] = float(voiced.max() - voiced.min())
    else:
        row["f0_mean"] = row["f0_std"] = row["f0_range"] = 0.0

    # Voiced ratio
    voiced_mask = f0_values > 0
    row["voiced_ratio"] = float(voiced_mask.sum() / (len(f0_values) + 1e-9))

    # ── Jitter ───────────────────────────────────────────────────────────────
    point_proc = call(snd, "To PointProcess (periodic, cc)", 60.0, 400.0)
    try:
        row["jitter_local"] = float(call(point_proc, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
    except Exception:
        row["jitter_local"] = 0.0

    # ── Shimmer ──────────────────────────────────────────────────────────────
    try:
        row["shimmer_local"] = float(call([snd, point_proc], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    except Exception:
        row["shimmer_local"] = 0.0

    # NaN safety
    for k, v in row.items():
        if np.isnan(v) or np.isinf(v):
            row[k] = 0.0


def _extract_librosa_features(y: np.ndarray, sr: int, row: dict, duration: float):
    """Fallback: extract prosodic features using only librosa (no Praat)."""
    hop_length = 160

    # F0 via pYIN
    f0, voiced_flag, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, hop_length=hop_length)
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced = f0[f0 > 0]

    if len(voiced) > 0:
        row["f0_mean"]  = float(voiced.mean())
        row["f0_std"]   = float(voiced.std())
        row["f0_range"] = float(voiced.max() - voiced.min())
    else:
        row["f0_mean"] = row["f0_std"] = row["f0_range"] = 0.0

    # Voiced ratio
    voiced_mask = f0 > 0
    row["voiced_ratio"] = float(voiced_mask.sum() / (len(f0) + 1e-9))

    # Jitter (from F0)
    if len(voiced) > 1:
        periods = 1.0 / voiced
        row["jitter_local"] = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-9))
    else:
        row["jitter_local"] = 0.0

    # Shimmer (from RMS)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if len(rms) > 1:
        row["shimmer_local"] = float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-9))
    else:
        row["shimmer_local"] = 0.0

    # NaN safety
    for k, v in row.items():
        if np.isnan(v) or np.isinf(v):
            row[k] = 0.0


def get_feature_names() -> list[str]:
    """Return ordered list of feature column names."""
    names = [
        "f0_mean", "f0_std", "f0_range",
        "voiced_ratio",
        "jitter_local",
        "shimmer_local",
        "energy_mean", "energy_std",
    ]
    return names

try:
    import torch
    import torchaudio
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def extract_prosodic_batch(waveforms: np.ndarray | torch.Tensor, sr: int = SR, device: str = 'cuda') -> list[dict]:
    """
    GPU-accelerated extraction of prosodic features for a batch of audio.
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
    
    duration = waveforms.size(1) / sr
    hop_length = 160
    
    # ── 1. F0 (Pitch) via Torchaudio ──
    # returns shape (batch, freq_frames)
    pitch = torchaudio.functional.detect_pitch_frequency(waveforms, sample_rate=sr)
    
    # ── 2. RMS Energy ──
    unfolded = waveforms.unfold(-1, hop_length, hop_length)
    rms_frames = torch.sqrt(torch.mean(unfolded**2, dim=-1))
    
    # Assembly via fast CPU iterators for the non-matrix stuff (peak counting, filtering)
    pitch_np = pitch.cpu().numpy()
    rms_np = rms_frames.cpu().numpy()
    
    results = []
    for b in range(batch_size):
        row = {}
        
        # F0 stats
        f0 = pitch_np[b]
        voiced = f0[f0 > 0]
        if len(voiced) > 0:
            row["f0_mean"]  = float(voiced.mean())
            row["f0_std"]   = float(voiced.std())
            row["f0_range"] = float(voiced.max() - voiced.min())
        else:
            row["f0_mean"] = row["f0_std"] = row["f0_range"] = 0.0
            
        # Voiced ratio
        row["voiced_ratio"] = float(len(voiced) / (len(f0) + 1e-9))
        
        # Jitter local
        if len(voiced) > 1:
            periods = 1.0 / voiced
            row["jitter_local"] = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-9))
        else:
            row["jitter_local"] = 0.0
            
        # Shimmer local
        rms = rms_np[b]
        if len(rms) > 1:
            row["shimmer_local"] = float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-9))
        else:
            row["shimmer_local"] = 0.0
            
        # Energy global
        row["energy_mean"] = float(np.nan_to_num(rms.mean()))
        row["energy_std"]  = float(np.nan_to_num(rms.std()))
        
        # Syllable rate (peaks in smoothed energy)
        if len(rms) > 5:
            smoothed = np.convolve(rms, np.ones(5)/5, mode='same')
            from scipy.signal import find_peaks
            peaks_idx, _ = find_peaks(smoothed, height=smoothed.mean())
            row["syllable_rate"] = float(len(peaks_idx) / (duration + 1e-6))
        else:
            row["syllable_rate"] = 0.0
            
        # NaN safety
        for k, v in row.items():
            if np.isnan(v) or np.isinf(v):
                row[k] = 0.0
                
        results.append(row)
        
    return results
