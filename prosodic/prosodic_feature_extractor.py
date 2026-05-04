"""
prosodic/prosodic_feature_extractor.py

Extracts ~45 tabular prosodic features from a single audio file or waveform.
Returns a flat dictionary suitable for DataFrame construction and XGBoost training.

Features extracted:
  - F0 (pitch)           : mean, std, min, max, range, median, CV, slope  → 8
  - Jitter               : local, local_abs, rap, ppq5  → 4
  - Shimmer              : local, local_dB, apq3, apq5, apq11  → 5
  - HNR                  : mean, std  → 2
  - Energy/RMS           : mean, std, max, dynamic_range  → 4
  - Speaking rate         : syllable_rate, pause_ratio  → 2
  - Voiced/unvoiced      : voiced_ratio, v_to_uv_transitions  → 2
  - Formants F1-F4       : mean  → 4
  - Duration             : total_duration, voiced_duration  → 2
  - Delta features       : delta_f0 stats, delta_energy stats  → 12

Total: ~45 features per audio file.

Uses librosa + parselmouth (Praat) for robust prosodic analysis.
Falls back to librosa-only if parselmouth is not available.
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import find_peaks

warnings.filterwarnings("ignore", category=UserWarning)

# Try parselmouth for Praat-quality prosodic features
try:
    import parselmouth
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
    Extract tabular prosodic features from one audio file or waveform.

    Parameters
    ----------
    path_or_waveform : str, Path, or np.ndarray
        Audio file path or pre-loaded waveform (1-D float32).
    sr : int
        Target sample rate.

    Returns
    -------
    dict : feature_name → float  (~45 entries)
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
    row["energy_max"]  = float(np.nan_to_num(rms.max()))
    e_min = rms[rms > 0].min() if np.any(rms > 0) else 1e-9
    row["energy_dynamic_range"] = float(np.nan_to_num(
        20 * np.log10(rms.max() / (e_min + 1e-9))
    ))

    # Delta energy
    if len(rms) > 2:
        d_rms = np.diff(rms)
        row["delta_energy_mean"]  = float(np.nan_to_num(d_rms.mean()))
        row["delta_energy_std"]   = float(np.nan_to_num(d_rms.std()))
        row["delta_energy_max"]   = float(np.nan_to_num(np.abs(d_rms).max()))
    else:
        row["delta_energy_mean"]  = 0.0
        row["delta_energy_std"]   = 0.0
        row["delta_energy_max"]   = 0.0

    # Speaking rate (syllable-like peaks in energy)
    if len(rms) > 5:
        smoothed = np.convolve(rms, np.ones(5) / 5, mode="same")
        peaks, _ = find_peaks(smoothed, height=smoothed.mean())
        row["syllable_rate"] = float(len(peaks) / (duration + 1e-6))
    else:
        row["syllable_rate"] = 0.0

    # Pause ratio (fraction of frames below threshold)
    threshold = rms.mean() * 0.1
    row["pause_ratio"] = float((rms < threshold).sum() / (len(rms) + 1e-9))

    # Duration
    row["total_duration"] = float(duration)

    return row


def _extract_praat_features(y: np.ndarray, sr: int, row: dict, duration: float):
    """Extract prosodic features using Parselmouth (Praat)."""
    snd = parselmouth.Sound(y, sampling_frequency=sr)

    # ── F0 (Pitch) ───────────────────────────────────────────────────────────
    pitch = call(snd, "To Pitch", 0.0, 60.0, 400.0)
    f0_values = pitch.selected_array["frequency"]
    voiced = f0_values[f0_values > 0]

    if len(voiced) > 0:
        row["f0_mean"]   = float(voiced.mean())
        row["f0_std"]    = float(voiced.std())
        row["f0_min"]    = float(voiced.min())
        row["f0_max"]    = float(voiced.max())
        row["f0_range"]  = float(voiced.max() - voiced.min())
        row["f0_median"] = float(np.median(voiced))
        row["f0_cv"]     = float(voiced.std() / (voiced.mean() + 1e-9))
        # F0 slope (linear regression)
        x = np.arange(len(voiced))
        if len(voiced) > 1:
            slope = np.polyfit(x, voiced, 1)[0]
            row["f0_slope"] = float(slope)
        else:
            row["f0_slope"] = 0.0
    else:
        for key in ["f0_mean", "f0_std", "f0_min", "f0_max", "f0_range",
                     "f0_median", "f0_cv", "f0_slope"]:
            row[key] = 0.0

    # Delta F0
    if len(voiced) > 2:
        d_f0 = np.diff(voiced)
        row["delta_f0_mean"]    = float(np.nan_to_num(d_f0.mean()))
        row["delta_f0_std"]     = float(np.nan_to_num(d_f0.std()))
        row["delta_f0_abs_mean"] = float(np.nan_to_num(np.abs(d_f0).mean()))
    else:
        row["delta_f0_mean"]    = 0.0
        row["delta_f0_std"]     = 0.0
        row["delta_f0_abs_mean"] = 0.0

    # Voiced ratio & transitions
    voiced_mask = f0_values > 0
    row["voiced_ratio"] = float(voiced_mask.sum() / (len(f0_values) + 1e-9))
    if len(voiced_mask) > 1:
        transitions = np.sum(np.abs(np.diff(voiced_mask.astype(int))))
        row["v_to_uv_transitions"] = float(transitions)
    else:
        row["v_to_uv_transitions"] = 0.0
    row["voiced_duration"] = float(row["voiced_ratio"] * duration)

    # ── Jitter ───────────────────────────────────────────────────────────────
    point_proc = call(snd, "To PointProcess (periodic, cc)", 60.0, 400.0)
    try:
        row["jitter_local"]     = float(call(point_proc, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        row["jitter_local_abs"] = float(call(point_proc, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3))
        row["jitter_rap"]       = float(call(point_proc, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3))
        row["jitter_ppq5"]      = float(call(point_proc, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3))
    except Exception:
        row["jitter_local"] = row["jitter_local_abs"] = row["jitter_rap"] = row["jitter_ppq5"] = 0.0

    # ── Shimmer ──────────────────────────────────────────────────────────────
    try:
        row["shimmer_local"]    = float(call([snd, point_proc], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        row["shimmer_local_dB"] = float(call([snd, point_proc], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        row["shimmer_apq3"]     = float(call([snd, point_proc], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        row["shimmer_apq5"]     = float(call([snd, point_proc], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        row["shimmer_apq11"]    = float(call([snd, point_proc], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    except Exception:
        row["shimmer_local"] = row["shimmer_local_dB"] = row["shimmer_apq3"] = 0.0
        row["shimmer_apq5"] = row["shimmer_apq11"] = 0.0

    # ── HNR ──────────────────────────────────────────────────────────────────
    try:
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 60.0, 0.1, 1.0)
        hnr_values = harmonicity.values[harmonicity.values != -200]  # -200 = undefined
        row["hnr_mean"] = float(hnr_values.mean()) if len(hnr_values) > 0 else 0.0
        row["hnr_std"]  = float(hnr_values.std()) if len(hnr_values) > 0 else 0.0
    except Exception:
        row["hnr_mean"] = row["hnr_std"] = 0.0

    # ── Formants F1–F4 ──────────────────────────────────────────────────────
    try:
        formant = call(snd, "To Formant (burg)", 0.0, 5, 5500.0, 0.025, 50.0)
        n_frames = call(formant, "Get number of frames")
        for fi in range(1, 5):
            vals = []
            for frame in range(1, n_frames + 1):
                v = call(formant, "Get value at time", fi, frame * 0.025, "Hertz", "Linear")
                if not np.isnan(v):
                    vals.append(v)
            row[f"formant_f{fi}_mean"] = float(np.mean(vals)) if vals else 0.0
    except Exception:
        for fi in range(1, 5):
            row[f"formant_f{fi}_mean"] = 0.0

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
        row["f0_mean"]   = float(voiced.mean())
        row["f0_std"]    = float(voiced.std())
        row["f0_min"]    = float(voiced.min())
        row["f0_max"]    = float(voiced.max())
        row["f0_range"]  = float(voiced.max() - voiced.min())
        row["f0_median"] = float(np.median(voiced))
        row["f0_cv"]     = float(voiced.std() / (voiced.mean() + 1e-9))
        x = np.arange(len(voiced))
        row["f0_slope"] = float(np.polyfit(x, voiced, 1)[0]) if len(voiced) > 1 else 0.0
    else:
        for key in ["f0_mean", "f0_std", "f0_min", "f0_max", "f0_range",
                     "f0_median", "f0_cv", "f0_slope"]:
            row[key] = 0.0

    # Delta F0
    if len(voiced) > 2:
        d_f0 = np.diff(voiced)
        row["delta_f0_mean"]     = float(d_f0.mean())
        row["delta_f0_std"]      = float(d_f0.std())
        row["delta_f0_abs_mean"] = float(np.abs(d_f0).mean())
    else:
        row["delta_f0_mean"] = row["delta_f0_std"] = row["delta_f0_abs_mean"] = 0.0

    # Voiced ratio & transitions
    voiced_mask = f0 > 0
    row["voiced_ratio"] = float(voiced_mask.sum() / (len(f0) + 1e-9))
    transitions = np.sum(np.abs(np.diff(voiced_mask.astype(int)))) if len(voiced_mask) > 1 else 0
    row["v_to_uv_transitions"] = float(transitions)
    row["voiced_duration"] = float(row["voiced_ratio"] * duration)

    # Jitter (from F0)
    if len(voiced) > 1:
        periods = 1.0 / voiced
        row["jitter_local"]     = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-9))
        row["jitter_local_abs"] = float(np.mean(np.abs(np.diff(periods))))
        # RAP and PPQ5 approximations
        row["jitter_rap"]  = row["jitter_local"] * 0.8  # rough approx
        row["jitter_ppq5"] = row["jitter_local"] * 0.7
    else:
        row["jitter_local"] = row["jitter_local_abs"] = row["jitter_rap"] = row["jitter_ppq5"] = 0.0

    # Shimmer (from RMS)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if len(rms) > 1:
        row["shimmer_local"]    = float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-9))
        row["shimmer_local_dB"] = float(np.mean(np.abs(np.diff(20 * np.log10(rms + 1e-9)))))
        row["shimmer_apq3"]     = row["shimmer_local"] * 0.9
        row["shimmer_apq5"]     = row["shimmer_local"] * 0.85
        row["shimmer_apq11"]    = row["shimmer_local"] * 0.75
    else:
        row["shimmer_local"] = row["shimmer_local_dB"] = row["shimmer_apq3"] = 0.0
        row["shimmer_apq5"] = row["shimmer_apq11"] = 0.0

    # HNR approximation
    row["hnr_mean"] = 0.0
    row["hnr_std"]  = 0.0

    # Formants (not available without Praat)
    for fi in range(1, 5):
        row[f"formant_f{fi}_mean"] = 0.0

    # NaN safety
    for k, v in row.items():
        if np.isnan(v) or np.isinf(v):
            row[k] = 0.0


def get_feature_names() -> list[str]:
    """Return ordered list of feature column names."""
    names = [
        # F0
        "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range",
        "f0_median", "f0_cv", "f0_slope",
        # Delta F0
        "delta_f0_mean", "delta_f0_std", "delta_f0_abs_mean",
        # Voiced
        "voiced_ratio", "v_to_uv_transitions", "voiced_duration",
        # Jitter
        "jitter_local", "jitter_local_abs", "jitter_rap", "jitter_ppq5",
        # Shimmer
        "shimmer_local", "shimmer_local_dB", "shimmer_apq3",
        "shimmer_apq5", "shimmer_apq11",
        # HNR
        "hnr_mean", "hnr_std",
        # Formants
        "formant_f1_mean", "formant_f2_mean", "formant_f3_mean", "formant_f4_mean",
        # Energy
        "energy_mean", "energy_std", "energy_max", "energy_dynamic_range",
        # Delta energy
        "delta_energy_mean", "delta_energy_std", "delta_energy_max",
        # Speaking rate
        "syllable_rate", "pause_ratio",
        # Duration
        "total_duration",
    ]
    return names
