"""
preprocessor/feature_extraction.py
Spectral and prosodic feature extraction from audio chunks.
"""

from __future__ import annotations
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Spectral features
# ─────────────────────────────────────────────────────────────────────────────

def compute_stft(
    chunk: np.ndarray,
    sr: int = 16_000,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    """Return magnitude STFT, shape (n_fft//2+1, T)."""
    S = librosa.stft(chunk, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return np.abs(S).astype(np.float32)


def compute_mel_spectrogram(
    chunk: np.ndarray,
    sr: int = 16_000,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
) -> np.ndarray:
    """Mel-spectrogram in log scale, shape (n_mels, T)."""
    mel = librosa.feature.melspectrogram(
        y=chunk, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def compute_mfcc(
    chunk: np.ndarray,
    sr: int = 16_000,
    n_mfcc: int = 20,
    n_fft: int = 512,
    hop_length: int = 160,
) -> np.ndarray:
    """MFCC, shape (n_mfcc, T)."""
    return librosa.feature.mfcc(
        y=chunk, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    ).astype(np.float32)


def compute_lfcc(
    chunk: np.ndarray,
    sr: int = 16_000,
    n_lfcc: int = 20,
    n_filters: int = 70,
    n_fft: int = 512,
    hop_length: int = 160,
) -> np.ndarray:
    """
    Linear Frequency Cepstral Coefficients (LFCC).
    Pipeline: STFT → linear filterbank → log → DCT.
    Returns shape (n_lfcc, T).
    """
    mag = compute_stft(chunk, sr=sr, n_fft=n_fft, hop_length=hop_length)  # (F, T)
    freq_bins = mag.shape[0]

    # Linear filterbank (uniform triangular filters)
    freq_points = np.linspace(0, freq_bins - 1, n_filters + 2).astype(int)
    filterbank = np.zeros((n_filters, freq_bins), dtype=np.float32)
    for m in range(1, n_filters + 1):
        left, center, right = freq_points[m - 1], freq_points[m], freq_points[m + 1]
        for k in range(left, center + 1):
            if center != left:
                filterbank[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right + 1):
            if right != center:
                filterbank[m - 1, k] = (right - k) / (right - center)

    filtered = np.dot(filterbank, mag)                  # (n_filters, T)
    log_filtered = np.log(filtered + 1e-8)
    # DCT-II via matrix multiply (type-2 DCT)
    n_f = log_filtered.shape[0]
    dct_matrix = np.cos(
        np.pi / n_f * (np.arange(n_lfcc)[:, None] + 0.5) * np.arange(n_f)[None, :]
    ).astype(np.float32)
    return np.dot(dct_matrix, log_filtered).astype(np.float32)   # (n_lfcc, T)


def compute_cqcc(
    chunk: np.ndarray,
    sr: int = 16_000,
    n_cqcc: int = 20,
    hop_length: int = 160,
    n_bins: int = 84,
    bins_per_octave: int = 12,
) -> np.ndarray:
    """
    Constant-Q Cepstral Coefficients (CQCC).
    Pipeline: CQT → log magnitude → DCT.
    Returns shape (n_cqcc, T).
    """
    cqt = np.abs(
        librosa.cqt(
            chunk, sr=sr, hop_length=hop_length,
            n_bins=n_bins, bins_per_octave=bins_per_octave,
        )
    ).astype(np.float32)  # (n_bins, T)

    log_cqt = np.log(cqt + 1e-8)
    n_b = log_cqt.shape[0]
    dct_matrix = np.cos(
        np.pi / n_b * (np.arange(n_cqcc)[:, None] + 0.5) * np.arange(n_b)[None, :]
    ).astype(np.float32)
    return np.dot(dct_matrix, log_cqt).astype(np.float32)   # (n_cqcc, T)


def extract_spectral_features(
    chunk: np.ndarray,
    sr: int = 16_000,
    n_mels: int = 40,
    n_mfcc: int = 20,
    n_lfcc: int = 20,
    n_cqcc: int = 20,
) -> np.ndarray:
    """
    Concatenate Mel + MFCC + LFCC + CQCC along the feature axis.
    All features are trimmed / padded to the same T.
    Returns shape (n_features, T).
    """
    mel  = compute_mel_spectrogram(chunk, sr=sr, n_mels=n_mels)
    mfcc = compute_mfcc(chunk, sr=sr, n_mfcc=n_mfcc)
    lfcc = compute_lfcc(chunk, sr=sr, n_lfcc=n_lfcc)
    cqcc = compute_cqcc(chunk, sr=sr, n_cqcc=n_cqcc)

    T = min(mel.shape[1], mfcc.shape[1], lfcc.shape[1], cqcc.shape[1])
    feats = np.concatenate(
        [mel[:, :T], mfcc[:, :T], lfcc[:, :T], cqcc[:, :T]], axis=0
    )
    # z-score normalise per feature
    mean = feats.mean(axis=1, keepdims=True)
    std  = feats.std(axis=1, keepdims=True) + 1e-8
    return ((feats - mean) / std).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Prosodic features
# ─────────────────────────────────────────────────────────────────────────────

def compute_f0(
    chunk: np.ndarray,
    sr: int = 16_000,
    hop_length: int = 160,
    fmin: float = 60.0,
    fmax: float = 400.0,
) -> np.ndarray:
    """
    Fundamental frequency (F0) contour via pYIN.
    Unvoiced frames are set to 0.
    Returns shape (T,).
    """
    f0, voiced_flag, _ = librosa.pyin(
        chunk, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0.astype(np.float32)


def compute_jitter(f0: np.ndarray) -> float:
    """Local jitter: mean absolute difference of consecutive voiced F0 periods."""
    voiced = f0[f0 > 0]
    if len(voiced) < 2:
        return 0.0
    periods = 1.0 / voiced
    return float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-9))


def compute_shimmer(chunk: np.ndarray, sr: int = 16_000, hop_length: int = 160) -> float:
    """
    Local shimmer: mean absolute amplitude variation between consecutive frames.
    """
    rms = librosa.feature.rms(y=chunk, hop_length=hop_length)[0]
    if len(rms) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-9))


def compute_energy_contour(chunk: np.ndarray, hop_length: int = 160) -> np.ndarray:
    """RMS energy contour, shape (T,)."""
    return librosa.feature.rms(y=chunk, hop_length=hop_length)[0].astype(np.float32)


def compute_speaking_rate(chunk: np.ndarray, sr: int = 16_000, hop_length: int = 160) -> float:
    """
    Estimate syllable rate from energy peaks per second.
    """
    energy = compute_energy_contour(chunk, hop_length=hop_length)
    # smooth and detect peaks
    from scipy.signal import find_peaks
    smoothed = np.convolve(energy, np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(smoothed, height=smoothed.mean())
    duration = len(chunk) / sr
    return float(len(peaks) / (duration + 1e-6))


def extract_prosodic_features(
    chunk: np.ndarray,
    sr: int = 16_000,
    hop_length: int = 160,
    n_features_target: int = 5,
) -> np.ndarray:
    """
    Build a prosodic feature matrix of shape (5, T):
      0 – F0 (normalised)
      1 – delta F0 (first difference)
      2 – RMS energy
      3 – delta RMS
      4 – voiced/unvoiced flag (0/1)
    """
    f0     = compute_f0(chunk, sr=sr, hop_length=hop_length)        # (T,)
    energy = compute_energy_contour(chunk, hop_length=hop_length)   # (T,)

    T = min(len(f0), len(energy))
    f0, energy = f0[:T], energy[:T]

    voiced = (f0 > 0).astype(np.float32)

    # normalise
    f0_norm = (f0 - f0[f0 > 0].mean() if voiced.sum() > 0 else f0) / (f0.std() + 1e-8)
    e_norm  = (energy - energy.mean()) / (energy.std() + 1e-8)

    df0 = np.concatenate([[0], np.diff(f0_norm)])
    de  = np.concatenate([[0], np.diff(e_norm)])

    feats = np.stack([f0_norm, df0, e_norm, de, voiced], axis=0)  # (5, T)
    return feats.astype(np.float32)
