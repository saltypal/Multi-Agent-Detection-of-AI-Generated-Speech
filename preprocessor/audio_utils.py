"""
preprocessor/audio_utils.py
Core audio I/O and waveform manipulation.
"""

from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Load & normalise
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(
    path: str | Path,
    target_sr: int = 16_000,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load any audio file (FLAC, WAV, MP3 …), resample to *target_sr*,
    convert to mono, and normalise amplitude to [-1, 1].

    Returns
    -------
    waveform : np.ndarray  shape (n_samples,)
    sr       : int         always == target_sr
    """
    waveform, sr = librosa.load(str(path), sr=target_sr, mono=mono)
    # Peak normalise
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak
    return waveform.astype(np.float32), target_sr


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_audio(
    waveform: np.ndarray,
    sr: int = 16_000,
    chunk_duration: float = 3.0,
    overlap: float = 0.5,
) -> list[np.ndarray]:
    """
    Split *waveform* into fixed-size overlapping chunks.

    Parameters
    ----------
    waveform       : 1-D float32 array
    sr             : sample rate (Hz)
    chunk_duration : length of each chunk in seconds
    overlap        : overlap between consecutive chunks in seconds

    Returns
    -------
    List of 1-D float32 arrays, each of length ``chunk_duration * sr``.
    Short files that yield no full chunk still return [padded_waveform].
    """
    chunk_len = int(chunk_duration * sr)
    hop_len = int((chunk_duration - overlap) * sr)

    if len(waveform) <= chunk_len:
        return [pad_or_trim(waveform, chunk_len)]

    chunks = []
    start = 0
    while start + chunk_len <= len(waveform):
        chunks.append(waveform[start: start + chunk_len].copy())
        start += hop_len

    # include tail if substantial (> half a chunk)
    tail = waveform[start:]
    if len(tail) > chunk_len // 2:
        chunks.append(pad_or_trim(tail, chunk_len))

    return chunks


def pad_or_trim(waveform: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad (zeros, right) or trim (right) *waveform* to exactly *target_length*.
    """
    n = len(waveform)
    if n == target_length:
        return waveform
    if n > target_length:
        return waveform[:target_length]
    return np.pad(waveform, (0, target_length - n), mode="constant")


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def get_duration(path: str | Path) -> float:
    """Return audio duration in seconds without loading the full file."""
    info = sf.info(str(path))
    return info.duration
