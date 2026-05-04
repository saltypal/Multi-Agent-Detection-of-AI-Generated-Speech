"""
preprocessor/dataloader.py

Auto-detects dataset folder structure and collects audio file paths with labels.
Purely a data-discovery utility — no PyTorch dependencies.

Key exports
-----------
detect_dataset_structure(root_dir) -> DatasetLayout
collect_files(split_dir, layout, data_factor, ...) -> list[(Path, int)]
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# Dataset structure auto-detection
# ─────────────────────────────────────────────────────────────────────────────

_SPLIT_CANDIDATES: dict[str, list[str]] = {
    "train": ["train", "flac_T", "training", "Train", "TRAIN"],
    "val":   ["val", "dev", "flac_D", "development", "Val", "DEV"],
    "test":  ["test", "eval", "flac_E_eval", "evaluation", "Test", "EVAL"],
}

_LABEL_CANDIDATES: dict[str, list[str]] = {
    "bonafide": ["bonafide", "genuine", "real", "bona-fide", "Bonafide"],
    "spoof":    ["spoof", "fake", "synthetic", "spoofed", "Spoof"],
}


@dataclass
class DatasetLayout:
    """Describes the resolved dataset paths for each split."""
    train_dir: Path | None
    val_dir:   Path | None
    test_dir:  Path | None
    label_strategy: Literal["subdir", "tsv"]
    bonafide_name: str
    spoof_name:    str

    def get(self, split: str) -> Path | None:
        return {"train": self.train_dir, "val": self.val_dir, "test": self.test_dir}.get(split)

    def __str__(self) -> str:
        lines = ["DatasetLayout:"]
        for role in ("train", "val", "test"):
            p = self.get(role)
            lines.append(f"  {role:5s} → {p if p else '(not found)'}")
        lines.append(f"  label_strategy → {self.label_strategy}")
        lines.append(f"  bonafide subfolder → '{self.bonafide_name}'")
        lines.append(f"  spoof subfolder    → '{self.spoof_name}'")
        return "\n".join(lines)


def detect_dataset_structure(root_dir: str | Path) -> DatasetLayout:
    """
    Scan *root_dir* and auto-detect split folders and label subfolders.
    Raises FileNotFoundError if root_dir doesn't exist.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    resolved: dict[str, Path | None] = {"train": None, "val": None, "test": None}
    for role, candidates in _SPLIT_CANDIDATES.items():
        for name in candidates:
            p = root / name
            if p.is_dir():
                resolved[role] = p
                break

    if all(v is None for v in resolved.values()):
        resolved["train"] = root
        print(f"[DataLoader] No split subdirs found. Treating '{root}' as train.")

    sample_split_dir = next(p for p in resolved.values() if p is not None)
    bonafide_name, spoof_name = _resolve_label_dirs(sample_split_dir)

    strategy = "subdir" if bonafide_name else "tsv"

    layout = DatasetLayout(
        train_dir=resolved["train"],
        val_dir=resolved["val"],
        test_dir=resolved["test"],
        label_strategy=strategy,
        bonafide_name=bonafide_name or "",
        spoof_name=spoof_name or "",
    )
    print(layout)
    return layout


def _resolve_label_dirs(split_dir: Path) -> tuple[str | None, str | None]:
    """Return (bonafide_subfolder_name, spoof_subfolder_name) or (None, None)."""
    subdirs = {d.name: d for d in split_dir.iterdir() if d.is_dir()}
    bona, spf = None, None
    for name in subdirs:
        if any(c.lower() == name.lower() for c in _LABEL_CANDIDATES["bonafide"]):
            bona = name
        if any(c.lower() == name.lower() for c in _LABEL_CANDIDATES["spoof"]):
            spf = name
    return bona, spf


def collect_files(
    split_dir: Path,
    layout: DatasetLayout,
    data_factor: float = 1.0,
    rng_seed: int = 42,
    extensions: tuple[str, ...] = (".flac", ".wav", ".mp3"),
) -> list[tuple[Path, int]]:
    """
    Collect (file_path, label) pairs from *split_dir*.
    Applies *data_factor* sampling, balanced across bonafide/spoof.

    Parameters
    ----------
    split_dir   : path to a split folder (e.g., dataset/train/)
    layout      : DatasetLayout from detect_dataset_structure()
    data_factor : fraction of files to keep (0.0–1.0)
    rng_seed    : random seed for reproducibility
    extensions  : file extensions to look for

    Returns sorted list of (path, label) for reproducibility.
    label: 0 = bonafide, 1 = spoof
    """
    rng = random.Random(rng_seed)

    if layout.label_strategy == "subdir":
        bona_dir = split_dir / layout.bonafide_name
        spoof_dir = split_dir / layout.spoof_name

        bona_files = []
        spoof_files = []
        for ext in extensions:
            bona_files.extend(sorted(bona_dir.glob(f"**/*{ext}")))
            spoof_files.extend(sorted(spoof_dir.glob(f"**/*{ext}")))

        bona_n = max(1, int(len(bona_files) * data_factor))
        spoof_n = max(1, int(len(spoof_files) * data_factor))

        rng.shuffle(bona_files)
        rng.shuffle(spoof_files)

        pairs = (
            [(f, 0) for f in bona_files[:bona_n]]
            + [(f, 1) for f in spoof_files[:spoof_n]]
        )
    else:
        tsv_files = list(split_dir.glob("*.tsv"))
        if not tsv_files:
            raise FileNotFoundError(f"No bonafide/spoof subdirs and no .tsv found in {split_dir}")
        tsv = tsv_files[0]
        pairs = _parse_tsv_labels(tsv, split_dir)
        n = max(1, int(len(pairs) * data_factor))
        rng.shuffle(pairs)
        pairs = pairs[:n]

    rng.shuffle(pairs)
    return pairs


def _parse_tsv_labels(tsv_path: Path, audio_dir: Path) -> list[tuple[Path, int]]:
    """Parse ASVspoof5 TSV metadata file. Returns (audio_path, label) pairs."""
    pairs = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            fname = parts[1]
            key = parts[-1].lower()
            label = 0 if "bonafide" in key else 1
            audio_path = audio_dir / (fname + ".flac")
            if not audio_path.exists():
                audio_path = audio_dir / fname
            if audio_path.exists():
                pairs.append((audio_path, label))
    return pairs
