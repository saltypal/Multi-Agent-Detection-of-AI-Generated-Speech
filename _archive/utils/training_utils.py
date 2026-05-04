"""
utils/training_utils.py
Shared training utilities for all agent training scripts.

Provides:
  - TrainingTimer      : per-batch and per-epoch ETA tracking
  - auto_download_dataset : Kaggle API auto-downloader
  - detect_device      : TPU > GPU > CPU auto-detection
  - save_checkpoint / load_checkpoint / find_latest_checkpoint
  - plot_training_curves  : 3-panel (loss / EER / AUC) plot
  - run_final_evaluation  : post-training test-set evaluation
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_device():
    """Return (device, device_type_str). Priority: TPU > GPU > CPU."""
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"[Device] TPU detected: {device}")
        return device, "xla"
    except ImportError:
        pass
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU detected: {torch.cuda.get_device_name(0)}")
        return device, "cuda"
    print("[Device] No accelerator — using CPU")
    return torch.device("cpu"), "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset auto-download
# ─────────────────────────────────────────────────────────────────────────────

def auto_download_dataset(
    data_dir: str,
    kaggle_dataset: str = "aniket202411001/asvspoof5-flac",
):
    """Download ASVspoof5 from Kaggle if data_dir is missing or empty."""
    data_path = Path(data_dir)
    if data_path.exists() and any(data_path.iterdir()):
        print(f"[Dataset] Found at '{data_dir}'. Skipping download.")
        return

    print(f"[Dataset] Not found at '{data_dir}'. Downloading from Kaggle …")
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[Dataset] Installing kaggle …")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "kaggle", "-q"],
            check=True,
        )

    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", kaggle_dataset, "--unzip", "-p", str(data_path),
    ]
    print(f"[Dataset] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Dataset] STDERR:\n{result.stderr}")
        raise RuntimeError(
            "Dataset download failed.\n"
            "Make sure KAGGLE_USERNAME and KAGGLE_KEY are set, "
            "or ~/.kaggle/kaggle.json exists."
        )
    print(f"[Dataset] Download complete → '{data_dir}'")


# ─────────────────────────────────────────────────────────────────────────────
# Training Timer (ETA)
# ─────────────────────────────────────────────────────────────────────────────

class TrainingTimer:
    """
    Tracks wall-clock time and estimates training completion.

    Usage
    -----
    timer = TrainingTimer(total_epochs=30, steps_per_epoch=len(train_loader))
    for epoch in range(start_epoch, total_epochs):
        timer.start_epoch(epoch)
        for i, batch in enumerate(train_loader):
            loss = ...  # training step
            print(timer.step(loss=loss), end="", flush=True)
        print()
        print(timer.epoch_summary())
    """

    def __init__(self, total_epochs: int, steps_per_epoch: int, alpha: float = 0.15):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.alpha = alpha           # EMA smoothing factor
        self._session_start = time.time()
        self._epoch_start: float | None = None
        self._step_start: float | None = None
        self._current_epoch = 0
        self._current_step = 0
        self._ema_step_time: float | None = None

    def start_epoch(self, epoch_idx: int):
        self._current_epoch = epoch_idx
        self._current_step = 0
        self._epoch_start = time.time()
        self._step_start = time.time()

    def step(self, loss: float | None = None) -> str:
        """Call after each batch. Returns a formatted progress string."""
        now = time.time()
        step_time = now - self._step_start
        self._step_start = now
        self._current_step += 1

        # Exponential moving average of step time
        if self._ema_step_time is None:
            self._ema_step_time = step_time
        else:
            self._ema_step_time = (
                (1 - self.alpha) * self._ema_step_time + self.alpha * step_time
            )

        steps_left = (
            (self.steps_per_epoch - self._current_step)
            + (self.total_epochs - self._current_epoch - 1) * self.steps_per_epoch
        )
        eta_secs = steps_left * self._ema_step_time
        elapsed_secs = now - self._session_start

        loss_str = f"Loss: {loss:.4f} | " if loss is not None else ""
        bar = self._bar(self._current_step, self.steps_per_epoch)
        return (
            f"\rEpoch {self._current_epoch + 1}/{self.total_epochs} {bar} "
            f"{self._current_step}/{self.steps_per_epoch} | "
            f"{loss_str}"
            f"ETA: {self._fmt(eta_secs)} | Elapsed: {self._fmt(elapsed_secs)}   "
        )

    def epoch_summary(self) -> str:
        epoch_secs = time.time() - self._epoch_start
        remaining_epochs = self.total_epochs - self._current_epoch - 1
        est_remaining = remaining_epochs * epoch_secs
        return (
            f"  ✓ Epoch {self._current_epoch + 1}/{self.total_epochs} "
            f"finished in {self._fmt(epoch_secs)} | "
            f"Est. remaining: {self._fmt(est_remaining)}"
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(secs: float) -> str:
        secs = max(0, int(secs))
        if secs >= 3600:
            return f"{secs // 3600}h {(secs % 3600) // 60}m"
        if secs >= 60:
            return f"{secs // 60}m {secs % 60:02d}s"
        return f"{secs}s"

    @staticmethod
    def _bar(current: int, total: int, width: int = 18) -> str:
        filled = int(width * current / max(total, 1))
        return f"[{'=' * filled}{'>' if filled < width else '='}{'.' * (width - filled - 1)}]"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    state = torch.load(p, map_location="cpu")
    print(f"[Checkpoint] Loaded ← {path}  (epoch {state.get('epoch', '?')})")
    return state


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    d = Path(checkpoint_dir)
    if not d.exists():
        return None
    epochs = sorted(d.glob("epoch_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    if epochs:
        return str(epochs[-1])
    best = d / "best.pt"
    return str(best) if best.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
# Training curve plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    train_losses,
    val_losses,
    val_eers=None,
    val_aucs=None,
    save_path: str = "curves.png",
    best_epoch: int | None = None,
    agent_name: str = "Agent",
):
    """
    3-panel figure: (Train/Val Loss) | (Val EER ↓) | (Val AUC ↑).
    Saved as PNG and displayed inline if running in Jupyter/Colab.
    """
    panels = [True, val_eers is not None, val_aucs is not None]
    n = sum(panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
    if n == 1:
        axes = [axes]

    epochs = list(range(1, len(train_losses) + 1))
    c = {"train": "#4C9EE8", "val": "#E8734C", "eer": "#C94CB8", "auc": "#48B882"}
    ax_idx = 0

    # Loss
    ax = axes[ax_idx]; ax_idx += 1
    ax.plot(epochs, train_losses, color=c["train"], lw=2, label="Train")
    ax.plot(epochs, val_losses, color=c["val"], lw=2, ls="--", label="Val")
    if best_epoch:
        ax.axvline(best_epoch, color="#999", ls=":", lw=1.5, label=f"Best ep {best_epoch}")
    ax.set(title=f"{agent_name} — Loss", xlabel="Epoch", ylabel="BCE Loss")
    ax.legend(); ax.grid(alpha=0.25)

    # EER
    if val_eers is not None:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(epochs, val_eers, color=c["eer"], lw=2, marker="o", ms=3)
        if best_epoch:
            ax.axvline(best_epoch, color="#999", ls=":", lw=1.5)
        ax.set(title=f"{agent_name} — Val EER ↓", xlabel="Epoch", ylabel="EER")
        ax.grid(alpha=0.25)

    # AUC
    if val_aucs is not None:
        ax = axes[ax_idx]
        ax.plot(epochs, val_aucs, color=c["auc"], lw=2, marker="s", ms=3)
        if best_epoch:
            ax.axvline(best_epoch, color="#999", ls=":", lw=1.5)
        ax.set(title=f"{agent_name} — Val AUC ↑", xlabel="Epoch", ylabel="AUC")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {save_path}")

    try:
        from IPython.display import Image, display
        display(Image(save_path))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Post-training evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_final_evaluation(
    model,
    test_loader,
    device,
    device_type: str,
    save_dir: str,
    agent_name: str = "Agent",
) -> dict:
    """
    Run model on full test split. Computes EER, AUC, Accuracy, F1.
    Saves eval_report.json and confusion_matrix.png to save_dir.
    """
    _root = Path(__file__).parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from fusion.evaluation import compute_eer, compute_auc
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, classification_report,
    )

    model.eval()
    all_labels, all_scores = [], []

    n = len(test_loader.dataset)
    print(f"\n[Eval] Evaluating {agent_name} on {n} test samples …")

    with torch.no_grad():
        for batch in test_loader:
            feats, labels = batch
            feats = feats.to(device)
            out = model(feats)
            if out.dim() > 1:
                out = out.squeeze(-1)
            all_scores.extend(out.cpu().numpy().tolist())
            all_labels.extend(
                labels.numpy().tolist() if hasattr(labels, "numpy") else list(labels)
            )

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    preds = (all_scores >= 0.5).astype(int)

    eer = compute_eer(all_labels, all_scores)
    auc = compute_auc(all_labels, all_scores)
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, zero_division=0)
    cm = confusion_matrix(all_labels, preds)

    # ── print report ──────────────────────────────────────────
    sep = "─" * 45
    print(f"\n{sep}")
    print(f"  Final Evaluation — {agent_name}")
    print(sep)
    print(f"  EER      : {eer * 100:.2f}%")
    print(f"  AUC      : {auc:.4f}")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  F1 Score : {f1:.4f}")
    print(sep)
    print(classification_report(all_labels, preds, target_names=["bonafide", "spoof"]))

    # ── save confusion matrix ──────────────────────────────────
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["bonafide", "spoof"],
        yticklabels=["bonafide", "spoof"],
        title=f"{agent_name} — Confusion Matrix",
        xlabel="Predicted", ylabel="True",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    cm_path = save_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Confusion matrix → {cm_path}")

    # ── save JSON report ───────────────────────────────────────
    report = {
        "agent": agent_name,
        "eer": round(float(eer), 6),
        "auc": round(float(auc), 6),
        "accuracy": round(float(acc), 6),
        "f1": round(float(f1), 6),
        "confusion_matrix": cm.tolist(),
    }
    rp_path = save_dir / "eval_report.json"
    with open(rp_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Eval] Report       → {rp_path}")

    try:
        from IPython.display import Image, display
        display(Image(str(cm_path)))
    except Exception:
        pass

    return report
