"""
Spectral Structured Model — XGBoost classifier on tabular spectral features.

Usage:
    # Train (from CSVs produced by extract_features.ipynb)
    python spectral_s_model.py --train --data_dir /path/to/csvs --out_dir ./results

    # Inference on a single audio file
    python spectral_s_model.py --predict --model best_xgb.json --audio /path/to/file.flac
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from sklearn.metrics import (
    roc_curve, accuracy_score, f1_score, auc as sklearn_auc,
    classification_report,
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# Try XGBoost first, fall back to sklearn GradientBoosting
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False
    print("[!] xgboost not found — using sklearn GradientBoostingClassifier")

SR = 16000

# ═══════════════════════════════════════════════════════════════════════════════
# Feature extraction (mirrors extract_features.ipynb logic exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_spectral_row(path_or_waveform, sr=SR):
    """Extract ~110 spectral features from one audio file or waveform → dict."""
    if isinstance(path_or_waveform, (str, Path)):
        y, _ = librosa.load(str(path_or_waveform), sr=sr, mono=True)
    else:
        y = path_or_waveform
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak

    row = {}

    # 1. MFCC (20 coeffs) — mean, std
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=512, hop_length=160)
    for i in range(20):
        row[f"mfcc{i}_mean"] = float(mfcc[i].mean())
        row[f"mfcc{i}_std"] = float(mfcc[i].std())
    # Delta MFCC
    dmfcc = librosa.feature.delta(mfcc)
    for i in range(20):
        row[f"dmfcc{i}_mean"] = float(dmfcc[i].mean())
        row[f"dmfcc{i}_std"] = float(dmfcc[i].std())

    # 2. Spectral descriptors
    for name, fn in [
        ("centroid", librosa.feature.spectral_centroid),
        ("bandwidth", librosa.feature.spectral_bandwidth),
        ("rolloff", librosa.feature.spectral_rolloff),
        ("flatness", librosa.feature.spectral_flatness),
    ]:
        feat = fn(y=y, sr=sr)[0]
        row[f"{name}_mean"] = float(feat.mean())
        row[f"{name}_std"] = float(feat.std())
        row[f"{name}_max"] = float(feat.max())

    # 3. Spectral contrast (7 bands)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(contrast.shape[0]):
        row[f"contrast{i}_mean"] = float(contrast[i].mean())
        row[f"contrast{i}_std"] = float(contrast[i].std())

    # 4. Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    row["zcr_mean"] = float(zcr.mean())
    row["zcr_std"] = float(zcr.std())

    # 5. RMS energy
    rms = librosa.feature.rms(y=y)[0]
    row["rms_mean"] = float(rms.mean())
    row["rms_std"] = float(rms.std())
    row["rms_max"] = float(rms.max())

    # 6. Chroma (12 bins)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        row[f"chroma{i}_mean"] = float(chroma[i].mean())

    # 7. Mel spectrogram global stats
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    row["mel_mean"] = float(mel_db.mean())
    row["mel_std"] = float(mel_db.std())
    row["mel_max"] = float(mel_db.max())
    row["mel_min"] = float(mel_db.min())

    return row


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    try:
        return brentq(lambda x: interp1d(fpr, fnr - fpr)(x), 0, 1)
    except Exception:
        return float(np.mean(np.abs(fnr - fpr)))


def save_results(out_dir, eer, auc_val, acc, f1):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.txt", "w") as f:
        f.write(f"EER: {eer*100:.4f}%\nAUC: {auc_val:.4f}\nAccuracy: {acc*100:.4f}%\nF1: {f1:.4f}\n")
    with open(out_dir / "results.json", "w") as f:
        json.dump({"eer": eer, "auc": auc_val, "accuracy": acc, "f1": f1}, f, indent=4)
    print(f"Results saved → {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def get_feature_cols(df):
    """Return list of numeric feature columns (everything except label & filename)."""
    return [c for c in df.columns if c not in ("label", "filename")]


def train_model(data_dir: str, out_dir: str):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    feat_cols = get_feature_cols(train_df)
    X_train, y_train = train_df[feat_cols].values, train_df["label"].values
    X_val, y_val = val_df[feat_cols].values, val_df["label"].values
    X_test, y_test = test_df[feat_cols].values, test_df["label"].values

    # Replace NaN/inf
    for X in [X_train, X_val, X_test]:
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    if HAS_XGB:
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            early_stopping_rounds=20,
            use_label_encoder=False,
            tree_method="hist",  # fast
            n_jobs=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True,
        )
        model.save_model(str(out_dir / "best_xgb.json"))
        proba = model.predict_proba(X_test)[:, 1]
    else:
        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
        )
        model.fit(X_train, y_train)
        import joblib
        joblib.dump(model, out_dir / "best_xgb.joblib")
        proba = model.predict_proba(X_test)[:, 1]

    preds = (proba >= 0.5).astype(int)
    eer = compute_eer(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba, pos_label=1)
    auc_val = sklearn_auc(fpr, tpr)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"\n{'='*50}")
    print(f"Test EER:  {eer*100:.2f}%")
    print(f"Test AUC:  {auc_val:.4f}")
    print(f"Test Acc:  {acc*100:.2f}%")
    print(f"Test F1:   {f1:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_test, preds, target_names=["bonafide", "spoof"]))

    save_results(out_dir, eer, auc_val, acc, f1)

    # Save feature column order for inference
    with open(out_dir / "feature_cols.json", "w") as f:
        json.dump(feat_cols, f)

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════════

def predict_file(model_path: str, audio_path: str):
    """Load model + predict a single audio file → probability of spoof."""
    model_path = Path(model_path)

    if HAS_XGB:
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
    else:
        import joblib
        model = joblib.load(model_path)

    row = extract_spectral_row(audio_path)

    # Load feature column order
    cols_file = model_path.parent / "feature_cols.json"
    if cols_file.exists():
        with open(cols_file) as f:
            feat_cols = json.load(f)
    else:
        feat_cols = sorted(row.keys())

    X = np.array([[row.get(c, 0.0) for c in feat_cols]])
    np.nan_to_num(X, copy=False)

    proba = model.predict_proba(X)[0, 1]
    return float(proba)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectral XGBoost Model")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--out_dir", type=str, default="./results")
    parser.add_argument("--model", type=str, default="results/best_xgb.json")
    parser.add_argument("--audio", type=str, default="")
    args = parser.parse_args()

    if args.train:
        train_model(args.data_dir, args.out_dir)
    elif args.predict and args.audio:
        score = predict_file(args.model, args.audio)
        label = "SPOOF" if score >= 0.5 else "BONAFIDE"
        print(f"Score: {score:.4f}  →  {label}")
    else:
        parser.print_help()
