"""
spectral/spectral_model.py

XGBoost-based classifier for spectral deepfake detection.
Operates on tabular features produced by spectral_feature_extractor.

Key API:
    train_model(train_csv, test_csv, out_dir)  → trained model
    predict_file(model_path, audio_path)        → float probability
    predict_row(model, feature_dict)            → float probability
    load_model(path)                            → XGBClassifier
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, accuracy_score, f1_score,
    auc as sklearn_auc, classification_report,
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# Try XGBoost, fall back to sklearn
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False
    print("[!] xgboost not found — falling back to sklearn GradientBoostingClassifier")


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_eer(labels, scores):
    """Equal Error Rate — threshold where FAR == FRR."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    try:
        return brentq(lambda x: interp1d(fpr, fnr - fpr)(x), 0, 1)
    except Exception:
        return float(np.mean(np.abs(fnr - fpr)))


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return list of numeric feature columns (excludes label, filename, split)."""
    return [c for c in df.columns if c not in ("label", "filename", "split")]


def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str | Path,
    val_df: pd.DataFrame | None = None,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    use_gpu: bool = False,
):
    """
    Train XGBoost on spectral features.

    Parameters
    ----------
    train_df  : DataFrame with feature columns + 'label'
    test_df   : DataFrame with feature columns + 'label'
    out_dir   : directory to save model + results
    val_df    : optional validation DataFrame for early stopping
    use_gpu   : if True, use 'gpu_hist' tree method

    Returns
    -------
    Trained model object
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_cols = get_feature_cols(train_df)
    X_train, y_train = train_df[feat_cols].values, train_df["label"].values
    X_test, y_test = test_df[feat_cols].values, test_df["label"].values

    # Clean NaN/inf
    for X in [X_train, X_test]:
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    eval_set = [(X_test, y_test)]
    if val_df is not None:
        X_val = val_df[feat_cols].values
        np.nan_to_num(X_val, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        y_val = val_df["label"].values
        eval_set = [(X_val, y_val)]

    print(f"[Spectral] Train: {X_train.shape}  Test: {X_test.shape}")

    if HAS_XGB:
        tree_method = "gpu_hist" if use_gpu else "hist"
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            early_stopping_rounds=20,
            use_label_encoder=False,
            tree_method=tree_method,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        model.save_model(str(out_dir / "best_xgb.json"))
        proba = model.predict_proba(X_test)[:, 1]
    else:
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
        )
        model.fit(X_train, y_train)
        import joblib
        joblib.dump(model, out_dir / "best_xgb.joblib")
        proba = model.predict_proba(X_test)[:, 1]

    # Evaluate
    preds = (proba >= 0.5).astype(int)
    eer = compute_eer(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba, pos_label=1)
    auc_val = sklearn_auc(fpr, tpr)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"\n{'='*50}")
    print(f"  Spectral Model — Test Results")
    print(f"{'='*50}")
    print(f"  EER      : {eer*100:.2f}%")
    print(f"  AUC      : {auc_val:.4f}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  F1 Score : {f1:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_test, preds, target_names=["bonafide", "spoof"]))

    # Save results
    results = {"eer": float(eer), "auc": float(auc_val), "accuracy": float(acc), "f1": float(f1)}
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
    with open(out_dir / "results.txt", "w") as f:
        f.write(f"EER: {eer*100:.4f}%\nAUC: {auc_val:.4f}\nAccuracy: {acc*100:.4f}%\nF1: {f1:.4f}\n")

    # Save feature column order for inference
    with open(out_dir / "feature_cols.json", "w") as f:
        json.dump(feat_cols, f)

    print(f"[Spectral] Results saved → {out_dir}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str | Path):
    """Load a trained XGBoost model from disk."""
    model_path = Path(model_path)
    if HAS_XGB:
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
    else:
        import joblib
        model = joblib.load(model_path)
    return model


def predict_row(model, feature_dict: dict, feat_cols: list[str] | None = None) -> float:
    """Predict spoof probability from a feature dictionary."""
    if feat_cols is None:
        feat_cols = sorted(feature_dict.keys())
    X = np.array([[feature_dict.get(c, 0.0) for c in feat_cols]])
    np.nan_to_num(X, copy=False)
    return float(model.predict_proba(X)[0, 1])


def predict_file(model_path: str | Path, audio_path: str | Path) -> float:
    """Load model and predict spoof probability for a single audio file."""
    from spectral.spectral_feature_extractor import extract_spectral_row

    model_path = Path(model_path)
    model = load_model(model_path)

    # Load feature column order
    cols_file = model_path.parent / "feature_cols.json"
    feat_cols = None
    if cols_file.exists():
        with open(cols_file) as f:
            feat_cols = json.load(f)

    row = extract_spectral_row(audio_path)
    return predict_row(model, row, feat_cols)
