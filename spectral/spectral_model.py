"""
spectral/spectral_model.py

Inference module for the Spectral deepfake detection agent.
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[!] xgboost not found.")


class SpectralAgent:
    def __init__(self, model_dir: str | Path):
        """Initialize the Spectral Agent and load the trained XGBoost model."""
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "best_xgb.json"
        self.cols_path = self.model_dir / "feature_cols.json"
        
        self.feat_cols = None
        if self.cols_path.exists():
            with open(self.cols_path) as f:
                self.feat_cols = json.load(f)
                
        if HAS_XGB:
            self.model = xgb.XGBClassifier()
            if self.model_path.exists():
                self.model.load_model(str(self.model_path))
            else:
                print(f"[!] Warning: Model file not found at {self.model_path}")
        else:
            self.model = None
            print("[!] Cannot load XGBoost model because xgboost is not installed.")

    def predict_features(self, feature_dict: dict) -> float:
        """Predict spoof probability directly from a dictionary of extracted features."""
        if not self.model:
            return 0.5
        
        # Use saved feature columns order, otherwise default to sorted keys
        cols = self.feat_cols if self.feat_cols else sorted(feature_dict.keys())
        X = np.array([[feature_dict.get(c, 0.0) for c in cols]])
        np.nan_to_num(X, copy=False)
        return float(self.model.predict_proba(X)[0, 1])

    def predict(self, audio_path: str | Path) -> float:
        """End-to-end prediction: Audio -> Extract Features -> Probability."""
        from spectral.spectral_feature_extractor import extract_spectral_row
        row = extract_spectral_row(audio_path)
        return self.predict_features(row)
