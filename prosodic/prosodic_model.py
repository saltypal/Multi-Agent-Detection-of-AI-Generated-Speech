"""
prosodic/prosodic_model.py

Inference module for the Prosodic deepfake detection agent.
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")


try:

    # pyrefly: ignore [missing-import]
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    # pyrefly: ignore [missing-import]
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    # pyrefly: ignore [missing-import]
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False


class ProsodicAgent:
    def __init__(self, model_dir: str | Path):
        """Initialize the Prosodic Agent and load all available trained booster models."""
        self.model_dir = Path(model_dir)
        self.cols_path = self.model_dir / "feature_cols.json"
        
        self.feat_cols = None
        if self.cols_path.exists():
            with open(self.cols_path) as f:
                self.feat_cols = json.load(f)
                
        # Load XGBoost
        self.xgb_model = None
        self.xgb_path = self.model_dir / "best_prosodic_xgb.json"
        if HAS_XGB and self.xgb_path.exists():
            try:
                import torch
                dev = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.xgb_model = xgb.XGBClassifier(device=dev)
                self.xgb_model.load_model(str(self.xgb_path))
            except Exception as e:
                print(f"[!] Warning: Failed to load XGBoost: {e}")
                
        # Load LightGBM
        self.lgb_model = None
        self.lgb_path = self.model_dir / "best_prosodic_lgb.txt"
        if HAS_LGB and self.lgb_path.exists():
            try:
                self.lgb_model = lgb.Booster(model_file=str(self.lgb_path))
            except Exception as e:
                print(f"[!] Warning: Failed to load LightGBM: {e}")
                
        # Load CatBoost
        self.cb_model = None
        self.cb_path = self.model_dir / "best_prosodic_cat.json"
        if HAS_CB and self.cb_path.exists():
            try:
                self.cb_model = cb.CatBoostClassifier()
                self.cb_model.load_model(str(self.cb_path))
            except Exception as e:
                print(f"[!] Warning: Failed to load CatBoost: {e}")
                
        print(f"[*] Prosodic Agent loaded: XGB={self.xgb_model is not None}, LGB={self.lgb_model is not None}, CB={self.cb_model is not None}")

    def predict_features(self, feature_dict: dict, use_best_only: bool = True) -> float:
        """Predict spoof probability directly. Defaults to using the best-performing booster (XGBoost, EER: 46.59%)."""
        # Use saved feature columns order, otherwise default to sorted keys
        cols = self.feat_cols if self.feat_cols else sorted(feature_dict.keys())
        X = np.array([[feature_dict.get(c, 0.0) for c in cols]])
        np.nan_to_num(X, copy=False)
        
        # If use_best_only is True and XGBoost is loaded, use it directly (best EER)
        if use_best_only and self.xgb_model:
            return float(self.xgb_model.predict_proba(X)[0, 1])
            
        probs = []
        if self.xgb_model:
            try:
                probs.append(float(self.xgb_model.predict_proba(X)[0, 1]))
            except Exception:
                pass
        if self.lgb_model:
            try:
                probs.append(float(self.lgb_model.predict(X)[0]))
            except Exception:
                pass
        if self.cb_model:
            try:
                probs.append(float(self.cb_model.predict_proba(X)[0, 1]))
            except Exception:
                pass
                
        if len(probs) == 0:
            return 0.5
            
        return float(np.mean(probs))

    def predict(self, audio_path: str | Path) -> float:
        """End-to-end prediction: Audio -> Extract Features -> Probability."""
        from prosodic.prosodic_feature_extractor import extract_prosodic_row
        row = extract_prosodic_row(audio_path)
        return self.predict_features(row)

    def predict_with_shap(self, audio_path: str | Path) -> tuple[float, dict[str, float]]:
        """End-to-end prediction with native Tree-SHAP value contributions for features."""
        from prosodic.prosodic_feature_extractor import extract_prosodic_row
        row = extract_prosodic_row(audio_path)
        
        cols = self.feat_cols if self.feat_cols else sorted(row.keys())
        X = np.array([[row.get(c, 0.0) for c in cols]])
        np.nan_to_num(X, copy=False)
        
        prob = float(self.xgb_model.predict_proba(X)[0, 1]) if self.xgb_model else 0.5
        
        shap_dict = {}
        if self.xgb_model:
            try:
                import xgboost as xgb
                booster = self.xgb_model.get_booster()
                dmat = xgb.DMatrix(X, feature_names=cols)
                contribs = booster.predict(dmat, pred_contribs=True)[0]
                
                # Zip feature names with SHAP values, excluding the last bias value
                shap_dict = {cols[i]: float(contribs[i]) for i in range(len(cols))}
            except Exception as e:
                print(f"[!] Warning calculating Prosodic Tree-SHAP: {e}")
                
        return prob, shap_dict
