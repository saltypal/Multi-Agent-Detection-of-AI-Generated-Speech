import os
import sys
import torch
# pyrefly: ignore [missing-import]
import joblib
import numpy as np
from pathlib import Path

# Add project root to sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral.spectral_model import SpectralAgent
from prosodic.prosodic_model import ProsodicAgent
from linguistic.linguistic_model import LinguisticAgent
from ssl_agent.ssl_model import SSLAgent

class MultiAgentDetector:
    """
    Final Integrated Deepfake Detection System.
    Fuses 4 specialized agents (Spectral, Prosodic, Linguistic, SSL) 
    using a Meta-Classifier.
    """
    def __init__(self, models_root="trained_models"):
        self.root = Path(models_root)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[*] Initializing Multi-Agent System (Device: {self.device})...")
        
        # 1. Initialize Agents
        self.spec_agent = SpectralAgent(self.root / "spectral_model")
        self.pros_agent = ProsodicAgent(self.root / "prosodic_model")
        self.ling_agent = LinguisticAgent(self.root / "linguistic_bert_model", device=self.device)
        self.ssl_agent  = SSLAgent(device=self.device)
        
        # 2. Load Fusion Meta-Classifier (Prioritize Random Forest)
        fusion_path = self.root / "fusion model" / "rf_fusion_model.pkl"
        if not fusion_path.exists():
            fusion_path = self.root / "fusion model" / "fusion_meta_model.pkl"
            
        self.global_importances = None
        if fusion_path.exists():
            print(f"[*] Loading Fusion Meta-Model from {fusion_path}...")
            self.meta_model = joblib.load(fusion_path)
            # Warm up SHAP explainer
            try:
                import shap
                self.shap_explainer = shap.TreeExplainer(self.meta_model)
                print("[*] SHAP TreeExplainer initialized for fusion model.")
            except Exception as e:
                print(f"[!] Warning: Failed to initialize SHAP TreeExplainer: {e}")
                self.shap_explainer = None
                
            # Extract global agent importances
            if hasattr(self.meta_model, 'feature_importances_'):
                self.global_importances = {
                    'Spectral': float(self.meta_model.feature_importances_[0]),
                    'Prosodic': float(self.meta_model.feature_importances_[1]),
                    'Linguistic': float(self.meta_model.feature_importances_[2]),
                    'SSL': float(self.meta_model.feature_importances_[3])
                }
        else:
            print("[!] Fusion model not found. Using simple average instead.")
            self.meta_model = None
            self.shap_explainer = None

    def detect(self, audio_path):
        """
        Runs the full multi-agent pipeline on a single audio file.
        Returns: {
            'final_decision': 'SPOOF' | 'BONAFIDE',
            'confidence': float,
            'agent_scores': dict,
            'transcript': str
        }
        """
        # Run individual agents with native signal Tree-SHAP contributions
        p_spec, spec_shaps = self.spec_agent.predict_with_shap(audio_path)
        p_pros, pros_shaps = self.pros_agent.predict_with_shap(audio_path)
        
        # Extract transcript once and evaluate text
        transcript = self.ling_agent.transcribe(audio_path)
        if transcript:
            p_ling = self.ling_agent.predict_text(transcript)
        else:
            transcript = "[No speech detected]"
            p_ling = 0.0
            
        p_ssl  = self.ssl_agent.predict(audio_path)
        
        scores = {
            'Spectral': p_spec,
            'Prosodic': p_pros,
            'Linguistic': p_ling,
            'SSL': p_ssl
        }
        
        # Helper to isolate top 3 SHAP feature contributions
        def get_top_contributors(shap_dict, num=3):
            if not shap_dict:
                return {"fake": [], "real": []}
            sorted_features = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
            fake_contribs = [{"feature": f, "value": float(v)} for f, v in sorted_features[:num] if v > 0.001]
            real_contribs = [{"feature": f, "value": float(v)} for f, v in reversed(sorted_features) if v < -0.001][:num]
            return {"fake": fake_contribs, "real": real_contribs}
            
        # Fusion & SHAP explainability
        if self.meta_model:
            # Inputs must be in the same order as training
            X = np.array([[p_spec, p_pros, p_ling, p_ssl]])
            final_prob = self.meta_model.predict_proba(X)[0][1]
            
            # Compute Tree-SHAP values
            if hasattr(self, 'shap_explainer') and self.shap_explainer is not None:
                try:
                    raw_shaps = self.shap_explainer.shap_values(X)
                    if isinstance(raw_shaps, list):
                        # sklearn RF returns [Class 0 SHAPs, Class 1 SHAPs]
                        local_shaps = raw_shaps[1][0]
                    else:
                        if len(raw_shaps.shape) == 3:
                            local_shaps = raw_shaps[0, :, 1]
                        else:
                            local_shaps = raw_shaps[0]
                    
                    shap_values = {
                        'Spectral': float(local_shaps[0]),
                        'Prosodic': float(local_shaps[1]),
                        'Linguistic': float(local_shaps[2]),
                        'SSL': float(local_shaps[3])
                    }
                except Exception as e:
                    print(f"[!] Warning: Failed to calculate SHAP values: {e}")
                    shap_values = {
                        'Spectral': float(p_spec - 0.5),
                        'Prosodic': float(p_pros - 0.5),
                        'Linguistic': float(p_ling - 0.5),
                        'SSL': float(p_ssl - 0.5)
                    }
            elif hasattr(self.meta_model, 'coef_'):
                # Fallback for original Linear Regression
                coefs = self.meta_model.coef_[0]
                shap_values = {
                    'Spectral': float(coefs[0] * p_spec),
                    'Prosodic': float(coefs[1] * p_pros),
                    'Linguistic': float(coefs[2] * p_ling),
                    'SSL': float(coefs[3] * p_ssl)
                }
            else:
                shap_values = {
                    'Spectral': float(p_spec - 0.5),
                    'Prosodic': float(p_pros - 0.5),
                    'Linguistic': float(p_ling - 0.5),
                    'SSL': float(p_ssl - 0.5)
                }
        else:
            # Simple Average Fallback & Mock SHAP deviation
            final_prob = np.mean([p_spec, p_pros, p_ling, p_ssl])
            shap_values = {
                'Spectral': float(p_spec - 0.5),
                'Prosodic': float(p_pros - 0.5),
                'Linguistic': float(p_ling - 0.5),
                'SSL': float(p_ssl - 0.5)
            }
            
        decision = "SPOOF" if final_prob > 0.5 else "BONAFIDE"
        confidence = final_prob if final_prob > 0.5 else (1 - final_prob)
        
        return {
            'final_decision': decision,
            'confidence': float(confidence),
            'agent_scores': scores,
            'transcript': transcript,
            'shap_values': shap_values,
            'global_importances': self.global_importances,
            'spectral_shaps': get_top_contributors(spec_shaps),
            'prosodic_shaps': get_top_contributors(pros_shaps)
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent Deepfake Speech Detection")
    parser.add_argument("audio", help="Path to audio file (.flac or .wav)")
    args = parser.parse_args()
    
    mad = MultiAgentDetector()
    result = mad.detect(args.audio)
    
    print("\n" + "="*40)
    print(f"   FINAL DECISION: {result['final_decision']}")
    print(f"   CONFIDENCE:     {result['confidence']:.2%}")
    print("="*40)
    print("Agent Breakdowns:")
    for agent, score in result['agent_scores'].items():
        print(f" - {agent:<12}: {score:.4f}")
    print("="*40)
    print("SHAP TreeExplainer (Meta-Classifier Contributions):")
    for agent, shap_val in result['shap_values'].items():
        print(f" - {agent:<12}: {shap_val:+.4f}")
    if result.get('global_importances'):
        print("="*40)
        print("Global Model Importances:")
        for agent, imp in result['global_importances'].items():
            print(f" - {agent:<12}: {imp:.4f}")
    print("="*40)
