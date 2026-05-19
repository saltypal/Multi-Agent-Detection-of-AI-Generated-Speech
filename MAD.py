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
        
        # 2. Load Fusion Meta-Classifier
        fusion_path = self.root / "fusion model" / "fusion_meta_model.pkl"
        if fusion_path.exists():
            print(f"[*] Loading Fusion Meta-Model from {fusion_path}...")
            self.meta_model = joblib.load(fusion_path)
        else:
            print("[!] Fusion model not found. Using simple average instead.")
            self.meta_model = None

    def detect(self, audio_path):
        """
        Runs the full multi-agent pipeline on a single audio file.
        Returns: {
            'final_decision': 'SPOOF' | 'BONAFIDE',
            'confidence': float,
            'agent_scores': dict
        }
        """
        # Run individual agents
        p_spec = self.spec_agent.predict(audio_path)
        p_pros = self.pros_agent.predict(audio_path)
        p_ling = self.ling_agent.predict(audio_path)
        p_ssl  = self.ssl_agent.predict(audio_path)
        
        scores = {
            'Spectral': p_spec,
            'Prosodic': p_pros,
            'Linguistic': p_ling,
            'SSL': p_ssl
        }
        
        # Fusion
        if self.meta_model:
            # Inputs must be in the same order as training
            X = np.array([[p_spec, p_pros, p_ling, p_ssl]])
            final_prob = self.meta_model.predict_proba(X)[0][1]
        else:
            # Simple Average Fallback
            final_prob = np.mean([p_spec, p_pros, p_ling, p_ssl])
            
        decision = "SPOOF" if final_prob > 0.5 else "BONAFIDE"
        confidence = final_prob if final_prob > 0.5 else (1 - final_prob)
        
        return {
            'final_decision': decision,
            'confidence': float(confidence),
            'agent_scores': scores
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
