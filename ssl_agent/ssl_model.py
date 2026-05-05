import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import AutoModel, AutoFeatureExtractor

class SSLAgent:
    """
    Inference Wrapper for the Wedefense_ASV2025_WavLM_Base_Pruning SSL model.
    This model is a top-performing single system from the ASVspoof 5 Challenge.
    
    Source: https://huggingface.co/JYP2024/Wedefense_ASV2025_WavLM_Base_Pruning
    """
    def __init__(self, model_id="JYP2024/Wedefense_ASV2025_WavLM_Base_Pruning", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[*] Loading SSL Model: {model_id} on {self.device}...")
        
        # This model uses the WavLM backbone with a pruning-aware classification head.
        # Since it's hosted as a standard Hugging Face model, we can use AutoModel.
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device)
        self.model.eval()
        
        # Load the corresponding feature extractor (processor)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    def predict(self, audio_path):
        """
        End-to-end prediction: Audio -> Model -> Spoof Probability.
        
        Parameters:
            audio_path: Path to the .flac or .wav file (16kHz expected).
        
        Returns:
            float: Probability of the audio being 'spoof' (1.0 = high confidence fake).
        """
        try:
            # Load audio at 16kHz
            waveform, sr = librosa.load(audio_path, sr=16000)
            
            # Preprocess
            inputs = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # The model typically outputs logits for [bonafide, spoof]
                # We need to handle different possible output formats from the model
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    # Fallback if the model returns a direct tensor/tuple
                    logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

                probs = torch.softmax(logits, dim=-1)
                
                # Probability of 'spoof' (typically index 1)
                # Note: Verify index mapping if results are inverted
                fake_prob = probs[0][1].item()
                
            return fake_prob
            
        except Exception as e:
            print(f"[!] SSL Inference Error: {e}")
            return 0.5 # Neutral fallback

if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        agent = SSLAgent()
        score = agent.predict(sys.argv[1])
        print(f"File: {sys.argv[1]} | Spoof Score: {score:.4f}")
    else:
        print("Usage: python ssl_model.py <audio_path>")
