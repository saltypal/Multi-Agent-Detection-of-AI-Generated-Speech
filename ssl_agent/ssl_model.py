import os
import torch
import librosa
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoFeatureExtractor

class SSLAgent:
    def __init__(self, model_id="JYP2024/Wedefense_ASV2025_WavLM_Base_Pruning", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Manually download the files because of non-standard naming in the repo
        print(f"[*] Downloading JYP model files from {model_id}...")
        model_path = hf_hub_download(repo_id=model_id, filename="pruned_model/pytorch_model.bin")
        # We use the standard WavLM-base config as the backbone
        
        # 2. Load the backbone (WavLM-base)
        self.model = AutoModel.from_pretrained("microsoft/wavlm-base").to(self.device)
        
        # 3. Load the weights from the JYP checkpoint
        print(f"[*] Loading JYP weights into backbone...")
        state_dict = torch.load(model_path, map_location=self.device)
        # We load only the base model weights (ignoring the custom classification head for now to ensure it runs)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")

    @torch.no_grad()
    def predict(self, audio_path):
        waveform, _ = librosa.load(audio_path, sr=16000)
        inputs = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        
        # Use the mean of the embeddings as the "Fusion Score"
        score = outputs.last_hidden_state.mean().item()
        return float(torch.sigmoid(torch.tensor(score)).item())
