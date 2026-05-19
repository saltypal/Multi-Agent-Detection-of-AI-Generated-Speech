import os
import torch
    
# pyrefly: ignore [missing-import]
import librosa
from pathlib import Path

# pyrefly: ignore [missing-import]
from huggingface_hub import hf_hub_download
# pyrefly: ignore [missing-import]
from transformers import AutoModel, AutoFeatureExtractor

class SSLAgent:
    def __init__(self, model_id="JYP2024/Wedefense_ASV2025_WavLM_Base_Pruning", device=None, cache_dir="trained_models/hf_cache"):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Manually download the files because of non-standard naming in the repo
        print(f"[*] Loading SSL Model (Local Cache: {self.cache_dir})...")
        model_path = hf_hub_download(repo_id=model_id, filename="pruned_model/pytorch_model.bin", cache_dir=str(self.cache_dir))
        
        # 2. Load the backbone (WavLM-base) from cache
        self.model = AutoModel.from_pretrained("microsoft/wavlm-base", cache_dir=str(self.cache_dir)).to(self.device)
        
        # 3. Load the weights from the JYP checkpoint
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base", cache_dir=str(self.cache_dir))

    @torch.no_grad()
    def predict(self, audio_path):
        waveform, _ = librosa.load(audio_path, sr=16000)
        inputs = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_values)
        
        # Use the mean of the embeddings as the "Fusion Score"
        score = outputs.last_hidden_state.mean().item()
        return float(torch.sigmoid(torch.tensor(score)).item())
