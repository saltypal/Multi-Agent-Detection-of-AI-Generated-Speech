"""
MAD.py - Multi-Agent Detection of AI-Generated Speech
Main inference entry point.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import yaml
import numpy as np

# Import agents
from spectral.spectral_model import SpectralModel
from prosodic.prosodic_model import ProsodicModel
from linguistic.linguistic_model import LinguisticClassifier, extract_text_features
from preprocessor.audio_utils import load_audio, chunk_audio
from preprocessor.feature_extraction import extract_spectral_features, extract_prosodic_features
from fusion.explainability import generate_explanation, print_explanation

# Default weights if results.txt is not found
DEFAULT_WEIGHTS = {
    "spectral": 0.4,
    "prosodic": 0.2,
    "linguistic": 0.4
}

class MADSystem:
    def __init__(self, config_path="config.yaml", device=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = self._load_all_models()
        self.weights = self._calculate_fusion_weights()

    def _load_all_models(self):
        models = {}
        project_root = Path(".")

        # Spectral
        spec_path = project_root / "spectral" / "results" / "best.pt"
        if spec_path.exists():
            m = SpectralModel(n_features=100).to(self.device)
            ckpt = torch.load(spec_path, map_location=self.device)
            m.load_state_dict(ckpt["model_state_dict"])
            models["spectral"] = m.eval()
            print("[MAD] Spectral model loaded.")

        # Prosodic
        pros_path = project_root / "prosodic" / "results" / "best.pt"
        if pros_path.exists():
            m = ProsodicModel(n_features=5).to(self.device)
            ckpt = torch.load(pros_path, map_location=self.device)
            m.load_state_dict(ckpt["model_state_dict"])
            models["prosodic"] = m.eval()
            print("[MAD] Prosodic model loaded.")

        # Linguistic
        ling_path = project_root / "linguistic" / "results" / "best.pt"
        if ling_path.exists():
            m = LinguisticClassifier(bert_name=self.config["models"].get("bert", "bert-base-uncased")).to(self.device)
            ckpt = torch.load(ling_path, map_location=self.device)
            m.load_state_dict(ckpt["model_state_dict"])
            models["linguistic"] = m.eval()
            print("[MAD] Linguistic model loaded.")

        return models

    def _calculate_fusion_weights(self):
        """
        Reads results.txt from each agent's results folder.
        Uses 1/EER as a basis for weighting.
        """
        eers = {}
        for agent in ["spectral", "prosodic", "linguistic"]:
            res_path = Path(agent) / "results" / "results.txt"
            if res_path.exists():
                try:
                    with open(res_path, 'r') as f:
                        for line in f:
                            if "EER" in line:
                                # Expecting "EER: 0.05" or similar
                                val = float(line.split(":")[-1].strip().replace("%", ""))
                                if "%" in line: val /= 100.0
                                eers[agent] = val
                                break
                except:
                    pass
        
        if not eers:
            print("[MAD] No results.txt found for agents. Using default weights.")
            return DEFAULT_WEIGHTS
        
        # Calculate weights based on 1/EER (lower EER -> higher weight)
        inv_eers = {k: 1.0 / (v + 1e-6) for k, v in eers.items()}
        total = sum(inv_eers.values())
        weights = {k: v / total for k, v in inv_eers.items()}
        
        # Merge with defaults for missing ones
        final_weights = {}
        for agent in DEFAULT_WEIGHTS:
            final_weights[agent] = weights.get(agent, 0.0)
        
        # Re-normalize if some agents are missing
        final_total = sum(final_weights.values())
        if final_total > 0:
            final_weights = {k: v / final_total for k, v in final_weights.items()}
        else:
            final_weights = DEFAULT_WEIGHTS

        print(f"[MAD] Calculated Fusion Weights: {final_weights}")
        return final_weights

    def _run_spectral(self, chunks):
        if "spectral" not in self.models: return 0.5, {"explanation": "Model not loaded"}
        from spectral.spectral_explainability import explain_spectral
        
        feats = [extract_spectral_features(c) for c in chunks]
        score = self.models["spectral"].predict(feats)
        expl = explain_spectral(score, feats[0])
        return score, expl

    def _run_prosodic(self, chunks):
        if "prosodic" not in self.models: return 0.5, {"explanation": "Model not loaded"}
        from prosodic.prosodic_explainability import explain_prosodic
        
        feats = [extract_prosodic_features(c) for c in chunks]
        score = self.models["prosodic"].predict(feats)
        expl = explain_prosodic(score, chunks[0])
        return score, expl

    def _run_linguistic(self, waveform):
        if "linguistic" not in self.models: return 0.5, {"explanation": "Model not loaded"}
        from linguistic.linguistic_model import LinguisticModel
        from linguistic.linguistic_explainability import explain_linguistic
        
        # We need Whisper to transcribe
        # For simplicity in this example, assume we have a helper or use LinguisticModel's logic
        # Ideally, MADSystem would have a Whisper instance or load one
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        w_name = self.config["models"].get("whisper", "openai/whisper-small")
        proc = WhisperProcessor.from_pretrained(w_name)
        w_model = WhisperForConditionalGeneration.from_pretrained(w_name).to(self.device)
        
        inputs = proc(waveform, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        with torch.no_grad():
            ids = w_model.generate(inputs)
        transcript = proc.batch_decode(ids, skip_special_tokens=True)[0]
        
        # Now use the BERT classifier
        bert_name = self.config["models"].get("bert", "bert-base-uncased")
        from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel
        tokenizer = AutoTokenizer.from_pretrained(bert_name)
        g_tok = GPT2Tokenizer.from_pretrained("gpt2")
        g_mod = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device).eval()
        
        hand = extract_text_features(transcript, g_tok, g_mod, device=str(self.device))
        hand_t = torch.from_numpy(hand).unsqueeze(0).to(self.device)
        
        enc = tokenizer(transcript, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        with torch.no_grad():
            score = self.models["linguistic"](enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device), hand_t).item()
        
        expl = explain_linguistic(score, transcript, hand)
        return score, expl

    def detect(self, audio_path):
        waveform, sr = load_audio(audio_path)
        chunks = chunk_audio(waveform, sr)
        
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_spec = executor.submit(self._run_spectral, chunks)
            future_pros = executor.submit(self._run_prosodic, chunks)
            future_ling = executor.submit(self._run_linguistic, waveform)
            
            results["spectral"] = future_spec.result()
            results["prosodic"] = future_pros.result()
            results["linguistic"] = future_ling.result()
            
        # Fusion
        final_score = 0.0
        agent_scores = {}
        agent_expls = {}
        for agent, (score, expl) in results.items():
            final_score += score * self.weights.get(agent, 0.0)
            agent_scores[agent] = score
            agent_expls[agent] = expl
            
        decision = "spoof" if final_score >= 0.5 else "bonafide"
        
        explanation = generate_explanation(
            final_score, decision, agent_scores, agent_expls
        )
        
        return explanation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    
    mad = MADSystem(args.config)
    result = mad.detect(args.audio)
    print_explanation(result)
