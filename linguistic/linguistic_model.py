"""
linguistic/linguistic_model.py

Linguistic Agent for Deepfake Detection.
Uses OpenAI Whisper for ASR and a fine-tuned BERT for text classification.
"""

import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import numpy as np

class LinguisticAgent:
    def __init__(self, model_path=None, device=None):
        """
        Initialize the Linguistic Agent.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Whisper directly (skipping the buggy pipeline)
        print(f"[*] Loading Whisper-Tiny on {self.device}...")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(self.device)
        
        # Load BERT for classification
        if model_path:
            print(f"[*] Loading fine-tuned BERT from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.classifier = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        else:
            print("[*] Loading base BERT (for inference only)...")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.classifier = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(self.device)
            
    def transcribe(self, audio_path):
        """Transcribe audio by bypassing the pipeline entirely."""
        try:
            # 1. Load audio with librosa
            audio, _ = librosa.load(audio_path, sr=16000)
            
            # 2. Extract features
            input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
            
            # 3. Generate transcription tokens
            predicted_ids = self.whisper_model.generate(input_features)
            
            # 4. Decode to text
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription
        except Exception as e:
            print(f"[!] Transcription error on {audio_path}: {e}")
            return ""
    
    def predict_text(self, text):
        """Classify a piece of text as real (0) or fake (1)."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # Probability of 'fake' (label 1)
            fake_prob = probs[0][1].item()
        return fake_prob

    def predict(self, audio_path):
        """End-to-end prediction: Audio -> Text -> Probability."""
        text = self.transcribe(audio_path)
        if not text or not text.strip():
            return 0.0 # Return 0 (Bonafide) if transcription fails as requested
        return self.predict_text(text)

def load_model(path):
    return LinguisticAgent(model_path=path)
