"""
linguistic/linguistic_model.py

Linguistic Agent for Deepfake Detection.
Uses OpenAI Whisper for ASR and a fine-tuned BERT for text classification.
"""

import torch
import librosa
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import numpy as np

class LinguisticAgent:
    def __init__(self, model_path=None, device=None):
        """
        Initialize the Linguistic Agent.
        If model_path is provided, it loads the fine-tuned BERT model.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # 1. Initialize Whisper for transcription
        # We use the 'small' model as it's a good balance of speed and accuracy
        print(f"[*] Loading Whisper-small on {self.device}...")
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=0 if self.device == "cuda" else -1
        )
        
        # 2. Initialize BERT for classification
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        if model_path and Path(model_path).exists():
            print(f"[*] Loading fine-tuned BERT from {model_path}...")
            self.classifier = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        else:
            print("[!] No fine-tuned model found. Using base BERT (needs training).")
            self.classifier = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(self.device)
            
    def transcribe(self, audio_path):
        """Transcribe an audio file by pre-loading it into an explicit raw dictionary."""
        try:
            # Load audio manually first to bypass the 'num_frames' header bug
            audio, _ = librosa.load(audio_path, sr=16000)
            
            # Pass as an explicit dictionary to prevent the pipeline from guessing file metadata
            result = self.transcriber({"raw": audio, "sampling_rate": 16000})
            return result["text"]
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
