"""
linguistic/linguistic_model.py
Whisper (ASR) + BERT text classifier pipeline for linguistic deepfake detection.
"""

from __future__ import annotations
import re
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel,
    WhisperProcessor, WhisperForConditionalGeneration,
    GPT2Tokenizer, GPT2LMHeadModel,
)


# ─────────────────────────────────────────────────────────────────────────────
# BERT-based text classifier head
# ─────────────────────────────────────────────────────────────────────────────

class LinguisticClassifier(nn.Module):
    """
    Takes BERT [CLS] embedding (768) + hand-crafted text features (5)
    → concatenate → FC → sigmoid probability.
    """

    N_HAND = 5   # perplexity, repetition, disfluency, ttr, sent_len

    def __init__(self, bert_name: str = "bert-base-uncased", dropout: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        bert_dim = self.bert.config.hidden_size   # 768

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_dim + self.N_HAND, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hand_features: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (B,) probabilities."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]    # (B, 768)
        x = torch.cat([cls, hand_features], dim=-1)   # (B, 768+5)
        return torch.sigmoid(self.head(x).squeeze(-1))


# ─────────────────────────────────────────────────────────────────────────────
# Text feature extractor (hand-crafted)
# ─────────────────────────────────────────────────────────────────────────────

_DISFLUENCY_PATTERNS = re.compile(
    r'\b(um+|uh+|er+|ah+|hmm+|like|you know|i mean|sort of|kind of)\b',
    re.IGNORECASE,
)


def _compute_perplexity(text: str, tokenizer, model, device: str = "cpu") -> float:
    """Estimate perplexity using GPT-2 (lower perplexity = more fluent)."""
    if not text.strip():
        return 1000.0
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            loss = model(**enc, labels=enc["input_ids"]).loss
        return float(torch.exp(loss).item())
    except Exception:
        return 500.0


def extract_text_features(
    transcript: str,
    gpt2_tokenizer=None,
    gpt2_model=None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Returns np.ndarray of shape (5,):
      [0] perplexity (clamped log-scale)
      [1] bigram repetition rate
      [2] disfluency rate
      [3] type-token ratio
      [4] normalised mean sentence length
    """
    words = transcript.lower().split() if transcript.strip() else []
    n_words = len(words) + 1e-9

    # Perplexity
    if gpt2_tokenizer is not None and gpt2_model is not None:
        ppl = _compute_perplexity(transcript, gpt2_tokenizer, gpt2_model, device)
    else:
        ppl = 100.0
    ppl_norm = min(math.log(ppl + 1) / 10.0, 1.0)  # log-normalise to [0,1]

    # Bigram repetition rate
    if len(words) >= 2:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        unique_bigrams = len(set(bigrams))
        rep_rate = 1.0 - unique_bigrams / (len(bigrams) + 1e-9)
    else:
        rep_rate = 0.0

    # Disfluency rate
    disfluency_matches = len(_DISFLUENCY_PATTERNS.findall(transcript))
    disfluency_rate = disfluency_matches / n_words

    # Type-token ratio
    ttr = len(set(words)) / n_words

    # Mean sentence length (normalised)
    sentences = re.split(r'[.!?]+', transcript.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        mean_sent_len = sum(len(s.split()) for s in sentences) / len(sentences)
    else:
        mean_sent_len = n_words
    sent_len_norm = min(mean_sent_len / 50.0, 1.0)

    return np.array([ppl_norm, rep_rate, disfluency_rate, 1.0 - ttr, sent_len_norm],
                    dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Full Linguistic pipeline
# ─────────────────────────────────────────────────────────────────────────────

class LinguisticModel:
    """
    Full inference pipeline:
      audio file → Whisper ASR → transcript → BERT + hand features → p_ling
    """

    def __init__(
        self,
        whisper_name: str = "openai/whisper-small",
        bert_name: str = "bert-base-uncased",
        gpt2_name: str = "gpt2",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Linguistic] Loading Whisper ({whisper_name}) …")
        self.whisper_proc = WhisperProcessor.from_pretrained(whisper_name)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_name)
        self.whisper_model.eval().to(self.device)

        print(f"[Linguistic] Loading BERT classifier ({bert_name}) …")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        self.classifier = LinguisticClassifier(bert_name=bert_name).to(self.device)

        print(f"[Linguistic] Loading GPT-2 ({gpt2_name}) for perplexity …")
        self.gpt2_tok = GPT2Tokenizer.from_pretrained(gpt2_name)
        self.gpt2_mod = GPT2LMHeadModel.from_pretrained(gpt2_name).eval().to(self.device)

    def transcribe(self, audio: np.ndarray, sr: int = 16_000) -> str:
        """Transcribe raw waveform to text using Whisper."""
        inputs = self.whisper_proc(audio, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        with torch.no_grad():
            ids = self.whisper_model.generate(input_features)
        return self.whisper_proc.batch_decode(ids, skip_special_tokens=True)[0]

    def predict(self, transcript: str, return_features: bool = False):
        """
        Parameters
        ----------
        transcript : ASR transcript string
        Returns p_ling ∈ [0,1] (and optionally hand features).
        """
        hand = extract_text_features(
            transcript, self.gpt2_tok, self.gpt2_mod, self.device
        )
        hand_t = torch.from_numpy(hand).unsqueeze(0).to(self.device)  # (1,5)

        enc = self.tokenizer(
            transcript, return_tensors="pt",
            truncation=True, max_length=512, padding=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attn_mask = enc["attention_mask"].to(self.device)

        self.classifier.eval()
        with torch.no_grad():
            prob = self.classifier(input_ids, attn_mask, hand_t).item()

        if return_features:
            return float(prob), hand
        return float(prob)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_trained_classifier(
    checkpoint_path: str | Path,
    device: str = "cpu",
    bert_name: str = "bert-base-uncased",
) -> LinguisticClassifier:
    state = torch.load(checkpoint_path, map_location=device)
    clf = LinguisticClassifier(bert_name=bert_name)
    clf.load_state_dict(state["model_state_dict"])
    clf.to(device).eval()
    print(f"[Linguistic] Classifier loaded from {checkpoint_path}  (epoch {state.get('epoch','?')})")
    return clf
