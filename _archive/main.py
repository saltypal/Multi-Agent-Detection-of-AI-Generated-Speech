"""
main.py
Multi-agent deepfake speech detection — inference pipeline.

Usage
-----
  python main.py --audio path/to/audio.flac
  python main.py --audio path/to/audio.flac --config config.yaml --output result.json
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_models(cfg: dict, device):
    """Load all trained agent models from checkpoints."""
    ckpt_dir = Path(cfg["checkpoints"]["dir"])
    models = {}

    # Spectral
    from spectral.spectral_model import SpectralModel
    spec_ckpt = ckpt_dir / "spectral" / "best.pt"
    if spec_ckpt.exists():
        m = SpectralModel(n_features=100)
        state = torch.load(spec_ckpt, map_location="cpu")
        m.load_state_dict(state["model_state_dict"])
        models["spectral"] = m.eval().to(device)
        print(f"[Main] Spectral model loaded from {spec_ckpt}")
    else:
        print(f"[Main] WARNING: Spectral checkpoint not found at {spec_ckpt}")

    # Prosodic
    from prosodic.prosodic_model import ProsodicModel
    pros_ckpt = ckpt_dir / "prosodic" / "best.pt"
    if pros_ckpt.exists():
        m = ProsodicModel(n_features=5)
        state = torch.load(pros_ckpt, map_location="cpu")
        m.load_state_dict(state["model_state_dict"])
        models["prosodic"] = m.eval().to(device)
        print(f"[Main] Prosodic model loaded from {pros_ckpt}")
    else:
        print(f"[Main] WARNING: Prosodic checkpoint not found at {pros_ckpt}")

    # Linguistic
    from linguistic.linguistic_model import LinguisticClassifier
    ling_ckpt = ckpt_dir / "linguistic" / "best.pt"
    if ling_ckpt.exists():
        m = LinguisticClassifier(bert_name=cfg["models"].get("bert", "bert-base-uncased"))
        state = torch.load(ling_ckpt, map_location="cpu")
        m.load_state_dict(state["model_state_dict"])
        models["linguistic_clf"] = m.eval().to(device)
        print(f"[Main] Linguistic classifier loaded from {ling_ckpt}")
    else:
        print(f"[Main] WARNING: Linguistic checkpoint not found at {ling_ckpt}")

    # Fusion agent
    fusion_pkl = ckpt_dir / "fusion" / "fusion_agent.pkl"
    if fusion_pkl.exists():
        from fusion.decision_agent import DecisionAgent
        models["fusion"] = DecisionAgent.load(str(fusion_pkl))
    else:
        from fusion.decision_agent import DecisionAgent
        models["fusion"] = DecisionAgent(
            mode=cfg["fusion"]["mode"],
            weights=cfg["fusion"]["weights"],
        )
        print("[Main] Using default fusion weights (no trained meta-classifier found)")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Per-agent inference functions (run in parallel threads)
# ─────────────────────────────────────────────────────────────────────────────

def _run_spectral(models, chunks, sr, cfg):
    from preprocessor.feature_extraction import extract_spectral_features
    from spectral.spectral_explainability import explain_spectral

    model = models.get("spectral")
    if model is None:
        return "spectral", 0.5, {}

    feature_chunks = [extract_spectral_features(c, sr=sr) for c in chunks]
    score = model.predict(feature_chunks)
    expl = explain_spectral(score, feature_chunks[0]) if feature_chunks else {}
    return "spectral", score, expl


def _run_prosodic(models, chunks, sr, cfg):
    from preprocessor.feature_extraction import extract_prosodic_features
    from prosodic.prosodic_explainability import explain_prosodic

    model = models.get("prosodic")
    if model is None:
        return "prosodic", 0.5, {}

    feature_chunks = [extract_prosodic_features(c, sr=sr) for c in chunks]
    score = model.predict(feature_chunks)
    expl = explain_prosodic(score, chunks[0], sr=sr) if chunks else {}
    return "prosodic", score, expl


def _run_linguistic(models, audio_path, waveform, sr, cfg):
    from preprocessor.audio_utils import load_audio
    from linguistic.linguistic_model import extract_text_features
    from linguistic.linguistic_explainability import explain_linguistic
    from transformers import (
        WhisperProcessor, WhisperForConditionalGeneration,
        AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel,
    )
    import torch

    clf = models.get("linguistic_clf")
    if clf is None:
        return "linguistic", 0.5, {}

    device_str = str(next(clf.parameters()).device)

    whisper_name = cfg["models"].get("whisper", "openai/whisper-small")
    w_proc  = WhisperProcessor.from_pretrained(whisper_name)
    w_model = WhisperForConditionalGeneration.from_pretrained(whisper_name).eval()
    w_model = w_model.to(device_str)

    inputs = w_proc(waveform, sampling_rate=sr, return_tensors="pt").input_features.to(device_str)
    with torch.no_grad():
        ids = w_model.generate(inputs)
    transcript = w_proc.batch_decode(ids, skip_special_tokens=True)[0]

    bert_name = cfg["models"].get("bert", "bert-base-uncased")
    gpt2_name = cfg["models"].get("gpt2", "gpt2")
    bert_tok  = AutoTokenizer.from_pretrained(bert_name)
    gpt2_tok  = GPT2Tokenizer.from_pretrained(gpt2_name)
    gpt2_mod  = GPT2LMHeadModel.from_pretrained(gpt2_name).eval().to(device_str)

    hand = extract_text_features(transcript, gpt2_tok, gpt2_mod, device=device_str)
    hand_t = torch.from_numpy(hand).unsqueeze(0).to(device_str)

    enc = bert_tok(transcript, return_tensors="pt", truncation=True,
                   max_length=512, padding="max_length")
    input_ids = enc["input_ids"].to(device_str)
    attn_mask = enc["attention_mask"].to(device_str)

    with torch.no_grad():
        score = clf(input_ids, attn_mask, hand_t).item()

    expl = explain_linguistic(score, transcript, hand)
    return "linguistic", score, expl


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(audio_path: str, cfg: dict, output_path: str | None = None) -> dict:
    t0 = time.time()

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Device: {device}")

    # ── Load audio ────────────────────────────────────────────────────────────
    from preprocessor.audio_utils import load_audio, chunk_audio
    sr = cfg["audio"]["sample_rate"]
    waveform, _ = load_audio(audio_path, target_sr=sr)
    chunks = chunk_audio(waveform, sr,
                         cfg["audio"]["chunk_duration"],
                         cfg["audio"]["chunk_overlap"])
    print(f"[Main] Loaded '{audio_path}' → {len(chunks)} chunks")

    # ── Load models ───────────────────────────────────────────────────────────
    models = _load_models(cfg, device)

    # ── Parallel agent inference ──────────────────────────────────────────────
    print("[Main] Running agents in parallel …")
    agent_scores: dict[str, float] = {}
    agent_expls:  dict[str, dict]  = {}

    tasks = {
        "spectral":   (_run_spectral,  (models, chunks, sr, cfg)),
        "prosodic":   (_run_prosodic,  (models, chunks, sr, cfg)),
        "linguistic": (_run_linguistic,(models, audio_path, waveform, sr, cfg)),
    }

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(fn, *args): name
            for name, (fn, args) in tasks.items()
        }
        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                name, score, expl = future.result()
                agent_scores[name] = score
                agent_expls[name]  = expl
                print(f"[Main]   {name:12s}: score={score:.4f}")
            except Exception as e:
                print(f"[Main]   {agent_name}: ERROR — {e}")
                agent_scores[agent_name] = 0.5
                agent_expls[agent_name]  = {"explanation": f"Error: {e}"}

    # ── Fusion ────────────────────────────────────────────────────────────────
    fusion_agent = models["fusion"]
    decision = fusion_agent.decide(agent_scores)

    # ── Explanation ───────────────────────────────────────────────────────────
    from fusion.explainability import generate_explanation, print_explanation, save_explanation

    result = generate_explanation(
        p_final=decision["p_final"],
        decision=decision["decision"],
        agent_scores=agent_scores,
        agent_explanations=agent_expls,
        threshold=cfg["fusion"].get("decision_threshold", 0.5),
    )
    result["inference_time_s"] = round(time.time() - t0, 2)

    print_explanation(result)

    if output_path:
        save_explanation(result, output_path)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Deepfake Speech Detector")
    parser.add_argument("--audio",  required=True, help="Path to input audio file")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output", default=None, help="Save JSON result to this path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_detection(args.audio, cfg, output_path=args.output)
    print("\n" + json.dumps(result, indent=2))
