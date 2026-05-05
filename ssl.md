# SSL Agent Strategy: WavLM-Base-Pruning

We are incorporating the `JYP2024/Wedefense_ASV2025_WavLM_Base_Pruning` model as our SSL (Self-Supervised Learning) Agent. This model is highly valuable because it is one of the top-performing single systems from the official ASVspoof 5 Challenge leaderboard.

## Why this specific model?
1. **Raw Waveform Input**: It operates directly on the raw `.flac` audio at 16kHz. We do not need to manually engineer features for it.
2. **Already Fine-Tuned for Deepfakes**: Unlike a generic WavLM model (which only understands generic speech), this specific variant has already been explicitly trained on ASVspoof 5 using a Hybrid Pruning strategy.
3. **Advanced Backend**: It utilizes Multi-Head Factorized Attentive Pooling (MHFA) on top of the WavLM transformer embeddings to detect microscopic acoustic artifacts left by TTS engines.

## How we will integrate it
Because this model is already fully trained on our exact problem domain, **we do not need to fine-tune it**. We will use it purely as a robust, pre-trained inference engine.

### Pipeline Steps
1. **Architecture**: We will write an inference wrapper (`ssl_model.py`) that loads the Microsoft `wavlm-base` backbone and applies the JYP2024 pruned weights.
2. **Feature Extraction**: Pass raw 16kHz audio arrays into the model.
3. **Probability Score**: The model outputs logits. We will apply a Softmax function to convert this to a continuous float between `0.0` (100% human) and `1.0` (100% AI).
4. **Fusion**: This highly accurate probability will be combined with the Spectral, Prosodic, and Linguistic probabilities in the final meta-classifier.
