# Multi-Agent Detection of AI-Generated (Deepfake) Speech

> Detect AI-generated speech using multiple specialized agents (spectral, prosodic, linguistic) whose outputs are fused by a decision agent.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [How Speech Signals Are Pre-Processed](#2-how-speech-signals-are-pre-processed)
3. [Speech Signal Components — Spectral, Prosodic, Linguistic & Beyond](#3-speech-signal-components--spectral-prosodic-linguistic--beyond)
4. [Models for Each Agent — Architectures & How They Work](#4-models-for-each-agent--architectures--how-they-work)
5. [Other Processable Speech Components](#5-other-processable-speech-components)
6. [Model Outputs, Evaluation & Performance](#6-model-outputs-evaluation--performance)
7. [Voting / Fusion — How All Models' Results Are Combined](#7-voting--fusion--how-all-models-results-are-combined)
8. [Explainability — How Each Agent's Decision Is Interpreted](#8-explainability--how-each-agents-decision-is-interpreted)
9. [Robustness to Unseen Generators](#9-robustness-to-unseen-generators)
10. [Which Metric — EER vs AUC — and Why](#10-which-metric--eer-vs-auc--and-why)
11. [Final Tech Stack — End-to-End Pipeline](#11-final-tech-stack--end-to-end-pipeline)
12. [Comparative Literature Table](#12-comparative-literature-table)
13. [References](#13-references)

---

## 1. Problem Statement

Audio deepfakes (AI-generated speech via Text-to-Speech / Voice Conversion) are increasingly indistinguishable from organic human speech, fooling both automated speaker verification (ASV) systems and human listeners. Single-model detectors typically overfit to the specific synthesis algorithms seen during training—performance collapses on novel, unseen generators.

**Key Limitation:** A monolithic detector trained on known attacks (e.g. Tacotron2, WaveNet) may achieve <1% EER in-domain but >20% EER on a new TTS system it has never encountered.

**Our Contribution:**
- **Multi-agent architecture** — specialized detectors (spectral, prosodic, linguistic, segmental) each capture different artifact types
- **Decision-level fusion** with weighted voting / meta-classifier
- **Per-agent explainability** — trace _which_ speech characteristic flagged a sample as fake
- **Robustness protocols** — leave-one-attack-out, cross-dataset evaluation, adversarial hardening

**Datasets:** ASVspoof 5 (2024) — 2,000 speakers, 32 TTS/VC attack algorithms, crowdsourced real-world conditions; ASVspoof 2025 Logical Access corpora.

---

## 2. How Speech Signals Are Pre-Processed

Before any model sees an audio sample, a standardized preprocessing pipeline transforms raw waveforms into ML-ready representations.

### 2.1 Raw Audio Ingestion

| Step | What Happens | Typical Values |
|------|-------------|----------------|
| **Sampling** | Resample to uniform rate | 16 kHz (ASVspoof standard) |
| **Bit Depth** | Normalize to float32 | [-1.0, 1.0] range |
| **Channel** | Convert stereo → mono | Single channel |
| **Duration Normalization** | Pad short / truncate long utterances | 4–6 seconds typical |

### 2.2 Noise & Silence Handling

| Step | Method | Purpose |
|------|--------|---------|
| **Voice Activity Detection (VAD)** | WebRTC VAD, energy-based, or Silero VAD | Remove leading/trailing silence and non-speech segments |
| **Pre-emphasis** | First-order filter: `y[n] = x[n] - α·x[n-1]` where α ≈ 0.97 | Boost high-frequency energy, compensate for spectral roll-off of human vocal tract |
| **Denoising** (optional) | Spectral subtraction, Wiener filtering, or neural denoiser | Reduce background/channel noise for cleaner features |
| **Normalization** | Peak normalization or z-score (mean=0, std=1) | Consistent amplitude across samples |

### 2.3 Feature Extraction (Transforms)

This is the critical step — raw waveform is converted into feature representations the models consume:

#### **A. Spectrogram-Based Features**

```
Raw Waveform ──► Framing (25ms windows, 10ms hop)
                  ──► Windowing (Hamming)
                  ──► FFT (512 or 1024 points)
                  ──► Power Spectrum
                  ──► Filter Bank (Mel / Linear / CQT)
                  ──► Log Compression
                  ──► [Optional] DCT → Cepstral Coefficients
```

| Feature | Filter Bank | Output Shape | What It Captures |
|---------|------------|--------------|------------------|
| **MFCC** (Mel-Frequency Cepstral Coefficients) | Mel-scale (triangular) | (T × 13-40) | Compact spectral envelope on perceptual scale; most widely used in speech |
| **LFCC** (Linear-Frequency Cepstral Coefficients) | Linear-spaced | (T × 20-60) | Uniform frequency resolution; better for high-frequency artifact detection |
| **CQCC** (Constant-Q Cepstral Coefficients) | Constant-Q transform | (T × 20-60) | Log-spaced frequency bins; captures both low and high frequency details with variable resolution |
| **Mel Spectrogram** | Mel-scale | (T × 80-128) | 2D time-frequency image; used as input to CNNs like ResNet or ViT |
| **Log Power Spectrogram** | None (raw FFT) | (T × 257-513) | Full spectral detail; used in LCNN and other models |

#### **B. Self-Supervised Learning (SSL) Embeddings**

Instead of hand-crafted features, modern SOTA systems feed raw waveforms directly into pre-trained SSL models:

```
Raw Waveform (16kHz) ──► [SSL Encoder] ──► Frame-level embeddings (T × D)
                                              e.g., T×768 for WavLM-Base
                                              or T×1024 for WavLM-Large
```

| SSL Model | Pre-training | Embedding Dim | Key Advantage |
|-----------|-------------|---------------|---------------|
| **WavLM** (Microsoft) | Masked speech prediction + denoising | 768 / 1024 | SOTA for anti-spoofing; captures both acoustic and semantic info; best ASVspoof5 results |
| **Wav2Vec 2.0** (Meta) | Contrastive + masked prediction | 768 / 1024 | Strong general audio representation |
| **HuBERT** (Meta) | Offline cluster + masked prediction | 768 / 1024 | Good phonetic representations |
| **XLSR-53 / XLS-R** | Multilingual Wav2Vec 2.0 | 1024 | Cross-lingual transfer |
| **Whisper Encoder** (OpenAI) | Supervised ASR on 680k hours | 512-1280 | Strong robustness to noise/channels |

**Key insight:** An ensemble of four WavLM models (fine-tuned and late-fused) secured **6.56% / 17.08% EER** on the two ASVspoof-5 evaluations, topping the 2024 challenge leaderboard.

#### **C. Prosodic Feature Extraction**

```
Raw Waveform ──► Pitch Tracker (CREPE / YIN / Praat)  ──► F0 contour
             ──► Jitter/Shimmer Extractor (Praat/openSMILE) ──► Stats
             ──► Energy Tracker ──► RMS, HNR
             ──► Duration Model ──► Phoneme lengths, pause stats
```

| Feature | Tool | What It Measures |
|---------|------|-----------------|
| **F0 (Fundamental Frequency)** | Praat, CREPE, pYIN | Pitch — vibration rate of vocal cords |
| **Jitter** | Praat, openSMILE | Cycle-to-cycle variation in F0 period (micro-perturbation) |
| **Shimmer** | Praat, openSMILE | Cycle-to-cycle variation in amplitude |
| **HNR** (Harmonics-to-Noise Ratio) | Praat | Ratio of periodic vs. aperiodic energy; voice quality |
| **Speaking Rate** | Forced alignment (MFA) | Syllables/phonemes per second |
| **Pause Distribution** | VAD + alignment | Duration and placement of pauses |

#### **D. Linguistic Feature Extraction**

```
Raw Waveform ──► ASR (Whisper / wav2vec2-CTC) ──► Transcript
             ──► Language Model (BERT / GPT) ──► Token embeddings
             ──► Phoneme Recognizer ──► Phoneme sequence
```

### 2.4 Data Augmentation (Training-Time)

| Technique | What It Does | Reference |
|-----------|-------------|-----------|
| **RawBoost** | Adds colored noise + convolutive artifacts to raw waveforms | Tak et al., 2022 |
| **SpecAugment** | Time/frequency masking on spectrograms | Park et al., 2019 |
| **Codec Augmentation** | Re-encode through MP3/Opus/AMR codecs | ASVspoof5 protocol |
| **Room Impulse Response (RIR)** | Simulate reverberant environments | Improves channel robustness |
| **Speed Perturbation** | 0.9x / 1.1x playback speed | Standard Kaldi augmentation |
| **Voice Conversion Augmentation** | Apply VC to bonafide data to create semi-synthetic training samples | Improves generator diversity |

---

## 3. Speech Signal Components — Spectral, Prosodic, Linguistic & Beyond

### 3.1 Spectral (Frequency-Domain) Analysis

**Definition:** Spectral analysis examines the **frequency content** of speech at each moment in time. The spectrum of a speech frame reveals which frequencies (harmonics, formants, noise components) are present and at what amplitudes.

**Why it detects deepfakes:** Synthetic speech generators (TTS/VC) introduce subtle artifacts in the frequency domain:
- **Harmonic truncation** — overtones cut off above a certain frequency (e.g., many neural vocoders struggle above 8 kHz)
- **Phase discontinuities** — incoherent phase spectrum between frames
- **Vocoder fingerprints** — each neural vocoder (WaveNet, HiFi-GAN, WaveGlow) leaves a spectral signature in formant bandwidths, spectral tilt, and noise floor patterns
- **Over-smoothing** — mel-spectrogram-based generation smears fine spectral detail

**What models look at:**
- Short-time spectral shape (MFCCs/LFCCs capture this)
- Spectral sub-band energy distribution
- Temporal evolution of spectral features (spectro-temporal patterns)
- High-frequency artifact presence/absence

### 3.2 Prosodic (Suprasegmental) Analysis

**Definition:** Prosody encompasses the **suprasegmental** features of speech — properties that span beyond individual phonemes and characterize how something is said, not what is said. Key prosodic features include **pitch (F0), rhythm, stress, intonation, speaking rate, and voice quality measures (jitter, shimmer, HNR).**

**Why it detects deepfakes:** Current TTS systems often produce prosody that is:
- **Over-regularized** — unnaturally smooth pitch contours without the micro-perturbations (jitter/shimmer) of real vocal cord vibration
- **Statistically unnatural** — jitter values in deepfake speech are often lower than in real speech (too "perfect")
- **Rhythmically rigid** — synthetic speech often has more uniform phoneme durations and fewer natural hesitations/micro-pauses

**Key prosodic features for detection (Warren et al., 2025):**

| Feature | What It Measures | Detection Signal |
|---------|-----------------|------------------|
| Mean F0 | Average pitch | Abnormal pitch range for speaker |
| F0 Std Dev | Pitch variability | Too smooth = synthetic |
| Jitter (%) | F0 period perturbation | Real: ~0.5-1.0%; Fake: often lower |
| Shimmer (%) | Amplitude perturbation | Real: ~3-5%; Fake: often lower |
| HNR (dB) | Harmonic vs noise | Fake may have higher HNR (too clean) |
| Speaking Rate | Syllables/sec | Often unnaturally consistent in fakes |

**Warren et al. (2025)** demonstrated that a prosody-only detector using just 6 features achieved **93% accuracy** on ASVspoof 2021, and showed **significantly more robustness against adversarial attacks** than spectral-only models (only 7% degradation vs. 99.3% for LFCC-based models under L∞ attack).

### 3.3 Linguistic (Content-Level) Analysis

**Definition:** Linguistic analysis examines the **content, semantics, and realization patterns** of speech — how words, phonemes, and language structure are produced. This includes both the textual content (lexical/semantic level) and how acoustic style correlates with linguistic content.

**Why it detects deepfakes:**
- **Style–Linguistics Mismatch (SLIM):** In real speech, there is a natural correlation between _how_ something is said (acoustic style — speaker identity, emotion, prosody) and _what_ is said (linguistic content — phonemes, words). Deepfake generators often break this dependency because the style and content are independently controlled.
- **Pronunciation anomalies:** Synthetic speech may produce phoneme sequences that are subtly unnatural — wrong coarticulation patterns, unusual vowel reduction, or dialect inconsistencies.
- **Lexical sensitivity:** Nguyen et al. (2025) showed that tiny text-level changes (synonym substitution) can dramatically fool existing detectors, indicating that some models inadvertently learn linguistic patterns.

**SLIM Framework (Zhu et al., NeurIPS 2024):**
```
                    ┌─── Style Encoder ──────► Style Embedding ──┐
Raw Waveform ──► SSL │                                            ├──► Alignment Score
                    └─── Linguistics Encoder ──► Ling Embedding ──┘
                         (via SSL + Projector)

Real speech:  High style-linguistics alignment (correlated)
Fake speech:  Low style-linguistics alignment (mismatch detected!)
```

### 3.4 Segmental / Articulatory Analysis

**Definition:** Segmental features operate at the **individual phoneme level** — formant frequencies (F1, F2, F3), voice onset time (VOT), formant transitions, and coarticulation patterns.

**Why it detects deepfakes:** Yang et al. (2026) demonstrated that **vowel formant trajectories** serve as "articulatory fingerprints" — each real speaker has consistent formant patterns that deepfakes fail to replicate accurately. Their formant-based classifiers achieved **lower EER and calibration loss (Cllr)** than MFCC-based systems on both controlled and in-the-wild deepfakes.

### 3.5 Phase-Domain Analysis

**Definition:** While most features use magnitude spectrum, **phase features** capture the phase relationships between frequency components. These include group delay, instantaneous frequency, and phase-based spectrograms.

**Why it detects deepfakes:** Neural vocoders often generate magnitude spectra well but produce incoherent or overly smooth phase spectra. Phase analysis can detect artifacts invisible in magnitude-only features.

---

## 4. Models for Each Agent — Architectures & How They Work

### 4.1 Spectral Agent Models

#### **A. AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)**
- **Paper:** Jung et al., ICASSP 2022 → Interspeech extensions
- **Architecture:**
  ```
  Raw Waveform ──► RawNet2 Encoder (SincConv + ResBlocks)
                   ──► Spectral Graph Branch (HS-GAL layers)
                   ──► Temporal Graph Branch (HS-GAL layers)
                   ──► Max Graph Operation (MGO) — competitive selection
                   ──► Readout (concat max + avg + stack node)
                   ──► Output Layer ──► Score (real vs fake)
  ```
- **Key innovations:**
  - **Heterogeneous Stacking Graph Attention Layer (HS-GAL):** Models artifacts spanning both temporal and spectral domains as a heterogeneous graph
  - **Max Graph Operation (MGO):** Two parallel branches model temporal and spectral sub-graphs; element-wise maximum selects the most discriminative artifacts
  - **Stack Node:** Accumulates heterogeneous information across the graph
- **Input:** Raw waveform (no manual feature extraction)
- **Output:** Single scalar score (higher → more likely bonafide)
- **Performance:** ~1-3% EER on ASVspoof 2019 LA; strong baseline for ASVspoof 5
- **Parameters:** ~300K (very lightweight)

#### **B. RawNet2**
- **Paper:** Tak et al., 2021
- **Architecture:**
  ```
  Raw Waveform ──► SincConv Layer (learnable bandpass filters)
                   ──► ResBlocks with FMS (Feature Map Scaling)
                   ──► GRU Layer
                   ──► Fully Connected ──► Score
  ```
- **Key idea:** Learns filter bank parameters directly from raw audio (SincNet front-end). The sinc-based convolutional layer forces learned filters to be bandpass filters, making the model interpretable at the first layer.
- **Output:** Binary score
- **Performance:** ~5-8% EER on ASVspoof 2019 LA

#### **C. LCNN (Light Convolutional Neural Network)**
- **Architecture:**
  ```
  Spectrogram (LFCC/MFCC) ──► Conv Layers with Max-Feature-Map (MFM)
                              ──► Fully Connected ──► Score
  ```
- **Key idea:** MFM activation selects the maximum of two feature maps at each layer — acts as a competitive mechanism for feature selection.
- **Input:** Hand-crafted spectral features (LFCC, LFCC+Δ, MFCC)
- **Performance:** Baseline for ASVspoof challenges; ~3-6% EER

#### **D. ResNet / ResNeXt on Spectrograms**
- **Architecture:** Standard ResNet-18/34 or ResNeXt adapted for 2D spectrogram input
- **Input:** Mel spectrogram or LFCC as a 2D "image"
- **Performance:** ResNeXt with LFCC+MFCC+CQCC fusion achieved **1.05% EER** on ASVspoof 2019 LA

#### **E. SSL-Based Backend (WavLM + Classifier Head)**
- **Architecture (SOTA for ASVspoof 5):**
  ```
  Raw Waveform ──► WavLM-Large (frozen or fine-tuned)
                   ──► Weighted-sum of hidden layers (learnable weights)
                   ──► Backend Head:
                       Option A: MLP (Linear → ReLU → Linear → Sigmoid)
                       Option B: AASIST-style graph module
                       Option C: Conformer layers → Attentive pooling → Linear
                   ──► Score
  ```
- **Why WavLM dominates:** Pre-trained on 94K hours with masked speech denoising — captures both acoustic details and high-level structures. Hidden layers 15-21 contain the most anti-spoofing-relevant information.
- **Performance:** Single WavLM-Large + MLP: ~2-4% EER on ASVspoof 2019 LA; Ensemble of 4 WavLM systems: **6.56% EER on ASVspoof 5 Progress set** (leaderboard winner)

#### **F. Scalable AASIST (2025)**
- **Paper:** arXiv:2507.11777
- **Improvement:** Refines graph attention for efficiency and scalability; reduces computational overhead while maintaining detection accuracy

### 4.2 Prosodic Agent Models

#### **A. MLP on Hand-Crafted Prosodic Features (Warren et al., 2025)**
- **Architecture:**
  ```
  Audio ──► Praat / openSMILE extraction
            ──► 6 Features: [Mean F0, Std F0, Jitter, Shimmer, HNR, Speaking Rate]
            ──► MLP (2–3 hidden layers, ReLU, Dropout)
            ──► Attention Mechanism (for explainability)
            ──► Sigmoid ──► Score
  ```
- **Output:** Real/Fake probability + attention weights showing which prosodic feature mattered most
- **Performance:** 93% accuracy, **24.7% EER** on ASVspoof 2021 (prosody alone)
- **Advantage:** Highly **interpretable** and **adversarially robust** (only 7% degradation under L∞ attack)

#### **B. Gradient Boosted Decision Trees (GBDT / XGBoost / LightGBM)**
- **Input:** Extended prosodic feature set (50+ features from openSMILE eGeMAPS config): F0 stats, jitter variants, shimmer variants, HNR, loudness, formant bandwidth stats, spectral flux, voice quality
- **Architecture:** Ensemble of decision trees with gradient boosting
- **Advantage:** Fast inference, built-in feature importance, works well on tabular prosodic features

#### **C. RNN/LSTM on F0 Contour**
- **Input:** Frame-level F0 trajectory (full pitch contour over time)
- **Architecture:** Bidirectional LSTM capturing temporal dynamics of pitch
- **Advantage:** Captures long-range prosodic patterns (intonation contours, pitch resets at phrase boundaries)

### 4.3 Linguistic Agent Models

#### **A. SLIM (Style-Linguistics Mismatch) — Zhu et al., NeurIPS 2024**
- **Architecture:**
  ```
  Raw Waveform ──► WavLM / HuBERT (SSL backbone)
                   ├──► Style Projector ──► d-dimensional style embedding
                   └──► Linguistics Projector ──► d-dimensional linguistics embedding
                   ──► Cosine Similarity / MLP ──► Mismatch Score
  ```
  - **Stage 1 (Pre-training):** Learn to decompose SSL representations into style and linguistics subspaces using contrastive learning on real speech (maximize alignment for same utterance)
  - **Stage 2 (Detection):** Measure mismatch — real speech has high style-linguistics correlation; deepfakes show low correlation
- **Key insight:** The dependency features capture the natural relationship between "how it sounds" and "what is said" — deepfakes disrupt this.
- **Performance:** Outperforms baselines on out-of-domain tests; improved generalization to unseen attacks

#### **B. ASR Transcript + Language Model Anomaly Detection**
- **Architecture:**
  ```
  Audio ──► Whisper (ASR) ──► Transcript
        ──► BERT/GPT-2 ──► Perplexity score / embedding
        ──► Compare: pronunciation patterns, word-level confidence, timing alignment
  ```
- **Idea:** Synthetic speech may produce transcripts with unusual confidence patterns (too uniform) or slight pronunciation differences detectable via ASR-level analysis

#### **C. Phoneme-Level Embedding Networks**
- **Input:** Forced-aligned phoneme sequences with timing
- **Architecture:** Transformer on phoneme sequence + duration features
- **Detects:** Unnatural coarticulation, wrong allophone selection, abnormal phoneme duration distributions

### 4.4 Segmental / Articulatory Agent

#### **Formant-Based Classifier (Yang et al., 2026)**
- **Architecture:**
  ```
  Audio ──► Forced Alignment (MFA) ──► Segment by vowel
        ──► Formant extraction (F1, F2, F3 for each vowel)
        ──► Speaker-specific Logistic Regression
        ──► Per-vowel scores ──► Aggregated score
  ```
- **Performance:** Consistently lower EER and Cllr than MFCC-based systems on SpoofCeleb and DF In-the-Wild datasets
- **Advantage:** Highly interpretable — can point to specific vowels with anomalous formant patterns

### 4.5 Summary: What Each Agent Returns

| Agent | Input | Model | Output | Interpretability |
|-------|-------|-------|--------|-----------------|
| **Spectral** | Raw waveform or spectrogram | AASIST / WavLM+MLP / ResNet | Scalar score ∈ [0,1] | Attention heatmaps on time-freq regions |
| **Prosodic** | Extracted prosody features | MLP / GBDT | Scalar score ∈ [0,1] | Feature importance (jitter, shimmer, F0) |
| **Linguistic** | SSL embeddings (style + ling) | SLIM / ASR+LM | Mismatch score ∈ [0,1] | Style-vs-content alignment visualization |
| **Segmental** | Formant trajectories per vowel | Logistic Regression / SVM | Per-vowel + aggregate score | Which vowel, which formant is anomalous |

---

## 5. Other Processable Speech Components

Beyond spectral, prosodic, and linguistic, additional speech dimensions can be exploited:

| Component | Features | Detection Signal | Models |
|-----------|----------|-----------------|--------|
| **Phase Spectrum** | Group delay, instantaneous frequency, phase spectrogram | Neural vocoders produce incoherent phase | Phase-aware CNNs, IF-based models |
| **Excitation Signal** | Glottal flow (via inverse filtering), glottal pulse shape | LPC residual reveals source characteristics | LP inverse filter + GRU |
| **Breathing / Pauses** | Breath event detection, inter-breath intervals, breath loudness | Deepfakes lack or over-regularize breath | RNN breath detector + rule engine |
| **Channel / Codec Artifacts** | Re-encoding traces, compression artifacts, bandwidth analysis | Double-compressed or codec-specific patterns | CNN on coded spectrograms |
| **Emotional Consistency** | Emotion recognition (arousal, valence, dominance) | Style-emotion mismatch in generated speech | SER (Speech Emotion Recognition) models |
| **Speaker Embedding Consistency** | x-vector / d-vector / ECAPA-TDNN speaker embeddings | Speaker embedding drift across an utterance | Segmental speaker verification |

---

## 6. Model Outputs, Evaluation & Performance

### 6.1 What Each Model Returns

Every agent produces a **countermeasure (CM) score** — a scalar value where:
- **Higher score** → more likely **bonafide** (real)
- **Lower score** → more likely **spoof** (fake)

At a chosen threshold τ, the decision is:
```
if CM_score > τ: ACCEPT (bonafide)
else:           REJECT (spoof)
```

### 6.2 Evaluation Metrics

| Metric | Formula / Definition | What It Measures | Used In |
|--------|---------------------|------------------|---------|
| **EER** (Equal Error Rate) | Threshold where FAR = FRR | Single-number summary of detection tradeoff; lower is better | ASVspoof primary evaluation |
| **min-tDCF** (minimum tandem Detection Cost Function) | Weighted cost of CM errors on ASV system | Impact of spoofing CM on downstream ASV; accounts for real-world costs | ASVspoof 2019–2021 primary |
| **minDCF** | Similar to tDCF but stand-alone | Application-agnostic detection cost | ASVspoof 5 Track 1 (primary) |
| **AUC** (Area Under ROC Curve) | Area under FPR-vs-TPR curve | Overall discrimination ability across all thresholds | General ML evaluation |
| **Cllr** (Log-likelihood cost) | Calibration-sensitive cost | Measures both discrimination AND calibration quality | ASVspoof 5, forensics |
| **Accuracy** | (TP + TN) / Total | Simple classification correctness | Quick benchmarking |
| **F1-Score** | Harmonic mean of Precision & Recall | Balanced precision-recall tradeoff | When classes are imbalanced |

### 6.3 Performance Comparison Across Literature

| Paper | Year | Features / Agent Type | Model / Architecture | Dataset | EER (%) | Other Metrics |
|-------|------|----------------------|---------------------|---------|---------|--------------|
| Kulkarni et al. | 2024 | SSL (WavLM, HuBERT) ensemble | 4-system score fusion | ASVspoof 5 | **6.56** (Prog) / 17.08 (Eval) | Leaderboard winner |
| ResNeXt + Multi-Feature | 2025 | LFCC + MFCC + CQCC | ResNeXt-based CNN | ASVspoof 2019 LA | **1.05** | min-tDCF: 0.028 |
| Warren et al. | 2025 | Prosody only (6 features) | MLP + attention | ASVspoof 2021 LA | 24.7 | Acc: 93%, adversarial-robust |
| Yang et al. | 2026 | Vowel formants (segmental) | Speaker-specific LR | SpoofCeleb + DF-ITW | <10 (varies) | Lower Cllr than MFCC |
| Zhu et al. (SLIM) | 2024 | Style-Linguistics mismatch | Contrastive SSL | ASVspoof 2019 LA | Improved OOD | Better generalization |
| Liu et al. | 2025 | SSL + retrieval augmentation | k-NN ensemble + fusion | DeepFake-Eval-2024 | ~12 (fusion) | Cross-dataset evaluation |
| Celik et al. | 2026 | Wavelet + pattern features | kNN/SVM + feature selection | ASVspoof 2019/2021 | 0.97–10.85 | Acc: 89.2–99.2% |
| Yang et al. (Poincaré) | 2025 | Hyperbolic prototypical | Poin-HierNet | ASVspoof 2019/2021/ITW | SOTA (lowest) | Hyperbolic metric learning |
| Jiang et al. | 2025 | Spectral + SSL (cross-attn fusion) | Multi-view collab network | ASVspoof 2019/2021 | 38% relative reduction | Cross-attention > concat |
| AASIST | 2022 | Raw waveform (spectro-temporal) | RawNet2 + Graph Networks | ASVspoof 2019 LA | ~1-3 | min-tDCF: ~0.03 |

---

## 7. Voting / Fusion — How All Models' Results Are Combined

The **Decision Agent** (meta-agent) takes the outputs from all specialized agents and produces a final verdict. There are multiple strategies, from simple to sophisticated:

### 7.1 Score-Level Fusion (Simple but Effective)

```
Final_Score = w₁·S_spectral + w₂·S_prosodic + w₃·S_linguistic + w₄·S_segmental
```

| Method | Description | Pros | Cons |
|--------|------------|------|------|
| **Simple Average** | Equal weights: w₁ = w₂ = w₃ = w₄ = 0.25 | No training needed; surprisingly effective | Ignores agent quality differences |
| **Weighted Average** | Weights learned on validation set (grid search / Bayesian optimization) | Respects agent reliability | Weights may overfit to validation data |
| **Score Normalization + Fusion** | Z-norm or min-max each agent's score, then combine | Accounts for different score distributions | Extra normalization step |

**Evidence:** Kulkarni et al. (2024) showed that simple **score averaging** of 4 SSL-based detectors significantly improved out-of-domain robustness compared to any single system.

### 7.2 Embedding-Level Fusion (Feature Concatenation)

```
Agent1_embedding ─┐
Agent2_embedding ──┼──► Concatenate / Cross-Attention ──► Classifier ──► Score
Agent3_embedding ─┘
```

| Method | Description | Performance |
|--------|------------|------------|
| **Concatenation** | Stack agent embeddings → MLP classifier | Baseline fusion |
| **Cross-Attention Fusion** | Agents attend to each other's representations | **38% relative EER reduction** vs single-stream (Jiang et al., 2025) |
| **Gating Mechanism** | Learned gates dynamically weight features | Adaptive per-sample weighting |
| **Attentive Fusion** | Self-Supervised Multi-Fusion Attentive Classifier | Learns which features matter per utterance |

### 7.3 Meta-Classifier (Stacking)

```
Agent1_score ─┐                              ┌─── Final Score
Agent2_score ──┼──► Meta-Classifier (RF/MLP) ──┤
Agent3_score ─┘                              └─── Confidence
Agent4_score ─┘
```

| Meta-Classifier | Description | Used In |
|----------------|------------|---------|
| **Random Forest** | Takes agent scores as features; robust to overfitting | DeepAgent (multi-modal A/V deepfake); analogous for audio agents |
| **Logistic Regression** | Simple weighted combination with learned threshold | Standard ensemble stacking |
| **Small Neural Network** | 2-layer MLP on score vector | When non-linear interactions between agents matter |
| **Gradient Boosted Trees** | XGBoost/LightGBM on score+confidence features | Handles heterogeneous agent outputs well |

### 7.4 Majority Voting (Hard Decision Fusion)

```
Each agent makes binary decision: 1 (fake) or 0 (real)
Final = Majority vote: if ≥ k agents say "fake" → FAKE
```

| Variant | Rule | Use Case |
|---------|------|----------|
| **Simple Majority** | ≥ 50% agents agree | Equal trust in all agents |
| **Weighted Majority** | Weight votes by agent's historical accuracy | Better agents get more say |
| **Unanimous** | All agents must agree on "fake" | High-precision setting (minimize false alarms) |
| **Any-one (OR)** | Any single agent can flag as fake | High-recall setting (catch all fakes) |

### 7.5 Selective / Adaptive Fusion (Advanced)

```
Input ──► OOD Detector (is this in-domain or out-of-domain?)
          │
          ├── In-domain:  Use Spectral Agent (high confidence)
          └── OOD:        Fall back to Ensemble / Retrieval-Augmented Agent
```

**Liu et al. (2025)** proposed this approach: a small k-NN OOD detector decides whether to trust the original SSL model or the retrieval-augmented ensemble. This "selective fusion" provides the best of both worlds.

### 7.6 Retrieval-Augmented Fusion

```
Test utterance ──► SSL Encoder ──► Embedding
                ──► k-NN search in reference database
                ──► Aggregate k-NN labels/scores
                ──► Linear fusion with base CM score
                ──► Final Score
```

- **Liu et al. (2025):** Build a database of (feature, score) pairs from known data. At inference, find k-NN neighbors and aggregate their labels. Combined with base classifier via linear/selective fusion.
- **Advantage:** Adapts to new attacks **without retraining** — just add reference samples to the database.
- **Performance:** ~12% EER on DeepFake-Eval-2024 (cross-domain)

### 7.7 Recommended Fusion Pipeline for This Project

```
┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Spectral   │  │   Prosodic   │  │  Linguistic  │  │  Segmental   │
│    Agent    │  │    Agent     │  │    Agent     │  │    Agent     │
│ (WavLM+MLP)│  │ (MLP/GBDT)  │  │   (SLIM)    │  │(Formant+LR) │
└──────┬──────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                │                 │                  │
   score₁           score₂           score₃             score₄
   embed₁           features₂        embed₃             features₄
       │                │                 │                  │
       └────────┬───────┴────────┬────────┘──────────────────┘
                │                │
        ┌───────▼────────┐ ┌────▼─────────────┐
        │ Score-Level    │ │ Embedding-Level  │
        │ Weighted Avg   │ │ Cross-Attention  │
        └───────┬────────┘ └────┬─────────────┘
                │               │
                └───────┬───────┘
                   ┌────▼──────────────┐
                   │  Meta-Classifier  │
                   │  (XGBoost / MLP)  │
                   └────┬──────────────┘
                        │
                   Final Verdict + Confidence + Per-Agent Explanation
```

---

## 8. Explainability — How Each Agent's Decision Is Interpreted

Multi-agent systems are inherently more interpretable than monolithic black-boxes because each agent's decision can be traced to a specific speech dimension.

### 8.1 Per-Agent Explainability Methods

| Agent | Method | What It Reveals | Output |
|-------|--------|----------------|--------|
| **Spectral** | Attention heatmaps (from AASIST's graph attention) | Which time-frequency regions triggered detection | 2D heatmap over spectrogram |
| **Spectral** | SHAP on SSL layer outputs | Which hidden layers/features contributed most | SHAP bar chart |
| **Spectral** | GradCAM on spectrogram-CNN | Spatial regions of spectrogram most important | Highlighted spectrogram |
| **Prosodic** | Attention weights (Warren et al.) | Which prosodic features drove the decision | Bar chart: Jitter > Shimmer > Mean F0 |
| **Prosodic** | SHAP / Feature Importance from GBDT | Quantitative feature contribution | SHAP waterfall plot |
| **Linguistic** | Style-Linguistics alignment score (SLIM) | Where style-content correlation breaks down | Temporal alignment visualization |
| **Linguistic** | ASR confidence per word | Words with unusual recognition confidence | Highlighted transcript |
| **Segmental** | Per-vowel formant deviation | Specific vowels with anomalous formants | Formant plot with anomaly markers |

### 8.2 System-Level Explanation Generation

The meta-agent aggregates per-agent explanations into a human-readable report:

```
═══════════════════════════════════════════════════════════
         DEEPFAKE DETECTION REPORT
═══════════════════════════════════════════════════════════
Verdict:    FAKE (Confidence: 94.3%)

Agent Votes:
  [FAKE ] Spectral Agent    — Score: 0.12  (threshold: 0.50)
  [FAKE ] Prosodic Agent    — Score: 0.23  (threshold: 0.50)
  [REAL ] Linguistic Agent  — Score: 0.61  (threshold: 0.50)
  [FAKE ] Segmental Agent   — Score: 0.08  (threshold: 0.50)

Key Evidence:
  1. SPECTRAL: Anomalous energy drop above 7.5 kHz (vocoder truncation)
  2. PROSODIC: Jitter = 0.12% (abnormally low; real speech: 0.5-1.0%)
              Shimmer = 0.8% (abnormally low; real speech: 3-5%)
  3. SEGMENTAL: Vowel /a/ formant F2 deviation = 340 Hz from speaker norm
  4. LINGUISTIC: Style-content alignment within normal range (no flag)

Agent Agreement: 3/4 agents flag as FAKE
Consensus Strength: HIGH (diverse evidence from multiple domains)
═══════════════════════════════════════════════════════════
```

### 8.3 XAI Techniques Applied

| Technique | Type | Application | Library |
|-----------|------|-------------|---------|
| **SHAP** (SHapley Additive exPlanations) | Post-hoc, model-agnostic | Feature contribution for any agent | `shap` |
| **LIME** (Local Interpretable Model-agnostic Explanations) | Post-hoc, model-agnostic | Local explanations for spectrogram-based models | `lime` |
| **GradCAM** | Gradient-based | Heatmaps for CNN-based spectral agent | PyTorch hooks |
| **Attention Weights** | Intrinsic | Built into AASIST, Transformers, MLP+attention | Model architecture |
| **Feature Importance** | Intrinsic | GBDT-based prosodic agent | XGBoost/LightGBM `.feature_importances_` |
| **Integrated Gradients** | Gradient-based | Attribution for any differentiable model | `captum` |

---

## 9. Robustness to Unseen Generators

The central challenge: a detector trained on WaveNet + Tacotron must still work on VALL-E, XTTS, or future TTS systems it has never seen.

### 9.1 Evaluation Protocols for Robustness

| Protocol | Description | What It Tests |
|----------|------------|---------------|
| **Leave-One-Attack-Out (LOAO)** | Train on N-1 attack types, test on the held-out one | Single unknown generator |
| **Cross-Dataset Evaluation** | Train on ASVspoof 2019, test on ASVspoof 5 or DF-In-The-Wild | Domain shift (different speakers, conditions, generators) |
| **Unknown Split in ASVspoof** | Train on subset of generators, evaluate on "progress" and "eval" sets with unseen generators | Official challenge protocol |
| **Cross-Language Evaluation** | Train on English, test on Bengali/Chinese/etc. | Language-invariant detection |
| **Adversarial Attack Evaluation** | Apply L∞, FGSM, PGD attacks to test samples | Susceptibility to adversarial manipulation |

### 9.2 Techniques for Improving Robustness

| Technique | How It Works | Evidence |
|-----------|-------------|----------|
| **Multi-Agent Ensemble** | Different agents overfit to different artifacts → ensemble covers more ground | Kulkarni et al.: ensemble significantly reduces EER on unseen attacks |
| **Meta-Learning (MAML / ProtoNet)** | Learn attack-invariant features that adapt to new attacks with very few samples | Kukanov et al. (2025): ASDG method for domain generalization |
| **Data Augmentation (RawBoost, Codec, RIR)** | Expose model to diverse acoustic conditions during training | Tak et al., 2022: RawBoost improves generalization |
| **Domain Generalization** | Train with domain adversarial loss — learn features that are invariant across generator types | arXiv:2408.13341: meta-learning + disentangled training |
| **Retrieval-Augmented Detection** | k-NN against reference database adapts at inference without retraining | Liu et al., 2025: works on zero-day attacks |
| **Contrastive Learning** | Learn representations where real vs. fake differences are generator-agnostic | SLIM leverages contrastive pre-training |
| **OOD Detection + Selective Fusion** | Route OOD samples to more robust agents/ensembles | Liu et al., 2025: selective fusion with OOD routing |
| **Adversarial Training** | Include adversarial examples in training data | arXiv:2408.13341: adversarial hardening |
| **Multi-Task Learning** | Jointly train for detection + generator identification → learn richer features | Forces models to capture generator-specific patterns |
| **Calibration** | Use Platt scaling / temperature scaling so scores are well-calibrated on new data | Improves reliability of threshold selection |

### 9.3 Why Multi-Agent Architecture Helps Robustness

```
               Known Generator (WaveNet)   │  Unseen Generator (VALL-E)
─────────────────────────────────────────────────────────────────────────
Spectral Agent  →  Detects vocoder artifact │  May miss new artifact pattern  ✗
Prosodic Agent  →  Detects smooth jitter    │  Still detects smooth jitter    ✓
Linguistic Agent → Detects style mismatch   │  Still detects style mismatch   ✓  
Segmental Agent →  Detects formant anomaly  │  May detect formant issues      ~
─────────────────────────────────────────────────────────────────────────
Single Model    →  EER: 2%                  │  EER: 25% (catastrophic drop)
Multi-Agent     →  EER: 3%                  │  EER: 10% (graceful degradation) ✓
```

The key insight: **different generators fail differently**. A new TTS might produce perfect spectrograms but still have unnatural prosody. Multi-agent systems capture diverse artifacts, so even if one agent is fooled, others compensate.

---

## 10. Which Metric — EER vs AUC — and Why

### 10.1 Definitions

**EER (Equal Error Rate):**
- The operating point where **False Acceptance Rate (FAR) = False Rejection Rate (FRR)**
- FAR = proportion of fakes incorrectly accepted as real
- FRR = proportion of reals incorrectly rejected as fake
- **Lower EER = better detector**
- Single scalar that doesn't require choosing a threshold

**AUC (Area Under ROC Curve):**
- Area under the curve plotting **True Positive Rate vs. False Positive Rate** at all thresholds
- Ranges from 0.5 (random) to 1.0 (perfect)
- **Higher AUC = better detector**
- Summarizes performance across all possible operating points

### 10.2 Head-to-Head Comparison

| Criterion | EER | AUC |
|-----------|-----|-----|
| **Threshold dependence** | Implicitly defines a specific threshold (FAR=FRR point) | Threshold-free (integrates over all thresholds) |
| **Ease of interpretation** | "5% EER" = 5% of both fakes pass and reals fail | "0.98 AUC" = 98% chance model ranks a random real above a random fake |
| **Sensitivity to extremes** | Focuses on a single operating point | Can be inflated by good performance at irrelevant thresholds |
| **Spoofing community standard** | ✅ **Standard metric** in ASVspoof series since inception | Used as supplementary metric |
| **Calibration sensitivity** | Somewhat — depends on score distribution near threshold | No — purely rank-based |
| **Practical relevance** | Directly corresponds to an operational decision boundary | Doesn't tell you "how good is my detector at a specific threshold?" |
| **Cost-aware** | No (equal weight to both error types) | No (but minDCF / tDCF address this) |

### 10.3 Our Recommendation: **Prioritize EER** (with minDCF as secondary)

**Why EER:**
1. **Community standard** — ASVspoof 2019, 2021, 2024, 2025 all use EER as a primary or key metric. All comparison with prior work is in EER.
2. **Operational relevance** — In deployment, you pick ONE threshold. EER tells you performance at the "fair" operating point. AUC averages across thresholds you'll never use.
3. **Sensitivity to the decision region** — For security applications, we care about performance near the decision boundary, not at extreme thresholds. EER focuses exactly there.
4. **ASVspoof 5 uses minDCF (primary) + EER** — The official evaluation uses cost-aware metrics (minDCF, actDCF, Cllr) with EER as interpretive support.
5. **Comparable across studies** — Nearly every paper in Table 6.3 reports EER, enabling direct comparison.

**When to also report AUC:**
- When comparing with general ML community work
- For datasets without established EER convention
- When you want to show the model is robust across many operating points (not just the EER point)

**Final metric suite for this project:**

| Metric | Role | Priority |
|--------|------|----------|
| **EER** | Primary comparison metric | ★★★ |
| **minDCF** | Primary cost-aware metric (ASVspoof 5 standard) | ★★★ |
| **Cllr** | Calibration quality metric | ★★ |
| **AUC** | Supplementary: overall discrimination | ★★ |
| **Accuracy** | Quick sanity check | ★ |

---

## 11. Final Tech Stack — End-to-End Pipeline

### 11.1 Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                    │
│  Raw Audio (.flac/.wav, 16kHz, mono)                                       │
│  ──► VAD ──► Pre-emphasis ──► Normalization ──► Duration Alignment          │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                    ┌───────────┼───────────┬──────────────┐
                    ▼           ▼           ▼              ▼
┌──────────────────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────┐
│   SPECTRAL AGENT     │ │ PROSODIC │ │LINGUISTIC │ │  SEGMENTAL   │
│                      │ │  AGENT   │ │  AGENT    │ │   AGENT      │
│ ┌──────────────────┐ │ │          │ │           │ │              │
│ │ PRIMARY:         │ │ │ PRIMARY: │ │ PRIMARY:  │ │ PRIMARY:     │
│ │ WavLM-Large      │ │ │ MLP on   │ │ SLIM      │ │ Formant-LR   │
│ │ + AASIST backend │ │ │ prosody  │ │ (style-   │ │ (per-vowel)  │
│ │                  │ │ │ features │ │  ling      │ │              │
│ │ ALT 1:           │ │ │          │ │  mismatch) │ │ ALT:         │
│ │ AASIST (raw)     │ │ │ ALT 1:   │ │           │ │ Formant-SVM  │
│ │                  │ │ │ XGBoost  │ │ ALT 1:    │ │ Coarticulation│
│ │ ALT 2:           │ │ │ on eGe-  │ │ ASR +     │ │ feature MLP  │
│ │ ResNeXt on       │ │ │ MAPS     │ │ LM anomaly│ │              │
│ │ LFCC+MFCC+CQCC   │ │ │          │ │           │ │              │
│ │                  │ │ │ ALT 2:   │ │ ALT 2:    │ │              │
│ │ ALT 3:           │ │ │ Bi-LSTM  │ │ Phoneme   │ │              │
│ │ RawNet2 + LCNN   │ │ │ on F0    │ │ Transformer│ │              │
│ │                  │ │ │ contour  │ │           │ │              │
│ └──────────────────┘ │ └──────────┘ └───────────┘ └──────────────┘
│  Output:             │  Output:     Output:        Output:
│  score₁ + embed₁    │  score₂      score₃         score₄
│  + attention map     │  + feat imp  + alignment    + per-vowel
└──────────┬───────────┘──────┬──────────┬────────────────┬────────
           │                  │          │                │
           └──────────┬───────┴──────────┴────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DECISION AGENT (FUSION)                    │
│                                                                 │
│  PRIMARY: Two-Stage Fusion                                     │
│    Stage 1: Score-level weighted average (learned weights)      │
│    Stage 2: Meta-classifier (XGBoost on scores + confidences)  │
│                                                                 │
│  ALT 1: Cross-Attention Embedding Fusion + MLP                 │
│  ALT 2: Selective Fusion (OOD routing)                         │
│  ALT 3: Retrieval-Augmented k-NN + Linear Fusion              │
│  ALT 4: Simple Majority Voting (baseline)                      │
│                                                                 │
│  Output: Final verdict + confidence + per-agent explanations   │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EXPLAINABILITY MODULE                        │
│                                                                 │
│  PRIMARY: Per-agent SHAP + Aggregated Report                   │
│  ALT 1: GradCAM (spectral) + Feature Importance (prosodic)    │
│  ALT 2: LIME per agent + attention heatmaps                   │
│  ALT 3: Integrated Gradients (differentiable agents)           │
│                                                                 │
│  Output: Human-readable report with visualizations             │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 Software & Library Stack

| Layer | Component | Primary Choice | Alternatives | What It Returns |
|-------|-----------|---------------|-------------|----------------|
| **Framework** | Deep learning | PyTorch 2.x | TensorFlow, JAX | Model training & inference |
| **Audio I/O** | Loading/saving | `torchaudio` / `librosa` | `soundfile`, `scipy.io.wavfile` | Waveform tensor |
| **Preprocessing** | Feature extraction | `torchaudio.transforms` | `python_speech_features`, `librosa.feature` | MFCC/LFCC/Mel-spec tensors |
| **SSL Models** | Pre-trained encoders | `transformers` (HuggingFace) — WavLM-Large | `fairseq`, `s3prl` | Frame-level embeddings (T×D) |
| **Prosody** | Feature extraction | `Praat` via `parselmouth` | `openSMILE`, `pyin` via librosa | F0, jitter, shimmer, HNR vectors |
| **ASR** | Transcription | `whisper` (OpenAI) | `wav2vec2-CTC`, Kaldi | Transcript + word-level timestamps |
| **Forced Alignment** | Phoneme alignment | `Montreal Forced Aligner (MFA)` | `Kaldi`, `Gentle` | Phoneme-level time boundaries |
| **Formant Analysis** | Segmental features | `parselmouth` (Praat) | `VoiceSauce`, custom LPC | F1, F2, F3 per phoneme |
| **Graph Networks** | AASIST backend | `torch_geometric` | `dgl` | Graph-based classification |
| **Gradient Boosting** | Tabular models | `xgboost` / `lightgbm` | `catboost`, `sklearn.ensemble` | Score + feature importance |
| **Meta-Classifier** | Fusion | `xgboost` or `torch.nn.Module` | `sklearn` LogisticRegression | Final score + confidence |
| **Explainability** | XAI | `shap` + `captum` | `lime`, custom attention extraction | Feature attributions, heatmaps |
| **Evaluation** | Metrics | `sklearn.metrics` + custom EER function | `bob.measure`, `pyeer` | EER, AUC, minDCF, Cllr values |
| **Experiment Tracking** | MLOps | `wandb` | `mlflow`, `tensorboard` | Training curves, hyperparams |
| **Data Loading** | Pipeline | `torch.utils.data` DataLoader | `webdataset`, `DALI` | Batched, shuffled samples |
| **Augmentation** | Training-time | RawBoost (custom), `torch-audiomentations` | `audiomentations`, `sox`-based | Augmented waveforms |

### 11.3 Step-by-Step Pipeline Explained

| Step | Input | Process | Output | Primary Tool |
|------|-------|---------|--------|-------------|
| 1. **Ingest** | .flac file | Load, resample to 16kHz, mono | Float32 tensor [1, N] | torchaudio |
| 2. **Preprocess** | Waveform | VAD → Pre-emphasis → Normalize → Pad/Truncate to 64000 samples (4s) | Clean waveform [1, 64000] | Silero VAD + torchaudio |
| 3a. **Spectral Features** | Clean waveform | → WavLM-Large → Weighted layer sum | Embedding [T, 1024] | HuggingFace transformers |
| 3b. **Spectral Features (Alt)** | Clean waveform | → STFT → Mel filterbank → Log → MFCC | MFCC matrix [T, 40] | torchaudio.transforms |
| 3c. **Prosodic Features** | Clean waveform | → Praat: F0, jitter, shimmer, HNR, rate → Statistics | Feature vector [1, 12] | parselmouth |
| 3d. **Linguistic Features** | Clean waveform | → WavLM → Style Projector + Ling Projector | Two embeddings [1, 256] | Custom SLIM model |
| 3e. **Segmental Features** | Clean waveform | → MFA alignment → Per-vowel formants | Formant matrix [V, 3] (V vowels) | MFA + parselmouth |
| 4a. **Spectral Agent** | Embedding [T, 1024] | → AASIST graph backend → Score | score₁ ∈ ℝ | torch_geometric |
| 4b. **Prosodic Agent** | Feature [1, 12] | → MLP (64→32→1) + Attention | score₂ ∈ ℝ | PyTorch |
| 4c. **Linguistic Agent** | Style + Ling embeds | → Cosine similarity + MLP | score₃ ∈ ℝ | PyTorch |
| 4d. **Segmental Agent** | Formant [V, 3] | → Per-vowel LR → Average | score₄ ∈ ℝ | sklearn |
| 5. **Fusion** | [score₁, score₂, score₃, score₄] | → Weighted average → Meta-classifier | Final score + confidence | XGBoost |
| 6. **Explain** | Scores + internal features | → SHAP per agent + Aggregation | Human-readable report | shap + captum |
| 7. **Evaluate** | Scores vs. labels | → EER, minDCF, Cllr, AUC computation | Metric values | sklearn + custom |

### 11.4 Training Strategy

| Phase | What | Duration | Details |
|-------|------|----------|---------|
| **Phase 0** | SSL pre-training (skip if using pre-trained WavLM) | — | Use HuggingFace checkpoint |
| **Phase 1** | Train each agent independently | 20-50 epochs each | Binary cross-entropy loss; early stopping on validation EER |
| **Phase 2** | Train meta-classifier on frozen agent outputs | 5-10 epochs | Feed agent scores into XGBoost/MLP; use validation set for weight learning |
| **Phase 3** | Optional end-to-end fine-tuning | 5-10 epochs (low LR) | Unfreeze top layers of spectral agent; joint optimization |
| **Phase 4** | Robustness hardening | Extra augmentation passes | RawBoost, adversarial training, codec augmentation |
| **Phase 5** | Calibration | Post-training | Platt scaling or temperature scaling for each agent's scores |

### 11.5 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 1× RTX 3060 (12GB) for AASIST-only | 1× A100 (80GB) for WavLM-Large fine-tuning |
| **RAM** | 16 GB | 64 GB (for large dataset loading) |
| **Storage** | 50 GB (ASVspoof 5 dataset) | 200 GB (dataset + checkpoints + augmented data) |
| **CPU** | 8 cores | 16+ cores (for Praat/openSMILE feature extraction) |

---

## 12. Comparative Literature Table

| # | Paper | Year | arXiv / DOI | Agent Type | Key Innovation | EER (%) | Dataset |
|---|-------|------|-------------|-----------|----------------|---------|---------|
| 1 | AASIST (Jung et al.) | 2022 | 2110.01200 | Spectral | Graph attention on spectro-temporal features | ~1-3 | ASVspoof 2019 LA |
| 2 | RawBoost (Tak et al.) | 2022 | 2111.04433 | Augmentation | Raw data boosting for anti-spoofing | Improved baselines | ASVspoof 2021 |
| 3 | WavLM (Chen et al.) | 2022 | 2110.13900 | SSL Backbone | Masked speech denoising pre-training | Foundation model | General |
| 4 | Pitch Imperfect (Warren et al.) | 2025 | 2502.14726 | Prosodic | 6-feature prosody detector, adversarial robust | 24.7 | ASVspoof 2021 |
| 5 | SLIM (Zhu et al.) | 2024 | NeurIPS 2024 | Linguistic | Style-linguistics mismatch detection | OOD improved | ASVspoof 2019 LA |
| 6 | Kulkarni et al. | 2024 | ASVspoof5 | Ensemble | 4× WavLM score fusion | 6.56 | ASVspoof 5 |
| 7 | WavLM Backends (Kang et al.) | 2024 | 2409.05032 | Spectral | WavLM backend exploration for ASVspoof 5 | Competitive | ASVspoof 5 |
| 8 | Multi-Fusion Attentive (Wu et al.) | 2023 | 2312.08089 | Fusion | WavLM + multi-fusion attentive classifier | Improved | ASVspoof 2021 |
| 9 | Retrieval-Augmented (Liu et al.) | 2025 | 2509.21728 / 2404.13892 | Retrieval | k-NN retrieval + selective fusion | ~12 | Cross-dataset |
| 10 | Meta-Learning (Kukanov et al.) | 2025 | 2410.20578 | Robustness | ASDG for unseen attack generalization | Improved OOD | ASVspoof 2019/2021 |
| 11 | ResNeXt Multi-Feature | 2025 | ScienceDirect | Spectral | LFCC+MFCC+CQCC with ResNeXt | 1.05 | ASVspoof 2019 |
| 12 | Yang et al. (Formant) | 2026 | — | Segmental | Vowel formant articulatory fingerprints | <10 | SpoofCeleb, ITW |
| 13 | Scalable AASIST | 2025 | 2507.11777 | Spectral | Efficient graph attention refinement | — | ASVspoof |
| 14 | SpeechFake (Multilingual) | 2025 | 2507.21463 | Dataset | Large-scale multilingual deepfake dataset | Benchmark | Multilingual |
| 15 | ASVspoof 5 Database | 2024 | 2502.08857 | Dataset | 2000 speakers, 32 attacks, crowdsourced | Benchmark | ASVspoof 5 |
| 16 | Meta-Learning + Adversarial | 2024 | 2408.13341 | Robustness | Meta-learning + disentangled adversarial training | Improved OOD | ASVspoof |
| 17 | Zero-Day via Retrieval (Liu et al.) | 2025 | 2509.21728 | Retrieval | Profile matching for zero-day attacks | ~15 (OOD) | AI4T |
| 18 | Bengali Transfer Learning | 2025 | 2512.21702 | Cross-lingual | Zero-shot to zero-lies, transfer learning | — | Bengali deepfake |
| 19 | Multi-View Collaborative (AAAI) | 2025 | AAAI Proc. | Fusion | Multi-view collaborative learning network | 38% rel. improvement | ASVspoof 2019/2021 |
| 20 | Audio Deepfakes Survey (Zhang) | 2025 | MDPI Sensors | Survey | Comprehensive review + future directions | — | — |

---

## 13. References

1. **Jung, W., et al.** (2022). "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks." *ICASSP / Interspeech*. [arXiv:2110.01200](https://arxiv.org/abs/2110.01200)

2. **Tak, H., et al.** (2022). "RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing." *ICASSP*. [arXiv:2111.04433](https://arxiv.org/abs/2111.04433)

3. **Chen, S., et al.** (2022). "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing." *IEEE JSTSP*. [arXiv:2110.13900](https://arxiv.org/abs/2110.13900)

4. **Warren, K., et al.** (2025). "Pitch Imperfect: Detecting Audio Deepfakes Through Acoustic Prosodic Analysis." [arXiv:2502.14726](https://arxiv.org/abs/2502.14726)

5. **Zhu, Y., et al.** (2024). "SLIM: Style-Linguistics Mismatch Model for Generalized Audio Deepfake Detection." *NeurIPS 2024*. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7d6930fd71740eae21224a5ffb70cb8c-Abstract-Conference.html)

6. **Kulkarni, S., et al.** (2024). "WavLM model ensemble for audio deepfake detection." *ASVspoof 5 Challenge*. [arXiv:2408.07414](https://arxiv.org/abs/2408.07414)

7. **Kang, W., et al.** (2024). "Exploring WavLM Back-ends for Speech Spoofing and Deepfake Detection." [arXiv:2409.05032](https://arxiv.org/abs/2409.05032)

8. **Wu, H., et al.** (2023). "Audio Deepfake Detection with Self-Supervised WavLM and Multi-Fusion Attentive Classifier." [arXiv:2312.08089](https://arxiv.org/abs/2312.08089)

9. **Liu, Z., et al.** (2025). "Zero-Day Audio DeepFake Detection via Retrieval Augmentation and Profile Matching." [arXiv:2509.21728](https://arxiv.org/abs/2509.21728)

10. **Liu, Z., et al.** (2024). "Retrieval-Augmented Audio Deepfake Detection." [arXiv:2404.13892](https://arxiv.org/abs/2404.13892)

11. **Kukanov, I., et al.** (2025). "Meta-Learning Approaches for Improving Detection of Unseen Speech Deepfakes." [arXiv:2410.20578](https://arxiv.org/abs/2410.20578)

12. **Wang, X., et al.** (2024). "ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech." [arXiv:2502.08857](https://arxiv.org/abs/2502.08857)

13. **Yang, Y., et al.** (2026). "Segmental Articulatory Features for Deepfake Speech Detection." *IEEE TASLP*.

14. **Jiang, L., et al.** (2025). "A Comprehensive Approach to Deepfake Audio Detection: Using Feature Fusion and Deep Learning." *IEEE Xplore*.

15. **Li, J., et al.** (2025). "Towards Scalable AASIST: Refining Graph Attention for Speech Deepfake Detection." [arXiv:2507.11777](https://arxiv.org/abs/2507.11777)

16. **Zhang, L.** (2025). "Audio Deepfake Detection: What Has Been Achieved and What Lies Ahead." *MDPI Sensors*.

17. **Chen, X., et al.** (2024). "Toward Improving Synthetic Audio Spoofing Detection Robustness via Meta-Learning and Disentangled Training With Adversarial Examples." [arXiv:2408.13341](https://arxiv.org/abs/2408.13341)

18. **Yang, Y., et al.** (2025). "Multi-View Collaborative Learning Network for Speech Deepfake Detection." *AAAI Conference Proceedings*.

19. **Govindu, A., et al.** (2023/2025). "Deepfake audio detection and justification with Explainable Artificial Intelligence (XAI)." *Research Square / Journal*.

20. **Momin, M.S., et al.** (2025). "Explainable AI techniques in context of deepfake detection: A comprehensive survey." *Image and Vision Computing*.

21. **Rahman, A., et al.** (2025). "Zero-Shot to Zero-Lies: Detecting Bengali Deepfake Audio through Transfer Learning." [arXiv:2512.21702](https://arxiv.org/abs/2512.21702)

22. **Li, Y., et al.** (2025). "SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset." [arXiv:2507.21463](https://arxiv.org/abs/2507.21463)

23. **Celik, T., et al.** (2026). "Hand-crafted wavelet and pattern features for audio deepfake detection." *Journal*.

24. **Yang, H., et al.** (2025). "Poin-HierNet: Hierarchical Poincaré Prototypical Networks for Deepfake Detection." *Conference*.

---

**Dataset:** ASVspoof 5 (2024) — freely available at [ASVspoof.org](https://www.asvspoof.org/)

**License:** This project is for academic/research purposes.

**Last Updated:** March 2026