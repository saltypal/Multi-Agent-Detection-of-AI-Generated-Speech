# Teach + Code Tutor Mode

This document defines the tutoring method to use across the notebook series in this project.

## Core Rule

One notebook, one topic, one task at a time.

Do not jump ahead to the next notebook or the next concept until the current topic is clear.

## Teaching Style

1. Start with an agenda for the notebook.
2. Explain the concept from first principles.
3. Explain every function used:
   - what it does
   - why it is needed
   - what it returns
   - what the important inputs mean
4. Explain the math behind the operation when relevant.
5. Show only one or two short code snippets.
6. Then do one small implementation task.
7. Verify the result before moving on.

## Notebook Workflow

For each notebook, follow this exact flow:

### 1. Agenda
State the single topic for the notebook and the single task we will complete first.

### 2. Concept Explanation
Explain the idea clearly before touching code.

### 3. Function-by-Function Breakdown
For every function or library call used, explain:
- purpose
- inputs
- outputs
- return value
- side effects, if any

### 4. Math Intuition
If the notebook has signal processing, feature extraction, or model logic, explain the underlying math in simple terms first, then more formally.

### 5. Minimal Code
Use only one or two snippets at a time.
Keep the snippet focused on the current task.

### 6. One Task Only
Implement just the current task.
Do not mix in extra features.

### 7. Check Understanding
Summarize what was learned and confirm the result before continuing.

## Scope for This Project

The notebook sequence is treated as a progression:

Signal -> Representation -> Features -> Agents -> Fusion -> System

We will work through the notebooks in order:

- 00_data_loading_and_visualization
- 01_preprocessing_pipeline
- 02_stft_and_spectrograms
- 03_spectral_features
- 04_prosodic_features
- 05_pitch_estimation_methods
- 06_dft_fft_understanding
- 07_time_frequency_analysis
- 08_spectral_agent_prototype
- 09_prosodic_agent_prototype
- 10_linguistic_agent_prototype
- 11_segmental_features
- 12_phase_analysis
- 13_ssl_embeddings_exploration
- 14_feature_comparison
- 15_agent_outputs_and_scores
- 16_fusion_methods

## Response Format to Use in Future Chats

When working in this mode, responses should usually include:

- a short agenda
- one concept explanation
- one or two code snippets
- one task to complete next

## Goal

The goal is not just to finish notebooks.
The goal is to build a real understanding of the signal processing and modeling pipeline step by step.

Good. Now we formalize this like a **full curriculum + lab blueprint**—not just a list.
Each notebook is broken into **atomic tasks**, and each task is something you should deeply understand (concept + math + behavior + implementation).

This is essentially your **self-built Speech Processing specialization**.

---

# 🧠 MASTER ROADMAP — ALL NOTEBOOKS (IN-DEPTH)

We follow strict hierarchy:

```text
Signal → Preprocessing → Time-Frequency → Features → Agents → Fusion
```

---

# 📘 NOTEBOOK 00 — Data Loading & Signal Understanding

## 🎯 Goal:

Understand **what speech data actually is**

### Tasks:

### 0.1 What is a discrete-time signal

* Continuous vs discrete signal
* Sampling theorem (Nyquist intuition)

### 0.2 Audio file structure

* Channels (mono/stereo)
* Bit depth
* Sampling rate meaning

### 0.3 Load audio

* `torchaudio.load()`
* Tensor structure `[channels, samples]`

### 0.4 Waveform visualization

* Amplitude vs time
* Convert index → seconds

### 0.5 Interpretation

* Silence vs speech
* Loud vs soft
* Frequency intuition (dense oscillations)

---

# 📘 NOTEBOOK 01 — Preprocessing Pipeline

## 🎯 Goal:

Standardize raw signal → **ML-ready signal**

---

### Tasks:

### 1.1 Resampling

* Why 16kHz standard
* Aliasing concept
* Downsampling vs upsampling

---

### 1.2 Normalization

* Peak normalization
* RMS normalization
* Why amplitude consistency matters

---

### 1.3 Pre-emphasis filter

[
y[n] = x[n] - \alpha x[n-1]
]

* High-frequency boosting
* Vocal tract compensation
* Effect visualization

---

### 1.4 Voice Activity Detection (VAD)

* Energy-based VAD
* Thresholding
* Silence trimming

---

### 1.5 Framing preparation

* Why we split signal
* Window length (25ms)
* Hop length (10ms)

---

# 📘 NOTEBOOK 02 — STFT & Spectrograms

## 🎯 Goal:

Convert signal → **time-frequency representation**

---

### Tasks:

### 2.1 Framing mathematically

* Stationarity assumption
* Frame extraction

---

### 2.2 Windowing

* Hamming window
* Spectral leakage

---

### 2.3 Fourier Transform intuition

* Time → frequency conversion
* Sinusoidal decomposition

---

### 2.4 FFT computation

* Why FFT (O(N log N))
* Difference from DFT 

---

### 2.5 STFT formulation

[
X(m, k) = \sum x[n] w[n-m] e^{-j2\pi kn/N}
]

* Sliding window transform

---

### 2.6 Spectrogram generation

* Magnitude spectrum
* Log scaling

---

### 2.7 Interpretation

* Formants
* Harmonics
* Speech patterns

---

# 📘 NOTEBOOK 03 — Spectral Features

## 🎯 Goal:

Extract **compact frequency representations**

---

### Tasks:

### 3.1 Mel scale

* Human perception
* Linear vs log frequency

---

### 3.2 Mel filter banks

* Triangular filters
* Energy aggregation

---

### 3.3 MFCC pipeline

* Log → DCT
* Why decorrelation

---

### 3.4 LFCC

* Linear filters
* High-frequency importance

---

### 3.5 Log spectrogram

* Raw power representation

---

### 3.6 Feature comparison

* MFCC vs Mel vs LFCC
* Information loss analysis

---

# 📘 NOTEBOOK 04 — Prosodic Features

## 🎯 Goal:

Understand **speech beyond frequency**

---

### Tasks:

### 4.1 Pitch (F0)

* Vocal cord vibration
* Periodicity

---

### 4.2 Energy

* RMS energy
* Loudness representation

---

### 4.3 Jitter

Cycle-to-cycle variation:

* Stability of vocal cords 

---

### 4.4 Shimmer

* Amplitude variation

---

### 4.5 HNR

* Harmonic vs noise ratio

---

### 4.6 Speaking rate

* Duration-based features

---

### 4.7 Interpretation

* Human vs synthetic differences

---

# 📘 NOTEBOOK 05 — Pitch Estimation Deep Dive

## 🎯 Goal:

Understand **how pitch is computed**

---

### Tasks:

### 5.1 Autocorrelation

[
R(\tau) = \sum x[n] x[n-\tau]
]

* Period detection

---

### 5.2 Zero Crossing Rate

* Sign changes
* Fast approximation

---

### 5.3 AMDF

* Difference-based matching

---

### 5.4 Compare methods

* Accuracy vs speed

---

### 5.5 Numerical problems

Use:


---

# 📘 NOTEBOOK 06 — DFT & FFT Deep Dive

## 🎯 Goal:

Remove “black box FFT thinking”

---

### Tasks:

### 6.1 DFT definition

[
X(k) = \sum x(n)e^{-j2\pi kn/N}
]

---

### 6.2 Manual computation

* Small sequences

Use:


---

### 6.3 Frequency bins meaning

* k index interpretation

---

### 6.4 FFT algorithm

* Divide & conquer
* Complexity reduction 

---

### 6.5 Visualization

* Compare DFT vs FFT outputs

---

# 📘 NOTEBOOK 07 — Time-Frequency Tradeoffs

## 🎯 Goal:

Understand **resolution limits**

---

### Tasks:

### 7.1 Window size effect

* Short window → time precision
* Long window → frequency precision

---

### 7.2 Heisenberg uncertainty in signals

* Cannot optimize both

---

### 7.3 Practical tuning

* Speech-specific windowing

---

# 📘 NOTEBOOK 08 — Spectral Agent Prototype

## 🎯 Goal:

Convert spectral features → detection

---

### Tasks:

### 8.1 Feature selection

* MFCC vs spectrogram

---

### 8.2 Feature visualization

* Patterns in fake vs real

---

### 8.3 Baseline classifier

* Logistic regression / small NN

---

### 8.4 Error analysis

* Where spectral fails

---

# 📘 NOTEBOOK 09 — Prosodic Agent

## 🎯 Goal:

Use **voice behavior patterns**

---

### Tasks:

### 9.1 Feature vector creation

* Combine prosodic stats

---

### 9.2 Distribution analysis

* Real vs fake differences

---

### 9.3 Simple model

* MLP / tree-based

---

### 9.4 Interpretability

* Feature importance

---

# 📘 NOTEBOOK 10 — Linguistic Agent

## 🎯 Goal:

Understand **speech content vs style**

---

### Tasks:

### 10.1 ASR pipeline

* Audio → text

---

### 10.2 Word-level timing

* Alignment

---

### 10.3 Linguistic embeddings

* Semantic representation

---

### 10.4 Style vs content mismatch

* SLIM intuition

---

# 📘 NOTEBOOK 11 — Segmental Features

## 🎯 Goal:

Speech as **articulatory system**

---

### Tasks:

### 11.1 Phoneme segmentation

* Forced alignment

---

### 11.2 Formants (F1, F2, F3)

* Vocal tract resonances

---

### 11.3 Vowel analysis

* Speaker-specific patterns

---

# 📘 NOTEBOOK 12 — Phase Analysis

## 🎯 Goal:

Go beyond magnitude

---

### Tasks:

### 12.1 Phase spectrum

* What phase represents

---

### 12.2 Group delay

* Temporal structure

---

### 12.3 Why deepfakes fail here

* Phase inconsistency

---

# 📘 NOTEBOOK 13 — SSL Embeddings

## 🎯 Goal:

Understand modern representations

---

### Tasks:

### 13.1 WavLM embeddings

* Frame-level representation

---

### 13.2 Layer analysis

* What each layer captures

---

### 13.3 Clustering

* Real vs fake separation

---

# 📘 NOTEBOOK 14 — Feature Comparison

## 🎯 Goal:

Unify understanding

---

### Tasks:

### 14.1 Compare all features

* Spectral vs prosodic vs linguistic

---

### 14.2 Redundancy analysis

* What overlaps

---

### 14.3 Complementarity

* Why multi-agent works

---

# 📘 NOTEBOOK 15 — Agent Outputs

## 🎯 Goal:

Simulate system behavior

---

### Tasks:

### 15.1 Score generation

* Each agent → probability

---

### 15.2 Agreement analysis

* When agents disagree

---

### 15.3 Failure cases

* Edge scenarios

---

# 📘 NOTEBOOK 16 — Fusion Methods

## 🎯 Goal:

Final decision system

---

### Tasks:

### 16.1 Score-level fusion

* Weighted averaging

---

### 16.2 Voting methods

* Majority / threshold

---

### 16.3 Meta-classifier

* Learn combination

---

### 16.4 Robustness testing

* Unseen scenarios


---

# ⚠️ RULES GOING FORWARD

1. We go **one notebook at a time**
2. Inside notebook → **one task at a time**
3. You should:

   * predict output before running code
   * question every transformation

---

