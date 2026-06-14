# Multi-Agent Deepfake Speech Detection System (MAD-SDS)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green.svg)](https://xgboost.readthedocs.io/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange.svg)](https://huggingface.co/docs/transformers/index)
[![ASVspoof 5](https://img.shields.io/badge/ASVspoof_5-Supported-red.svg)](https://www.asvspoof.org/)

MAD-SDS is a state-of-the-art **multi-agent forensic speech detection system** designed to identify artificial voice clones, synthetic text-to-speech (TTS), and voice conversion attacks. Powered by federated agent intelligence, real-time C++ Tree-SHAP explainability, and a robust self-supervised speech backbone, MAD-SDS delivers deep, instant acoustic and semantic audits.

---

## End-to-End System Architecture

MAD-SDS runs on a federated multi-agent architecture. Instead of relying on a single fallible neural loop, the system orchestrates four highly specialized "expert" agents. Their individual probability vectors are analyzed and combined using a calibrated **Logistic Regression Meta-Classifier (Decision Agent)**.

```mermaid
graph TD
    A[Raw Audio Input] --> B{Agent Ensemble}
    
    subgraph Signal Analysis (Tree-SHAP)
        B -->|MFCC/Flux/CQT| C[Spectral Agent - XGBoost]
        B -->|Pitch/Jitter/Shimmer| D[Prosodic Agent - XGBoost]
    end
    
    subgraph Contextual Semantics
        B -->|ASR Whisper + DistilBERT| E[Linguistic Agent]
    end
    
    subgraph Self-Supervised Acoustic Robustness
        B -->|WavLM Pruned Embeddings| F[SSL Agent]
    end
    
    C -->|P_spec + Tree-SHAP| G[Decision Fusion - Logistic Regression]
    D -->|P_pros + Tree-SHAP| G
    E -->|P_ling| G
    F -->|P_ssl| G
    
    G --> H{Final Decision}
    H -->|Confidence > 50%| I[ SPOOF]
    H -->|Confidence <= 50%| J[BONAFIDE]
    
    style I fill:#f43f5e,stroke:#e11d48,stroke-width:2px,color:#fff
    style J fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    style G fill:#6366f1,stroke:#4f46e5,stroke-width:2px,color:#fff
```

---

## The Multi-Agent Ensemble

### 1. Spectral Agent (The "Texture" Detector)
*   **Booster Model:** XGBoost (e.g. LightGBM, CatBoost support).
*   **Acoustic Target:** Evaluates frequency-domain signatures and vocoder anomalies.
*   **Signal Features:**
    *   **MFCCs (Mel-Frequency Cepstral Coefficients):** Captures physical vocal tract resonance. Voice clones typically exhibit micro-discontinuities during rapid consonant-to-vowel transitions.
    *   **Spectral Flux:** Measures the velocity of the power spectrum envelope change. AI models tend to produce unnaturally uniform frequency changes.
    *   **Spectral Centroid:** Identifies the "spectral center of mass" to catch artificial high-frequency "metallic" spikes.
*   **Explainability:** Integrated with **native C++ Tree-SHAP** (`pred_contribs=True`) to map signal features that argue for or against synthesis directly onto the UI.

### 2. Prosodic Agent (The "Naturalness" Detector)
*   **Booster Model:** XGBoost.
*   **Acoustic Target:** Evaluates the intonation, breathiness, and emotional rhythm of speech.
*   **Signal Features:**
    *   **Pitch Contour (F0F_0):** Measures fundamental frequency variations. Synthetic text-to-speech generators often display robotic monotonicity (flat intonation).
    *   **Jitter:** Measures micro-frequency variations in pitch timing. Synthesized speech often sounds "too perfect" (low jitter) because it lacks biological vocal-fold fluctuations.
    *   **Shimmer:** Measures breath-related micro-variations in pitch amplitude.
    *   **Harmonics-to-Noise Ratio (HNR):** Quantifies vocal clarity versus turbulent breath noise.

### 3. Linguistic Agent (The "Context" Detector)
*   **Pipeline:** OpenAI Whisper-Tiny (ASR) + Fine-tuned DistilBERT Classifier.
*   **Acoustic Target:** Identifies synthetic semantic structures and unnatural vocabulary loops.
*   **Workflow:** The raw audio is transcribed using Whisper-Tiny in a single-pass inference step. The transcribed text is analyzed by our fine-tuned DistilBERT model to flag grammatical or lexical deepfake signatures.

### 4.  SSL Agent (The Acoustic Robustness Shield)
*   **Bakbone Model:** Pruned WavLM Base (`JYP2024/Wedefense_ASV2025_WavLM_Base_Pruning`).
*   **Why we need a Robustness Model:** Hand-engineered features (like MFCCs) struggle under telephone compression, room reverberation, and background noise. WavLM is pre-trained on over 94,000 hours of distorted speech using **Masked Speech Denoising**, forcing it to separate environmental noise from structural voice patterns.
*   **Performance:** Achieves an Equal Error Rate (EER) of **under 1.8%** on the competitive **ASVspoof 5 Challenge** dataset.

---

## Class Imbalance & Tabular SMOTE
The ASVspoof 5 dataset contains a heavy ratio of Spoof (~80%) to Bonafide (~20%) speech samples.
*   **The Threat:** Neural networks and boosting classifiers trained on this directly would become lazy, guessing "Spoof" to achieve high accuracy while failing to identify genuine human speech.
*   **The Safeguard (SMOTE):** We apply **Synthetic Minority Over-sampling Technique (SMOTE)** in the tabular feature space during Spectral and Prosodic agent calibration. This creates synthetic Bonafide samples along minority vectors, forcing the decision boundaries to learn the exact physical properties of real human voices rather than memorizing the spoof majority.

---

##  Decision Fusion & Shapley Explanations
When an audit is triggered, the system calculates exact log-odds contributions (ϕi=βi⋅Pi\phi_i = \beta_i \cdot P_i) using the coefficients of the Logistic Regression Meta-Classifier:
*   **Linear Shapley Impact:** The UI dynamically renders how much weight each individual agent had in establishing the final verdict (e.g., `+34% (Argues FAKE)` or `-12% (Argues REAL)`).
*   **Tree-SHAP Badges:** Signals that pushed the XGBoost models over the threshold are visualized as glowing, color-coded badges (e.g. `STFT Skewness (+0.14)` in rose-red or `Mel-Bin 12 (-0.08)` in emerald-green).

---
