# 🧠 Fusion Module Blueprint

The Fusion Agent is the final decision-maker of our deepfake detection pipeline. Its job is to ingest the separate opinions of our four independent specialist agents and output a single, highly accurate classification.

## 1. The Power of Probabilities

> [!WARNING]  
> **Do not use "Yes/No" hard labels for the individual agents!**

To answer your question: Yes, the XGBoost classifiers and the BERT model **must** output continuous probabilities (e.g., `0.92` or `0.45`), not just binary `1` or `0` classifications. 

If an agent outputs a hard "Yes" (1), the Fusion module has no idea if the agent is 51% sure or 99.9% sure. By feeding continuous probabilities (which I have already coded `spectral_model.py`, `prosodic_model.py`, and BERT to output) into the Fusion module, we preserve crucial confidence information.

### The Input Features
For every audio file, the Fusion model receives exactly 4 features:
- **$P_{spec}$** $\in [0, 1]$ (XGBoost Spectral)
- **$P_{pros}$** $\in [0, 1]$ (XGBoost Prosodic)
- **$P_{ling}$** $\in [0, 1]$ (BERT Linguistic)
- **$P_{ssl}$** $\in [0, 1]$ (WavLM Deep Learning)

## 2. Meta-Classifier Options

We are essentially building a machine learning model *on top* of machine learning models (Stacking / Meta-Classification). 

### Option A: Logistic Regression (Recommended)
A linear combination: `Output = Sigmoid(w1*P_spec + w2*P_pros + w3*P_ling + w4*P_ssl + bias)`
> [!TIP]  
> This is highly recommended for our use case. It is lightning fast, mathematically prevents overfitting (since there are only 4 weights to learn), and gives us beautiful interpretability. We can look at the 4 weights to see exactly which agent the system trusts the most!

### Option B: LightGBM / XGBoost Meta-Model
A small decision tree trained on the 4 probabilities.
- **Pros**: Can learn complex conditional rules. (e.g., *"If the linguistic agent says 0.99, trust it unconditionally, but if it says 0.50, ignore it and rely entirely on the spectral agent."*)
- **Cons**: Slightly higher risk of overfitting the validation set compared to linear regression.

## 3. Implementation Workflow

When you are ready to build the Fusion module, we will follow these exact steps:

1. **Extract OOF (Out-of-Fold) Predictions**: We run the entire `train` and `test` dataset audio through all four fully-trained agents to collect their probability scores.
2. **Build the Fusion Dataset**: We generate a simple CSV named `fusion_features.csv`:
   | filename | p_spec | p_pros | p_ling | p_ssl | true_label |
   |---|---|---|---|---|---|
   | file1.flac | 0.95 | 0.88 | 0.51 | 0.98 | 1 |
   | file2.flac | 0.05 | 0.12 | 0.49 | 0.02 | 0 |
3. **Train & Plot**: We train the Logistic Regression meta-model on this CSV and output the final, combined EER and AUC plots.
