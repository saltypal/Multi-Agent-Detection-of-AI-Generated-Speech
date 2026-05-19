# about dataset 

============================================================
Analyzing & Fixing spectral_train.csv...
============================================================
[BEFORE] Total Rows: 182,357
  - Spoof (AI): 182,357 (100.00%)
  - Bonafide (Real): 0 (0.00%)

[AFTER] Total Rows: 182,357
  - Spoof (AI): 163,560 (89.69%)
  - Bonafide (Real): 18,797 (10.31%)
  - Imbalance Ratio (Spoof:Bonafide): 8.70:1
✅ Successfully corrected and saved spectral_train.csv.
⚠️ File not found: spectral_dev.csv (Skipping)

============================================================
Analyzing & Fixing spectral_eval.csv...
============================================================
[BEFORE] Total Rows: 408,022
  - Spoof (AI): 408,022 (100.00%)
  - Bonafide (Real): 0 (0.00%)

[AFTER] Total Rows: 408,022
  - Spoof (AI): 325,176 (79.70%)
  - Bonafide (Real): 82,846 (20.30%)
  - Imbalance Ratio (Spoof:Bonafide): 3.93:1
✅ Successfully corrected and saved spectral_eval.csv.

============================================================
Analyzing & Fixing prosodic_train.csv...
============================================================
[BEFORE] Total Rows: 182,357
  - Spoof (AI): 182,357 (100.00%)
  - Bonafide (Real): 0 (0.00%)

[AFTER] Total Rows: 182,357
  - Spoof (AI): 163,560 (89.69%)
  - Bonafide (Real): 18,797 (10.31%)
  - Imbalance Ratio (Spoof:Bonafide): 8.70:1
✅ Successfully corrected and saved prosodic_train.csv.
⚠️ File not found: prosodic_dev.csv (Skipping)

============================================================
Analyzing & Fixing prosodic_eval.csv...
============================================================
[BEFORE] Total Rows: 408,022
  - Spoof (AI): 408,022 (100.00%)
  - Bonafide (Real): 0 (0.00%)

[AFTER] Total Rows: 408,022
  - Spoof (AI): 325,176 (79.70%)
  - Bonafide (Real): 82,846 (20.30%)
  - Imbalance Ratio (Spoof:Bonafide): 3.93:1
✅ Successfully corrected and saved prosodic_eval.csv.


# Imbalance Ratios: An imbalance of 8.7:1 (Train) and 3.9:1 (Eval) is exactly what the ASVspoof 5 official dataset distributions are supposed to look like. 



## EVALUATION METRICS
The Equal Error Rate (EER) is the standard metric used in the ASVspoof (Automatic Speaker Verification Spoofing) challenges to measure the performance of anti-spoofing and deepfake detection systems. It represents the point where the False Acceptance Rate (genuine speech classified as fake) and False Rejection Rate (fake speech classified as genuine) are equal




========================================================

 
======================================================================
🚀 STARTING TRAINING PIPELINE FOR: SPECTRAL AGENT
======================================================================
⚠️ dev.csv not found! Automatically carving 10% Dev set from Train...
📊 Data Splits -> Train: 164,121 | Dev: 18,236 | Eval: 408,022

⚖️ Class Distribution:
  - Class 0 [Bonafide]: 16,917 (10.31%)
  - Class 1 [Spoof]: 147,204 (89.69%)
Applying SMOTE to perfectly balance classes (1:1)... 
  - Balanced Train Shape: (294408, 35) (Spoof: 147,204, Bonafide: 147,204)

--- 🪵 Training XGBoost ---
[0]	validation_0-logloss:0.66334	validation_0-aucpr:0.98846
[100]	validation_0-logloss:0.17655	validation_0-aucpr:0.99719
[200]	validation_0-logloss:0.12953	validation_0-aucpr:0.99823
[300]	validation_0-logloss:0.10537	validation_0-aucpr:0.99873
[400]	validation_0-logloss:0.08904	validation_0-aucpr:0.99903
[499]	validation_0-logloss:0.07769	validation_0-aucpr:0.99921

--- ⚡ Training LightGBM ---

--- 🐈 Training CatBoost ---
Default metric period is 5 because AUC is/are not implemented for GPU
0:	test: 0.9156275	best: 0.9156275 (0)	total: 27.6ms	remaining: 13.8s
100:	test: 0.9693716	best: 0.9693716 (100)	total: 872ms	remaining: 3.44s
200:	test: 0.9778825	best: 0.9778825 (200)	total: 1.59s	remaining: 2.37s
300:	test: 0.9825198	best: 0.9825198 (300)	total: 2.3s	remaining: 1.52s
400:	test: 0.9854469	best: 0.9854469 (400)	total: 3.04s	remaining: 749ms
499:	test: 0.9874484	best: 0.9874484 (499)	total: 3.78s	remaining: 0us
bestTest = 0.9874483943
bestIteration = 499
/usr/local/lib/python3.12/dist-packages/xgboost/core.py:751: UserWarning: [15:50:08] WARNING: /__w/xgboost/xgboost/src/common/error_msg.cc:62: Falling back to prediction using DMatrix due to mismatched devices. This might lead to higher memory usage and slower performance. XGBoost is running on: cuda:0, while the input data is on: cpu.
Potential solutions:
- Use a data structure that matches the device ordinal in the booster.
- Set the device for booster before call to inplace_predict.

This warning will only be shown once.

  return func(**kwargs)

📌 Model Performance: XGBoost
  - EER      : 38.96%
  - AUC      : 0.6551
  - Accuracy : 50.29%
/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
  warnings.warn(

📌 Model Performance: LightGBM
  - EER      : 39.22%
  - AUC      : 0.6497
  - Accuracy : 48.36%

📌 Model Performance: CatBoost
  - EER      : 39.31%
  - AUC      : 0.6481
  - Accuracy : 49.87%

======================================================================
🧠 ACOUSTIC EXPLAINABILITY & DIAGNOSTICS
======================================================================
/tmp/ipykernel_9032/23724010.py:150: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x='importance', y='feature', data=xgb_imp, ax=axes[0], palette='magma')
/tmp/ipykernel_9032/23724010.py:156: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x='importance', y='feature', data=lgb_imp, ax=axes[1], palette='viridis')
/tmp/ipykernel_9032/23724010.py:162: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x='importance', y='feature', data=cb_imp, ax=axes[2], palette='mako')
🎉 Complete evaluation metrics, ROC curves, and Explainability analysis saved to Drive!


 
======================================================================
🚀 STARTING TRAINING PIPELINE FOR: PROSODIC AGENT
======================================================================
⚠️ dev.csv not found! Automatically carving 10% Dev set from Train...
📊 Data Splits -> Train: 164,121 | Dev: 18,236 | Eval: 408,022

⚖️ Class Distribution:
  - Class 0 [Bonafide]: 16,917 (10.31%)
  - Class 1 [Spoof]: 147,204 (89.69%)
Applying SMOTE to perfectly balance classes (1:1)... 
  - Balanced Train Shape: (294408, 9) (Spoof: 147,204, Bonafide: 147,204)

--- 🪵 Training XGBoost ---
[0]	validation_0-logloss:0.67987	validation_0-aucpr:0.93490
[100]	validation_0-logloss:0.36908	validation_0-aucpr:0.97344
[200]	validation_0-logloss:0.30158	validation_0-aucpr:0.97530
[300]	validation_0-logloss:0.28075	validation_0-aucpr:0.97675
[400]	validation_0-logloss:0.27004	validation_0-aucpr:0.97779
[499]	validation_0-logloss:0.26415	validation_0-aucpr:0.97836

--- ⚡ Training LightGBM ---

--- 🐈 Training CatBoost ---
Default metric period is 5 because AUC is/are not implemented for GPU
0:	test: 0.6730539	best: 0.6730539 (0)	total: 11.6ms	remaining: 5.78s
100:	test: 0.8071680	best: 0.8073351 (99)	total: 775ms	remaining: 3.06s
200:	test: 0.8274898	best: 0.8274898 (200)	total: 1.42s	remaining: 2.11s
300:	test: 0.8379420	best: 0.8379420 (300)	total: 3.12s	remaining: 2.06s
400:	test: 0.8446828	best: 0.8446828 (400)	total: 5.43s	remaining: 1.34s
499:	test: 0.8488727	best: 0.8488727 (499)	total: 6.5s	remaining: 0us
bestTest = 0.8488726616
bestIteration = 499

📌 Model Performance: XGBoost
  - EER      : 46.59%
  - AUC      : 0.5441
  - Accuracy : 65.06%
/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
  warnings.warn(

📌 Model Performance: LightGBM
  - EER      : 46.76%
  - AUC      : 0.5414
  - Accuracy : 66.78%

📌 Model Performance: CatBoost
  - EER      : 46.99%
  - AUC      : 0.5392
  - Accuracy : 67.04%

======================================================================
🧠 ACOUSTIC EXPLAINABILITY & DIAGNOSTICS
======================================================================
/tmp/ipykernel_9032/23724010.py:150: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x='importance', y='feature', data=xgb_imp, ax=axes[0], palette='magma')
/tmp/ipykernel_9032/23724010.py:156: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x='importance', y='feature', data=lgb_imp, ax=axes[1], palette='viridis')
/tmp/ipykernel_9032/23724010.py:162: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x='importance', y='feature', data=cb_imp, ax=axes[2], palette='mako')
🎉 Complete evaluation metrics, ROC curves, and Explainability analysis saved to Drive!
