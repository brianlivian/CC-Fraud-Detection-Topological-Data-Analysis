# 📊 TDA-Based Fraud Detection (Weekend Project)

This repository contains a weekend project exploring how **Topological Data Analysis (TDA)** techniques — specifically **sliding window embeddings of PCA vectors** followed by **L1 norm calculations of persistence landscapes** — can be used to detect credit card fraud. 

The project uses the well-known [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and applies an experimental approach combining **topological feature extraction**, **SMOTETomek balancing**, and **LightGBM with Optuna hyperparameter tuning**.

---

## 🧠 Project Motivation

This was a fun, exploratory project to brush up on **TDA concepts**, particularly:
- Sliding window embeddings using Giotto-TDA
- Vietoris-Rips persistence diagrams
- Persistence landscapes and their L1/AUC norms
- Statistical significance tests (Welch's t-test)

Surprisingly, **TDA-derived features** (L1 norms of the persistence landscape curves) showed meaningful correlation with the fraud label and were competitive predictors in the final fraud classification model.

---

## 🔬 Method Summary

1. **Point Cloud Generation**  
   Apply **sliding window embeddings** of size 2–9 to each PCA vector (columns V1–V28), forming point clouds per row.

2. **TDA Extraction**  
   For each point cloud, compute:
   - Vietoris-Rips persistence diagrams
   - Persistence landscapes
   - L1/AUC norms over 100 sampled filtration values

3. **Statistical Testing**  
   Welch's t-test was used to determine whether the TDA-derived norms differ significantly between fraud and non-fraud classes. Many passed at the 5% level.

4. **Modeling Pipeline**
   - SMOTETomek used to balance training data
   - LightGBM classifier trained and tuned with Optuna
   - Cross-validation metrics: accuracy, recall, precision, F1, AUC

5. **Final Feature Set**
   - Top engineered features from LightGBM included both traditional PCA-derived columns (e.g., `V14`) and TDA-derived norms like `N2`.

---

## 📈 Key Finding

One of the **TDA-based features (N2)** — derived from the L1 norm of the persistence landscape for 2-sized sliding window embeddings — emerged as a surprisingly strong indicator for fraud.
