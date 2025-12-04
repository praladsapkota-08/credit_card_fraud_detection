# Credit Card Fraud Detection

## 📌 Project Overview
This project implements a **machine learning pipeline for detecting fraudulent credit card transactions**.  
It uses the `creditcard.csv` dataset, applies preprocessing, handles class imbalance with **SMOTE**, and evaluates multiple models including **Logistic Regression, XGBoost, and Random Forest**.  

The goal is to build a robust fraud detection system that balances accuracy with precision/recall, given the highly imbalanced nature of fraud datasets.

---

## ⚙️ Workflow Summary
1. **Data Loading & Exploration**
   - Load dataset (`creditcard.csv`) using pandas.
   - Explore with `.head()`, `.info()`, `.describe()`.
   - Visualize distributions (`Amount`, `Time`) using seaborn/matplotlib.
   - Create new feature `hours` from transaction time.

2. **Preprocessing**
   - Log transform `Amount` to reduce skewness.
   - Standardize `Amount` and `hours` using `StandardScaler`.

3. **Handling Imbalance**
   - Fraud cases are rare → apply **SMOTE (Synthetic Minority Oversampling Technique)** to balance training data.

4. **Model Training & Evaluation**
   - **Logistic Regression**: baseline model.
   - **XGBoost Classifier**: trained with and without SMOTE, tuned hyperparameters.
   - **Random Forest Classifier**: additional ensemble model.
   - Metrics: **Confusion Matrix, Classification Report, Accuracy, AUPRC (Area Under Precision‑Recall Curve)**.
   - Cross‑validation for stability.

5. **Model Persistence**
   - Save best performing model (`xgb_class_smote`) using `joblib` for deployment.

---

## 🧪 Models Compared
| Model | Technique | Key Notes |
|-------|-----------|-----------|
| Logistic Regression | With SMOTE | Baseline linear model, interpretable. |
| XGBoost (no SMOTE) | Weighted training | Handles imbalance via `scale_pos_weight`. |
| XGBoost (with SMOTE) | Tuned hyperparameters | Best performing model, saved for deployment. |
| Random Forest | With SMOTE | Ensemble approach, compared against XGBoost. |

---

## 📊 Evaluation Metrics
- **Confusion Matrix**: visualize fraud vs non‑fraud predictions.  
- **Classification Report**: precision, recall, F1‑score.  
- **AUPRC (Average Precision Score)**: chosen because fraud detection is an imbalanced classification problem.  
- **Accuracy**: reported but less reliable due to imbalance.  
- **Cross‑Validation**: ensures model generalization.

---

## 🚀 Why This Approach?
- Fraud detection datasets are **highly imbalanced** (fraud cases are <1%).  
- Standard accuracy is misleading → focus on **precision/recall and AUPRC**.  
- **SMOTE** helps balance training data by generating synthetic fraud samples.  
- **XGBoost** is powerful for tabular data, handles imbalance well, and provides strong performance.  
- **Random Forest** adds ensemble diversity for comparison.  
- Final model (`xgb_class_smote`) is tuned and persisted for real‑world use.

---


