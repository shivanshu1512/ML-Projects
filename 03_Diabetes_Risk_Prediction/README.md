# 🩺 Diabetes Risk Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![sklearn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Predict diabetes risk using the **PIMA Indians Diabetes Dataset**. This project correctly handles the dataset's notorious zero-value problem, adds smart feature engineering, and includes a Neural Network (MLP) alongside tree models.

---

## 📊 Dataset

**PIMA Indians Diabetes** — 768 patients × 8 features  
Source: `diabetes.csv` (Kaggle / UCI)

| Feature | Note |
|---------|------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose (2h oral test) |
| BloodPressure | Diastolic BP (mm Hg) |
| SkinThickness | Triceps skinfold (mm) |
| Insulin | 2-hour serum insulin |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Genetic risk score |
| Age | Years |
| **Outcome** | **1=Diabetic, 0=Healthy** |

**Positive rate: ~34.9%**

---

## 🏗️ What's Improved

| Original | This version |
|----------|-------------|
| Ignores zero values | **Zeros replaced with median** (medically impossible) |
| Raw features | BMI category, age group, glucose×BMI interaction |
| Single model | LR + RF + **MLP (neural net)** + XGBoost |
| No threshold tuning | **F1-optimal decision threshold** |
| No CLI | Full `argparse` CLI |

---

## 🚀 Quick Start

```bash
cd 03_Diabetes_Risk_Prediction
pip install -r requirements.txt

# Train (default: XGBoost or best available)
python train.py --data diabetes.csv

# Find best decision threshold automatically
python train.py --data diabetes.csv --tune_threshold

# Compare all models
python train.py --data diabetes.csv --compare

# Use MLP neural network
python train.py --data diabetes.csv --model mlp
```

---

## 📁 Structure

```
03_Diabetes_Risk_Prediction/
├── src/
│   ├── data.py      # PIMA preprocessing + feature engineering
│   └── model.py     # LR, RF, MLP, XGBoost + threshold tuning
├── train.py         # CLI entry point
├── results/         # ROC + confusion matrix plots
└── requirements.txt
```

---

## 📈 Results (XGBoost, default threshold=0.5)

| Metric | Score |
|--------|-------|
| ROC-AUC | ~0.835 |
| F1 | ~0.672 |
| Precision | ~0.724 |
| Recall | ~0.628 |
