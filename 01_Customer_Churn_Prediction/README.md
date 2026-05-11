# 📉 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![sklearn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Predict which telecom customers are likely to churn using the **IBM Telco Customer Churn** dataset. Multiple classification models are compared with proper cross-validation, and all results are automatically saved as plots.

---

## 📊 Dataset

**Telco Customer Churn** — 7,043 customers × 21 features  
Source: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

| Feature Group | Examples |
|---------------|---------|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Services | PhoneService, InternetService, StreamingTV |
| Account info | tenure, Contract, PaperlessBilling, PaymentMethod |
| Charges | MonthlyCharges, TotalCharges |
| **Target** | **Churn** (Yes/No) |

**Churn rate: ~26.5%** (imbalanced — handled via stratified splits)

---

## 🏗️ What's Improved

| Original | This version |
|----------|-------------|
| Single model (Logistic Reg) | **5 models** compared with CV |
| Single train/test split | **Stratified 5-fold cross-validation** |
| Basic encoding | Feature engineering + pipeline |
| No plots saved | ROC, PR curve, confusion matrix, feature importance |
| No CLI | Full `argparse` CLI |

---

## 🚀 Quick Start

```bash
cd 01_Customer_Churn_Prediction
pip install -r requirements.txt

# Train best model (auto-selected)
python train.py

# Compare ALL models with 5-fold CV
python train.py --compare

# Use a specific model
python train.py --model xgboost
```

---

## 📁 Structure

```
01_Customer_Churn_Prediction/
├── src/
│   ├── data.py       # Load, clean, engineer features
│   ├── model.py      # All models + cross-validation
│   └── evaluate.py   # ROC/PR/confusion plots
├── train.py          # CLI entry point
├── results/          # Auto-created: plots + CV table
└── requirements.txt
```

---

## 📈 Results (XGBoost, 5-fold CV)

| Metric | Score |
|--------|-------|
| ROC-AUC | ~0.848 |
| F1 | ~0.614 |
| Precision | ~0.672 |
| Recall | ~0.567 |

*Results may vary slightly due to random seeds.*

---

## 🔧 CLI Options

```
--data      PATH    CSV file path         [../WA_Fn-UseC_-Telco-Customer-Churn.csv]
--model     STR     Model key             [auto]
--compare           Compare all models with CV
--cv        INT     K-fold splits         [5]
--test_size FLOAT   Holdout fraction      [0.2]
--seed      INT     Random seed           [42]
```
