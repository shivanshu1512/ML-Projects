# 🧠 Autism Spectrum Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![sklearn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Machine learning model to detect Autism Spectrum Disorder (ASD) using the **AQ-10 screening questionnaire** dataset. Focuses on **sensitivity** and **specificity** — the metrics that matter most in medical classification.

> ⚠️ **Disclaimer**: This model is for educational purposes only. It does **not** constitute medical advice or clinical diagnosis.

---

## 📊 Dataset

**Autism Screening** dataset from UCI Machine Learning Repository  
~800 instances × 21 features — adults and/or children

| Feature Group | Examples |
|---------------|---------|
| AQ-10 scores | A1_Score … A10_Score (0 or 1) |
| Demographics | age, gender, ethnicity, country |
| History | jaundice at birth, family history of autism |
| **Target** | **Class/ASD** (YES/NO) |

---

## 🏗️ What's Improved

| Original | This version |
|----------|-------------|
| Single model | SVM, Random Forest, XGBoost compared |
| Accuracy only | **Sensitivity + Specificity** reported |
| No CV | Stratified 5-fold cross-validation |
| Basic encoding | AQ-10 total score feature + age groups |
| No CLI | Full `argparse` CLI |

---

## 🚀 Quick Start

```bash
cd 02_Autism_Spectrum_Detection
pip install -r requirements.txt

# Train (provide your dataset CSV)
python train.py --data autism_data.csv

# Compare all models
python train.py --data autism_data.csv --compare

# Use SVM
python train.py --data autism_data.csv --model svm
```

---

## 📁 Structure

```
02_Autism_Spectrum_Detection/
├── src/
│   ├── data.py     # Load, clean, engineer AQ-10 features
│   └── model.py    # SVM / RF / XGBoost + sensitivity/specificity
├── train.py        # CLI entry point
├── results/        # Auto-created: ROC + confusion matrix
└── requirements.txt
```

---

## 📈 Key Metrics Explained

- **Sensitivity** (Recall for ASD=YES): % of actual ASD cases correctly identified → minimise false negatives
- **Specificity** (Recall for ASD=NO): % of actual non-ASD cases correctly identified → minimise false positives

In medical screening, **high sensitivity is critical** to avoid missing true cases.
