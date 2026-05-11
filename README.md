# 🤖 ML Projects

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![sklearn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A collection of end-to-end Machine Learning projects — built while learning ML, structured like a professional portfolio.**

</div>

---

## 📂 Projects

| # | Project | Task | Models | Highlights |
|---|---------|------|--------|-----------|
| [01](./01_Customer_Churn_Prediction/) | **Customer Churn Prediction** | Binary Classification | LR, RF, XGBoost, LightGBM | 5-fold CV, ROC/PR plots, feature importance |
| [02](./02_Autism_Spectrum_Detection/) | **Autism Spectrum Detection** | Binary Classification | SVM, RF, XGBoost | Sensitivity & Specificity, medical focus |
| [03](./03_Diabetes_Risk_Prediction/) | **Diabetes Risk Prediction** | Binary Classification | LR, RF, MLP, XGBoost | Zero-imputation, threshold tuning, neural net |
| [04](./04_LSTM_Time_Series_Forecasting/) | **LSTM Time-Series Forecasting** | Regression / Forecasting | BiLSTM (PyTorch) | yfinance auto-download, multi-step forecast, RMSE/MAPE |

---

## 🏗️ Repository Structure

```
ML-Projects/
│
├── 01_Customer_Churn_Prediction/
│   ├── src/           # data.py, model.py, evaluate.py
│   ├── train.py       # python train.py [--compare] [--model xgboost]
│   ├── requirements.txt
│   └── README.md
│
├── 02_Autism_Spectrum_Detection/
│   ├── src/           # data.py, model.py
│   ├── train.py       # python train.py --data autism_data.csv
│   ├── requirements.txt
│   └── README.md
│
├── 03_Diabetes_Risk_Prediction/
│   ├── src/           # data.py, model.py
│   ├── train.py       # python train.py --data diabetes.csv [--tune_threshold]
│   ├── requirements.txt
│   └── README.md
│
├── 04_LSTM_Time_Series_Forecasting/
│   ├── src/           # data.py, model.py (BiLSTM)
│   ├── train.py       # python train.py --ticker AAPL --window 60
│   ├── requirements.txt
│   └── README.md
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← Dataset for Project 01
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start (any project)

```bash
# 1. Clone
git clone https://github.com/shivanshu1512/ML-Projects.git
cd ML-Projects

# 2. Pick a project
cd 01_Customer_Churn_Prediction

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train
python train.py
```

---

## 📈 What Each Project Demonstrates

### 01 — Customer Churn Prediction
- **Feature engineering**: tenure buckets, avg monthly spend
- **Multiple models** compared with Stratified K-Fold CV
- **Visualisations**: ROC curve, PR curve, confusion matrix, feature importance

### 02 — Autism Spectrum Detection
- **Medical ML focus**: sensitivity vs specificity trade-off
- **AQ-10 score** engineered as a strong domain feature
- SVM with RBF kernel + class balancing

### 03 — Diabetes Risk Prediction
- **Zero-value imputation** (Glucose, BMI = 0 is biologically impossible)
- **MLP (sklearn)** neural network alongside tree models
- **Optimal threshold tuning** via Precision-Recall curve

### 04 — LSTM Time-Series Forecasting
- **Bidirectional LSTM** in PyTorch (better than vanilla LSTM)
- **yfinance** integration — train on any stock in seconds
- **Multi-step forecasting** (predict N days ahead)
- Metrics: RMSE, MAE, MAPE, Directional Accuracy

---

## 🛠️ Technologies

| Library | Usage |
|---------|-------|
| `scikit-learn` | Classical ML models, CV, preprocessing |
| `xgboost` / `lightgbm` | Gradient boosting models |
| `PyTorch` | BiLSTM architecture |
| `yfinance` | Stock price data download |
| `matplotlib` / `seaborn` | All visualisations |
| `pandas` / `numpy` | Data wrangling |

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ❤️ by <a href="https://github.com/shivanshu1512">shivanshu1512</a> while learning Machine Learning
</div>
