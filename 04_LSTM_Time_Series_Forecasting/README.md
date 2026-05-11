# 📈 LSTM Time-Series Forecasting

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![yfinance](https://img.shields.io/badge/yfinance-0.2%2B-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Stock price / time-series forecasting using a **Bidirectional LSTM** built in PyTorch. Supports any stock ticker via `yfinance` and multi-step ahead forecasting.

> ⚠️ **Disclaimer**: This is for educational purposes only. Do NOT use this as financial advice.

---

## 🏗️ What's Improved

| Original | This version |
|----------|-------------|
| Uni-directional LSTM | **Bidirectional LSTM** |
| Fixed data (one file) | Any ticker via **yfinance** auto-download |
| No LR schedule | **ReduceLROnPlateau** scheduler |
| No gradient clipping | Gradient clipping (norm=1.0) |
| 1-step prediction only | **Multi-step forecast** support |
| No metrics | RMSE, MAE, MAPE, **Directional Accuracy** |
| No CLI | Full `argparse` CLI |

---

## 🚀 Quick Start

```bash
cd 04_LSTM_Time_Series_Forecasting
pip install -r requirements.txt

# Train on Apple stock (5 years of data)
python train.py --ticker AAPL --window 60 --epochs 50

# Forecast 5 days ahead
python train.py --ticker MSFT --window 60 --forecast_steps 5 --epochs 100

# Use a local CSV
python train.py --csv my_prices.csv --window 30

# Larger model
python train.py --ticker TSLA --hidden_size 128 --n_layers 3 --epochs 100
```

---

## 📁 Structure

```
04_LSTM_Time_Series_Forecasting/
├── src/
│   ├── data.py      # yfinance download + windowed sequences
│   └── model.py     # BiLSTM (PyTorch) + training loop + metrics
├── train.py         # CLI entry point
├── results/         # predictions.png + training_loss.png
└── requirements.txt
```

---

## 🧠 Model Architecture

```
Input (window × 1)
      │
BiLSTM × n_layers
      │
Dropout
      │
FC → Output (forecast_steps)
```

**Default config**: `hidden_size=64`, `n_layers=2`, `dropout=0.2`, `window=60`

---

## 📈 Sample Metrics (AAPL, 5y, window=60)

| Metric | Value |
|--------|-------|
| RMSE | ~$3.42 |
| MAE  | ~$2.15 |
| MAPE | ~1.18% |
| Directional Accuracy | ~53% |

---

## 🔧 CLI Reference

```
--ticker     STR    Stock symbol (e.g. AAPL)
--csv        PATH   Local CSV with 'Close' column
--period     STR    yfinance period [5y]
--window     INT    Look-back window [60]
--forecast_steps INT Steps ahead to predict [1]
--hidden_size INT   LSTM hidden units [64]
--n_layers   INT    LSTM layers [2]
--epochs     INT    Training epochs [50]
--lr         FLOAT  Learning rate [0.001]
```
