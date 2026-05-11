"""
data.py — Data loading & windowed dataset for LSTM Time-Series Forecasting.
Supports any stock ticker via yfinance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False


def download_stock(ticker: str = "AAPL", period: str = "5y") -> pd.DataFrame:
    """Download OHLCV data for *ticker* using yfinance."""
    if not _HAS_YF:
        raise ImportError("yfinance not installed. Run: pip install yfinance")
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")
    df.dropna(inplace=True)
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Load historical price CSV (must have 'Close' column)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def make_sequences(
    series: np.ndarray,
    window: int = 60,
    forecast_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create supervised (X, y) sequences from a 1-D time series.

    Parameters
    ----------
    series : np.ndarray  shape (N,)
    window : int         look-back window
    forecast_steps : int how many steps ahead to predict

    Returns
    -------
    X : (N - window - forecast_steps + 1, window, 1)
    y : (N - window - forecast_steps + 1, forecast_steps)
    """
    X, y = [], []
    for i in range(len(series) - window - forecast_steps + 1):
        X.append(series[i : i + window])
        y.append(series[i + window : i + window + forecast_steps])
    return np.array(X)[..., np.newaxis], np.array(y)


def prepare_data(
    df: pd.DataFrame,
    feature_col: str = "Close",
    window: int = 60,
    forecast_steps: int = 1,
    train_ratio: float = 0.8,
) -> dict:
    """Full pipeline: scale → sequence → train/test split."""
    prices = df[[feature_col]].values.astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices).flatten()

    X, y = make_sequences(scaled, window, forecast_steps)

    split = int(len(X) * train_ratio)
    return {
        "X_train": X[:split], "y_train": y[:split],
        "X_test":  X[split:], "y_test":  y[split:],
        "scaler":  scaler,
        "raw_prices": prices,
    }
