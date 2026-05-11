"""
train.py — CLI entry point for LSTM Time-Series Forecasting.

Usage
-----
# Download AAPL and train
python train.py --ticker AAPL --window 60 --epochs 50

# Use a local CSV
python train.py --csv prices.csv --window 30 --forecast_steps 5
"""
from __future__ import annotations

import argparse, logging, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
sys.path.insert(0, ".")

from src.data import download_stock, load_csv, prepare_data
from src.model import build_model, train, predict, compute_metrics


def parse_args():
    p = argparse.ArgumentParser(description="LSTM Time-Series Forecaster",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ticker", help="Stock ticker symbol (e.g. AAPL, MSFT, TSLA)")
    g.add_argument("--csv",    help="Path to local CSV with 'Close' column")
    p.add_argument("--period",  default="5y", help="yfinance period (1y/2y/5y)")
    p.add_argument("--window",  type=int, default=60,  help="Look-back window (days)")
    p.add_argument("--forecast_steps", type=int, default=1, help="Days to forecast ahead")
    p.add_argument("--hidden_size",    type=int, default=64)
    p.add_argument("--n_layers",       type=int, default=2)
    p.add_argument("--dropout",        type=float, default=0.2)
    p.add_argument("--epochs",         type=int, default=50)
    p.add_argument("--batch_size",     type=int, default=32)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--no_bidirectional", action="store_true")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def save_plots(y_true, y_pred, scaler, ticker: str, losses: list[float]):
    out = Path("results"); out.mkdir(exist_ok=True)

    # Inverse transform
    y_true_p = scaler.inverse_transform(y_true.reshape(-1,1)).flatten()
    y_pred_p = scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()

    # Prediction vs actual
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_p, label="Actual", lw=1.5)
    plt.plot(y_pred_p, label="Predicted", lw=1.5, alpha=0.8)
    plt.title(f"LSTM Forecast — {ticker}"); plt.xlabel("Time step"); plt.ylabel("Price (USD)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out/"predictions.png", dpi=150); plt.close()

    # Training loss
    plt.figure(figsize=(7, 4))
    plt.plot(losses, lw=2, color="darkorange"); plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("Training Loss"); plt.tight_layout()
    plt.savefig(out/"training_loss.png", dpi=150); plt.close()
    logger.info("Plots saved to results/")


def main():
    args = parse_args()

    # Load data
    if args.ticker:
        logger.info("Downloading %s (%s) …", args.ticker, args.period)
        df = download_stock(args.ticker, args.period)
        label = args.ticker
    else:
        logger.info("Loading CSV: %s", args.csv)
        df = load_csv(args.csv)
        label = Path(args.csv).stem

    logger.info("Rows: %d | Date range: %s → %s",
                len(df), df.index[0].date(), df.index[-1].date())

    data = prepare_data(df, window=args.window, forecast_steps=args.forecast_steps)
    logger.info("Sequences — Train: %d | Test: %d", len(data["X_train"]), len(data["X_test"]))

    # Build model
    model = build_model(
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        forecast_steps=args.forecast_steps,
        bidirectional=not args.no_bidirectional,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("BiLSTM parameters: %d", total_params)

    # Train
    losses = train(model, data["X_train"], data["y_train"],
                   epochs=args.epochs, batch_size=args.batch_size,
                   lr=args.lr, device=args.device)

    # Evaluate
    y_pred = predict(model, data["X_test"], args.device)
    metrics = compute_metrics(data["y_test"], y_pred, data["scaler"])

    print(f"\n{'='*50}")
    print(f"  Ticker       : {label}")
    print(f"  Window       : {args.window} days")
    print(f"  Forecast     : {args.forecast_steps} step(s)")
    print(f"  RMSE         : ${metrics['RMSE']:.2f}")
    print(f"  MAE          : ${metrics['MAE']:.2f}")
    print(f"  MAPE         : {metrics['MAPE']:.2f}%")
    print(f"  Dir. Accuracy: {metrics['Dir_Acc']:.2%}")
    print(f"{'='*50}\n")

    save_plots(data["y_test"][:, 0], y_pred[:, 0], data["scaler"], label, losses)


if __name__ == "__main__":
    main()
