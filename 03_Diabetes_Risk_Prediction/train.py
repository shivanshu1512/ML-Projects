"""
train.py — CLI entry point for Diabetes Risk Prediction.

Usage
-----
python train.py --data diabetes.csv
python train.py --data diabetes.csv --model xgboost --compare --tune_threshold
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
sys.path.insert(0, ".")

from src.data import load_data, preprocess, get_splits
from src.model import get_models, cross_validate_all, train_best, evaluate, find_best_threshold


def parse_args():
    p = argparse.ArgumentParser(description="Diabetes Risk Predictor",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default="diabetes.csv", help="Path to diabetes CSV")
    p.add_argument("--model", default=None, help="Model key")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--tune_threshold", action="store_true", help="Find optimal decision threshold")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def save_plots(metrics, y_test, model_name):
    out = Path("results")
    out.mkdir(exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, metrics["y_prob"])
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(4,3))
    ConfusionMatrixDisplay(metrics["conf_mat"],
                           display_labels=["No Diabetes","Diabetes"]).plot(ax=ax, colorbar=False, cmap="Greens")
    ax.set_title(f"Confusion — {model_name}")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()
    logger.info("Plots saved to results/")


def main():
    args = parse_args()
    logger.info("Loading: %s", args.data)
    df = load_data(args.data)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test, _ = get_splits(X, y, random_state=args.seed)
    logger.info("Train: %d | Test: %d | Features: %d | Positive rate: %.1f%%",
                len(X_train), len(X_test), X.shape[1], y.mean()*100)

    models = get_models(args.seed)
    if args.compare:
        logger.info("Cross-validation comparison …")
        cv_df = cross_validate_all(models, X_train, y_train, args.cv)
        print("\n" + cv_df.to_string(index=False))

    name, model = train_best(models, X_train, y_train, args.model)
    logger.info("Trained: %s", name)

    # Threshold tuning
    threshold = 0.5
    if args.tune_threshold:
        y_prob_train = model.predict_proba(X_train)[:, 1]
        threshold = find_best_threshold(y_train, y_prob_train)
        logger.info("Optimal threshold (train F1): %.3f", threshold)

    metrics = evaluate(model, X_test, y_test, threshold=threshold)
    print(f"\n{'='*52}")
    print(f"  Model     : {name}")
    print(f"  Threshold : {threshold:.3f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"{'='*52}\n")
    print(metrics["report"])
    save_plots(metrics, y_test, name)


if __name__ == "__main__":
    main()
