"""
train.py — CLI entry point for Customer Churn Prediction.

Usage
-----
# Default (uses best available model)
python train.py

# Specify model and data path
python train.py --model xgboost --data ../WA_Fn-UseC_-Telco-Customer-Churn.csv

# Compare all models with CV
python train.py --compare
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# allow running from project root
sys.path.insert(0, ".")

from src.data import load_data, preprocess, get_splits
from src.model import get_models, cross_validate_all, train_best, evaluate
from src.evaluate import (
    plot_roc, plot_pr, plot_confusion, plot_feature_importance, save_cv_table
)


def parse_args():
    p = argparse.ArgumentParser(description="Customer Churn Predictor",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default="../WA_Fn-UseC_-Telco-Customer-Churn.csv",
                   help="Path to the Telco CSV file")
    p.add_argument("--model", default=None,
                   help="Model key: logistic_regression | random_forest | gradient_boosting | xgboost | lightgbm")
    p.add_argument("--compare", action="store_true",
                   help="Run cross-validation comparison across all models")
    p.add_argument("--cv", type=int, default=5, help="K-fold CV splits")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("Loading data from: %s", args.data)
    df = load_data(args.data)
    logger.info("  Shape: %s | Churn rate: %.1f%%",
                df.shape, df["Churn"].value_counts(normalize=True).get("Yes", 0) * 100)

    X, y = preprocess(df)
    X_train, X_test, y_train, y_test, _ = get_splits(X, y, args.test_size, args.seed)
    logger.info("  Train: %d | Test: %d | Features: %d", len(X_train), len(X_test), X.shape[1])

    models = get_models(args.seed)

    # ── Optional: cross-validate all models ────────────────────────────
    if args.compare:
        logger.info("Running %d-fold CV comparison across %d models …", args.cv, len(models))
        cv_df = cross_validate_all(models, X_train, y_train, cv=args.cv)
        print("\n" + cv_df.to_string(index=False))
        save_cv_table(cv_df)

    # ── Train best model ───────────────────────────────────────────────
    best_name, best_model = train_best(models, X_train, y_train, best_name=args.model)
    logger.info("Trained: %s", best_name)

    # ── Evaluate ───────────────────────────────────────────────────────
    metrics = evaluate(best_model, X_test, y_test)
    print(f"\n{'='*50}")
    print(f"  Model   : {best_name}")
    print(f"  ROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"  F1      : {metrics['f1']:.4f}")
    print(f"{'='*50}\n")
    print(metrics["report"])

    # ── Save plots ─────────────────────────────────────────────────────
    logger.info("Saving result plots to results/ …")
    plot_roc(y_test, metrics["y_prob"], best_name)
    plot_pr(y_test, metrics["y_prob"], best_name)
    plot_confusion(metrics["conf_mat"], best_name)
    plot_feature_importance(best_model, list(X.columns))
    logger.info("Done ✓")


if __name__ == "__main__":
    main()
