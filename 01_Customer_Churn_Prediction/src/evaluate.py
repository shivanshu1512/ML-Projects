"""
evaluate.py — Evaluation plots for Customer Churn Prediction.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, ConfusionMatrixDisplay


RESULTS_DIR = Path("results")


def _ensure_dir():
    RESULTS_DIR.mkdir(exist_ok=True)


def plot_roc(y_test, y_prob, model_name: str = "model") -> None:
    _ensure_dir()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'roc_curve.png'}")


def plot_pr(y_test, y_prob, model_name: str = "model") -> None:
    _ensure_dir()
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, lw=2, color="darkorange")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve — {model_name}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pr_curve.png", dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'pr_curve.png'}")


def plot_confusion(conf_mat, model_name: str = "model") -> None:
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(conf_mat, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'confusion_matrix.png'}")


def plot_feature_importance(model, feature_names: list[str], top_n: int = 20) -> None:
    _ensure_dir()
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(8, 6))
    plt.barh([feature_names[i] for i in idx], importances[idx], color="steelblue")
    plt.xlabel("Importance")
    plt.title(f"Top-{top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'feature_importance.png'}")


def save_cv_table(df: pd.DataFrame) -> None:
    _ensure_dir()
    df.to_csv(RESULTS_DIR / "cv_comparison.csv", index=False)
    print(f"  Saved: {RESULTS_DIR / 'cv_comparison.csv'}")
