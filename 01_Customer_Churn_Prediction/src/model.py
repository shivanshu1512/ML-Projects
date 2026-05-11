"""
model.py — Model training & comparison for Customer Churn Prediction.

Models: Logistic Regression, Random Forest, XGBoost, LightGBM
Uses StratifiedKFold cross-validation for reliable estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report, confusion_matrix
)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


def get_models(random_state: int = 42) -> dict:
    """Return dict of all candidate models."""
    models: dict = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=random_state, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            random_state=random_state
        ),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            use_label_encoder=False, eval_metric="logloss",
            random_state=random_state
        )
    if _HAS_LGB:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            random_state=random_state, verbose=-1
        )
    return models


def cross_validate_all(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> pd.DataFrame:
    """Run stratified K-fold CV on every model. Returns a comparison DataFrame."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []
    for name, model in models.items():
        auc   = cross_val_score(model, X, y, cv=skf, scoring="roc_auc",  n_jobs=-1)
        f1    = cross_val_score(model, X, y, cv=skf, scoring="f1",       n_jobs=-1)
        prec  = cross_val_score(model, X, y, cv=skf, scoring="precision", n_jobs=-1)
        rec   = cross_val_score(model, X, y, cv=skf, scoring="recall",    n_jobs=-1)
        results.append({
            "Model":     name,
            "ROC-AUC":   f"{auc.mean():.4f} ± {auc.std():.4f}",
            "F1":        f"{f1.mean():.4f} ± {f1.std():.4f}",
            "Precision": f"{prec.mean():.4f} ± {prec.std():.4f}",
            "Recall":    f"{rec.mean():.4f} ± {rec.std():.4f}",
            "_auc_mean": auc.mean(),
        })
    df = pd.DataFrame(results).sort_values("_auc_mean", ascending=False).drop(
        columns=["_auc_mean"]
    )
    return df


def train_best(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_name: str | None = None,
) -> tuple:
    """Train the best model on full training data. Returns (name, fitted_model)."""
    if best_name is None:
        # Default: prefer XGBoost → LightGBM → GBM
        for preferred in ["xgboost", "lightgbm", "gradient_boosting"]:
            if preferred in models:
                best_name = preferred
                break
    model = models[best_name]
    model.fit(X_train, y_train)
    return best_name, model


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a fitted model on the test set."""
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc":  roc_auc_score(y_test, y_prob),
        "f1":       f1_score(y_test, y_pred),
        "report":   classification_report(y_test, y_pred),
        "conf_mat": confusion_matrix(y_test, y_pred),
        "y_prob":   y_prob,
        "y_pred":   y_pred,
    }
