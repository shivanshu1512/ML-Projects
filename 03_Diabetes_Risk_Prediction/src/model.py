"""
model.py — Models for Diabetes Risk Prediction.
Includes: Logistic Regression, Random Forest, XGBoost, MLP (sklearn)
with threshold tuning and calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, precision_recall_curve
)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def get_models(random_state: int = 42) -> dict:
    models: dict = {
        "logistic_regression": LogisticRegression(max_iter=1000, C=0.5, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced",
            random_state=random_state, n_jobs=-1
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu",
            max_iter=500, random_state=random_state, early_stopping=True
        ),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            use_label_encoder=False, eval_metric="logloss",
            random_state=random_state
        )
    return models


def cross_validate_all(models: dict, X, y, cv: int = 5) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        f1  = cross_val_score(model, X, y, cv=skf, scoring="f1",      n_jobs=-1)
        rows.append({
            "Model":   name,
            "ROC-AUC": f"{auc.mean():.4f} ± {auc.std():.4f}",
            "F1":      f"{f1.mean():.4f} ± {f1.std():.4f}",
            "_auc":    auc.mean(),
        })
    return pd.DataFrame(rows).sort_values("_auc", ascending=False).drop(columns=["_auc"])


def find_best_threshold(y_true, y_prob) -> float:
    """Find threshold that maximises F1 score on the validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


def train_best(models: dict, X_train, y_train, best_name: str | None = None):
    if best_name is None:
        for pref in ["xgboost", "mlp", "random_forest"]:
            if pref in models:
                best_name = pref
                break
    model = models[best_name]
    model.fit(X_train, y_train)
    return best_name, model


def evaluate(model, X_test, y_test, threshold: float = 0.5) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "f1":        f1_score(y_test, y_pred),
        "threshold": threshold,
        "report":    classification_report(y_test, y_pred,
                                            target_names=["No Diabetes", "Diabetes"]),
        "conf_mat":  confusion_matrix(y_test, y_pred),
        "y_prob":    y_prob,
        "y_pred":    y_pred,
    }
