"""
model.py — Model training for Autism Spectrum Detection.

Uses Sensitivity & Specificity as primary metrics (medical context).
Models: SVM, Random Forest, XGBoost with GridSearchCV.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score
)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def get_models(random_state: int = 42) -> dict:
    models: dict = {
        "svm": SVC(probability=True, kernel="rbf", C=1.0, gamma="scale",
                   class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced",
            random_state=random_state, n_jobs=-1
        ),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            scale_pos_weight=1, use_label_encoder=False,
            eval_metric="logloss", random_state=random_state
        )
    return models


def sensitivity_specificity(y_true, y_pred) -> tuple[float, float]:
    """Return (sensitivity, specificity) from binary predictions."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def cross_validate_all(models: dict, X, y, cv: int = 5) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []
    for name, model in models.items():
        auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        f1  = cross_val_score(model, X, y, cv=skf, scoring="f1",      n_jobs=-1)
        results.append({
            "Model":    name,
            "ROC-AUC":  f"{auc.mean():.4f} ± {auc.std():.4f}",
            "F1":       f"{f1.mean():.4f} ± {f1.std():.4f}",
            "_auc":     auc.mean(),
        })
    return pd.DataFrame(results).sort_values("_auc", ascending=False).drop(columns=["_auc"])


def train_best(models: dict, X_train, y_train, best_name: str | None = None):
    if best_name is None:
        for pref in ["xgboost", "random_forest", "svm"]:
            if pref in models:
                best_name = pref
                break
    model = models[best_name]
    model.fit(X_train, y_train)
    return best_name, model


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    sens, spec = sensitivity_specificity(y_test, y_pred)
    return {
        "roc_auc":     roc_auc_score(y_test, y_prob),
        "f1":          f1_score(y_test, y_pred),
        "sensitivity": sens,
        "specificity": spec,
        "report":      classification_report(y_test, y_pred, target_names=["No ASD", "ASD"]),
        "conf_mat":    confusion_matrix(y_test, y_pred),
        "y_prob":      y_prob,
        "y_pred":      y_pred,
    }
