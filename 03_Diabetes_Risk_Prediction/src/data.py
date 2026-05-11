"""
data.py — Data loading & preprocessing for Diabetes Risk Prediction.
Dataset: PIMA Indians Diabetes (Kaggle / UCI)
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


COLUMNS = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
           "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]

# PIMA dataset: these columns should NOT be 0 (medically impossible)
ZERO_INVALID = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]


def load_data(path: str = "diabetes.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Outcome" not in df.columns and "diabetes" in df.columns:
        df.rename(columns={"diabetes": "Outcome"}, inplace=True)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean & engineer PIMA diabetes features.

    Steps
    -----
    - Replace biologically impossible zeros with NaN, impute with median
    - BMI category feature
    - Age group feature
    - Glucose × Insulin interaction term
    - Log-transform skewed features (Insulin, SkinThickness)
    """
    df = df.copy()

    # Replace 0 → NaN in medically invalid columns
    for col in ZERO_INVALID:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # Impute with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Feature engineering
    df["bmi_category"] = pd.cut(df["BMI"],
                                 bins=[0, 18.5, 25, 30, 100],
                                 labels=["underweight","normal","overweight","obese"])
    df["age_group"]    = pd.cut(df["Age"],
                                 bins=[0, 30, 45, 60, 100],
                                 labels=["young","middle","senior","elderly"])
    df["glucose_bmi"]  = df["Glucose"] * df["BMI"]
    df["log_insulin"]  = np.log1p(df["Insulin"])

    # Encode categoricals
    df = pd.get_dummies(df, columns=["bmi_category", "age_group"], drop_first=True)

    y = df["Outcome"].astype(int)
    X = df.drop(columns=["Outcome"])
    return X, y


def get_splits(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns)
    return X_train_s, X_test_s, y_train.reset_index(drop=True), y_test.reset_index(drop=True), scaler
