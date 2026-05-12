"""
data.py — Data loading & feature engineering for Customer Churn Prediction.
Dataset: Telco Customer Churn (IBM Watson Analytics)
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    """Load the raw Telco churn CSV."""
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean and engineer features.

    Steps
    -----
    - Drop customerID (not predictive)
    - Fix TotalCharges (coerce whitespace → NaN → fill with 0)
    - Binary encode Yes/No columns
    - One-hot encode multi-class categoricals
    - Engineer: tenure_bucket, avg_monthly_spend
    """
    df = df.copy()

    # Drop non-predictive column
    df.drop(columns=["customerID"], inplace=True)

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Binary columns → 0/1
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService",
                   "PaperlessBilling", "Churn"]
    for col in binary_cols:
        df[col] = (df[col].str.strip().str.lower().isin(["yes", "male", "1"])).astype(int)

    # Feature engineering
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=[0,12,24,48,72],
                                  labels=["0-1yr","1-2yr","2-4yr","4-6yr"])
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # One-hot encode remaining categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols + ["tenure_bucket"], drop_first=True)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def get_splits(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split + StandardScaler."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns)
    return X_train_s, X_test_s, y_train, y_test, scaler
