"""
data.py — Data loading & preprocessing for Autism Spectrum Detection.
Dataset: Autism Screening Adult / Child (AQ-10 questionnaire)
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


AUTISM_COLS = [
    "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
    "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
    "age", "gender", "ethnicity", "jaundice", "autism",
    "country_of_res", "used_app_before", "result", "age_desc",
    "relation", "Class/ASD"
]


def load_data(path: str) -> pd.DataFrame:
    """Load autism screening CSV (adult or child dataset)."""
    df = pd.read_csv(path, na_values=["?", "", " "])
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess autism screening data.

    Steps
    -----
    - Drop 'result' (leakage — it's the sum of AQ scores)
    - Drop 'used_app_before', 'age_desc' (low signal)
    - Fill missing ages with median
    - Encode binary categoricals (Yes/No, m/f)
    - One-hot remaining categoricals
    - Derive: aq10_total = sum of A1..A10 scores
    """
    df = df.copy()

    # Drop leaky / low-signal columns
    drop_cols = [c for c in ["result", "used_app_before", "age_desc"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # AQ-10 total score (strong signal)
    aq_cols = [c for c in df.columns if c.startswith("A") and "_Score" in c]
    if aq_cols:
        df["aq10_total"] = df[aq_cols].sum(axis=1)

    # Fill missing age
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["age"].fillna(df["age"].median(), inplace=True)
        df["age_group"] = pd.cut(df["age"], bins=[0, 12, 18, 40, 100],
                                  labels=["child", "teen", "adult", "senior"])

    # Target
    target_col = next((c for c in df.columns if "Class" in c or "ASD" in c), None)
    if target_col is None:
        raise ValueError("Cannot find target column (Class/ASD) in dataset")
    y = (df[target_col].str.strip().str.upper() == "YES").astype(int)
    df.drop(columns=[target_col], inplace=True)

    # Binary columns → 0/1
    for col in ["gender", "jaundice", "autism"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower().map(
                {"yes": 1, "no": 0, "m": 1, "f": 0, "male": 1, "female": 0}
            ).fillna(0).astype(int)

    # One-hot remaining objects
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols + (["age_group"] if "age_group" in df.columns else []),
                        drop_first=True)
    df.fillna(0, inplace=True)

    return df, y


def get_splits(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns)
    return X_train_s, X_test_s, y_train.reset_index(drop=True), y_test.reset_index(drop=True), scaler
