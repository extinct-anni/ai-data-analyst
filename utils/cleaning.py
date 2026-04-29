"""
cleaning.py — Automated data cleaning: missing values, outliers, duplicates
"""

import pandas as pd
import numpy as np
from typing import Tuple


# ─── Missing Values ────────────────────────────────────────────────────────────

def get_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted DataFrame showing missing value counts and percentages."""
    missing = df.isna().sum()
    pct = (df.isna().mean() * 100).round(2)
    report = pd.DataFrame({"Missing Count": missing, "Missing %": pct})
    return report[report["Missing Count"] > 0].sort_values("Missing %", ascending=False)


def fill_missing(df: pd.DataFrame, strategy: str = "mean") -> Tuple[pd.DataFrame, list]:
    """
    Fill missing values.
    strategy: 'mean', 'median', 'mode', 'drop'
    Returns (cleaned_df, list_of_actions_taken)
    """
    df = df.copy()
    actions = []

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    if strategy == "drop":
        before = len(df)
        df.dropna(inplace=True)
        actions.append(f"Dropped {before - len(df)} rows with missing values.")
        return df, actions

    for col in num_cols:
        if df[col].isna().sum() == 0:
            continue
        count = df[col].isna().sum()
        if strategy == "mean":
            df[col].fillna(df[col].mean(), inplace=True)
            actions.append(f"[{col}] Filled {count} NaNs with mean ({df[col].mean():.2f})")
        elif strategy == "median":
            df[col].fillna(df[col].median(), inplace=True)
            actions.append(f"[{col}] Filled {count} NaNs with median ({df[col].median():.2f})")
        elif strategy == "mode":
            df[col].fillna(df[col].mode()[0], inplace=True)
            actions.append(f"[{col}] Filled {count} NaNs with mode")

    for col in cat_cols:
        if df[col].isna().sum() == 0:
            continue
        count = df[col].isna().sum()
        fill_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
        df[col].fillna(fill_val, inplace=True)
        actions.append(f"[{col}] Filled {count} NaNs with mode ('{fill_val}')")

    return df, actions


# ─── Outlier Detection ─────────────────────────────────────────────────────────

def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.Series:
    """Return boolean mask of outlier rows using IQR method."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (df[col] < lower) | (df[col] > upper)


def detect_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.Series:
    """Return boolean mask of outlier rows using Z-score method."""
    z = (df[col] - df[col].mean()) / df[col].std()
    return z.abs() > threshold


def get_outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return outlier counts for all numeric columns using IQR."""
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    rows = []
    for col in num_cols:
        mask = detect_outliers_iqr(df, col)
        count = mask.sum()
        if count > 0:
            rows.append({
                "Column": col,
                "Outlier Count": int(count),
                "Outlier %": round(count / len(df) * 100, 2),
                "Method": "IQR",
            })
    return pd.DataFrame(rows)


def remove_outliers(df: pd.DataFrame, col: str, method: str = "iqr") -> Tuple[pd.DataFrame, int]:
    """Remove outliers from a column. Returns (cleaned_df, rows_removed)."""
    before = len(df)
    if method == "iqr":
        mask = detect_outliers_iqr(df, col)
    else:
        mask = detect_outliers_zscore(df, col)
    df = df[~mask].copy()
    return df, before - len(df)


# ─── Duplicates ────────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate rows. Returns (cleaned_df, count_removed)."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)


# ─── Data Type Fixes ───────────────────────────────────────────────────────────

def fix_dtypes(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Try to auto-cast columns to better types."""
    df = df.copy()
    actions = []
    for col in df.select_dtypes(include=["object"]).columns:
        # Try numeric
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() / len(df) > 0.9:
            df[col] = converted
            actions.append(f"[{col}] Cast to numeric")
            continue
        # Try datetime
        try:
            converted_dt = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            if converted_dt.notna().sum() / len(df) > 0.7:
                df[col] = converted_dt
                actions.append(f"[{col}] Cast to datetime")
        except Exception:
            pass
    return df, actions
