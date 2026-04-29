"""
schema.py — Auto-detect column types and data profile
"""

import pandas as pd
import numpy as np


def detect_schema(df: pd.DataFrame) -> dict:
    """
    Returns a structured schema dict with column types, nulls, uniques.
    """
    numeric = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    boolean_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    # Try to parse object cols that look like dates
    for col in categorical[:]:
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            if parsed.notna().sum() / len(df) > 0.7:
                datetime_cols.append(col)
                categorical.remove(col)
        except Exception:
            pass

    profile = {}
    for col in df.columns:
        profile[col] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(df[col].isna().mean() * 100, 2),
            "unique_count": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(5).tolist(),
        }
        if col in numeric:
            profile[col].update({
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
            })

    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime_cols,
        "boolean": boolean_cols,
        "profile": profile,
        "shape": df.shape,
        "columns": df.columns.tolist(),
    }


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Extended describe with skew and kurtosis for numeric cols."""
    num_df = df.select_dtypes(include=["int64", "float64"])
    if num_df.empty:
        return pd.DataFrame()
    desc = num_df.describe().T
    desc["skew"] = num_df.skew()
    desc["kurtosis"] = num_df.kurtosis()
    desc["missing_%"] = (num_df.isna().mean() * 100).round(2)
    return desc.round(3)
