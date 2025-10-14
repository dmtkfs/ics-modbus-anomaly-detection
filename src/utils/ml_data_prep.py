from __future__ import annotations
import yaml
import pandas as pd
from src.utils.data_prep import encode_labels


def load_ml_config(path="configs/ml.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_supervised_xy(df: pd.DataFrame, ml_cfg: dict):
    X = df[ml_cfg["features"]].copy()
    # Convert to numeric if needed
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    if X.isna().any().any():
        before = len(X)
        mask = ~X.isna().any(axis=1)
        X = X[mask]
        print(f"[ml_data_prep] Dropped {before - len(X)} rows due to NaNs in features.")

    y = encode_labels(df, ml_cfg["label_map"])
    return X, y
