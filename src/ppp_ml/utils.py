from __future__ import annotations
import csv
import os
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def artifact_dir() -> Path:
    p = Path(os.getenv("PPP_ARTIFACTS_DIR", "models"))
    p.mkdir(parents=True, exist_ok=True)
    return p

def append_metrics_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "mae", "mse", "rmse", "notes"])
        if write_header:
            w.writeheader()
        w.writerow(row)

def select_sane_features(df: pd.DataFrame, feat_cols: list[str], min_non_null: int = 3) -> list[str]:
    """
    Keep only features that have at least `min_non_null` non-null values.
    This prevents one totally-missing regressor from nuking the whole frame.
    """
    good: list[str] = []
    for c in feat_cols:
        if c in df.columns and df[c].notna().sum() >= min_non_null:
            good.append(c)
    return good

def build_train_Xy(
    df: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    1) Restrict to rows with at least lag-1 present (if available),
    2) keep features with some data,
    3) cast to float64,
    4) median-impute remaining NaNs (column-wise),
    5) return X_df, y_ser, used_feats.
    """
    # Prefer keeping rows with pop_lag1 (when present)
    if "pop_lag1" in df.columns:
        df = df[df["pop_lag1"].notna()].copy()
    # If that makes it empty, fall back to the raw df
    if df.empty:
        df = df.copy()

    used_feats = select_sane_features(df, feat_cols, min_non_null=3)
    if not used_feats:
        # As a last resort, try with the original list — caller can handle empty later
        used_feats = feat_cols[:]

    X_df = df.loc[:, [c for c in used_feats if c in df.columns]].astype("float64")
    y_ser = df[target_col].astype("float64")

    # Column-wise median impute for any residual gaps
    med = X_df.median(numeric_only=True)
    X_df = X_df.fillna(med)

    # Drop any rows that are still fully NaN in y
    mask = y_ser.notna()
    X_df = X_df.loc[mask]
    y_ser = y_ser.loc[mask]

    return X_df, y_ser, used_feats


def regression_metrics(y_true: NDArray, y_pred: NDArray) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mse ** 0.5)
    r2 = float(r2_score(y_true, y_pred))
    # guard against division by ~0 in % errors on level data
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}