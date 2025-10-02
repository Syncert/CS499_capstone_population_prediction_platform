# src/ppp_ml/training_io.py
from __future__ import annotations
import pandas as pd
from sqlalchemy import text
from ppp_common.orm import engine
from sklearn.metrics import r2_score


AGG_FOLD = -1  # aggregate metric (non-CV)

# Columns we never want as features
EXCLUDE_COLS = (
    "year",
    "population",
    "geo_code",
    "pop_ma3",
    "pop_yoy_growth_pct",
    "pop_cagr_5yr_pct",
    "has_full_features"
)

def select_numeric_features(df: pd.DataFrame, exclude: tuple[str, ...] = EXCLUDE_COLS) -> list[str]:
    """
    Return only numeric feature columns, excluding identifiers and target.
    """
    feats: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    return feats

def attach_actuals(geo: str, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure ds (date) and yhat exist; join actuals for evaluation where available.
    Accepts either a 'year' column (int) or 'ds' (date-like).
    """
    df = forecast_df.copy()
    if "ds" not in df.columns:
        if "year" not in df.columns:
            raise ValueError("forecast_df must include 'ds' or 'year'")
        df["ds"] = pd.to_datetime(df["year"].astype(str) + "-12-31")
    # pull actuals
    with engine.connect() as cx:
        act = pd.read_sql(text("""
            select year, population as actual
            from core.population_observations
            where geo_code = :g
        """), cx, params={"g": geo})
    df["year"] = pd.to_datetime(df["ds"]).dt.year
    out = df.merge(act, on="year", how="left")
    # Final column order
    cols = ["ds", "yhat"] + [c for c in ["yhat_lo","yhat_hi","actual"] if c in out.columns]
    return out[cols]

def basic_test_metrics(forecast_df: pd.DataFrame) -> list[tuple[str,str,int,float]]:
    df = forecast_df.dropna(subset=["actual"])
    if df.empty:
        return []
    err = df["yhat"] - df["actual"]

    mae  = float(err.abs().mean())
    rmse = float((err**2).mean() ** 0.5)
    r2   = float(r2_score(df["actual"], df["yhat"]))

    metrics = [
        ("mae", "test", -1, mae),
        ("rmse","test", -1, rmse),
        ("r2",  "test", -1, r2),
    ]

    # optional MAPE
    if (df["actual"] != 0).any():
        mape = float((err.abs() / df["actual"].replace({0: pd.NA})).dropna().mean() * 100)
        metrics.append(("mape","test",-1,mape))

    return metrics