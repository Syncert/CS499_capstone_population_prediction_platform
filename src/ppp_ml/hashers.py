# src/ppp_ml/hashers.py
from __future__ import annotations
import hashlib
import pandas as pd

# Columns that influence training. Add/remove as your features evolve.
HASH_COLS = [
    "year",
    "population",
    "pop_lag1", "pop_lag5", "pop_ma3",
    "pop_yoy_growth_pct", "pop_cagr_5yr_pct",
    "unemployment_rate",
    "rent_cpi_index",
]

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize order + types so the same data => same bytes.
    - Keep only HASH_COLS that exist.
    - Sort by year (if present), else by all cols.
    - Cast to string with stable NA marker.
    """
    cols = [c for c in HASH_COLS if c in df.columns]
    key_df = df[cols].copy()

    if "year" in key_df.columns:
        key_df = key_df.sort_values("year")
    else:
        key_df = key_df.sort_values(by=cols)

    # Fill NA with a stable token, cast to string
    key_df = key_df.fillna("__NA__").astype(str)
    return key_df

def dataframe_hash(df: pd.DataFrame) -> str:
    """
    Stable md5 digest over the normalized feature slice.
    """
    if df is None or df.empty:
        return hashlib.md5(b"EMPTY").hexdigest()

    key_df = _normalize(df)
    # Join row-wise with separators to avoid accidental collisions
    # Example row: "2020|512341|511000|...|3.2|__NA__"
    payload_rows = ["|".join(row) for row in key_df.to_numpy().tolist()]
    payload = ("\n".join(payload_rows)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()
