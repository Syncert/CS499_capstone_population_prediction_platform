from __future__ import annotations
from typing import Tuple
import pandas as pd
from sqlalchemy import text
import sqlalchemy as sa

# single source of engine
from ppp_common.orm import engine  # type: ignore

def get_engine() -> sa.Engine:
    return engine

def load_feature_matrix(geo_code: str = "US") -> pd.DataFrame:
    """Fetch rows from ml.feature_matrix for one geo ordered by year."""
    q = text("""
        SELECT *
        FROM ml.feature_matrix
        WHERE geo_code = :g
        ORDER BY year
    """)
    with get_engine().connect() as cx:
        return pd.read_sql(q, cx, params={"g": geo_code})

def split_train_test_years(df: pd.DataFrame, split_year: int = 2020) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] < split_year].copy()
    test  = df[df["year"] >= split_year].copy()
    return train, test