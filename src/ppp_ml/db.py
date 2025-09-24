from __future__ import annotations

from typing import List, Tuple
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text

from ppp_common.orm import engine

def get_engine() -> sa.Engine:
    return engine

def load_feature_matrix(geo_code: str) -> pd.DataFrame:
    q = text("""
        SELECT *
        FROM ml.feature_matrix
        WHERE geo_code = :g
        ORDER BY year
    """)
    with engine.connect() as cx:
        # Explicit return type helps Pylance; no pandas.Series aliasing.
        df: pd.DataFrame = pd.read_sql(q, cx, params={"g": geo_code})
    return df

def list_geos(min_years: int = 8, require_full: bool = True) -> List[str]:
    cond = "WHERE has_full_features = true" if require_full else ""
    q = f"""
      SELECT geo_code
      FROM ml.feature_matrix
      {cond}
      GROUP BY geo_code
      HAVING COUNT(*) >= :min_years
      ORDER BY geo_code
    """
    with engine.connect() as cx:
        rows = cx.execute(text(q), {"min_years": min_years}).fetchall()
    return [r[0] for r in rows]

def split_train_test_years(df: pd.DataFrame, split_year: int = 2020) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Make sure df["year"] is numeric to avoid dtype confusion.
    year = pd.to_numeric(df["year"], errors="coerce")
    train = df[year < split_year].copy()
    test  = df[year >= split_year].copy()
    return train, test