from __future__ import annotations
from typing import Tuple, List, Optional
import pandas as pd
from sqlalchemy import text
import sqlalchemy as sa

# single source of engine
from ppp_common.orm import engine  # type: ignore

def get_engine() -> sa.Engine:
    return engine

#feature matrix has to be on a per geo_code basis
def load_feature_matrix(geo_code: str) -> pd.DataFrame:
    q = text("""
        SELECT *
        FROM ml.feature_matrix
        WHERE geo_code = :g
        ORDER BY year
    """)
    with engine.connect() as cx:
        return pd.read_sql(q, cx, params={"g": geo_code})

#helper for listing every geocode sequentially that has minimum amount of rows
def list_geos(min_years: int = 8, require_full: bool = True) -> List[str]:
    """
    Return geo_codes with enough rows; optionally require has_full_features=true.
    """
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
    train = df[df["year"] < split_year].copy()
    test  = df[df["year"] >= split_year].copy()
    return train, test