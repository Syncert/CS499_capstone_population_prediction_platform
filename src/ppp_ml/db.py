from __future__ import annotations
from typing import Tuple
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text

# Your shared engine (already configured to build the DB URL properly)
from ppp_common.orm import engine  # type: ignore

def get_engine() -> sa.Engine:
    return engine

def load_population_timeseries(geo_code: str = "US") -> pd.DataFrame:
    """
    Returns columns: ['geo_code','year','population'] from core.population_observations.
    """
    q = text("""
        SELECT geo_code, year, population
        FROM core.population_observations
        WHERE geo_code = :g
        ORDER BY year
    """)
    with get_engine().connect() as cx:
        df = pd.read_sql(q, cx, params={"g": geo_code})
    return df

def split_train_test_years(df: pd.DataFrame, split_year: int = 2020) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] < split_year].copy()
    test  = df[df["year"] >= split_year].copy()
    return train, test
