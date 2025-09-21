from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class SplitResult:
    split_year: int
    mae: float
    rmse: float


def expanding_years(df: pd.DataFrame, first_test_year: int, year_col: str = "year") -> List[int]:
    years = sorted(df[year_col].dropna().unique().tolist())
    return [y for y in years if y >= first_test_year]


def rolling_backtest(
    df: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    year_col: str,
    fit_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], object],
    predict_fn: Callable[[object, NDArray[np.float64]], NDArray[np.float64]],
    first_test_year: int,
) -> list[SplitResult]:
    out: list[SplitResult] = []

    for y in expanding_years(df, first_test_year, year_col):
        train = df[df[year_col] < y].copy()
        test  = df[df[year_col] == y].copy()
        if train.empty or test.empty:
            continue

        # keep pandas for selection; convert to NumPy right before model calls
        Xtr_df = train.loc[:, feat_cols].astype("float64").dropna()
        ytr_df = train.loc[Xtr_df.index, target_col].astype("float64")
        Xte_df = test.loc[:, feat_cols].astype("float64").dropna()
        yte_df = test.loc[Xte_df.index, target_col].astype("float64")

        Xtr: NDArray[np.float64] = Xtr_df.to_numpy()
        ytr: NDArray[np.float64] = ytr_df.to_numpy()
        Xte: NDArray[np.float64] = Xte_df.to_numpy()
        yte: NDArray[np.float64] = yte_df.to_numpy()

        m = fit_fn(Xtr, ytr)
        yp: NDArray[np.float64] = predict_fn(m, Xte)

        mae  = float(mean_absolute_error(yte, yp))
        rmse = float(mean_squared_error(yte, yp) ** 0.5)
        out.append(SplitResult(split_year=y, mae=mae, rmse=rmse))

    return out