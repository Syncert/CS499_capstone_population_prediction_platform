from __future__ import annotations
from dataclasses import dataclass
import pickle
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from ppp_ml.utils import build_train_Xy


@dataclass
class XGBRunResult:
    model_path: str
    feats: list[str]
    mae: float
    rmse: float


def fit_xgb(X: NDArray[np.float64], y: NDArray[np.float64]) -> xgb.XGBRegressor:
    """Fit an XGBRegressor with sensible defaults for tabular regression."""
    model: xgb.XGBRegressor = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=0,
        n_jobs=0,
        tree_method="hist",
    )
    model.fit(X, y)
    return model


def predict(model: xgb.XGBRegressor, X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Predict using a fitted XGBRegressor."""
    return model.predict(X)  # type: ignore[return-value]


def train_xgb_on_df(
    df: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    artifact_path: str
) -> XGBRunResult:
    """
    Train XGBoost on feature_matrix data and persist the model.
    Returns in-sample MAE/RMSE for quick comparison.
    """
    # Keep pandas for selection/NA handling; convert to NumPy for xgboost/sklearn
    X_df, y_df, used_feats = build_train_Xy(df, feat_cols, target_col)
    if X_df.empty:
        raise ValueError("No trainable rows after filtering/imputation. Check feature availability for this geo.")

    X = X_df.to_numpy(dtype="float64")
    y = y_df.to_numpy(dtype="float64")

    model = fit_xgb(X, y)
    pred  = predict(model, X)

    mse = float(mean_squared_error(y, pred))
    mae = float(mean_absolute_error(y, pred))
    rmse = float(mse ** 0.5)

    with open(artifact_path, "wb") as f:
        pickle.dump({"model": model, "features": used_feats}, f)

    return XGBRunResult(artifact_path, used_feats, mae, rmse)