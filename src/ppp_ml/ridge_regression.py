from __future__ import annotations
from dataclasses import dataclass
import pickle
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ppp_ml.utils import build_train_Xy 


@dataclass
class RidgeRunResult:
    model_path: str
    feats: list[str]
    mae: float
    rmse: float


def fit_ridge(X: NDArray[np.float64], y: NDArray[np.float64], alpha: float = 10.0) -> Pipeline:
    """Fit a Ridge regression pipeline with scaling."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=0))
    ])
    return pipe.fit(X, y)


def predict(model: Pipeline, X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Predict using a fitted Ridge pipeline."""
    return model.predict(X)  # type: ignore[return-value]


def train_ridge_on_df(
    df: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    artifact_path: str
) -> RidgeRunResult:
    """
    Train Ridge regression on feature_matrix data and persist the model.
    Returns in-sample MAE/RMSE for quick comparison.
    """
    X_df, y_df, used_feats = build_train_Xy(df, feat_cols, target_col)
    if X_df.empty:
        raise ValueError("No trainable rows after filtering/imputation. Check feature availability for this geo.")

    X = X_df.to_numpy(dtype="float64")
    y = y_df.to_numpy(dtype="float64")

    model = fit_ridge(X, y)
    pred  = predict(model, X)

    mse = float(mean_squared_error(y, pred))
    mae = float(mean_absolute_error(y, pred))
    rmse = float(mse ** 0.5)

    with open(artifact_path, "wb") as f:
        pickle.dump({"model": model, "features": used_feats}, f)

    return RidgeRunResult(artifact_path, used_feats, mae, rmse)