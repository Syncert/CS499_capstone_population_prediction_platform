from __future__ import annotations
from dataclasses import dataclass
import pickle
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from xgboost.callback import EarlyStopping as XgbEarlyStopping
from typing import Sequence, Protocol
from ppp_ml.utils import build_train_Xy


class _BoosterWrapper:
    def __init__(self, booster: xgb.Booster, feat_names: list[str]):
        self.booster = booster
        self._feat_names = feat_names

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        dm = xgb.DMatrix(X, feature_names=self._feat_names)
        # Prefer best_ntree_limit when present (classic API)
        if hasattr(self.booster, "best_ntree_limit") and getattr(self.booster, "best_ntree_limit"):
            return self.booster.predict(dm, ntree_limit=getattr(self.booster, "best_ntree_limit"))  # type: ignore[attr-defined]
        # Newer API: use iteration_range with best_iteration
        if hasattr(self.booster, "best_iteration") and getattr(self.booster, "best_iteration") is not None:
            bi = int(getattr(self.booster, "best_iteration"))
            return self.booster.predict(dm, iteration_range=(0, bi + 1))
        # Fallback: all trees
        return self.booster.predict(dm)  # type: ignore[return-value]


class HasPredict(Protocol):
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...

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


# replace your predict() signature
def predict(model: HasPredict, X: NDArray[np.float64]) -> NDArray[np.float64]:
    return model.predict(X)  # type: ignore[return-value]

def _monotone_vector(feat_names: Sequence[str]) -> list[int]:
    """
    Build monotone constraints for XGB: +1 for strictly non-decreasing features,
    0 for neutral. Tweak to taste.
    """
    inc_feats = {"pop_lag1", "pop_lag5", "pop_ma3"}  # population history should not decrease output
    # economic effects could be neutral or slightly negative; set 0 to avoid overconstraining
    # {"unemployment_rate"} might plausibly be negative, but keep 0 unless proven
    return [1 if f in inc_feats else 0 for f in feat_names]

def fit_xgb_timeaware(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    feat_names: list[str],
    val_tail: int = 4,
    neutral_monotone: bool = False
):
    if val_tail < 2 or val_tail >= len(y):
        val_tail = max(2, min(4, len(y) // 3))

    X_tr, X_val = X[:-val_tail], X[-val_tail:]
    y_tr, y_val = y[:-val_tail], y[-val_tail:]

    mono = [0] * len(feat_names) if neutral_monotone else _monotone_vector(feat_names)
    mono_str = "(" + ",".join(map(str, mono)) + ")"

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
    dvalid = xgb.DMatrix(X_val, label=y_val, feature_names=feat_names)

    # params.update({
    # "max_depth": 2,
    # "eta": 0.05,
    # "lambda": 1.0
    # })
    # booster = xgb.train(params, dtrain, num_boost_round=400,
    #                     evals=[(dvalid,"valid")], early_stopping_rounds=5, verbose_eval=False)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "base_score": float(np.mean(y_tr)),  # ≈ 0 for deltas
        "max_depth": 1, #doesn't navigate further since baseline is desired
        "eta": 0.03,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "lambda": 10.0,
        "alpha": 0.0,
        "min_child_weight": 1.0,
        "tree_method": "hist",
        "monotone_constraints": mono_str,
        "seed": 0,
        "nthread": 1,
    }

    evals_result = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=200,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=3,
        verbose_eval=False,
        evals_result=evals_result,
    )

    #logging
    best_iter = getattr(booster, "best_iteration", None)
    best_ntl  = getattr(booster, "best_ntree_limit", None)  # may be None on some builds
    best_score= getattr(booster, "best_score", None)

    print({
    "best_iteration": best_iter,
    "best_ntree_limit": best_ntl,
    "best_score": best_score,
    "eval_first10": evals_result.get("valid", {}).get("mae", [])[:10],
    })


    return _BoosterWrapper(booster, feat_names)


def train_xgb_on_df(
    df: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    artifact_path: str
) -> XGBRunResult:
    """
    Train XGBoost on feature_matrix data and persist the model.
    Uses a delta target (y - pop_lag1) internally to stabilize training,
    then reassembles the level for metrics/persistence.
    """

    minimal = ["pop_lag1", "pop_ma3", "pop_lag5"]
    feat_cols = [c for c in minimal if c in df.columns]  #adjusting training columns for sake of experiment

    # Ensure we have a stable minimal feature set for XGB
    if "pop_lag1" not in feat_cols:
        feat_cols = ["pop_lag1", "pop_ma3", "pop_lag5"] #do not need additional features since xgb drops them anyways and doesn't navigate down trees

    # Work on a copy
    work = df.copy()
    # 1) delta target
    work["__y_delta__"] = work[target_col] - work["pop_lag1"]
    target_for_fit = "__y_delta__"

    # 2) build matrices
    X_df, y_df, used_feats = build_train_Xy(work, feat_cols, target_for_fit)
   
    #logging
    print({"used_feats": used_feats})  # right after build_train_Xy

    if X_df.empty:
        raise ValueError("No trainable rows after filtering/imputation. Check feature availability for this geo.")
    X = X_df.to_numpy(dtype="float64")
    y_delta = y_df.to_numpy(dtype="float64")
    base = work.loc[X_df.index, "pop_lag1"].to_numpy(dtype="float64")  # align to X rows
    y_level = work.loc[X_df.index, target_col].to_numpy(dtype="float64")

    # 3) fit with time-aware routine; neutralize monotonicity (we're modeling deltas)
    model = fit_xgb_timeaware(X, y_delta, used_feats, val_tail=4, neutral_monotone=True)

    # 4) predict delta, then reassemble level
    pred_delta = predict(model, X)
    pred = pred_delta + base

    # ── Hard alignment & integrity checks
    idx = X_df.index
    y_level_aligned = work[target_col].reindex(idx).to_numpy(dtype="float64")
    base_aligned    = work["pop_lag1"].reindex(idx).to_numpy(dtype="float64")

    # quick invariants
    assert y_level_aligned.shape == pred.shape, (y_level_aligned.shape, pred.shape)
    assert not np.isnan(pred).any(), "pred has NaNs"
    assert not np.isnan(y_level_aligned).any(), "y has NaNs"

    # numpy MAE (ground truth) vs sklearn MAE
    mae_np  = float(np.mean(np.abs(y_level_aligned - pred)))
    mse_np  = float(np.mean((y_level_aligned - pred) ** 2))
    rmse_np = float(np.sqrt(mse_np))

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae_skl = float(mean_absolute_error(y_level_aligned, pred))
    mse_skl = float(mean_squared_error(y_level_aligned, pred))
    rmse_skl = float(mse_skl ** 0.5)

    print({
    "debug_shapes": {"y": y_level_aligned.shape, "pred": pred.shape},
    "debug_means":  {"y": float(np.mean(y_level_aligned)), "pred": float(np.mean(pred))},
    "debug_mae":    {"np": mae_np, "sk": mae_skl}
    })

    # 5) metrics on the level
    mse = mse_skl
    mae = mae_skl
    rmse = rmse_skl


    with open(artifact_path, "wb") as f:
        pickle.dump({"model": model, "features": used_feats, "use_delta": True}, f)

    return XGBRunResult(artifact_path, used_feats, mae, rmse)