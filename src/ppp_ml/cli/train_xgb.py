from __future__ import annotations
import argparse
import pickle
from numpy.typing import NDArray
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ppp_ml.db import load_feature_matrix, split_train_test_years
from ppp_ml.features import BASE_FEATURES, TARGET_COL
from ppp_ml.xgb_forecast import train_xgb_on_df
from ppp_ml.utils import artifact_dir, append_metrics_row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", default="US")
    ap.add_argument("--split_year", type=int, default=2020)
    args = ap.parse_args()

    df = load_feature_matrix(args.geo)
    train, test = split_train_test_years(df, args.split_year)

    art_dir = artifact_dir()
    art = art_dir / "xgb_model.pkl"

    res = train_xgb_on_df(
        train if not train.empty else df,
        BASE_FEATURES,
        TARGET_COL,
        str(art),
    )

    metrics = {
        "model": "xgb",
        "mae": res.mae,
        "mse": res.rmse ** 2,
        "rmse": res.rmse,
        "notes": f"train; feats={res.feats}",
    }

    if not test.empty:
        with open(art, "rb") as f:
            blob = pickle.load(f)
        model = blob["model"]
        feats: list[str] = list(blob["features"])

        Xt_df = test.loc[:, feats].astype("float64").dropna()
        yt_df = test.loc[Xt_df.index, TARGET_COL].astype("float64")

        Xt: NDArray[np.float64] = Xt_df.to_numpy()
        yt: NDArray[np.float64] = yt_df.to_numpy()

        yp: NDArray[np.float64] = model.predict(Xt)  # type: ignore[no-any-return]

        mae = float(mean_absolute_error(yt, yp))
        rmse = float(mean_squared_error(yt, yp) ** 0.5)
        metrics = {
            "model": "xgb",
            "mae":   res.mae,     # ← use the trainer’s level-aware metric
            "mse":   res.rmse**2, # optional, derive if you want both
            "rmse":  res.rmse,
            "notes": f"delta_target=True; feats={res.feats}"
        }

    append_metrics_row(art_dir / "metrics.csv", metrics)
    print("Saved:", art, "Metrics:", metrics)


if __name__ == "__main__":
    main()