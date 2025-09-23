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
    ap.add_argument("--geo", required=True)
    ap.add_argument("--split_year", type=int, default=2020)
    args = ap.parse_args()

    df = load_feature_matrix(args.geo)
    train, test = split_train_test_years(df, args.split_year)

    #define directory
    art_dir = artifact_dir() / "xgb"
    art_dir.mkdir(parents=True, exist_ok=True)

    art = art_dir / f"xgb_{args.geo}.pkl"

    res = train_xgb_on_df(
        train if not train.empty else df,
        BASE_FEATURES,
        TARGET_COL,
        str(art),
    )

    metrics = {
        "geo": args.geo,
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

        Xt = Xt_df.to_numpy()
        yt = yt_df.to_numpy()

        yp = model.predict(Xt)  # type: ignore[no-any-return]

        mae = float(mean_absolute_error(yt, yp))
        rmse = float(mean_squared_error(yt, yp) ** 0.5)
        metrics = {
            "geo": args.geo,
            "model": "xgb",
            "mae": mae,
            "mse": rmse**2,
            "rmse": rmse,
            "notes": f"test; feats={feats}",
        }

    append_metrics_row(art_dir / "metrics.csv", metrics)
    print("Saved:", art, "Metrics:", metrics)

if __name__ == "__main__":
    main()