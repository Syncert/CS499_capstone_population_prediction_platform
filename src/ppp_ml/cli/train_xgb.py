# src/ppp_ml/cli/train_xgb.py
from __future__ import annotations
import argparse
import pandas as pd
from xgboost import XGBRegressor

from ppp_ml.hashers import dataframe_hash
from ppp_ml.artifacts import record_run_start, record_metrics, record_forecasts, finish_and_point_artifact, capture_env, ensure_artifact_row, save_model_artifact
from ppp_ml.db import load_feature_matrix
from ppp_ml.training_io import attach_actuals, basic_test_metrics, select_numeric_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", required=True)
    ap.add_argument("--split_year", type=int, default=2020)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    args = ap.parse_args()

    df = load_feature_matrix(args.geo)
    if df.empty:
        raise SystemExit("no rows")

    h = dataframe_hash(df)
    ensure_artifact_row(args.geo, "xgb", h)
    rows = len(df)
    y0, y1 = int(df["year"].min()), int(df["year"].max())

    feature_cols = select_numeric_features(df)
    X = df[feature_cols].values
    y = df["population"].values
    years = df["year"].values

    model = XGBRegressor(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=0.9, colsample_bytree=0.9, random_state=42
    )
    model.fit(X[years <= args.split_year], y[years <= args.split_year])

    back = pd.DataFrame({"year": years, "yhat": model.predict(X)})
    if args.horizon > 0:
        last_feats = df.iloc[-1][feature_cols].values
        fut_years = list(range(int(y1)+1, int(y1)+1+args.horizon))
        futX = [last_feats for _ in fut_years]
        fut = pd.DataFrame({"year": fut_years, "yhat": model.predict(pd.DataFrame(futX, columns=feature_cols).values)})
        pred_df = pd.concat([back, fut], ignore_index=True)
    else:
        pred_df = back

    pred_df["ds"] = pd.to_datetime(pred_df["year"].astype(str) + "-12-31")
    forecast_df = attach_actuals(args.geo, pred_df[["ds","yhat"]])

    run_id = record_run_start(
        geo=args.geo, model="xgb", data_hash=h,
        rows=rows, year_min=y0, year_max=y1,
        split_year=args.split_year, horizon=args.horizon,
        params={
            "features": feature_cols,
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate
        },
        env=capture_env()
    )

    # write model artifact for API to load later
    artifact_path = save_model_artifact(
        model_name="xgb",
        geo=args.geo,
        estimator=model,
        features=feature_cols,
        use_delta=False,          # set True if your API expects delta-style models
        rename_map=None,
    )
    print(f"[ARTIFACT] {artifact_path}")

    record_metrics(run_id, basic_test_metrics(forecast_df))
    record_forecasts(run_id, args.geo, "xgb", forecast_df)
    finish_and_point_artifact(run_id)

if __name__ == "__main__":
    main()