from __future__ import annotations
import argparse, time
import pandas as pd
from prophet import Prophet

from ppp_ml.hashers import dataframe_hash
from ppp_ml.artifacts import record_run_start, record_metrics, record_forecasts, finish_and_point_artifact, capture_env, ensure_artifact_row
from ppp_ml.db import load_feature_matrix   # your existing accessor
from ppp_ml.training_io import attach_actuals, basic_test_metrics, select_numeric_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", required=True)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--split_year", type=int, default=2020)
    args = ap.parse_args()

    df = load_feature_matrix(args.geo)              # expects a pandas DF
    if df.empty:
        raise SystemExit("no rows for geo")

    # Hash + metadata
    h = dataframe_hash(df)
    ensure_artifact_row(args.geo, "prophet", h)

    rows = len(df)
    y0, y1 = int(df["year"].min()), int(df["year"].max())

    # Prepare Prophet format: ds,y
    hist = df[["year","population"]].copy()
    hist["ds"] = pd.to_datetime(hist["year"].astype(str) + "-12-31")
    hist = hist.rename(columns={"population":"y"})

    regressors = select_numeric_features(df) # all numeric features
    m = Prophet(growth="linear", changepoint_prior_scale=0.05)
    for r in regressors:
        m.add_regressor(r)

    # Build train frame with ds,y and only numeric regressors
    train = hist[hist["ds"].dt.year <= args.split_year].copy()
    for r in regressors:
        # align by year; avoid mixing strings
        s = df.set_index("year")[r]
        train[r] = s.reindex(train["ds"].dt.year.values).values

    t0 = time.time()
    m.fit(train)

    # Future frame: carry-forward last value for each numeric regressor
    future = m.make_future_dataframe(periods=args.horizon, freq="YE")
    future_years = future["ds"].dt.year
    for r in regressors:
        s = df.set_index("year")[r]
        last = s.iloc[-1]
        future[r] = (
                        s.reindex(future_years).ffill().fillna(last).infer_objects(copy=False).to_numpy()
                    )

    fcst = m.predict(future)
    # Build forecast_df with required columns
    forecast_df = pd.DataFrame({
        "ds": fcst["ds"],
        "yhat": fcst["yhat"],
        "yhat_lo": fcst.get("yhat_lower"),
        "yhat_hi": fcst.get("yhat_upper"),
    })
    forecast_df = attach_actuals(args.geo, forecast_df)

    # Record run
    run_id = record_run_start(
        geo=args.geo, model="prophet", data_hash=h,
        rows=rows, year_min=y0, year_max=y1,
        split_year=args.split_year, horizon=args.horizon,
        params={"used_regressors": regressors}, env=capture_env()
    )

    # Persist metrics & forecasts
    record_metrics(run_id, basic_test_metrics(forecast_df))
    record_forecasts(run_id, args.geo, "prophet", forecast_df)

    # Update pointers
    finish_and_point_artifact(run_id)

if __name__ == "__main__":
    main()
