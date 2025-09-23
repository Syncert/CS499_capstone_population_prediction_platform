from __future__ import annotations
import argparse
from ppp_ml.db import load_feature_matrix
from ppp_ml.prophet_forecast import train_and_forecast_prophet
from ppp_ml.utils import artifact_dir, append_metrics_row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", required=True)
    ap.add_argument("--horizon", type=int, default=10)
    args = ap.parse_args()

    df = load_feature_matrix(args.geo)

    #define directory
    art = artifact_dir() / "prophet"
    art.mkdir(parents=True, exist_ok=True)

    model_json = str(art / f"prophet_model_{args.geo}.json")
    forecast_csv = str(art / f"prophet_forecast_{args.geo}.csv")

    res = train_and_forecast_prophet(
        df=df, horizon_years=args.horizon,
        model_json_path=model_json, forecast_csv_path=forecast_csv
    )

    append_metrics_row(
        art / "metrics.csv",
        {"geo": args.geo, "model":"prophet", "mae":res.mae, "mse":res.rmse**2, "rmse":res.rmse,
         "notes": f"in-sample; regs={res.used_regressors}"}
    )
    print(f"Saved {model_json}  Forecast {forecast_csv}  MAE {res.mae:.2f}")

if __name__ == "__main__":
    main()