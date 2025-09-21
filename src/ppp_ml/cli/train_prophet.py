from __future__ import annotations
import argparse
from ppp_ml.db import load_feature_matrix
from ppp_ml.prophet_forecast import train_and_forecast_prophet
from ppp_ml.utils import artifact_dir, append_metrics_row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", default="US")
    ap.add_argument("--horizon", type=int, default=10)
    args = ap.parse_args()

    df = load_feature_matrix(args.geo)
    art_dir = artifact_dir()

    res = train_and_forecast_prophet(
        df=df,
        horizon_years=args.horizon,
        model_json_path=str(art_dir / "prophet_model.json"),
        forecast_csv_path=str(art_dir / "prophet_forecast.csv"),
    )

    append_metrics_row(art_dir / "metrics.csv",
        {"model":"prophet","mae":res.mae,"mse":res.rmse**2,"rmse":res.rmse,"notes":f"in-sample; regs={res.used_regressors}"})
    print("Saved:", res.model_path, "Forecast:", res.forecast_csv, "MAE:", res.mae)

if __name__ == "__main__":
    main()