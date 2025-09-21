from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from prophet import Prophet                          # pip package name: prophet
from prophet.serialize import model_to_json          # ← use typed, public serializer

@dataclass
class ProphetRunResult:
    model_path: str
    forecast_csv: str
    used_regressors: list[str]
    mae: float
    rmse: float

EXO = ["unemployment_rate", "rent_cpi_index"]

def _to_prophet_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["ds"] = pd.to_datetime(out["year"].astype(str) + "-12-31")
    out["y"]  = out["population"].astype(float)
    used: list[str] = []
    for r in EXO:
        if r in out.columns:
            out[r] = out[r].astype(float)
            used.append(r)
    return out[["ds", "y"] + used], used

def train_and_forecast_prophet(
    df: pd.DataFrame,
    horizon_years: int,
    model_json_path: str,
    forecast_csv_path: str
) -> ProphetRunResult:
    dfp, used = _to_prophet_frame(df)
    m = Prophet()
    for r in used:
        m.add_regressor(r)
    m.fit(dfp)

    last_year = int(df["year"].max())
    future = pd.DataFrame({"ds": pd.to_datetime([f"{y}-12-31" for y in range(last_year + 1, last_year + 1 + horizon_years)])})
    for r in used:
        last_val = float(dfp[r].dropna().iloc[-1]) if dfp[r].notna().any() else 0.0
        future[r] = last_val

    fcst = m.predict(pd.concat([dfp.drop(columns=["y"]), future], ignore_index=True))

    # Save artifacts (typed/public API)
    with open(model_json_path, "w") as f:
        f.write(model_to_json(m))
    fcst.to_csv(forecast_csv_path, index=False)

    merged = dfp.merge(fcst[["ds", "yhat"]], on="ds", how="left").dropna()
    err  = merged["y"] - merged["yhat"]
    mae  = float(err.abs().mean())
    rmse = float((err**2).mean() ** 0.5)

    return ProphetRunResult(model_json_path, forecast_csv_path, used, mae, rmse)