from __future__ import annotations
import json
import time
from datetime import datetime
from typing import Iterable, Optional
import pandas as pd
from sqlalchemy import text
import sqlalchemy as sa
from pathlib import Path
import os
import pickle
from ppp_common.orm import engine  # your existing Engine factory


def capture_env() -> dict:
    import platform, sys, subprocess
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        git_sha = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": getattr(pd, "__version__", None),
        "sqlalchemy": getattr(sa, "__version__", None),
        "git": git_sha,
    }

def get_artifact_hash(geo: str, model: str) -> str | None:
    """
    Look up the last recorded data_hash for a given (geo, model) in ml.model_artifacts.
    Returns None if no artifact row exists yet.
    """
    q = text("""
        select data_hash
        from ml.model_artifacts
        where geo_code = :g and model = :m
    """)
    with engine.connect() as cx:
        return cx.execute(q, {"g": geo, "m": model}).scalar()

def ensure_artifact_row(geo: str, model: str, data_hash: str) -> None:
    with engine.begin() as cx:
        cx.execute(text("""
          insert into ml.model_artifacts (geo_code, model, data_hash)
          values (:g, :m, :h)
          on conflict (geo_code, model) do nothing
        """), {"g": geo, "m": model, "h": data_hash})

def record_run_start(
    geo: str, model: str, data_hash: str,
    rows: int, year_min: int, year_max: int,
    split_year: int | None, horizon: int | None,
    params: dict, env: dict | None = None
) -> str:
    sql = text("""
      insert into ml.model_runs
        (geo_code, model, data_hash, rows, year_min, year_max,
         split_year, horizon, params, env)
      values (:g, :m, :h, :r, :ymin, :ymax, :split, :hz,
              CAST(:params AS jsonb), CAST(:env AS jsonb))
      returning run_id
    """)
    with engine.begin() as cx:
        return cx.execute(sql, {
            "g": geo, "m": model, "h": data_hash,
            "r": rows, "ymin": year_min, "ymax": year_max,
            "split": split_year, "hz": horizon,
            "params": json.dumps(params or {}),
            "env": json.dumps(env or capture_env()),
        }).scalar_one()

def refresh_headline(concurrently: bool = True) -> None:
    stmt = "refresh materialized view concurrently ml.model_headline" if concurrently \
           else "refresh materialized view ml.model_headline"
    # CONCURRENTLY must be autocommit (no transaction)
    if concurrently:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as cx:
            cx.execute(text(stmt))
    else:
        with engine.begin() as cx:
            cx.execute(text(stmt))

def record_metrics(run_id: str, metrics: Iterable[tuple[str, str, int, float]]) -> None:

    rows = [
        dict(rid=run_id, metric=m, scope=s, fold=f, val=float(v))
        for (m, s, f, v) in metrics
    ]
    if not rows:
        return

    # 1) write in a transaction
    with engine.begin() as cx:
        cx.execute(text("""
           insert into ml.model_metrics (run_id, metric, scope, fold, value)
           values (:rid, :metric, :scope, :fold, :val)
        """), rows)

    # 2) now that the insert COMMITTED, refresh the MV
    try:
        refresh_headline(concurrently=True)   # requires unique index on run_id
    except Exception:
        refresh_headline(concurrently=False)


def record_forecasts(
    run_id: str, geo: str, model: str, forecast_df: pd.DataFrame
) -> None:
    # Expect columns: ds, yhat [, yhat_lo, yhat_hi, actual]
    needed = {"ds", "yhat"}
    if not needed.issubset(set(forecast_df.columns)):
        raise ValueError(f"forecast_df missing required columns: {needed}")
    rows = []
    for r in forecast_df.itertuples(index=False):
        rows.append(dict(
            rid=run_id, g=geo, m=model,
            ds=pd.to_datetime(getattr(r, "ds")).date(),
            y=getattr(r, "yhat"),
            lo=getattr(r, "yhat_lo", None),
            hi=getattr(r, "yhat_hi", None),
            a=getattr(r, "actual", None),
        ))
    with engine.begin() as cx:
        cx.execute(text("""
          insert into ml.model_forecasts
            (run_id, geo_code, model, ds, yhat, yhat_lo, yhat_hi, actual)
          values (:rid,:g,:m,:ds,:y,:lo,:hi,:a)
        """), rows)

def finish_and_point_artifact(run_id: str) -> None:
    """
    1) Update latest pointer for the (geo, model) of this run_id.
    2) Recompute best pointer for that (geo, model) directly from ml.model_metrics
       using test RMSE (lower is better). No dependency on materialized view freshness.
    """
    with engine.begin() as cx:
        # 1) latest pointer
        cx.execute(text("""
            update ml.model_artifacts a
            set latest_run_id = r.run_id,
                trained_at    = now(),
                data_hash     = r.data_hash,
                rows          = r.rows,
                year_min      = r.year_min,
                year_max      = r.year_max,
                artifact_path = r.artifact_path
            from ml.model_runs r
            where a.geo_code = r.geo_code
              and a.model    = r.model
              and r.run_id   = :rid
        """), {"rid": run_id})

        # 2) best pointer (scoped to this (geo, model))
        cx.execute(text("""
            with cur as (
                select geo_code, model
                from ml.model_runs
                where run_id = :rid
            ),
            scores as (
                select r.geo_code, r.model, r.run_id,
                       min(m.value) filter (where m.metric='rmse' and m.scope='test') as rmse_test
                from ml.model_runs r
                join cur c
                  on c.geo_code = r.geo_code and c.model = r.model
                left join ml.model_metrics m
                  on m.run_id = r.run_id
                group by r.geo_code, r.model, r.run_id
            ),
            ranked as (
                select *, row_number() over (
                    partition by geo_code, model
                    order by rmse_test asc nulls last, run_id desc
                ) as rk
                from scores
            )
            update ml.model_artifacts a
            set best_run_id = r.run_id
            from ranked r
            where r.rk = 1
              and a.geo_code = r.geo_code
              and a.model    = r.model
        """), {"rid": run_id})

def _models_dir() -> Path:
    # where FastAPI looks by default
    return Path(os.getenv("MODELS_DIR", "models"))

def artifact_path_for(model: str, geo: str) -> Path:
    """
    Filesystem path we write to, matching what the API loader expects.
      - sklearn/xgb: models/<model>/<geo>.pkl
      - prophet:     models/prophet/prophet_model_<geo>.json
    """
    base = _models_dir()
    if model.lower() == "prophet":
        return base / "prophet" / f"prophet_model_{geo}.json"
    else:
        return base / model.lower() / f"{model.lower()}_{geo}.pkl"

def save_model_artifact(
    model_name: str,
    geo: str,
    *,
    estimator=None,
    features: list[str] | None = None,
    use_delta: bool | None = None,
    rename_map: dict[str, str] | None = None,
    prophet_model=None,                 # if model_name == "prophet"
) -> str:
    """
    Write the model to disk in the exact structure the API expects and
    return the absolute path as a string.

    For sklearn/xgb:
      payload = {"model": estimator, "features": [...], "use_delta": bool?, "rename_map": {...}?}

    For prophet:
      write JSON using prophet.serialize.model_to_json, file name prophet_model_<geo>.json
    """
    out = artifact_path_for(model_name, geo)
    out.parent.mkdir(parents=True, exist_ok=True)

    if model_name.lower() == "prophet":
        if prophet_model is None:
            raise ValueError("save_model_artifact: prophet_model is required for 'prophet'")
        # defer import to avoid hard dependency elsewhere
        from prophet.serialize import model_to_json
        out.write_text(model_to_json(prophet_model), encoding="utf-8")
    else:
        if estimator is None:
            raise ValueError("save_model_artifact: estimator is required for non-prophet models")
        payload = {"model": estimator}
        if features is not None:
            payload["features"] = list(features)
        if use_delta is not None:
            payload["use_delta"] = bool(use_delta)
        if rename_map:
            payload["rename_map"] = dict(rename_map)
        with out.open("wb") as f:
            pickle.dump(payload, f)

    return str(out.resolve())