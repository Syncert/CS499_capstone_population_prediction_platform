from __future__ import annotations
import os, requests, polars as pl
from sqlalchemy import text, Date, bindparam
from ppp_common.orm import engine
from ..lib.util import batch_id, artifacts_dir, write_json
from ..lib.layers import write_raw_event, write_stg_frame

FRED_KEY = os.environ["FRED_KEY"]
FRED_BASE = "https://api.stlouisfed.org/fred"

# Pick one series id:
#   CPIAUCSL  -> CPI All Urban Consumers, seasonally adjusted (index 1982-84=100)
#   CUSR0000SEHC -> CPI Shelter
SERIES_ID = os.getenv("FRED_SERIES_ID", "CUSR0000SEHC")

def fred_get(path: str, **params):
    p = {"api_key": FRED_KEY, "file_type": "json", **params}
    url = f"{FRED_BASE}/{path}"
    r = requests.get(url, params=p, timeout=30)
    r.raise_for_status()
    # RAW audit event
    write_raw_event(
        raw_table="raw.fred_calls",
        batch_id=params.get("_batch_id","unknown"),  # injected by caller
        endpoint=url,
        params={k:v for k,v in p.items() if k != "api_key"},
        payload=None,
        response_json=r.json(),
        status_code=r.status_code,
        notes=f"series={params.get('series_id')}"
    )
    return r.json()

def extract_observations(series_id: str, start="2009-01-01", *, batch: str):
    js = fred_get(
        "series/observations",
        series_id=series_id,
        observation_start=start,
        _batch_id=batch
    )
    obs = js["observations"]

    # Normalize to RAW (keep raw_value as text; "." means missing)
    if obs:
        rows = [{
            "etl_batch_id": batch,
            "series_id": series_id,
            "obs_date": o["date"],   # 'YYYY-MM-DD'
            "raw_value": o["value"], # text
        } for o in obs]

        stmt = text("""
            INSERT INTO raw.fred_observations (
                etl_batch_id, series_id, obs_date, raw_value
            )
            VALUES (
                :etl_batch_id, :series_id, :obs_date, :raw_value
            )
            ON CONFLICT DO NOTHING;
        """).bindparams(bindparam("obs_date", type_=Date))

        with engine.begin() as conn:
            conn.execute(stmt, rows)

    return obs


def transform(obs: list[dict]) -> pl.DataFrame:
    df = pl.DataFrame(obs)
    df = df.with_columns(
        pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False),
        pl.when(pl.col("value") == ".")
          .then(None)
          .otherwise(pl.col("value").cast(pl.Float64, strict=False))
          .alias("value_f"),
    )
    yearly = (
        df.with_columns(pl.col("date").dt.year().alias("year"))
          .group_by("year")
          .agg(pl.col("value_f").mean().alias("value"))
          .sort("year")
    )
    return yearly  # columns: year(int), value(float)

def validate(df: pl.DataFrame) -> dict:
    issues: list[str] = []
    if df.is_empty():
        issues.append("no observations")

    if "value" in df.columns:
        vals = df["value"].drop_nulls()
        if (not vals.is_empty()) and bool((vals < 0).any()):
            issues.append("negative CPI value")

    years_min: int | None = None
    years_max: int | None = None
    if (not df.is_empty()) and ("year" in df.columns):
        s = df["year"].drop_nulls()
        # If this ever comes in as Date/Datetime, extract the year first
        if s.dtype in (pl.Date, pl.Datetime):
            s = s.dt.year()
        # Coerce to integer dtype, then to a plain Python list for type certainty
        s = s.cast(pl.Int64, strict=False)
        ys = [int(y) for y in s.to_list() if y is not None]
        if ys:
            years_min, years_max = min(ys), max(ys)

    return {
        "source": "cpi",   # or "fred"
        "rows": df.height,
        "geos": None,
        "years": {"min": years_min, "max": years_max},
        "issues": issues,
    }


def load(df: pl.DataFrame, indicator_code: str, batch: str):
    with engine.begin() as conn:
        # Ensure US geography row exists
        conn.execute(text("""
            INSERT INTO core.geography(geo_code, geo_name, geo_type)
            VALUES ('US','United States','nation')
            ON CONFLICT (geo_code) DO NOTHING;
        """))
        # Upsert indicator values (uses your etl_batch_id column name)
        conn.execute(text("""
            INSERT INTO core.indicator_values
                (geo_code, year, indicator_code, value, source, unit, etl_batch_id)
            VALUES
                ('US', :year, :code, :val, 'FRED', 'index', :batch)
            ON CONFLICT (geo_code, year, indicator_code)
            DO UPDATE SET value = EXCLUDED.value, etl_batch_id = EXCLUDED.etl_batch_id;
        """), [{"year": int(y), "code": indicator_code, "val": float(v), "batch": batch}
              for y, v in zip(df["year"], df["value"])])
        # Keep MV fresh
        conn.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY ml.feature_matrix;"))

def main():
    batch = batch_id("fred")
    obs = extract_observations(SERIES_ID, start=os.getenv("FRED_START", "2009-01-01"), batch=batch)
    df  = transform(obs)

    # VALIDATE
    report = validate(df)
    write_json(report, artifacts_dir() / f"fred_{SERIES_ID}_validation_{batch}.json")

    # STG
    df_stg = df.with_columns([
        pl.lit(SERIES_ID).alias("series_id")
    ]).select(["year","value","series_id"])
    write_stg_frame("stg.cpi_yearly", df_stg, unique_cols=["series_id","year"], batch_id=batch)

    # CORE
    code = "CPI_SHELTER" if SERIES_ID == "CUSR0000SEHC" else "CPI_ALL_U"
    load(df, code, batch)

    print(f"FRED {SERIES_ID}: {df.height} rows; loaded into core.indicator_values as {code}.")

if __name__ == "__main__":
    main()