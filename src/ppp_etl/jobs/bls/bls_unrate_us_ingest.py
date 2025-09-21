from __future__ import annotations
import os, requests, polars as pl
from sqlalchemy import text
from ppp_common.orm import engine
from ..lib.util import batch_id, artifacts_dir, write_json
from ..lib.layers import write_raw_event, write_stg_frame
from requests.adapters import HTTPAdapter, Retry

BLS_KEY = os.environ["BUREAU_LABOR_STATISTICS_KEY"]
START   = int(os.getenv("BLS_START", "2009"))
END     = int(os.getenv("BLS_END",   "2024"))
API     = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
# CPS headline unemployment rate (SA). Use LNU04000000 if you want NSA.
SERIES_ID = os.getenv("BLS_US_UNRATE_SERIES_ID", "LNS14000000")

def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5,
                    status_forcelist=[429,500,502,503,504],
                    allowed_methods=["POST"])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

SESSION = _session()

def _post(series_ids: list[str]) -> dict:
    payload = {
        "seriesid": series_ids,
        "startyear": START,
        "endyear": END,
        "registrationkey": BLS_KEY,
    }
    r = SESSION.post(API, json=payload, timeout=90)
    r.raise_for_status()
    js = r.json()
    if js.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS error: {js.get('message') or js}")
    # raw audit
    write_raw_event(
        raw_table="raw.bls_calls",
        batch_id=os.environ.get("PPP_BATCH_ID","unknown"),
        endpoint=API,
        params={"seriesid": series_ids, "startyear": START, "endyear": END},
        payload={"seriesid": series_ids, "startyear": START, "endyear": END, "registrationkey": "***"},
        response_json=js, status_code=r.status_code,
        notes=f"CPS national unrate series"
    )
    return js

def extract_and_transform(*, batch: str) -> pl.DataFrame:
    js = _post([SERIES_ID])
    series = js.get("Results", {}).get("series", [])
    if not series:
        return pl.DataFrame(schema={"year": pl.Int64, "value": pl.Float64})
    obs = series[0].get("data", [])

    # Prefer BLS-provided annual averages (period M13). Fallback: avg months per year.
    annual = [o for o in obs if o.get("period") == "M13"]
    if not annual:
        df = pl.DataFrame(obs)
        if df.is_empty():
            return pl.DataFrame(schema={"year": pl.Int64, "value": pl.Float64})
        df = df.with_columns([
            pl.col("value").cast(pl.Float64, strict=False),
            pl.col("year").cast(pl.Int32, strict=False)
        ]).group_by("year").agg(pl.col("value").mean().alias("value")).sort("year")
        return df.select(pl.col("year").cast(pl.Int64), pl.col("value").cast(pl.Float64))

    return pl.DataFrame({
        "year": [int(a["year"]) for a in annual],
        "value": [float(a["value"]) for a in annual],
    }).sort("year")

def validate(df: pl.DataFrame) -> dict:
    issues = []
    if df.is_empty(): issues.append("no rows")
    if "value" in df.columns and not df["value"].drop_nulls().is_empty():
        if bool((df["value"].drop_nulls() < 0).any()):
            issues.append("negative rate")
    ys = df["year"].drop_nulls().cast(pl.Int64, strict=False).to_list() if ("year" in df.columns and not df.is_empty()) else []
    return {"source":"bls-cps-us","rows":df.height,"years":{"min":min(ys) if ys else None,"max":max(ys) if ys else None},"issues":issues}

def load(df: pl.DataFrame, *, batch: str):
    with engine.begin() as conn:
        # ensure US geography
        conn.execute(text("""
            INSERT INTO core.geography(geo_code, geo_name, geo_type)
            VALUES ('US','United States','nation')
            ON CONFLICT (geo_code) DO NOTHING;
        """))
        # upsert BLS_UNRATE for US
        conn.execute(text("""
            INSERT INTO core.indicator_values
                (geo_code, year, indicator_code, value, source, unit, etl_batch_id)
            VALUES
                ('US', :year, 'BLS_UNRATE', :val, 'BLS', 'percent', :batch)
            ON CONFLICT (geo_code, year, indicator_code)
            DO UPDATE SET value = EXCLUDED.value, etl_batch_id = EXCLUDED.etl_batch_id;
        """), [{"year": int(y), "val": float(v), "batch": batch} for y, v in zip(df["year"], df["value"])])
        conn.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY ml.feature_matrix;"))

def main():
    batch = batch_id("bls_unrate_us")
    os.environ["PPP_BATCH_ID"] = batch
    df = extract_and_transform(batch=batch)
    write_stg_frame("stg.unrate_us_bls", df.select(["year","value"]), unique_cols=["year"], batch_id=batch)
    write_json(validate(df), artifacts_dir() / f"bls_unrate_us_validation_{batch}.json")
    load(df, batch=batch)
    print(f"BLS CPS US unrate → core.indicator_values('US','BLS_UNRATE'): {df.height} years.")

if __name__ == "__main__":
    main()