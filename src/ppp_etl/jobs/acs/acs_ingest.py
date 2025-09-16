from __future__ import annotations
import os, requests
import polars as pl
from sqlalchemy import text
from ppp_common.orm import engine
from ..lib.util import batch_id, artifacts_dir, write_json

# Config (bare minimum)
ACS_YEAR = int(os.getenv("ACS_YEAR", "2021"))
ACS_URL  = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs1"

def fetch_total_pop_us() -> dict:
    params = {"get": "NAME,B01001_001E", "for": "us:1"}
    r = requests.get(ACS_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def to_staging_df(resp_json: dict) -> pl.DataFrame:
    header, *rows = resp_json
    idx = {col: i for i, col in enumerate(header)}
    rec = rows[0]
    df = pl.DataFrame({
        "year": [ACS_YEAR],
        "name": [rec[idx["NAME"]]],
        "pop":  [int(rec[idx["B01001_001E"]])],
    })
    return df

def validate_df(df: pl.DataFrame) -> dict:
    issues: list[str] = []

    # Pull out Python scalars (or None) in a way Pylance understands.
    pop_min = df["pop"].min()
    pop_max = df["pop"].max()
    year_min = df["year"].min()
    year_max = df["year"].max()
    year_nunique = df["year"].n_unique()  # already an int

    if pop_min is not None and isinstance(pop_min, (int, float)) and pop_min < 0:
        issues.append("population negative")
    if year_nunique != 1:
        issues.append("multi-year in batch")

    def to_int(x) -> int:
        return int(x) if isinstance(x, (int, float)) else 0

    return {
        "row_count": df.height,
        "year_min": to_int(year_min),
        "year_max": to_int(year_max),
        "pop_min": to_int(pop_min),
        "pop_max": to_int(pop_max),
        "issues": issues,
    }

def load_raw(resp_json: dict):
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO raw.acs1_total_population(resp_json) VALUES (:js::jsonb)"),
            {"js": pl.Series([resp_json]).struct.json_encode().item(0)},
        )

def upsert_core(df: pl.DataFrame, batch: str):
    with engine.begin() as conn:
        # ensure geography exists
        conn.execute(text("""
            INSERT INTO core.geography(geo_code, geo_name, geo_type)
            VALUES ('US','United States','nation')
            ON CONFLICT (geo_code) DO NOTHING;
        """))
        # upsert population
        conn.execute(text("""
            INSERT INTO core.population_observations(geo_code, year, population)
            VALUES ('US', :year, :pop)
            ON CONFLICT (geo_code, year)
            DO UPDATE SET population = EXCLUDED.population;
        """), [{"year": int(y), "pop": int(p)} for y,p in zip(df["year"], df["pop"])])

        # indicator mirror (ACS_TOTAL_POP)
        conn.execute(text("""
            INSERT INTO core.indicator_values(geo_code, year, indicator_code, value, source, unit, batch_id)
            VALUES ('US', :year, 'ACS_TOTAL_POP', :pop, 'ACS', 'count', :batch)
            ON CONFLICT (geo_code, year, indicator_code)
            DO UPDATE SET value = EXCLUDED.value, batch_id = EXCLUDED.batch_id;
        """), [{"year": int(y), "pop": int(p), "batch": batch} for y,p in zip(df["year"], df["pop"])])

        # refresh MV
        conn.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY ml.feature_matrix;"))

def write_validation_report(report: dict, batch: str):
    out = artifacts_dir() / f"week2_validation_{batch}.json"
    write_json(report, out)

def main():
    batch = batch_id("week2")
    resp = fetch_total_pop_us()
    df   = to_staging_df(resp)
    report = validate_df(df)
    write_validation_report(report, batch)
    load_raw(resp)
    upsert_core(df, batch)
    print(f"ACS {ACS_YEAR}: loaded {df.height} rows; report saved; MV refreshed.")

if __name__ == "__main__":
    main()
