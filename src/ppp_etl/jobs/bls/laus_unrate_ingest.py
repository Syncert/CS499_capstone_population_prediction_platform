from __future__ import annotations
import os, json, math, requests
import polars as pl
from typing import Iterable
from sqlalchemy import text
from ppp_common.orm import engine
from ..lib.util import batch_id, artifacts_dir, write_json
from time import sleep
from requests.adapters import HTTPAdapter, Retry

BLS_KEY   = os.environ["BUREAU_LABOR_STATISTICS_KEY"]
START     = int(os.getenv("BLS_START", "2009")) #year acs census starts
END       = int(os.getenv("BLS_END", "2024"))

API = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# robust POST session (helps with 429/5xx)
def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

SESSION = _session()

# ---- Series ID builders (LAUS)
# State unemployment rate (seasonally adjusted, annual average). Pattern is:
#   LAUST + SS + 00000000000003
def state_series_id(ss: str) -> str:
    return f"LAUST{ss}00000000000003"

# County unemployment rate (not seasonally adjusted, annual average). Pattern is:
#   LAUCN + SS + CCC + 0000000003
def county_series_id(ss: str, ccc: str) -> str:
    return f"LAUCN{ss}{ccc}0000000003"

def _post(series_ids: list[str]) -> dict:
    if not os.getenv("BUREAU_LABOR_STATISTICS_KEY"):
        raise RuntimeError("BUREAU_LABOR_STATISTICS_KEY is not set in environment.")
    payload = {
        "seriesid": series_ids,
        "startyear": START,
        "endyear": END,
        "registrationkey": BLS_KEY,
    }
    r = SESSION.post(API, json=payload, timeout=90)
    r.raise_for_status()
    js = r.json()
    # BLS wraps status/messages
    if js.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS error: {js.get('message') or js}")
    return js

# --- get states/counties from your DB geography (preferred) ---
def _db_states_and_counties() -> tuple[list[str], list[str]]:
    """Return (['01','02',..], ['01001','01003',..]) or ([],[]) if not available."""
    try:
        with engine.begin() as conn:
            states = [row[0] for row in conn.execute(
                text("select geo_code from core.geography where geo_type='state' order by 1")
            )]
            counties = [row[0] for row in conn.execute(
                text("select geo_code from core.geography where geo_type='county' order by 1")
            )]
        return states, counties
    except Exception:
        return [], []

def _census_states() -> list[str]:
    # get state fips from Census (reuses your internet access)
    url = f"https://api.census.gov/data/2020/pep/population"
    js = requests.get(url, params={"get": "NAME", "for": "state:*"}, timeout=60).json()
    hdr, *rows = js
    i = {c: k for k, c in enumerate(hdr)}
    return [row[i["state"]] for row in rows]

def _census_counties_for_state(ss: str) -> list[str]:
    url = f"https://api.census.gov/data/2020/pep/population"
    js = requests.get(url, params={"get": "NAME", "for": "county:*", "in": f"state:{ss}"}, timeout=60).json()
    hdr, *rows = js
    i = {c: k for k, c in enumerate(hdr)}
    return [row[i["county"]] for row in rows]

def extract_series_ids() -> tuple[list[str], list[str]]:
    states, counties = _db_states_and_counties()
    if states and counties:
        state_ids  = [state_series_id(ss) for ss in states]                   # ss: '01'
        county_ids = [county_series_id(c[:2], c[2:5]) for c in counties]      # c: '01001'
        return state_ids, county_ids

    # Fallback if DB is empty (e.g., first run before ACS5): use a static list
    # (50 states + DC + PR; add more if you want territories)
    states = [
        "01","02","04","05","06","08","09","10","11","12","13","15","16","17","18","19",
        "20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35",
        "36","37","38","39","40","41","42","44","45","46","47","48","49","50","51","53",
        "54","55","56","72"
    ]
    # If you need counties here too, you can query Census with the robust ACS helper you wrote,
    # but since you already load ACS5 first, the DB path above should be used most of the time.
    county_ids: list[str] = []  # skip counties in this rare fallback
    return [state_series_id(ss) for ss in states], county_ids

def extract() -> pl.DataFrame:
    state_ids, county_ids = extract_series_ids()
    all_ids = state_ids + county_ids

    # chunk requests (BLS limit ~50 series per call; keep it small)
    CHUNK = 50
    frames: list[pl.DataFrame] = []
    for i in range(0, len(all_ids), CHUNK):
        chunk = all_ids[i:i+CHUNK]
        js = _post(chunk)
        series = js.get("Results", {}).get("series", [])
        for s in series:
            sid = s["seriesID"]
            obs = s.get("data", [])
            # obs contains monthly points; filter to annual averages ("M13") or compute mean by year
            # LAUS ships annual averages with "period": "M13" (documented by BLS). Fallback: average months.
            annual = [o for o in obs if o.get("period") == "M13"]
            if not annual:
                # fallback: avg months by year
                df = pl.DataFrame(obs)
                if df.is_empty():
                    continue
                df = df.with_columns([
                    pl.col("value").cast(pl.Float64, strict=False),
                    pl.col("year").cast(pl.Int32, strict=False)
                ])
                df = (df.group_by("year").agg(pl.col("value").mean().alias("value")).sort("year"))
                annual = [{"year": str(int(y)), "value": str(float(v))} for y, v in zip(df["year"], df["value"])]

            # decode geo from series id
            if sid.startswith("LAUST"):
                geo_type = "state"; ss = sid[5:7]; geo_code = ss
            elif sid.startswith("LAUCN"):
                geo_type = "county"; ss = sid[5:7]; ccc = sid[7:10]; geo_code = ss + ccc
            else:
                continue

            frames.append(pl.DataFrame({
                "geo_code": [geo_code for _ in annual],
                "geo_type": [geo_type for _ in annual],
                "year": [int(a["year"]) for a in annual],
                "unrate": [float(a["value"]) for a in annual],
            }))

    # add nation (state average is not "nation"); BLS national series is LNU04000000 (but not LAUS).
    # You can optionally map a national series from FRED/BLS here. For now we leave nation to FRED or add separately.
    return pl.concat(frames) if frames else pl.DataFrame(schema={"geo_code": pl.String, "geo_type": pl.String, "year": pl.Int32, "unrate": pl.Float64})

def validate(df: pl.DataFrame) -> dict:
    issues: list[str] = []
    if df.is_empty():
        issues.append("no rows")

    if "unrate" in df.columns:
        vals = df["unrate"].drop_nulls()
        if (not vals.is_empty()) and bool((vals < 0).any()):
            issues.append("negative rate")

    years_min: int | None = None
    years_max: int | None = None
    if (not df.is_empty()) and ("year" in df.columns):
        s = df["year"].drop_nulls()
        # If year somehow came in as Date/Datetime, extract the calendar year
        if s.dtype in (pl.Date, pl.Datetime):
            s = s.dt.year()
        # Coerce to integer dtype, then to a plain Python list for type-checker certainty
        s = s.cast(pl.Int64, strict=False)
        ys = [int(y) for y in s.to_list() if y is not None]
        if ys:
            years_min, years_max = min(ys), max(ys)

    geos = 0
    if (not df.is_empty()) and ("geo_code" in df.columns):
        geos = int(df["geo_code"].n_unique())

    return {
        "source": "bls",
        "rows": df.height,
        "geos": geos,
        "years": {"min": years_min, "max": years_max},
        "issues": issues,
    }

def load(df: pl.DataFrame, batch: str):
    with engine.begin() as conn:
        geos = df.select(["geo_code", "geo_type"]).unique()
        conn.execute(text("""
            INSERT INTO core.geography (geo_code, geo_name, geo_type)
            VALUES (:code, :name, :type)
            ON CONFLICT (geo_code) DO NOTHING;
        """), [{"code": c, "name": c, "type": t} for c, t in zip(geos["geo_code"], geos["geo_type"])])

        conn.execute(text("""
            INSERT INTO core.indicator_values(geo_code, year, indicator_code, value, source, unit, etl_batch_id)
            VALUES (:code, :year, 'BLS_UNRATE', :val, 'BLS', 'percent', :batch)
            ON CONFLICT (geo_code, year, indicator_code)
            DO UPDATE SET value = EXCLUDED.value, etl_batch_id = EXCLUDED.etl_batch_id;
        """), [{"code": c, "year": int(y), "val": float(v), "batch": batch} for c, y, v in zip(df["geo_code"], df["year"], df["unrate"])])

        conn.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY ml.feature_matrix;"))

def main():
    batch = batch_id("bls_laus")
    df = extract()
    report = validate(df)
    write_json(report, artifacts_dir() / f"bls_laus_validation_{batch}.json")
    load(df, batch)
    print(f"BLS LAUS: {df.height} rows across {int(df['geo_code'].n_unique())} geos from {START}–{END}.")

if __name__ == "__main__":
    main()