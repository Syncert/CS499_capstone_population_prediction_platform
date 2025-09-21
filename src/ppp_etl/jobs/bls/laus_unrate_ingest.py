from __future__ import annotations
import os, requests
import polars as pl
from sqlalchemy import text
from ppp_common.orm import engine
from ..lib.util import batch_id, artifacts_dir, write_json
from time import sleep
from requests.adapters import HTTPAdapter, Retry
from ..lib.layers import write_raw_event, write_stg_frame

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
#   LASST + SS + 00000000000003
def state_series_id(ss: str) -> str:
    return f"LASST{ss}0000000000003"

# County unemployment rate (not seasonally adjusted, annual average). Pattern is:
#   LAUCN + SS + CCC + 0000000003
def county_series_id(ss: str, ccc: str) -> str:
    return f"LAUCN{ss}{ccc}0000000003"

def _post(series_ids: list[str]) -> tuple[dict, int]:
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
    if js.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS error: {js.get('message') or js}")
    return js, r.status_code

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

def extract(*, batch: str) -> pl.DataFrame:
    state_ids, county_ids = extract_series_ids()
    all_ids = state_ids + county_ids

    print(f"[LAUS] building series: states={len(state_ids)}, counties={len(county_ids)}")
    print(f"[LAUS] first 5 state IDs: {state_ids[:5]}")
    if county_ids:
        print(f"[LAUS] first 5 county IDs: {county_ids[:5]}")

    CHUNK = 50
    rows_out: list[dict] = []
    tot_states = tot_counties = 0

    for i in range(0, len(all_ids), CHUNK):
        chunk = all_ids[i:i+CHUNK]
        js, status = _post(chunk)

        write_raw_event(
            raw_table="raw.bls_calls",
            batch_id=batch,
            endpoint=API,
            params={"seriesid": chunk, "startyear": START, "endyear": END},
            payload={"seriesid": chunk, "startyear": START, "endyear": END, "registrationkey": "***"},
            response_json=js, status_code=status,
            notes=f"LAUS chunk size={len(chunk)}"
        )

        series = js.get("Results", {}).get("series", [])
        if i == 0:
            prefixes = sorted({s["seriesID"][:5] for s in series})
            print(f"[LAUS] first chunk series prefixes: {prefixes}")

        c_states  = sum(s["seriesID"].startswith("LASST") for s in series)
        c_counties = sum(s["seriesID"].startswith("LAUCN") for s in series)
        tot_states += c_states; tot_counties += c_counties
        print(f"[LAUS] chunk {i//CHUNK+1}: states={c_states}, counties={c_counties}")

        for s in series:
            sid = s["seriesID"]
            obs = s.get("data", []) or []

            if sid.startswith("LASST"):
                geo_type = "state"; ss = sid[5:7]; geo_code = ss
            elif sid.startswith("LAUCN"):
                geo_type = "county"; ss = sid[5:7]; ccc = sid[7:10]; geo_code = ss + ccc
            else:
                continue

            # --- Build annual values robustly: prefer M13; fallback = mean of months ---
            # NOTE: LAUS usually ships M13 annuals for LASST too, but this handles either case.
            annual = [o for o in obs if o.get("period") == "M13"]
            if not annual:
                # monthly fallback
                if not obs:
                    print(f"[LAUS][WARN] no obs for {sid} → skipping")
                    continue
                dfm = pl.DataFrame(obs)
                if dfm.is_empty():
                    print(f"[LAUS][WARN] empty DF for {sid} → skipping")
                    continue
                # ensure types
                dfm = dfm.with_columns([
                    pl.col("value").cast(pl.Float64, strict=False),
                    pl.col("year").cast(pl.Int32, strict=False)
                ])
                dfm = dfm.group_by("year").agg(pl.col("value").mean().alias("value")).sort("year")
                annual = [{"year": int(y), "value": float(v)} for y, v in zip(dfm["year"], dfm["value"])]

            if not annual:
                print(f"[LAUS][WARN] annual empty after fallback for {sid} → skipping")
                continue

            # write RAW points only if present (nice audit)
            if obs:
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO raw.bls_points (etl_batch_id, series_id, year, period, value)
                        VALUES (:etl_batch_id, :series_id, :year, :period, :value)
                        ON CONFLICT DO NOTHING;
                    """), [{
                        "etl_batch_id": batch,
                        "series_id": sid,
                        "year": int(o["year"]),
                        "period": o.get("period"),
                        "value": o.get("value"),
                    } for o in obs])

            # accumulate output rows
            for a in annual:
                rows_out.append({
                    "geo_code": geo_code,
                    "geo_type": geo_type,
                    "year": int(a["year"]),
                    "unrate": float(a["value"]),
                })

    print(f"[LAUS] totals across all chunks: states={tot_states}, counties={tot_counties}")

    df = pl.from_dicts(
        rows_out,
        schema={"geo_code": pl.String, "geo_type": pl.String, "year": pl.Int64, "unrate": pl.Float64}
    ) if rows_out else pl.DataFrame(schema={"geo_code": pl.String, "geo_type": pl.String, "year": pl.Int64, "unrate": pl.Float64})

    # final visibility
    if not df.is_empty():
        n_states  = int(df.filter(pl.col("geo_type")=="state").height)
        n_counties = int(df.filter(pl.col("geo_type")=="county").height)
        print(f"[LAUS] extract result rows: total={df.height}, states={n_states}, counties={n_counties}, unique geos={int(df['geo_code'].n_unique())}")
    else:
        print("[LAUS][WARN] extract returned empty frame")

    return df


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
    os.environ["PPP_BATCH_ID"] = batch  # so helpers share it

    # extract/transform (single call; keyword-only guards misuse)
    df = extract(batch=batch)

    # VALIDATE
    report = validate(df)
    write_json(report, artifacts_dir() / f"bls_laus_validation_{batch}.json")

    # WRITE STG (typed, deduped)
    write_stg_frame(
        "stg.laus_unrate",
        df.select(["geo_code","geo_type","year","unrate"]),
        unique_cols=["geo_code","year"],
        batch_id=batch
    )

    # LOAD CORE from STG snapshot for this batch (keeps core logic unchanged)
    with engine.begin() as conn:
        stg = (
            conn.execute(text("""
                SELECT
                    geo_code,
                    MIN(geo_type) AS geo_type,
                    year,
                    AVG(unrate)   AS unrate
                FROM stg.laus_unrate
                WHERE etl_batch_id = :batch
                GROUP BY geo_code, year
            """), {"batch": batch})
            .mappings()
            .all()
        )

    # RowMapping -> dict so Polars is happy
    stg_dicts = [dict(r) for r in stg]

    # Build a typed Polars frame
    df_core = pl.from_dicts(
        stg_dicts,
        schema={"geo_code": pl.String, "geo_type": pl.String, "year": pl.Int64, "unrate": pl.Float64}
    )

    load(df_core, batch)

    print(f"BLS LAUS: {df.height} rows across {int(df['geo_code'].n_unique())} geos from {START}–{END}.")


if __name__ == "__main__":
    main()