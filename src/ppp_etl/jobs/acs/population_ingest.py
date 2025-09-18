from __future__ import annotations
import os, requests
import polars as pl
from typing import Iterable
from sqlalchemy import text
from ppp_common.orm import engine
from ..lib.util import batch_id, artifacts_dir, write_json
from time import sleep
from requests.adapters import HTTPAdapter, Retry
from ..lib.layers import write_raw_event, write_stg_frame
import json

# ---- Config: Census API Key ----
#API Key
CENSUS_KEY = os.getenv("CENSUS_API_KEY")


# ---- Config: control year range via env; gracefully skip missing years ----
ACS1_START = int(os.getenv("ACS1_START", "2005"))  # ACS 1-year starts ~2005
ACS5_START = int(os.getenv("ACS5_START", "2009"))  # ACS 5-year starts ~2009
ACS_END    = int(os.getenv("ACS_END",    str(__import__("datetime").date.today().year)))

# If True, attempt counties for acs1 (many counties won’t exist there; we’ll just skip missing)
INCLUDE_ACS1_COUNTIES = os.getenv("INCLUDE_ACS1_COUNTIES", "false").lower() in {"1","true","yes"}

VAR = "B01001_001E"  # total population estimate variable


def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5, backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

SESSION = _session()

#helper for building out url for census api calls
def _acs_url(year: int, survey: str) -> str:
    # survey is "acs1" or "acs5"
    return f"https://api.census.gov/data/{year}/acs/{survey}"


def _avail(year: int, survey: str) -> bool:
    url = _acs_url(year, survey)
    try:
        # ask for the variable we really need
        r = requests.get(url, params={"get": VAR, "for": "us:1"}, timeout=10)
        # 400/404 means not available; some errors may still return 200 + {"error": ...}
        if not r.ok:
            return False
        js = r.json()
        # ensure we got a table-shaped response, not an error dict
        return isinstance(js, list) and len(js) > 0 and VAR in js[0]
    except requests.RequestException:
        return False


def _get(year: int, survey: str, params: dict) -> list[list[str]]:
    url = _acs_url(year, survey)
    params = dict(params)
    if CENSUS_KEY:
        params["key"] = CENSUS_KEY

    r = SESSION.get(url, params=params, timeout=60)
    raw_text = r.text  # keep for logging

    # Try hard to parse JSON: first r.json(), then json.loads(text).
    try:
        rj = r.json()
    except Exception:
        try:
            rj = json.loads(raw_text)
        except Exception:
            rj = {"non_json_body": raw_text[:1024]}

    # RAW call audit (log what we actually saw)
    write_raw_event(
        raw_table="raw.acs_calls",
        batch_id=os.getenv("PPP_BATCH_ID", "unknown"),
        endpoint=url,
        params={k: v for k, v in params.items() if k != "key"},
        payload=None,
        response_json=rj if isinstance(rj, (dict, list)) else {"non_json_body": str(rj)[:1024]},
        status_code=r.status_code,
        notes=f"{survey} {year} params={{{','.join(sorted(params.keys()))}}}"
    )

    r.raise_for_status()
    js = rj
    if not (isinstance(js, list) and js and isinstance(js[0], list)):
        raise ValueError(f"Census API error for {survey} {year}: {js}")
    return js

def _state_fips_list() -> list[str]:
    """
    Return 2-digit state FIPS codes.
    Strategy:
      1) Try ACS5 (more stable): {ACS_END, ACS_END-1, 2023}
      2) Try PEP population:     {ACS_END, ACS_END-1, 2022, 2021, 2020}
      3) Fallback to static list (50 + DC + PR).
    No terminal spam: only warn if PPP_VERBOSE=1.
    """
    def _warn(msg: str):
        if os.getenv("PPP_VERBOSE", "0") in {"1", "true", "yes"}:
            print(f"[WARN] {msg}")

    candidates: list[tuple[str, int, str]] = []

    # Prefer ACS5 first (very reliable for 'state:*')
    for y in [ACS_END, ACS_END - 1, 2023]:
        candidates.append(("acs", y, "acs5"))

    # Then try PEP population across a few years
    for y in [ACS_END, ACS_END - 1, 2022, 2021, 2020]:
        candidates.append(("pep", y, "population"))

    params_base = {"get": "NAME", "for": "state:*"}
    if CENSUS_KEY:
        params_base["key"] = CENSUS_KEY

    for family, year, dataset in candidates:
        if family == "acs":
            url = f"https://api.census.gov/data/{year}/acs/{dataset}"
        else:
            url = f"https://api.census.gov/data/{year}/pep/{dataset}"
        try:
            r = SESSION.get(url, params=params_base, timeout=60)
            if r.status_code != 200:
                continue
            js = r.json()
            if not (isinstance(js, list) and js and isinstance(js[0], list)):
                continue
            hdr, *rows = js
            i = {c: k for k, c in enumerate(hdr)}
            if "state" not in i:
                continue
            return [row[i["state"]] for row in rows]
        except Exception as e:
            # Quietly try the next candidate; only warn when verbose
            _warn(f"state list via {family.upper()} {year}/{dataset} failed: {e}")

    # Final static fallback (50 states + DC + PR)
    return [
        "01","02","04","05","06","08","09","10","11","12","13","15","16","17","18","19",
        "20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35",
        "36","37","38","39","40","41","42","44","45","46","47","48","49","50","51","53",
        "54","55","56","72"
    ]

def _index_or_die(hdr: list[str], needed: list[str]) -> dict[str, int]:
    i = {c: k for k, c in enumerate(hdr)}
    missing = [c for c in needed if c not in i]
    if missing:
        raise KeyError(f"Missing columns {missing} in {hdr}")
    return i

# ---------- single-year extractors ----------
def extract_us(year: int, survey: str) -> pl.DataFrame:
    js = _get(year, survey, {"get": f"NAME,{VAR}", "for": "us:1"})
    hdr, *rows = js
    i = _index_or_die(hdr, ["NAME", VAR])
    rec = rows[0]

    # RAW rows
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO raw.acs_rows (etl_batch_id, dataset, year, geo_level, state_fips,
                                      geo_code, geo_name, var_code, raw_value)
            VALUES (:batch, :ds, :yr, 'us', NULL, 'US', :name, :var, :val)
            ON CONFLICT DO NOTHING;
        """), {
            "batch": os.getenv("PPP_BATCH_ID","unknown"), "ds": survey, "yr": year,
            "name": rec[i["NAME"]], "var": VAR, "val": rec[i[VAR]]
        })

    return pl.DataFrame({
        "geo_code": ["US"],
        "geo_name": [rec[i["NAME"]]],
        "geo_type": ["nation"],
        "year": [year],
        "population": [int(rec[i[VAR]])],
        "source": [survey.upper()],
    })

def extract_states(year: int, survey: str) -> pl.DataFrame:
    js = _get(year, survey, {"get": f"NAME,{VAR}", "for": "state:*"})
    hdr, *rows = js
    i = {c: k for k, c in enumerate(hdr)}

    # RAW rows (one per state)
    with engine.begin() as conn:
        payload = [{
            "batch": os.getenv("PPP_BATCH_ID", "unknown"),
            "ds": survey,
            "yr": year,
            "geo_level": "state",
            "state_fips": row[i["state"]],
            "geo_code": row[i["state"]],
            "geo_name": row[i["NAME"]],
            "var": VAR,
            "val": row[i[VAR]],
        } for row in rows]
        conn.execute(text("""
            INSERT INTO raw.acs_rows (
                etl_batch_id, dataset, year, geo_level, state_fips,
                geo_code, geo_name, var_code, raw_value
            )
            VALUES (:batch, :ds, :yr, :geo_level, :state_fips,
                    :geo_code, :geo_name, :var, :val)
            ON CONFLICT DO NOTHING;
        """), payload)

    return pl.DataFrame({
        "geo_code": [row[i["state"]] for row in rows],  # 2-digit FIPS
        "geo_name": [row[i["NAME"]] for row in rows],
        "geo_type": ["state"] * len(rows),
        "year": [year] * len(rows),
        "population": [int(row[i[VAR]]) for row in rows],
        "source": [survey.upper()] * len(rows),
    })

def extract_counties(year: int, survey: str) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for ss in _state_fips_list():
        try:
            js = _get(year, survey, {
                "get": f"NAME,{VAR}",
                "for": "county:*",
                "in": f"state:{ss}"
            })
        except Exception as e:
            print(f"[WARN] {survey} {year} state {ss}: {e}")
            continue

        hdr, *rows = js
        if not rows:
            continue
        i = {c: k for k, c in enumerate(hdr)}

        # RAW rows (one per county)
        with engine.begin() as conn:
            payload = [{
                "batch": os.getenv("PPP_BATCH_ID", "unknown"),
                "ds": survey,
                "yr": year,
                "geo_level": "county",
                "state_fips": row[i["state"]],
                "geo_code": f"{row[i['state']]}{row[i['county']]}",
                "geo_name": row[i["NAME"]],
                "var": VAR,
                "val": row[i[VAR]],
            } for row in rows]
            conn.execute(text("""
                INSERT INTO raw.acs_rows (
                    etl_batch_id, dataset, year, geo_level, state_fips,
                    geo_code, geo_name, var_code, raw_value
                )
                VALUES (:batch, :ds, :yr, :geo_level, :state_fips,
                        :geo_code, :geo_name, :var, :val)
                ON CONFLICT DO NOTHING;
            """), payload)

        frames.append(pl.DataFrame({
            "geo_code": [f"{row[i['state']]}{row[i['county']]}" for row in rows],
            "geo_name": [row[i["NAME"]] for row in rows],
            "geo_type": ["county"] * len(rows),
            "year": [year] * len(rows),
            "population": [int(row[i[VAR]]) for row in rows],
            "source": [survey.upper()] * len(rows),
        }))

        sleep(0.1)  # be polite to the API

    return pl.concat(frames) if frames else _EMPTY

_EMPTY = pl.DataFrame(schema={
    "geo_code": pl.String, "geo_name": pl.String, "geo_type": pl.String,
    "year": pl.Int32, "population": pl.Int64, "source": pl.String
})

# ---------- multi-year drivers ----------
def extract_series(years: list[int], survey: str, include_counties: bool) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for y in years:
        try:
            # nation + states
            frames.append(extract_us(y, survey))
            frames.append(extract_states(y, survey))
            # counties: always for acs5; optional for acs1
            if include_counties or survey == "acs5":
                frames.append(extract_counties(y, survey))
        except (requests.HTTPError, requests.ConnectionError, ValueError, KeyError) as e:
            # ValueError/KeyError come from shape/column checks in _get / _index
            print(f"[WARN] skipping {survey} {y}: {e}")
            continue

    return (pl.concat(frames, how="diagonal_relaxed") if frames else _EMPTY)

def choose_canonical(df_all: pl.DataFrame) -> pl.DataFrame:
    """
    For each (geo_code, year), pick the best source:
      - nation/state: prefer ACS1 over ACS5
      - county:       prefer ACS5 over ACS1
    """
    if df_all.is_empty():
        return df_all
    pri = (pl.when(pl.col("geo_type") == "county")
             .then(pl.when(pl.col("source") == "ACS5").then(0).otherwise(1))
             .otherwise(pl.when(pl.col("source") == "ACS1").then(0).otherwise(1))
             .alias("priority"))
    df = df_all.with_columns(pri).sort(["geo_code", "year", "priority"])
    # take first row per (geo_code, year)
    return df.unique(subset=["geo_code", "year"], keep="first").drop("priority")

def validate(df: pl.DataFrame) -> dict:
    issues: list[str] = []
    if df.is_empty():
        issues.append("no rows")

    if "population" in df.columns:
        vals = df["population"].drop_nulls()
        if (not vals.is_empty()) and bool((vals < 0).any()):
            issues.append("negative population")

    years_min: int | None = None
    years_max: int | None = None
    if (not df.is_empty()) and ("year" in df.columns):
        s = df["year"].drop_nulls()
        # Convert dates/datetimes → year
        if s.dtype in (pl.Date, pl.Datetime):
            s = s.dt.year()
        # Coerce to integer dtype, then to plain Python ints for type-checker clarity
        s = s.cast(pl.Int64, strict=False)
        ys = [int(y) for y in s.to_list() if y is not None]
        if ys:
            years_min, years_max = min(ys), max(ys)

    geos = 0
    if (not df.is_empty()) and ("geo_code" in df.columns):
        geos = int(df["geo_code"].n_unique())

    return {
        "source": "acs",
        "rows": df.height,
        "geos": geos,
        "years": {"min": years_min, "max": years_max},
        "issues": issues,
    }

def load(df_core: pl.DataFrame, df_alliv: pl.DataFrame, batch: str):
    with engine.begin() as conn:
        # -------- geographies
        geos = df_core.select(["geo_code", "geo_name", "geo_type"]).unique()
        geo_payload = [{"code": c, "name": n, "type": t}
                       for c, n, t in zip(geos["geo_code"], geos["geo_name"], geos["geo_type"])]
        if geo_payload:  # ← guard
            conn.execute(text("""
                INSERT INTO core.geography(geo_code, geo_name, geo_type)
                VALUES (:code,:name,:type)
                ON CONFLICT (geo_code)
                DO UPDATE SET geo_name=EXCLUDED.geo_name, geo_type=EXCLUDED.geo_type;
            """), geo_payload)

        # -------- population facts
        pop_payload = [{"code": c, "year": int(y), "pop": int(p)}
                       for c, y, p in zip(df_core["geo_code"], df_core["year"], df_core["population"])]
        if pop_payload:  # ← guard
            conn.execute(text("""
                INSERT INTO core.population_observations(geo_code, year, population)
                VALUES (:code, :year, :pop)
                ON CONFLICT (geo_code, year)
                DO UPDATE SET population = EXCLUDED.population;
            """), pop_payload)

        # -------- indicator mirrors (ACS1/ACS5)
        if not df_alliv.is_empty():
            df_iv = df_alliv.with_columns(
                pl.when(pl.col("source") == "ACS1")
                  .then(pl.lit("ACS1_TOTAL_POP"))
                  .otherwise(pl.lit("ACS5_TOTAL_POP"))
                  .alias("indicator_code")
            )
            iv_payload = [{"code": c, "year": int(y), "icode": ic,
                           "val": float(v), "src": s, "batch": batch}
                          for c, y, v, s, ic in zip(
                              df_iv["geo_code"], df_iv["year"], df_iv["population"],
                              df_iv["source"], df_iv["indicator_code"]
                          )]
            if iv_payload:  # ← guard
                conn.execute(text("""
                    INSERT INTO core.indicator_values
                        (geo_code, year, indicator_code, value, source, unit, etl_batch_id)
                    VALUES
                        (:code, :year, :icode, :val, :src, 'count', :batch)
                    ON CONFLICT (geo_code, year, indicator_code)
                    DO UPDATE SET value = EXCLUDED.value,
                                  etl_batch_id = EXCLUDED.etl_batch_id;
                """), iv_payload)

        # Only refresh if something changed (optional optimization)
        if pop_payload or (not df_alliv.is_empty()):
            conn.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY ml.feature_matrix;"))

# #DEBUG HELPER FUNCTION
# def print_yearly_summary(df: pl.DataFrame, label: str):
#     if df.is_empty():
#         print(f"[SUM] {label}: 0 rows")
#         return
#     g = (df.group_by(["year", "source"])
#            .len()
#            .sort(["year", "source"]))
#     print(f"[SUM] {label} (Year | Source | Rows):")
#     print(g)

    # #DEBUG
    # print(f"[DBG] years1={years1} ({len(years1)}), years5={years5} ({len(years5)})")
    # print(f"[DBG] df1.height={df1.height}, df5.height={df5.height}, df_all.height={df_all.height}")
    # print("[DBG] counts by source:")
    # print(df_all.group_by("source").len().sort("source"))


def main():
    batch = batch_id("acs_multi")
    os.environ["PPP_BATCH_ID"] = batch  # ensure RAW call/row logs carry this batch id

    years1 = [y for y in range(ACS1_START, ACS_END + 1) if _avail(y, "acs1")]
    years5 = [y for y in range(ACS5_START, ACS_END + 1) if _avail(y, "acs5")]

    df1 = extract_series(years1, "acs1", include_counties=INCLUDE_ACS1_COUNTIES)
    df5 = extract_series(years5, "acs5", include_counties=True)

    df_all = pl.concat([df1, df5], how="diagonal_relaxed")
    df_core = choose_canonical(df_all)

    rep_core = validate(df_core)
    rep_all  = validate(df_all)
    write_json({
        "core": rep_core, "all_iv": rep_all,
        "years": {
            "acs1": (min(years1) if years1 else None, max(years1) if years1 else None),
            "acs5": (min(years5) if years5 else None, max(years5) if years5 else None),
        }},
        artifacts_dir() / f"acs_multi_validation_{batch}.json"
    )

    write_stg_frame(
        "stg.acs_population",
        df_all.select(["geo_code","geo_name","geo_type","year","source","population"]),
        unique_cols=["geo_code","year","source"],
        batch_id=batch
    )

    load(df_core, df_all, batch)

    print(f"ACS multi-year load: core={df_core.height} rows; iv={df_all.height} rows; "
          f"geos={int(df_core['geo_code'].n_unique())}; "
          f"years1={years1[:3]}...{years1[-3:] if years1 else []}; "
          f"years5={years5[:3]}...{years5[-3:] if years5 else []}.")

if __name__ == "__main__":
    main()