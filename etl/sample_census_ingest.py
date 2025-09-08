import os
import datetime
import requests
import psycopg2
from typing import Tuple, Any, Dict
from dotenv import load_dotenv

# Load .env from repo root if present
load_dotenv()

PG_DSN = os.getenv("PG_DSN")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")
BLS_KEY = os.getenv("BUREAU_LABOR_STATISTICS_KEY", "")

ACS_YEAR = int(os.getenv("ACS_YEAR", "2021"))
BATCH = f"baseline_{datetime.date.today().isoformat()}"

def fetch_acs_us_total_pop(year: int, api_key: str | None = None) -> int:
    """US total population from ACS 1-year (B01001_001E)."""
    base = f"https://api.census.gov/data/{year}/acs/acs1"
    params: Dict[str, str] = {"get": "NAME,B01001_001E", "for": "us:1"}
    if api_key:
        params["key"] = api_key  # str is fine here
    r = requests.get(base, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    header, rows = js[0], js[1:]
    pop_idx = header.index("B01001_001E")
    return int(rows[0][pop_idx])

def fetch_bls_unemployment(api_key: str | None = None) -> Tuple[float, int]:
    """
    Latest US unemployment rate (series LNS14000000, seasonally adjusted).
    Returns (rate_percent, year).
    """
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

    # IMPORTANT: typed as Dict[str, Any] so Pylance doesn't force list[str] values
    payload: Dict[str, Any] = {"seriesid": ["LNS14000000"]}
    if api_key and api_key != "__set_in_local_env__":
        payload["registrationkey"] = api_key  # string is valid here

    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    js = r.json()
    latest = js["Results"]["series"][0]["data"][0]
    return float(latest["value"]), int(latest["year"])

def upsert_population(conn, geo_code: str, year: int, population: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into core.population_observations(geo_code, year, population)
            values (%s, %s, %s)
            on conflict (geo_code, year)
            do update set population = excluded.population
            """,
            (geo_code, year, population),
        )

def upsert_indicator(conn, geo_code: str, year: int, code: str, value: float,
                     source: str, unit: str, batch: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "select core.upsert_indicator_value(%s,%s,%s,%s,%s,%s,%s);",
            (geo_code, year, code, value, source, unit, batch),
        )

def refresh_feature_matrix(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY ml.feature_matrix;")

def main() -> None:
    assert PG_DSN, "PG_DSN not set. Put it in .env (host=localhost if running script on host)."

    # connect once, autocommit on for convenience
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True
    try:
        # Census ingest
        pop = fetch_acs_us_total_pop(ACS_YEAR, CENSUS_API_KEY)
        print(f"Fetched ACS {ACS_YEAR} US population: {pop}")
        upsert_population(conn, "US", ACS_YEAR, pop)
        upsert_indicator(conn, "US", ACS_YEAR, "ACS_TOTAL_POP", pop, "ACS", "count", BATCH)

        # BLS ingest
        unrate, year = fetch_bls_unemployment(BLS_KEY)
        print(f"Fetched BLS {year} unemployment rate: {unrate}%")
        upsert_indicator(conn, "US", year, "BLS_UNRATE", unrate, "BLS", "percent", BATCH)

        # Refresh MV
        refresh_feature_matrix(conn)
        print("DB upserts complete and feature matrix refreshed.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
