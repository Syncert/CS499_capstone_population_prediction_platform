from __future__ import annotations
import json, hashlib, datetime as dt
from typing import Any, Iterable
import polars as pl
from sqlalchemy import text
from ppp_common.orm import engine

def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds") + "Z"

def request_hash(endpoint: str, params: dict | None, payload: dict | None) -> str:
    blob = json.dumps({"endpoint": endpoint, "params": params or {}, "payload": payload or {}}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()

def write_raw_event(
    raw_table: str,
    *,
    batch_id: str,
    endpoint: str,
    params: dict | None,
    payload: dict | None,
    response_json: dict | list | None,
    status_code: int | None,
    notes: str | None = None,
):
    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {raw_table}
                (etl_batch_id, request_hash, endpoint, params_json, payload_json,
                 http_status, response_json, notes)
            VALUES
                (:batch, :rh, :ep, :pjs, :bjs, :code, :rjs, :notes)
            ON CONFLICT (request_hash) DO NOTHING;
        """), {
            "batch": batch_id,
            "rh": request_hash(endpoint, params, payload),
            "ep": endpoint,
            "pjs": json.dumps(params or {}, sort_keys=True),
            "bjs": json.dumps(payload or {}, sort_keys=True),
            "code": status_code,
            "rjs": json.dumps(response_json) if response_json is not None else None,
            "notes": notes,
        })

def write_stg_frame(stg_table: str, df: pl.DataFrame, *, unique_cols: list[str], batch_id: str):
    """Idempotent upsert into STG by a unique business key."""
    if df.is_empty():
        return
    df = df.with_columns(pl.lit(batch_id).alias("etl_batch_id"))
    cols = df.columns
    inserts = [dict(zip(cols, row)) for row in df.iter_rows()]
    set_cols = [c for c in cols if c not in unique_cols]
    set_clause = ", ".join([f"{c}=EXCLUDED.{c}" for c in set_cols])
    key = ", ".join(unique_cols)
    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {stg_table} ({", ".join(cols)})
            VALUES ({", ".join([f":{c}" for c in cols])})
            ON CONFLICT ({key}) DO UPDATE SET {set_clause};
        """), inserts)
