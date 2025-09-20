#!/usr/bin/env bash
set -euo pipefail

# Toggles (1=run, 0=skip). Set via `environment:` in compose if needed.
: "${RUN_ACS:=1}"
: "${RUN_LAUS:=1}"
: "${RUN_FRED:=1}"

echo "[ETL] Starting one-shot ETL run…"

if [[ "$RUN_ACS" == "1" ]]; then
  echo "[ETL] ACS population ingest…"
  ppp-etl-acs-pop
fi

if [[ "$RUN_LAUS" == "1" ]]; then
  echo "[ETL] BLS LAUS ingest…"
  ppp-etl-bls-laus
fi

if [[ "$RUN_FRED" == "1" ]]; then
  echo "[ETL] FRED CPI ingest…"
  ppp-etl-fred-cpi
fi

echo "[ETL] All selected jobs completed."
