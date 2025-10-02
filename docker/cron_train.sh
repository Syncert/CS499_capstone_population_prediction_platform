#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root from this script’s location: repo/docker/cron_train.sh → repo/
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Cron has a tiny PATH; ensure docker is reachable
export SHELL=/bin/bash
export PATH="/usr/local/bin:/usr/bin:/bin"

LOG_DIR="${REPO_DIR}/logs"
ENV_FILE="${REPO_DIR}/.env"

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

# Ensure stack is up (idempotent)
docker compose --env-file "${ENV_FILE}" up -d

# Drift-aware trainer (only retrains when data hash changed)
docker compose --env-file "${ENV_FILE}" exec -T api \
  ppp-ml-train-all-models --min-years 8 --split-year 2020 --horizon 10 \
  >> "${LOG_DIR}/train_$(date +%F).log" 2>&1