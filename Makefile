# --- config -------------------------------------------------
SHELL := /bin/bash
ENV ?= .env
COMPOSE := docker compose --env-file $(ENV)
API := $(COMPOSE) exec -T api
DB  := $(COMPOSE) exec -T db

# Default targets show help
.DEFAULT_GOAL := help

help: ## Show available make targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'

# --- infra lifecycle ---------------------------------------
up: ## Build and start all core services
	$(COMPOSE) up -d --build

down: ## Stop services (keep volumes)
	$(COMPOSE) down

clean: ## Stop services and wipe volumes (DB reset!)
	$(COMPOSE) down -v

logs: ## Tail all service logs
	$(COMPOSE) logs -f

ps: ## List services
	$(COMPOSE) ps

db-shell: ## Open psql inside the db container
	$(DB) psql -U $$POSTGRES_USER -d $$POSTGRES_DB

refresh-mv: ## Refresh feature matrix + headline MVs
	$(DB) psql -U $$POSTGRES_USER -d $$POSTGRES_DB -c "REFRESH MATERIALIZED VIEW CONCURRENTLY ml.feature_matrix;"
	$(API) python -c "from ppp_ml.artifacts import refresh_headline; refresh_headline(True)"

# --- ETL ----------------------------------------------------
etl-acs: ## Ingest ACS population (entry point from pyproject)
	$(API) ppp-etl-acs-pop

etl-bls: ## Ingest BLS LAUS
	$(API) ppp-etl-bls-laus

etl-fred: ## Ingest FRED CPI (example)
	$(API) ppp-etl-fred-cpi

etl-all: etl-acs etl-bls etl-fred refresh-mv ## Run all ETL jobs then refresh MVs

# --- ML / drift --------------------------------------------
train: ## Train all geos for all models (hash-aware drift check)
	$(API) ppp-ml-train-all-models --min-years 8 --split-year 2020 --horizon 10

train-dry: ## Print what would train without running any jobs
	$(API) ppp-ml-train-all-models --min-years 8 --split-year 2020 --horizon 10 --dry-run

force-train: ## Retrain regardless of stored hash (emergency)
	$(API) ppp-ml-train-all-models --min-years 8 --split-year 2020 --horizon 10 --force

scorecard: ## Quick API smoke checks
	@curl -sf http://localhost:$${API_PORT:-8000}/health && echo " API /health OK"
	@curl -sf http://localhost:$${API_PORT:-8000}/scorecard?geo=US | jq . >/dev/null && echo " /scorecard OK"

# --- full-loop runners -------------------------------------
bootstrap: up wait-db etl-all train scorecard ## Bring up, load data, train, smoke test

wait-db: ## Block until DB is ready
	@echo "Waiting for Postgres…" ; \
	until $(DB) pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB >/dev/null 2>&1; do sleep 1; done ; \
	echo "Postgres is ready."