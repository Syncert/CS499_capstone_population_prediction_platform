SHELL := /bin/bash
COMPOSE := docker compose --env-file .env
API := $(COMPOSE) exec -T api
DB  := $(COMPOSE) exec -T db

up:        ## Build & start
	$(COMPOSE) up -d --build
down:      ## Stop
	$(COMPOSE) down
clean:     ## Stop & wipe volumes (DB reset!)
	$(COMPOSE) down -v
logs:      ## Follow logs
	$(COMPOSE) logs -f

wait-db:   ## Wait for Postgres
	@until $(DB) pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB >/dev/null 2>&1; do sleep 1; done

etl-all:   ## Run all ETL jobs
	$(API) ppp-etl-acs-pop
	$(API) ppp-etl-bls-laus
	$(API) ppp-etl-fred-cpi

train:     ## Train all geos/models (drift-aware)
	$(API) ppp-ml-train-all-models --min-years 8 --split-year 2020 --horizon 10

train-dry: ## Dry-run training plan
	$(API) ppp-ml-train-all-models --dry-run

bootstrap: ## Fresh start -> ETL -> train -> smoke test
	$(COMPOSE) down -v
	$(COMPOSE) up -d --build
	$(MAKE) wait-db
	$(MAKE) etl-all
	$(MAKE) train
	@curl -sf http://localhost:$${API_PORT:-8000}/health >/dev/null && echo "API healthy ✅"