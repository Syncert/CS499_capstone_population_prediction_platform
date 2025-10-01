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

# ───────── CRON helpers (host) ─────────
CRON_MINUTE ?= 30
CRON_HOUR   ?= 2
CRON_SHELL  := /bin/bash
CRON_PATH   := /usr/local/bin:/usr/bin:/bin
REPO_ABS    := $(shell pwd -P)
CRON_LINE   := $(CRON_MINUTE) $(CRON_HOUR) * * * $(REPO_ABS)/docker/cron_train.sh

cron-install: ## Install nightly cron (02:30 by default). Override CRON_HOUR/MINUTE if desired.
	@echo "Installing cron entry for repo: $(REPO_ABS)"
	@mkdir -p "$(REPO_ABS)/logs"
	@(crontab -l 2>/dev/null; \
	  echo "SHELL=$(CRON_SHELL)"; \
	  echo "PATH=$(CRON_PATH)"; \
	  echo "$(CRON_LINE)";) | crontab -
	@echo "Installed:"
	@crontab -l | sed -n '1,3p;$$p'

cron-remove: ## Remove our cron entry
	@echo "Removing cron entry for repo: $(REPO_ABS)"
	@{ crontab -l 2>/dev/null | grep -v "$(REPO_ABS)/docker/cron_train.sh" || true; } | crontab -
	@echo "Remaining crontab (if any):"
	-@crontab -l

cron-now: ## Run the cron job immediately (no schedule)
	@"$(REPO_ABS)/docker/cron_train.sh"

cron-test: ## Temporarily install an every-2-min schedule (easy to verify), then tail logs
	@echo "Installing every-2-min test cron…"
	@(crontab -l 2>/dev/null | grep -v "$(REPO_ABS)/docker/cron_train.sh" || true; \
	  echo "SHELL=$(CRON_SHELL)"; \
	  echo "PATH=$(CRON_PATH)"; \
	  echo "*/2 * * * * $(REPO_ABS)/docker/cron_train.sh";) | crontab -
	@echo "Tailing logs. Ctrl+C to stop."
	@touch "$(REPO_ABS)/logs/train_$$(date +%F).log"
	@tail -f "$(REPO_ABS)/logs/train_$$(date +%F).log"