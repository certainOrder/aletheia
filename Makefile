# Simple dev workflow using a local virtualenv

PY=python3.11
VENV=.venv
PIP=$(VENV)/bin/pip
PYTHON=$(VENV)/bin/python
PRECOMMIT=$(VENV)/bin/pre-commit
RUFF=$(VENV)/bin/ruff
BLACK=$(VENV)/bin/black
MYPY=$(VENV)/bin/mypy
PYTEST=$(VENV)/bin/pytest

.PHONY: help
help:
	@echo "Targets: setup, lint, typecheck, format, test, hooks, run-api"

$(VENV)/bin/activate:
	$(PY) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt

setup: $(VENV)/bin/activate hooks

hooks: $(VENV)/bin/activate
	$(PRECOMMIT) install

lint: $(VENV)/bin/activate
	$(RUFF) check app alembic

format: $(VENV)/bin/activate
	$(RUFF) check app alembic --fix || true
	$(RUFF) format app alembic
	$(BLACK) app alembic

typecheck: $(VENV)/bin/activate
	$(MYPY) app

test: $(VENV)/bin/activate
	$(PYTEST) -q --cov=app --cov-report=term-missing --cov-fail-under=85

run-api: $(VENV)/bin/activate
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# --- Docker Compose helpers ---

.PHONY: compose-up compose-down compose-rebuild compose-logs

compose-up:
	docker compose --env-file .env up -d --build

compose-down:
	docker compose down -v

compose-rebuild:
	docker compose build --no-cache
	docker compose up -d

compose-logs:
	docker compose logs -f

# --- Alembic helpers ---

ALEMBIC=$(VENV)/bin/alembic

.PHONY: migrate-rev migrate-up migrate-down

migrate-rev: $(VENV)/bin/activate
	$(ALEMBIC) revision -m "$(m)"

migrate-up: $(VENV)/bin/activate
	$(ALEMBIC) upgrade head

migrate-down: $(VENV)/bin/activate
	$(ALEMBIC) downgrade -1
