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
	$(RUFF) check .

format: $(VENV)/bin/activate
	$(RUFF) check . --fix || true
	$(RUFF) format .
	$(BLACK) .

typecheck: $(VENV)/bin/activate
	$(MYPY) .

test: $(VENV)/bin/activate
	$(PYTEST) -q --cov=.

run-api: $(VENV)/bin/activate
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
