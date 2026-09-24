# Common commands. Run `make help` to list them.

PYTHON ?= .venv/bin/python
MANAGE  = cd backend && ../$(PYTHON) manage.py

.PHONY: help install openfold migrate run test lint format

help:  ## List the available commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

install:  ## Create .venv and install the web app dependencies
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements-dev.txt

openfold:  ## Clone and install OpenFold3-MLX (pinned version + patches)
	./scripts/setup_openfold.sh

migrate:  ## Create / update the SQLite database
	$(MANAGE) migrate

run: migrate  ## Start the web app on http://127.0.0.1:8000
	$(MANAGE) runserver

test:  ## Run the test suite (OpenFold is not needed)
	$(MANAGE) test predictor

lint:  ## Check code style and common mistakes
	$(PYTHON) -m ruff check backend openfold_worker

format:  ## Auto-fix what the linter can fix
	$(PYTHON) -m ruff check --fix backend openfold_worker
