.PHONY: install install-dev test lint

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m unittest discover -s tests

lint:
	ruff check .
