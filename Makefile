.PHONY: install install-dev test lint paper paper-main paper-supplemental paper-clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m unittest discover -s tests

lint:
	ruff check .

paper: paper-main paper-supplemental

paper-main:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex

paper-supplemental:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/supplemental.tex

paper-clean:
	latexmk -c -cd paper/main.tex
	latexmk -c -cd paper/supplemental.tex
	$(RM) paper/main.bbl
	$(RM) paper/mainNotes.bib
	$(RM) paper/supplemental.bbl
	$(RM) paper/supplementalNotes.bib
