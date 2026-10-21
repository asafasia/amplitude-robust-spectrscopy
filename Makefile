.PHONY: install install-dev test lint paper paper-main paper-supplemental \
	paper-clean paper-data paper-data-core paper-figures-supplemental

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

paper-figures-supplemental:
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_supplemental_figures.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_supplemental_extended_cutoff.py

paper-clean:
	latexmk -c -cd paper/main.tex
	latexmk -c -cd paper/supplemental.tex
	$(RM) paper/main.bbl
	$(RM) paper/mainNotes.bib
	$(RM) paper/supplemental.bbl
	$(RM) paper/supplementalNotes.bib

paper-data-core:
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_main_ac_stark_correction_maps.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_main_ac_stark_shifts.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_main_drag_summary.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_main_pulse_improvement.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_main_duration_cutoff_scan.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_long_pulse_lorentzian_comparison.py

paper-data: paper-data-core
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_simulated_echo_lorentzian_duration_cutoff_comparison.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_duration_resolution_comparison.py
