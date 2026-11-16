.PHONY: install install-dev test lint paper paper-main paper-supplemental \
	paper-clean paper-data paper-data-core paper-figures-supplemental \
	paper-main-assets paper-supplemental-assets

PYTHON ?= python

MAIN_FIGURES := \
	01_pulse_mechanism.pdf \
	02_latest_q1_spectroscopy.pdf \
	03_latest_q1_pulse_improvement.pdf \
	04_duration_cutoff_scan.pdf
MAIN_LOCAL_FIGURES := $(addprefix paper/figures/generated/,$(MAIN_FIGURES))

SUPPLEMENTAL_FIGURES := \
	supplemental/s01_measurement_chain.pdf \
	supplemental/s02_spectrum_slices.pdf \
	supplemental/s03_duration_resolution.pdf \
	supplemental/midpoint_smoothing_leakage.pdf \
	supplemental/drag_beta_optimization.pdf \
	supplemental/s08_pulse_modifications.pdf \
	supplemental/edge_smoothing_results.pdf \
	supplemental/square_echo_comparison.pdf \
	supplemental/ramsey_fft_linewidth.pdf \
	supp_extended_cutoff_duration.pdf
SUPPLEMENTAL_LOCAL_FIGURES := $(addprefix paper/figures/generated/,$(SUPPLEMENTAL_FIGURES))

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m unittest discover -s tests

lint:
	ruff check .

paper: paper-main paper-supplemental

paper-main: paper-main-assets
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex

paper-supplemental: paper-supplemental-assets
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/supplemental.tex

paper-supplemental-assets: $(SUPPLEMENTAL_LOCAL_FIGURES)

paper-main-assets: $(MAIN_LOCAL_FIGURES)

paper/figures/generated/%.pdf: figures/paper/%.pdf
	@mkdir -p "$(@D)"
	cp "$<" "$@"

paper-figures-supplemental:
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_supplemental_figures.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_supplemental_extended_cutoff.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_supplemental_square_echo.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_supplemental_ramsey_fft.py

paper-clean:
	latexmk -c -cd paper/main.tex
	latexmk -c -cd paper/supplemental.tex
	$(RM) paper/main.bbl
	$(RM) paper/mainNotes.bib
	$(RM) paper/supplemental.bbl
	$(RM) paper/supplementalNotes.bib

paper-data-core:
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_supplemental_ramsey_fft.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_supplemental_square_echo.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_latest_paper_measurements.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_main_ac_stark_correction_maps.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_main_ac_stark_shifts.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_main_drag_summary.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_main_pulse_improvement.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_main_duration_cutoff_scan.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_long_pulse_lorentzian_comparison.py

paper-data: paper-data-core
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_supplemental_spectrum_slices.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_current_drag_supplement.py --simulate
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_current_edge_supplement.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_simulated_echo_lorentzian_duration_cutoff_comparison.py
	PYTHONPATH=. MPLBACKEND=Agg $(PYTHON) scripts/make_duration_resolution_comparison.py
