# Repository Guidelines

## Project Structure & Module Organization

Reusable Python code belongs in `echospec/`; regression tests live in `tests/`
and `tests/analysis/`. Put reproducible figure generators in `scripts/` and
exploratory work in `notebooks/`. Manuscript sources are under `paper/`, paper
figures under `figures/paper/`, and Matplotlib styles under `styles/`. Treat
`archive/` as historical reference rather than active code.

## Build, Test, and Development Commands

Create and activate a virtual environment, then install development tools:

```bash
python -m venv .venv
source .venv/bin/activate
make install-dev
```

- `python -m pytest` runs the complete test suite, including pytest-style tests.
- `make test` runs the legacy `unittest` discovery suite.
- `make lint` checks Python with Ruff.
- `make paper` builds both `paper/main.pdf` and `paper/supplemental.pdf` with
  `latexmk`; use `make paper-main` for the Letter only.
- `make paper-data-core` regenerates the core documented figure data. Longer
  simulations are included in `make paper-data`.
- `jupyter lab` opens the notebook workflows.

## Coding Style & Naming Conventions

Use Python 3.10 or newer, four-space indentation, and an 88-character target
line length. Run the configured Ruff checks before submitting. Use `snake_case`
for modules, functions, and variables; `PascalCase` for classes; and descriptive
`test_<behavior>` names. Keep reusable computation and data loading out of
notebooks. Manuscript section filenames use two-digit order prefixes, such as
`paper/sections/03_spectroscopy_results.tex`.

## Testing Guidelines

Add focused tests whenever shared simulation, fitting, pulse, or data-export
behavior changes. Prefer small deterministic synthetic arrays and NumPy
assertions; avoid depending on large local measurement files. There is no
configured coverage threshold, so prioritize meaningful regression cases.
Run `python -m pytest` and `make lint`; rebuild affected figures or PDFs when
their sources change.

## Data, Figures, and Provenance

OPX1000 measurements live in the sibling repository
`/Users/asafsolonnikov/Developer/data_opx1000`; treat it as read-only source
data. Keep reproducible analysis and plotting code here, writing derived paper
figures to `figures/paper/` or `paper/figures/` as appropriate. Plotting scripts
should accept `OPX1000_DATA_DIR` when practical and use the sibling repository
as the local default. Do not commit large raw acquisitions or temporary build
products. Update rendered figures and their documented arrays or provenance
sidecars together.

## Commit & Pull Request Guidelines

Use short, imperative commit subjects, for example `Add script and provenance
file for Figure 3`, and keep each commit scoped. Pull requests should explain
the goal, list validation commands and regenerated artifacts, and include
previews for visual changes. Link relevant issues or review notes and call out
experimental-data assumptions.
