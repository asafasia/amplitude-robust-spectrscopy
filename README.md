# Amplitude-Robust Spectroscopy

Simulation, analysis, and plotting utilities for amplitude-robust spectroscopy
experiments.

The reusable Python code lives primarily in `echospec/`. Notebooks, generated
figures, poster assets, and archived scripts are kept separate so the package can
be installed and tested independently from exploratory work.

## Setup

Create an environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For a runtime-only install:

```bash
python -m pip install -e .
```

## Common Commands

Run the tests:

```bash
python -m unittest discover -s tests
```

Run the linter:

```bash
ruff check .
```

Open notebooks:

```bash
jupyter lab
```

## Project Layout

```text
echospec/          reusable simulation, experiment, analysis, and plotting code
data_loader/       existing data-loading helpers pending migration into echospec
tests/             fast tests for shared behavior
notebooks/         exploratory and figure-production notebooks
scripts/           maintained scripts and archived legacy scripts
figures/           generated and final figures
graphs_for_poster/ poster-specific plotting assets
docs/              project documentation
data/              local data, ignored by git by default
```

See `docs/PROJECT_ORGANIZATION.md` for the target repository structure and
cleanup rules.

## Notes

- Keep reusable logic out of notebooks when it becomes part of a repeated
  workflow.
- Prefer adding a small test before changing shared simulation, pulse, fitting,
  or result-processing behavior.
- Keep raw data and large generated arrays outside git unless they are deliberate
  small fixtures.
