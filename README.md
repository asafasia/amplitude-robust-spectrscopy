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
styles/            Matplotlib styles for paper and IPS presentation figures
assets/poster/     poster-specific plotting assets
archive/           old scripts, scratch work, and superseded experiments
docs/              project documentation
data/              local data, ignored by git by default
```

See `docs/PROJECT_ORGANIZATION.md` for the target repository structure and
cleanup rules.

## Figure Workflows

Paper and IPS presentation figures should be developed in parallel but share
the same computation code:

```text
notebooks/paper/             paper-oriented notebooks
notebooks/presentation_ips/  slide-oriented notebooks
figures/paper/               paper figure exports
figures/presentation_ips/    IPS presentation figure exports
styles/paper.mplstyle        compact publication style
styles/presentation_ips.mplstyle
```

Use the shared figure helpers in notebooks:

```python
from echospec.figures import FigureVariant, apply_figure_style, save_figure

VARIANT = FigureVariant.PRESENTATION_IPS
apply_figure_style(VARIANT)

# build fig...
save_figure(fig, "figure_name", variant=VARIANT)
```

Keep data loading, simulation, and reusable plotting functions in `echospec/`.
Keep notebooks focused on choosing the figure target, arranging panels, and
exporting the result.

## Notes

- Keep reusable logic out of notebooks when it becomes part of a repeated
  workflow.
- Prefer adding a small test before changing shared simulation, pulse, fitting,
  or result-processing behavior.
- Keep raw data and large generated arrays outside git unless they are deliberate
  small fixtures.
