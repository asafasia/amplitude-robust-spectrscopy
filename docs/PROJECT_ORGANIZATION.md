# Project Organization

This repository is a research-code project with a reusable Python package at its
core. Keep stable code, exploratory work, generated data, and presentation
assets separate so experiments remain reproducible.

## Target layout

```text
amplitude-robust-spectroscopy/
  echospec/              # reusable package code
  tests/                 # fast unit and regression tests
  notebooks/
    exploratory/         # active exploration
    paper/               # notebooks used to produce paper figures
    archive/             # old notebooks kept for reference
  scripts/               # maintained command-line workflows
  examples/              # small reproducible examples
  figures/
    generated/           # reproducible generated output
    final/               # final exported figures
  assets/
    poster/              # poster-specific assets
    paper/               # paper-specific assets
  data/                  # local data, ignored by git by default
  docs/                  # project notes and contributor documentation
```

## Repository rules

- Put reusable simulation, analysis, plotting, and result code in `echospec/`.
- Put new tests in `tests/` before changing shared behavior.
- Keep notebooks focused on exploration or figure production; move old notebooks
  into an archive folder instead of leaving copies beside active work.
- Do not commit raw data or large generated arrays unless they are small,
  intentional fixtures.
- Do not keep generated figures in the repository root. Use `figures/generated/`
  for reproducible output and `figures/final/` for publication-ready exports.
- Avoid one-off scripts in the root. Promote useful workflows into `scripts/`
  or package entry points.
