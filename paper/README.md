# Paper source

The paper has two document entry points: `main.tex` for the PRL Letter and
`supplemental.tex` for the standalone Supplemental Material. Each file's
`\input` list mirrors its reading order.

## Layout

```text
paper/
  main.tex                 document class and ordered file list
  supplemental.tex         standalone Supplemental Material entry point
  preamble.tex             packages and PDF metadata
  references.bib           bibliography used by the manuscript
  frontmatter/
    abstract.tex
  sections/                numbered files for the main text
    01_introduction.tex
    02_pulse_protocol.tex
    03_spectroscopy_results.tex
    04_robustness_and_limits.tex
    05_conclusion.tex
  backmatter/
    acknowledgements.tex
  appendices/              supporting sections included by supplemental.tex
    A_measurement_setup.tex
    B_t2_limit.tex
    C_cutoff.tex
    D_adiabatic_basis.tex
    E_pulse_length.tex
    G_numerical_model.tex
    H_lorentzian_echo_comparison.tex
    I_simulation_experiment_comparison.tex
  figures/                 manuscript figures
  data/                    documented numerical and experimental paper data
  notes/                   review and revision notes
  archive/                 preserved legacy source material
```

## Build

From the repository root:

```bash
make paper
```

The PDFs are written to `paper/main.pdf` and `paper/supplemental.pdf`.
To build only one document, use `make paper-main` or
`make paper-supplemental`. To remove intermediate LaTeX files while keeping
the PDFs:

```bash
make paper-clean
```

## Reviewer data package

Migrated figure workflows write the arrays used in the Letter and Supplemental
Material to `paper/data/numerical/` and `paper/data/experimental/`. Every NPZ
archive has a JSON provenance sidecar. See `paper/data/README.md` for the data
contract and current migration status.

Figure generators refresh their rendered assets and documented data together;
do not copy build caches from `figures/paper/` into the reviewer package by
hand. Refresh the populated core package with
`make PYTHON=.venv/bin/python paper-data-core`; use the `paper-data` target to
also run the longer migrated simulations.

The measured main-text Figure 2 maps are generated from the six imported q1
OPX1000 runs and batch provenance under
`data/experimental/2026-08-25/six_detuning_amplitude_sweeps/` with:

```bash
PYTHONPATH=. MPLBACKEND=Agg python scripts/make_main_central_spectroscopy_experiment.py
```

The script validates the broad/narrow grids, 2000-shot counts, pulse parameters,
array orientation, and saved active-$x180$ calibrations before writing PDF,
PNG, and SVG assets to `figures/paper/`.

The experiment--simulation slice figure uses the q6 fixed-amplitude campaign
at 3, 20, and 40 MHz from the sibling `opx1000-codes` checkout.  Its documented
CSV and provenance record the common empirical display alignment applied only
for line-shape comparison.  After refreshing those imported arrays, render the
paper-styled figure with:

```bash
PYTHONPATH=. MPLBACKEND=Agg python scripts/make_figure3_left_exp_sim.py
```

The experimental long-pulse comparison is generated from the sibling
`data_opx1000` repository with:

```bash
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_long_pulse_lorentzian_comparison.py
```

Set `OPX1000_DATA_DIR` when the data repository is not located beside this
checkout. The script writes PDF, PNG, and SVG versions to `figures/paper/` and
the plotted arrays to `paper/data/experimental/`.

The experimental broad and focused cutoff maps are generated from the
`cutoff_amp_fwhm_map` and `echo_lorentzian_cutoff_sweep` campaigns with:

```bash
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_echo_lorentzian_cutoff_sweep.py
```

The script accepts `OPX1000_DATA_DIR`, applies the documented fit-quality
screen, and writes PDF, PNG, SVG, and a provenance JSON file to
`figures/paper/`, together with a portable data/provenance pair in
`paper/data/experimental/`.

The main-text echo-root-Lorentzian resolution and contrast comparison for
$L=5$, 10, 15, 20, and $30~\mu\mathrm{s}$ is generated with:

```bash
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_duration_resolution_comparison.py
```

It uses fixed cutoff $c=0.005$, fits the archived q1 OPX1000 duration series,
and compares it with decohering two-level Bloch simulations. It writes PDF,
PNG, SVG, compressed source arrays, and a best-operating-point CSV to
`figures/paper/`. Set `OPX1000_DATA_DIR` to use the source data checkout
instead of the archived run bundles.

The fitted-center stability panel for the measured amplitude operating window
is generated with:

```bash
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_amplitude_center_stability.py
```

It applies the Supplemental Gaussian estimator and center-independent quality
criteria, then writes PDF, PNG, and SVG versions to `figures/paper/`.

The square, one-column main-text three-level AC-Stark shift comparison is
generated with:

```bash
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_main_ac_stark_shifts.py
```

It reproduces the constant-drive dressed center and the finite-pulse root- and
echo-root-Lorentzian feature positions, and writes PDF, PNG, and SVG versions
to `figures/paper/`.

## Editing conventions

- Keep one logical section per file.
- Prefix main-text filenames with their two-digit reading order.
- Prefix appendix filenames with their appendix letter.
- Put packages and document-wide settings in `preamble.tex`, not section files.
- Put citations in `references.bib`; `archive/legacy_zotero_export.bib` is kept
  only as a source for entries that may be imported later.
- Keep all references cited by the Supplemental Material in the main Letter's
  reference list, as required by APS.
- Keep temporary or superseded figures under `figures/archive/`.
