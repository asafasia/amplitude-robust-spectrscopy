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

The supplemental spectrum slices (Fig. S2) use five acquired rows from the
primary-qubit narrow-detuning scans also shown in main-text Fig. 2. They target
3, 10, 20, 30, and 40 MHz; the captions give the actual acquired values.
The simulator evaluates those exact drives and detunings. Each experiment and
simulation trace is independently mean-centered, then stacked on a common
population scale without gain or frequency fitting. Raw populations, display
arrays, selection indices, and alignment metadata are retained alongside the
figure. Rebuild its data and rendering with:

```bash
PYTHONPATH=. MPLBACKEND=Agg python scripts/make_supplemental_spectrum_slices.py
```

To render just the reordered Figs. S2 and S3 from the saved arrays:

```bash
PYTHONPATH=. MPLBACKEND=Agg python scripts/make_supplemental_figures.py --only spectrum_slices duration_metrics
```

The earlier three-amplitude q6 slice CSV and its display alignment remain
historical artifacts; they are not the source of the current Fig. S2.


The experimental long-pulse comparison is generated from the sibling
`data_opx1000` repository with:

```bash
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_long_pulse_lorentzian_comparison.py
```

Set `OPX1000_DATA_DIR` when the data repository is not located beside this
checkout. The script writes PDF, PNG, and SVG versions to `figures/paper/` and
the plotted arrays to `paper/data/experimental/`.

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

The stacked, one-column Figure 5 center-shift and FWHM comparison is
generated with:

```bash
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_main_ac_stark_shifts.py
```

It uses the exact measured beta=0 and beta=-0.22 centers displayed in Figure 4
and retains the original simulated constant-drive dressed center. It writes
PDF, PNG, and SVG assets, plus separate experimental and numerical datasets
and provenance under `paper/data/`. Regenerate Figure 4 first when its
source data or center selection changes.
The lower panel fits Gaussian-dip linewidths from the same stored measured
spectra, using the source campaign's fitting procedure.

The supplemental DRAG coefficient figure combines a resolved simulation scan,
waveform quadratures, saved simulated centers, and the available two-setting
experimental comparison. Reproduce it with:

```bash
.venv/bin/python scripts/scan_drag_beta_center.py
MPLCONFIGDIR=/private/tmp/ars-matplotlib-cache .venv/bin/python scripts/make_supplemental_figures.py --only drag_beta_optimization
make paper-supplemental
```

The scan minimizes RMS central-dip detuning over seven amplitudes for the
20 us, cutoff 0.005, 16 ns sine-smoothed echo. Its saved arrays and numerical
convergence checks are under `outputs/drag_beta_center_scan/`; the figure arrays
and provenance are under `figures/paper/supplemental/drag_beta_optimization*`.
The lowest sampled coefficient is -0.495, close to the half-derivative value
-0.5. The experimental panel uses the existing 50 us, cutoff 0.00075 cache at
beta 0 and -0.22 with different midpoint smoothing. It is not an experimental
beta optimization, and its coefficients are quoted in the acquisition convention.

The supplemental edge-smoothing comparison uses the saved
`03_pulse_improvement.npz` hard-cutoff and cutoff-subtracted polynomial (p=5)
pulses, both at beta -0.5. Render it with:

```bash
MPLCONFIGDIR=/private/tmp/ars-matplotlib-cache .venv/bin/python scripts/make_supplemental_figures.py --only edge_smoothing_results
```

The four panels show the 100 MHz spectrum and drive-dependent linewidth,
contrast, and ideal SNR. SNR assumes 1000 total ideal binary measurements:
500 at the dip and 250 at each +/-0.45 MHz shoulder. It is not measured SNR.
Arrays, metrics, source hashes, and high-drive summaries are saved as
`figures/paper/supplemental/edge_smoothing_results*`. Older edge-ramp reports
also contain a plain polynomial window without cutoff subtraction; those
arrays describe a different pulse and are not substituted here.

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

The measured Figure 4 compares q6 beta=0 and beta=-0.22 from the September
5--6, 2026 calibration campaign. Regenerate its figure and experimental data
package with `python scripts/make_main_ac_stark_correction_maps.py`.
See `data/README.md` for raw-source import and fit provenance.


## Dense q6 pulse-length comparison

The added Supplemental section uses the September 9--10, 2026 q6 campaign
from `opx1000-codes/data/pulse_length_spectroscopy/overnight_dense_20260909`.
It includes combined linewidth-versus-Rabi-frequency curves through 25 us
and a standalone paired-linewidth plot, with a campaign-specific T2* reference
of 51.996 kHz (not the paper's q1 normalization).

```bash
python scripts/make_dense_pulse_length_comparison.py
```

The default uses the portable paper data package once available. To reimport
saved source analysis, use `--source-data-dir /path/to/opx1000-codes/data`
or `OPX1000_DATA_DIR`. The generator writes PDF, PNG and SVG figures and
refreshes `12_dense_pulse_length.npz` plus JSON provenance separately under
`paper/data/experimental/` and `paper/data/numerical/`. It preserves the
source half-height widths and quality masks; it does not refit spectra.
Build the supplement with `make paper-supplemental`.
