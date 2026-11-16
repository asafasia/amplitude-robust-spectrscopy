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

### Editor previews

Both documents use copies of their figure PDFs under
`paper/figures/generated/`, so the built-in editor can compile with `paper/`
as its project root. The original exports, numerical arrays, and provenance
remain under `figures/paper/`; the copies contain identical PDF bytes.
After regenerating a figure, refresh the local copies before
using the editor preview:

```bash
make paper-main-assets paper-supplemental-assets
```

`make paper-main` and `make paper-supplemental` also refresh their copies before compiling.
Keep the copies with the manuscript sources when sharing the `paper/` folder.

## Reviewer data package

### Updated Figure 2 and Figure 3 measurements (31 October-1 November 2026)

The current Letter uses the four completed 5,000-average, 200-by-200 q1
Figure 2 maps from `20261031_002217_547409_optimized_5000/figures2-3`, and
the complete 5,000-average, 300-by-200 Figure 3 set from
`20261031_191600_figure3_50ns_poly2_avg5000_amp300/maps`. Partial acquisitions are
excluded. Figure 2 retains its earlier numerical reference panels, cropped
to the new measured range; Figure 3 is now entirely experimental.

```bash
python scripts/make_latest_paper_measurements.py
make paper
```

The generator uses portable datasets in `paper/data/experimental/` after
their first import. To validate and reimport the completed raw acquisitions,
add `--refresh-source`; `--source-data-dir` or `OPX1000_DATA_DIR` can point to
the q1 campaign-data directory (default `experiments/paper_q1/data`). It
checks raw/processed equality, acquisition completion, settings, grids,
shot counts and waveform-derived peak axes. It performs no acquisition,
new fitting or simulation. Existing model arrays are exported separately
under `paper/data/numerical/02_latest_q1_reference_simulations.*`.
Use `--figure3-only --refresh-source` to replace the Figure 3 data and assets
while keeping the existing Figure 2 exports and figures.
Use `--figure2-only` to regenerate only Figure 2. Its left block groups the
broad and narrow root-Lorentzian maps; its right block groups the corresponding
echo-root-Lorentzian maps, with measurements above reference simulations.
All paper linewidth markers and normalizations use the shared reference in
`echospec/figures/coherence.py`. The value is stated once in the main text;
source calibration and simulation parameters remain in the data provenance.

The current figure assets are `02_latest_q1_spectroscopy` and
`03_latest_q1_pulse_improvement` under `figures/paper/`. Their provenance
sidecars retain the source hashes, fit masks, displayed color limits and
matched-amplitude statistics. The generator also writes
`latest_results_parameters.tex`, keeping quantitative manuscript claims
tied to the exported data. Figure 3 uses 20 us, 50 ns midpoint smoothing,
polynomial power 2 and acquisition beta 1.2557576. Its amplitude axis is
the actual complex-envelope peak, unlike the nominal axis of Figure 2.
Figure 3 displays the 0--60 MHz window (180 acquired amplitude rows), with
the three maps above the resonance-position, contrast and linewidth graphs.
The last panel shows only FWHM/Gamma_T2star, on a 0--12 scale, using
the common 7.6 us reference. The generated manuscript parameters include
normalized median linewidths calculated from the unrounded saved fit widths.
Its reported statistics use the 177 nonzero amplitude rows accepted for all
three pulses in this window. The full 0--100 MHz acquisition and saved fit
masks remain unchanged in the portable export. The shared map color scale is
0--0.8. The reference linewidth is 1/(pi * T2*).
The Figure 3 maps show vertical half-width markers, and the resonance panel
uses the same half-width as horizontal markers. Coherence values are displayed
to one decimal place, with unrounded measurements retained in data provenance.

### Other and archived figure workflows

Migrated figure workflows write the arrays used in the Letter and Supplemental
Material to `paper/data/numerical/` and `paper/data/experimental/`. Every NPZ
archive has a JSON provenance sidecar. See `paper/data/README.md` for the data
contract and current migration status.

Figure generators refresh their rendered assets and documented data together;
do not copy build caches from `figures/paper/` into the reviewer package by
hand. Refresh the populated core package with
`make PYTHON=.venv/bin/python paper-data-core`; use the `paper-data` target to
also run the longer migrated simulations.

The archived August Figure 2 maps are generated from the six imported q1
OPX1000 runs and batch provenance under
`data/experimental/2026-08-25/six_detuning_amplitude_sweeps/` with:

```bash
PYTHONPATH=. MPLBACKEND=Agg python scripts/make_main_central_spectroscopy_experiment.py
```

The script validates the broad/narrow grids, 2000-shot counts, pulse parameters,
array orientation, and saved active-$x180$ calibrations before writing PDF,
PNG, and SVG assets to `figures/paper/`.

The supplemental spectrum slices (Fig. S2) use five acquired rows from the
current Figure 2 q1 narrow-detuning maps. They target
3, 5, 10, 15, and 20 MHz within the displayed drive range, using 20 us
pulses, cutoff 0.005, and an abrupt echo inversion without DRAG or polynomial
shaping. All 200 acquired detunings are retained; labels give the acquired
drive values. The calculation uses the Figure 2 campaign relaxation fit
and the common paper coherence reference.
The simulator evaluates those exact drives and detunings. Each experiment and
simulation trace is independently mean-centered, then stacked on a common
population scale without gain or frequency fitting. Raw populations, display
arrays, selection indices, and alignment metadata are retained alongside the
figure. Small measured maps above the spectra identify the selected rows
with blue dashed guides and matching drive labels. All experimental slices
use blue connected points, simulations use dashed dark curves, and capped
bars show a population difference of 0.2. The maps use the main figure's
`magma` colormap and population scale from 0 to 0.8. Rendering checks that the map rows exactly
match the archived slices and records its sources separately in the S2
provenance sidecar. Rebuild its data and rendering with:

```bash
PYTHONPATH=. MPLBACKEND=Agg python scripts/make_supplemental_spectrum_slices.py
```

To render just the reordered Figs. S2 and S3 from the saved arrays:

```bash
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_supplemental_figures.py --only spectrum_slices duration_metrics
```

Figure S3 uses a compact two-column layout: the constant-drive Bloch
reference spans 1--100 kHz on the left, and the echo simulations and
measurements span 0.1--20 MHz on the right. Both amplitude axes are
logarithmic, with shared vertical scales within each row. The renderer is
`scripts/make_supplemental_duration_resolution.py`; it uses the saved
`03_echo_duration_resolution_comparison.npz` metrics and writes the displayed
arrays and provenance alongside `figures/paper/supplemental/s03_duration_resolution.*`.
No new fits or simulations are needed. The optional preview script uses the
same renderer and writes only to `figures/paper/previews/`.

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

Figure S4 uses the zero-DRAG midpoint-smoothing comparison recomputed through
50 MHz. The 20 us pulse, cutoff 0.005, coherence parameters, and 0/4/8/16 ns
sine transitions match the original comparison. The drive is the nominal
envelope scale, without peak renormalization. Reproduce the calculation and
figure with:

```bash
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_midpoint_leakage_50mhz.py
MPLBACKEND=Agg .venv/bin/python scripts/make_supplemental_figures.py --only midpoint_smoothing_leakage
make paper
```

The portable export is `paper/data/numerical/08_midpoint_smoothing_50mhz.{npz,json}`.
It stores all three final populations on 26 drives from 0 to 50 MHz and 201
detunings within +/-0.5 MHz. The 50 MHz slice is checked using a time step of
0.0625 ns instead of 0.125 ns and 401 detunings. The quoted leakage at 50 MHz
is the maximum final population in the second excited state over detuning;
the separate whole-grid maximum is also retained in the provenance. Neither
quantity is a maximum transient population or a measured leakage probability.
The original 42.72 MHz comparison remains in `outputs/sine_echo_transition_comparison/`.

Figure S6 combines the current q1 measurements from Figure 3 with new
three-level simulations at L=20 us, cutoff 0.005, a 50 ns sine midpoint,
and (where enabled) cutoff subtraction and a power-2 polynomial window.
Both simulations and measurements use the actual complex-envelope peak
amplitude. Reproduce it with:

```bash
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_current_drag_supplement.py --simulate
MPLCONFIGDIR=/private/tmp/ars-matplotlib-cache .venv/bin/python scripts/make_supplemental_figures.py --only drag_beta_optimization
make paper-supplemental
```

Omit `--simulate` to render the saved numerical export. The scan and convergence
checks are in `paper/data/numerical/05_current_drag_comparison.{npz,json}`;
plotted arrays and provenance are in
`figures/paper/supplemental/drag_beta_optimization*`. The numerical objective
is RMS center about zero over six amplitudes from 10 to 60 MHz. The best sampled
coefficients are +0.475 (hard cutoff) and +0.5 (polynomial) in the manuscript
convention `Q=-beta*dI/dt/alpha`, with signed `alpha=omega12-omega01<0`.
The archived polynomial backend and numerical source exports retain
`Q=-beta_source*dI/dt/abs(alpha)`; figure rendering converts
`beta=-beta_source` without changing sample order, waveforms, or populations.
The figure NPZ and provenance use the manuscript beta convention; the source
NPZ and JSON retain their original convention. Step refinement changes the
centers by less than 1 Hz.
The measured panel uses 177 common accepted nonzero rows up to 60 MHz and
subtracts each curve's mean; its RMS variations are approximately 24, 4, and
2 kHz. Its fixed acquisition beta is 1.2557575757575759, displayed as 1.25.
The experimental beta-optimization scan will be supplied separately; it is
not reconstructed from these three fixed-setting measurements. Earlier
simulation surveys retain their independently documented parameter sets.

The archived simulated pulse-shaping figure includes resonance-center traces
of all three saved 2D sweeps. Its arrays remain in use by supplemental
comparisons; they are separate from the new measured Figure 3. Regenerate
that simulation figure, center arrays, metrics, and provenance with:

```bash
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_main_pulse_improvement.py
make paper-main
```

Panel (d) uses the existing `dip_metrics` three-point quadratic interpolation
of each raw central minimum, without smoothing or recentering. The annotations
report RMS relative to zero detuning over the 90 common samples in 10--100 MHz:
6.54 kHz (echo), 0.31 kHz (+ DRAG), and 0.29 kHz (+ polynomial).
The underlying detuning spacing is 5 kHz; interpolated sub-grid values are
not an independent sub-kHz convergence check. The NPZ includes the three
`*_center_hz` arrays and `center_rms_mask`, and the provenance records both
RMS about zero and the standard deviation about each curve's mean.

Figure 4 shows the 20 us row of the power-2, logarithmic-drive survey in
`outputs/polynomial_duration_cutoff_poly2_log_40_50/n100`. The supplementary
figure contains all five durations (10, 20, 30, 40, and 50 us). Both use 100
geometric cutoffs from 1e-5 to 1 and 100 geometric nominal drives from 1 to
100 MHz; zero drive is not sampled. The survey retains a 16 ns sine inversion,
beta=+0.5 in the manuscript convention (archived beta_source=-0.5),
T1=51.24 us, T2=7.31 us, and nominal pre-smoothing normalization.
Widths are displayed relative to the common 7.6 us T2* reference. Both figures
use the central Gaussian width and depth from a local fit with a quadratic
background, a smooth curvature-scaled window, and soft-L1 residual loss.
Each spectrum is fitted independently. These results are stored in
`robust_fits/metrics.npz` and `robust_fits/provenance.json`; original populations
and direct half-depth metrics are preserved. Regenerate the fit exports,
both figures, and their provenance, then rebuild the manuscripts with:

```bash
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/refit_cutoff_survey.py --workers 6
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_supplemental_extended_cutoff.py
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_main_duration_cutoff_scan.py
make paper
```

To repeat or resume the new simulations and representative convergence checks, use
`.venv/bin/python scripts/extend_cutoff_duration_40_50.py --workers 6`.
This integrates 40 and 50 us with the original model and numerical settings;
the 10, 20, and 30 us rows and their refinement checkpoints are copied with
checksum verification from `outputs/polynomial_duration_cutoff_poly2_log/n100`.
The original 45/60 us results and power-5, linearly sampled dataset remain
available separately.
The main-figure generator verifies exact equality to the supplemental 20 us
row and the common fitted arrays, with source hashes checked against the
original simulation spectra. Both figures use geometric
cell edges, linewidth colors on [0,10], contrast colors on [0,1], and white
C=0.2 guides. Only the white contour field is smoothed, with a mask-normalized
Gaussian of sigma 1.5 cutoff samples and 3 amplitude samples on the logarithmic
grid (`echospec/figures/cutoff_contours.py`). Rejected cells are masked again.
Heatmaps, fit values, threshold masks, numerical summaries, and slice curves
are unchanged. The HTML fit explorer can toggle the original boundary.
The original acceptance mask is retained and additional fit-quality checks
can reject a cell. Use `--direct-metrics` with the supplemental generator
to render the archived crossing-based widths for a diagnostic comparison.
Figure S8 additionally shows a third column for
`Q = C * Gamma_T2star / FWHM`, on a common 0--0.2 scale. Red Q=0.10 contours
outline regions with Q>0.10 in all three columns; the exported arrays include
Q and its threshold mask, with rejected dips excluded.

Figure 4 panels (c,d) show cutoff slices for 10, 20, 30, 40, and 50 us at the
same saved amplitude nearest 50 MHz (49.770236 MHz). Thick solid orange curves
highlight 20 us and match the orange dashed lines in the maps. Rejected slice
samples are gaps. The linewidth slice axis extends to 15; dotted lines mark
the coherence reference and C=0.2. The NPZ, CSV and provenance record the 20 us
maps, all five slices, masks, selected amplitude and numerical summaries.
All five displayed slices are checked against the corresponding Fig. S8 arrays.
The supplemental text summarizes fixed-cutoff selections using probability
fitted contrast/FWHM; the full numerical results remain in the summary JSON.
Forty-five representative traces are checked with half the time step and twice
the detuning resolution. Their linewidth changes and acceptance comparisons
are recorded in `robust_fits/provenance.json`, alongside six finer spectra
around the former 50 us crossing switch. The latter also vary the fitting
window factor from 1.2 to 1.8; widths change by less than 4% in those cases.
These sampled checks do not bound errors over the full grid. The original
crossing-based convergence results remain in `validation/convergence.json`.

Figures S5 and S7 use the current 20 us pulse, cutoff 0.005, 50 ns sine
midpoint transition, and power-2 polynomial correction. S5 shows the analytic
envelope, in-phase waveform, and DRAG quadrature in three panels, before
the three DRAG-results panels in S6. S7 compares hard and corrected envelopes
at beta=+0.5 in the manuscript convention (archived beta_source=-0.5)
and the same actual complex-envelope peak.
Recalculate and render with:

```bash
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_current_edge_supplement.py
MPLCONFIGDIR=/private/tmp/ars-matplotlib-cache .venv/bin/python scripts/make_supplemental_figures.py --only pulse_modifications edge_smoothing_results
make paper-supplemental
```

The numerical export `paper/data/numerical/07_current_edge_comparison.*`
contains both three-level grids, an additional no-DRAG control at 100 MHz,
and independent time/frequency refinement at 10, 50, and 100 MHz.
S7 shows three panels in one row: the 100 MHz spectrum, drive-dependent
FWHM/Gamma_T2star, and contrast. The linewidth reference is 1/(pi*T2star)
with T2star=7.6 us, matching S3 and S8; the dashed guide marks unity.
Both raw linewidths in Hz and normalized linewidths are exported.
The numerical exports retain the auxiliary SNR
diagnostics, which are no longer plotted or discussed in the supplement.
Plotted arrays, metrics and source hashes are under
`figures/paper/supplemental/edge_smoothing_results*` and
`s08_pulse_modifications*` (the latter is the historical filename for S5).
At 100 MHz the updated simulation gives FWHM/Gamma_T2star of about 12.6 versus 5.1
and contrast 0.06 versus 0.27.

## Single square-echo comparison

Section S10 and Figure S9 distinguish resonant cancellation from a useful
spectroscopic response. The exact two-level derivation is in
`appendices/P_square_echo_comparison.tex`. The figure reuses the saved
three-level simulations in `outputs/square_root_echo_comparison_20us`, with
20 us duration, cutoff 0.005 for the root-Lorentzian, one abrupt inversion,
and no DRAG, polynomial shaping, or frequency correction. The comparison
matches duration and peak drive; pulse area and integrated energy differ.

```bash
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_supplemental_square_echo.py
make paper PYTHON=.venv/bin/python
```

The generator uses `paper/data/numerical/09_square_echo_comparison.*` once
available. Pass `--refresh-source` to reimport the existing simulation
archives and check their recorded response checksum; this never launches a
simulation. The export preserves the original population samples, source
indices and hashes, validation records, and central-window population
variation. The two maps use the same population color scale, with exact
saved slices at 3.05 and 10.05 MHz and a wider +/-10 MHz view. Curves are
neither refitted nor normalized. Repeated rotary echoes are outside this
single-inversion comparison, and no sensitivity advantage is inferred.

## Ramsey Fourier-line reference

Section S11, Table S2, and Figure S10 use an ideal exponentially decaying
Ramsey cosine with the common T2*=7.6 us reference. Regenerate them with:

```bash
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_supplemental_ramsey_fft.py
make paper PYTHON=.venv/bin/python
```

The generator saves `paper/data/numerical/10_ramsey_fft_linewidth.*`, the figure,
and `paper/ramsey_linewidth_table.tex` together. It distinguishes the real,
power, and magnitude spectra, using exact finite-record half-height crossings
for linewidths and actual sampled cosine FFTs for the plotted 20/40 us traces.
The FFT uses 10 ns spacing, a rectangular window, and 2^18 padded points.
Padding supplies a dense frequency grid but no additional information.
The separate experimental marker is the existing Figure 3 corrected-echo
median; its source hash and statistic are recorded in the figure sidecar,
without mixing experimental values into the numerical dataset. This is a
linewidth reference, not an equal-acquisition-time sensitivity comparison.

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
and a standalone paired-linewidth plot, using the shared paper linewidth reference.

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
