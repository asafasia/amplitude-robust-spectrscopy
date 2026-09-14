# Data used in the PRL Letter and Supplemental Material

This directory is the reviewer-facing data package for the manuscript. It
contains the numerical and experimental values that are actually plotted or
used to calculate plotted quantities. Rendered figures remain in
`figures/paper/`.

## Layout

```text
paper/data/
  numerical/       simulation grids, model outputs, and derived fit quantities
  experimental/    selected measured values and the processed quantities plotted
```

Every dataset is a pair:

- `<name>.npz` contains numeric arrays and can be loaded with
  `numpy.load(..., allow_pickle=False)`.
- `<name>.json` documents the generator, manuscript scope, source or model,
  selection rules, and the name, shape, and type of every array.

Array names include units where practical, such as `detuning_mhz`,
`duration_us`, and `fwhm_hz`. Population and Boolean-mask arrays are
dimensionless.

## Provenance rules

1. Numerical and experimental values are never mixed in one archive.
2. Experimental archives contain only the selected or processed values used in
   the paper. The immutable raw OPX1000 records remain in the sibling
   `data_opx1000` or `opx1000-codes/data` repository, as specified by each sidecar.
3. Experimental provenance uses paths relative to `OPX1000_DATA_DIR`, not
   machine-specific absolute paths.
4. Producers call `echospec.paper_data.save_paper_dataset`; data should not be
   copied into this directory by hand.
5. A figure generator and its data export belong to the same run: rerunning
   the documented command refreshes both.

## Initial migration

The maintained generators below export here:

| Dataset | Kind | Manuscript use | Status | Generator |
|---|---|---|---|---|
| `04_main_ac_stark_correction_maps` | experimental | Letter, Figure 4 | included | `scripts/make_main_ac_stark_correction_maps.py` |
| `04_main_ac_stark_shifts_square` | experimental and numerical (separate archives) | Letter, Figure 5 | included | `scripts/make_main_ac_stark_shifts.py` |
| `06_long_pulse_lorentzian_comparison` | experimental | Supplemental | included | `scripts/make_long_pulse_lorentzian_comparison.py` |
| `08_echo_lorentzian_cutoff_sweep` | experimental | Supplemental | included | `scripts/make_echo_lorentzian_cutoff_sweep.py` |
| `09_simulated_echo_lorentzian_<L>us` | numerical | Supplemental | producer migrated; regenerate to populate | `scripts/make_simulated_echo_lorentzian_duration_cutoff_comparison.py` |
| `10_simulated_duration_resolution_comparison` | numerical | Letter and Supplemental | producer migrated; regenerate to populate | `scripts/make_duration_resolution_comparison.py` |

Notebook-only figure workflows are migrated incrementally. Until each one uses
the shared exporter, its `.npz` cache in `figures/paper/` is a figure-build
cache, not part of this reviewer-facing package.

Run the currently populated, relatively fast exports with:

```bash
make PYTHON=.venv/bin/python paper-data-core
```

Run all migrated exports, including the longer numerical simulations, with:

```bash
make PYTHON=.venv/bin/python paper-data
```

## Figure 4: q6 DRAG comparison

The active experimental dataset replaces the former numerical Figure 4 cache.
The same-named files in `numerical/` and the old `figures/paper/*.npz` cache
are historical model outputs, no longer consumed by the Figure 4 generator.
The source campaign is `drag_beta_kappa_calibration/2026-09-05_20-37-12`
under `opx1000-codes/data`. To reimport the two linked acquisitions:

```bash
python scripts/make_main_ac_stark_correction_maps.py --source-data-dir /path/to/opx1000-codes/data
```

Without an explicit source or `OPX1000_DATA_DIR`, the generator uses the
experimental paper cache, so a checkout can reproduce the figure without
access to the raw acquisitions. Source-relative paths and SHA-256 checksums
are in the sidecar. The arrays retain the complete measured maps, recorded
fit centers and acceptance masks, and saved summary statistics. The center
RMS is about each run's inverse-variance-weighted mean, not about zero.
Those saved full-acquisition statistics remain unchanged. Figure 4 displays
10--60 MHz and subtracts each run's **arithmetic mean** accepted fitted center
within that range from both its detuning grid and center curve. The
`*_display_*` arrays and `display` provenance record the masks, offsets,
centered coordinates, and RMS residuals used for this display; the original
acquisition coordinates and populations are retained. The common centered
detuning window is -0.15 to +0.15 MHz, within both shifted acquisition grids.
Both maps share color limits derived from the population extrema within these
displayed ranges, rounding the minimum down and maximum up to multiples of
0.1 (currently 0.4--0.6). The sidecar records both extrema and rounded limits.
The two waveforms also differ in applied midpoint smoothing (0 versus 16 ns);
there is no kappa correction or resolved leakage measurement in this pair.

## Figure 5: measured centers and constant-drive simulation

Run `python scripts/make_main_ac_stark_shifts.py` after regenerating Figure 4.
The experimental export retains exactly Figure 4's accepted center points,
arithmetic-mean centering, and 10--60 MHz range, with original acquisition
centers and source row indices for traceability. No zero-amplitude point or
interpolated data are added. The inset uses a symmetric -16 to +16 kHz
display window; excursions outside it remain in the main panel and exported data.
The numerical export contains the constant-drive three-level simulation
at the q6 anharmonicity -237.95 MHz (f12-f01 convention), using the
237950000 Hz magnitude recorded in Figure 4's pulse metadata. It retains its
bare-transition zero. The prior simulated
root and echo curves are no longer included in Figure 5.

The lower panel shows measured FWHM for the same selected rows. The reusable
`echospec/analysis/calibration_gaussian.py` estimator reproduces the source
campaign's bounded negative-Gaussian fit: trim 40 samples from each edge,
smooth with a Gaussian of sigma 1 sample, and fit with `maxfev=800`.
FWHM is `2*sqrt(2*ln(2))*sigma`. Refitted centers must match the saved centers
within 1 Hz. All 162 selected rows in each run pass the source quality checks;
no width-based filtering is applied. The experimental export includes widths,
fit centers, contrast, R-squared, acceptance flags, and nominal covariance
errors. Those errors describe smoothed fits and are not plotted as independent
measurement uncertainties. The top panel retains the previous measured
centers and the constant-drive simulation with q6 anharmonicity.

Panel (b) also contains a constant-drive Bloch-model reference
`Gamma_T2*sqrt(1+(2*pi*rabi_MHz)^2*T1_us*T2_us)`, using the existing q1
parameters in `paper/coherence_parameters.tex`. Both red curves are lines
without markers. The logarithmic linewidth axes show kHz on the left and
`FWHM/Gamma_T2` on the right, with `Gamma_T2=43.5 kHz`; this is the paper's
q1 normalization reference, not a new q6 coherence measurement. The numerical
export records the linewidth curve, coherence parameters, and normalized widths.


## Dense q6 pulse-length series

`12_dense_pulse_length` is exported separately for experiment and simulation
by `scripts/make_dense_pulse_length_comparison.py`. The ten-length, 400-amplitude
arrays retain original amplitude indices, nominal amplitudes, calibrated Rabi
frequencies, half-height widths and crossings, contrast, and acceptance and
peak/dip masks. The experimental package additionally contains estimated SNR.
The figures select lengths through 25 us; the paired plot intersects acceptance
masks at identical length/amplitude indices (1278 pairs). The 30--40 us measured
rows have no accepted widths. Unresolved widths are NaN, not zero.

The source is the q6 September 9--10 campaign in `opx1000-codes/data`, not the
older q1 datasets. Sidecars record source-relative paths, SHA-256 hashes,
acquisition/device settings, numerical validation and the simulation detuning
mapping. This campaign uses beta=0 and disabled AC-Stark compensation; the stored
kappa value is inactive. The right linewidth axis uses its saved T2*=6.121784 us,
with reference 1/(pi*T2*)=51.996 kHz. The simulation arrays contain precomputed
qutrit predictions, not a refit; this generator reproduces the displayed figures
from the documented package without requiring the original hardware or solver.
