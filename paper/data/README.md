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
   `data_opx1000` repository.
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
| `04_main_ac_stark_correction_maps` | numerical | Letter | included | `scripts/make_main_ac_stark_correction_maps.py` |
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
