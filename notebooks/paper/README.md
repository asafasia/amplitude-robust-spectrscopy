# Paper Figure Notebooks

Use this folder for notebooks that produce paper-oriented figures:
compact panels, small fonts, full labels, and publication-ready exports.

Start each notebook with:

```python
from echospec.figures import FigureVariant, apply_figure_style, save_figure

VARIANT = FigureVariant.PAPER
apply_figure_style(VARIANT)
```

Save outputs with:

```python
save_figure(fig, "figure_name", variant=VARIANT)
```

Outputs go to `figures/paper/`.

## Current Notebooks

- `00_lorentzian_pulse.ipynb` - Lorentzian pulse time/frequency figures.
- `01_2d_spec_for_paper.ipynb` - 2D spectroscopy comparison.
- `02_2d_spectrsocopy copy.ipynb` - archived/duplicate 2D spectroscopy simulation variant pending cleanup.
- `03_fwhm.ipynb` - FWHM vs Rabi figure.
- `04_2d_sweep.ipynb` - amplitude/cutoff sweep figure.
- `05_2d_spectrsocopy.ipynb` - 2D spectroscopy and waterfall variants.
- `10_anharmonicity.ipynb` - anharmonicity analysis.
- `11_nine_plot_2.ipynb` - length comparison and spectroscopy comparison variants.
- `12_nine_plot.ipynb` - cutoff comparison and spectroscopy comparison variants.
- `13_three_states_T1.ipynb` - three-state T1 figures.
- `14_q1_t1_t2.ipynb` - q1 energy-relaxation and Ramsey-dephasing fits for the Supplemental Material.
- `echo_lorentzian_cutoff/echo_lorentzian_cutoff_explorer.ipynb` - raw OPX1000 cutoff sweeper with independently recalculated FWHM values.
- `20_DRAG.ipynb` - DRAG exploration.
- `21_Ramsey.ipynb` - Ramsey/FFT analysis.
- `22_TLS.ipynb` - TLS exploration.
- `23_gpu.ipynb` - GPU exploration.
- `24_jax.ipynb` - JAX exploration.
- `25_spectrsocopy.ipynb` - spectroscopy exploration.
- `30_main_pulse_mechanism.ipynb` - signed pulse, waveform FFT, and time-resolved echo cancellation.
- `31_main_central_spectroscopy.ipynb` - main Figures 2 and 3: 20-us spectroscopy maps at 50-MHz and 1-MHz spans, plus root-Lorentzian/echo spectral slices at three peak Rabi frequencies.
- `32_main_amplitude_robustness.ipynb` - main Figure 4 plus Supplemental comparisons: experimental/simulated sweeps, representative spectra, matched echo metrics, and the analytic Torrey constant-drive resolution--contrast window.
- `33_main_operating_window.ipynb` - cutoff--amplitude linewidth and width-to-signal operating maps.
- `state_.ipynb` - placeholder notebook.

## Cleanup Notes

The notebooks now use the shared paper style setup. New exports should use
`save_figure(...)` so outputs land in `figures/paper/`.

Some notebook names still preserve old typos or temporary names to avoid
breaking history during the move. Rename them in a separate cleanup pass once
the figure set is stable.
