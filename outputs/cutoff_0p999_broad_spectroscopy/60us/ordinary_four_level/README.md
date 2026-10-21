# Ordinary pulse, four-level spectroscopy

Duration 60 µs; cutoff 0.999; peak Rabi 0–100 MHz;
drive detuning −125 to +125 MHz. The modeled states are |0>, |1>, |2>, |3>.
There is no midpoint phase flip, DRAG, or Stark compensation.

The existing ordinary-pulse four-level calculation was reused exactly; no new
dynamics calculation was necessary. metadata.json records the source and config.
The four-level Duffing Lindblad model uses the rotating-wave approximation,
initial ground state, alpha/2pi = -217.106667324 MHz,
T1 = 27.155802304 µs, and T2 = 6.497862158 µs.

At 100 MHz Rabi:

- Dressed f01 shift (minimum dressed-state gap, envelope averaged): +16.7996 MHz.
- Quadratic analytical prediction: +23.0148 MHz.
- Smoothed P1 response maximum: approximately +28.1 MHz.

The analytical formula is delta_f = −A²<f²>/(2 alpha_f), where A = Omega0/2pi,
alpha_f = alpha/2pi and <f²> = 0.999333133 for this cutoff. The population maximum
also reflects saturation, leakage, and finite-pulse interference. Its shaded
range varies the Gaussian smoothing sigma from 0.25 to 1 MHz and is not a confidence interval.
The phase flip is absent throughout this result.

The broad grid has 0.5 MHz spacing; the f01 window has 0.1 MHz spacing;
local peak extraction uses 0.02 MHz spacing. The broad plots do not resolve
every coherent fringe. Numerical convergence was checked in the source run;
the raw ordinary states had trace error 1.8e-11.
This is a simulation, not a hardware measurement.

Files: four_level_spectroscopy.png/.pdf, four_level_populations.png,
spectroscopy.npz (final axis ordered P0, P1, P2, P3), refined_peak_spectra.npz,
resonance_analysis.npz, resonance_vs_amplitude.csv, and metadata.json.

Regenerate from the saved simulation:

    .venv/bin/python scripts/make_cutoff_broad_spectroscopy.py --duration-us 60 --plot-only --ordinary-only
