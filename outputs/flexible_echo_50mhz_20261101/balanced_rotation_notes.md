# Area-balanced continuous-rotation prototypes

Five fixed rotation mixes were fitted using the three odd phase basis functions
at 0.05, 0.2, and 1 µs. The source was `smooth_base/best.json`, with its existing
13.577586 ns quintic transition, 18.533826 ns core, and 16.992340 µs duration.
The source transition was not multiplied by 1.875 a second time.

Each fit imposed two real conditions: zero real and imaginary parts of the raw
complex control's integral. Bounded least squares started from zero odd-phase
coefficients, with coefficient bounds ±3. The analytic waveform was integrated
using 96-point Gauss-Legendre quadrature in each piecewise interval; independent
192-point quadrature audited the result. All normalized complete-drive area
residuals were below 5.3e-14 µs. Refining the quadrature changed raw areas by less
than 7.1e-17 µs. Full versus raw integrals agreed within 9.3e-17 µs, consistent
with the derivative correction integrating to zero for zero endpoints.

The first mix, −0.159832449, cancels the central DRAG quadrature of the source
geometry before odd-phase fitting. After fitting, the odd phase has a nonzero
central derivative, so that original local cancellation is no longer exact.

Exactly five native candidate simulations and one original-baseline simulation
used the root optimizer's 12 training amplitudes and detuning grid. Peak
normalization and the root `assess` function were used without modification.

| Rotation mix η | FWHM at 50 MHz | Contrast at 50 MHz | Maximum center magnitude across training amplitudes | Feasible? |
|---:|---:|---:|---:|:---:|
| −0.159832449 | 119.012 kHz | 0.38998 | 9.088 kHz | No |
| +0.15 | 119.655 kHz | 0.39335 | 6.326 kHz | No |
| −0.3 | 114.032 kHz | 0.36103 | 17.735 kHz | No |
| +1 | 70.971 kHz | 0.15584 | 42.800 kHz | No |
| −1 | 44.732 kHz | 0.05872 | 203.921 kHz | No |

The best penalized result is η=+0.15, with odd phase coefficients
`[0.020901034179922746, 0.01675832713920283, -0.00894790210263001]`.
It remains below the required contrast at 50 MHz and violates other checks.
The narrow features of the unit-semicircle trials are not successes: their
contrast collapses and the spectra shift substantially.

This is first-order compensation, not exact echo cancellation. Zero integrated
transverse drive cancels the first control-area term, while higher time-ordered
terms, multilevel drift, and dissipative evolution remain. The pilot demonstrates
why integrated area alone cannot establish useful amplitude robustness.

`balanced_rotation_seeds.json` contains all records, sorted by root loss;
`balanced_rotation_seeds.npz` contains complete populations in the original
rotation order. Configuration and exact source snapshots have separate
provenance files. The root optimizer added a CLI option during the pilot; its
matching original source was recovered from an existing snapshot. The `assess`,
`encode`, and `make_pulse` function ASTs were verified unchanged, and all saved
quality values were independently recomputed from the saved rows.
