# Main-text Figure 2 experimental data

These six q1 OPX1000 runs replace the simulated maps previously used in
main-text Figure 2. Each run directory preserves `results.npz`, `sweep.npz`,
the acquisition parameters and metadata, and the saved q1 calibration profile
needed to reconstruct the peak-Rabi-frequency axis.

| Run | Domain | Protocol | Detuning | Shots |
| --- | --- | --- | --- | ---: |
| `14-46-06-038411` | broad | constant-envelope echo (`c=0.99`) | -50 to +50 MHz, 0.5 MHz step | 200 |
| `14-16-19-887278` | broad | root-Lorentzian | -50 to +50 MHz, 0.5 MHz step | 200 |
| `14-22-03-566753` | broad | echo-root-Lorentzian | -50 to +50 MHz, 0.5 MHz step | 200 |
| `14-52-31-810398` | narrow | constant-envelope echo (`c=0.99`) | -0.5 to +0.5 MHz, 0.005 MHz step | 200 |
| `14-09-56-777281` | narrow | root-Lorentzian | -0.5 to +0.5 MHz, 0.005 MHz step | 216 completed (1000 requested) |
| `14-02-28-518579` | narrow | echo-root-Lorentzian | -0.5 to +0.5 MHz, 0.005 MHz step | 1000 |

All six runs use active reset, state-discriminated readout, a 20-us
root-Lorentzian waveform, an unscaled peak output amplitude of 0.2 V, and a 200-point linear
amplitude-factor grid from 0 to 0.995. The acquisition parameters record
`use_readout_mitigation=true`; the imported arrays are the saved measured-state
values and no additional correction is applied by the figure builder.
AC-Stark correction is disabled.

The two constant-envelope runs were acquired through the same root-Lorentzian
generator with `c=0.99` and `echo=true`. They therefore approximate a square
envelope but retain the midpoint phase inversion; Figure 2 labels them
"Constant (echo)" rather than treating them as a fixed-phase constant drive.
The other four runs use `c=0.005`; the ordinary root-Lorentzian runs have
`echo=false`, and the echo-root-Lorentzian runs have `echo=true`.

The narrow root-Lorentzian run requested 1000 shots, but every saved population
is quantized in exact increments of `1/216`; its run duration is also much
shorter than the matched narrow echo run. Figure 2 therefore reports 216 as
the completed effective shot count for that panel.

For `pulse_shape="root_lorentzian"`, the acquisition code constructs the
waveform from the total length and cutoff. The saved
`lorentzian_tau_in_ns=8.0` field belongs to the ordinary-Lorentzian option and
is not used by these root-Lorentzian runs. The effective root-Lorentzian scale
is approximately 50 ns.

Generate the paper asset with:

```bash
PYTHONPATH=. MPLBACKEND=Agg python scripts/make_main_central_spectroscopy_experiment.py
```
