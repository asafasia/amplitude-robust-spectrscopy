# Paper measurement rerun plan

Prepared 29 October 2026 for the current Letter and Supplemental Material.

The existing experimental results require four spectroscopy maps for Figure 2,
five duration maps for Figure S3, two maps for Figure S5(d), and ten maps for
the dense-duration study in Supplemental Section S8. Figure S2 reuses Figure 2
data. Repeat the device and readout calibrations alongside these acquisitions.

Figures 1, 3, 4, S4, S6, S7, and S8 are currently waveforms or simulations.
They have no old hardware acquisition to repeat. Proposed measurements for
testing those figures are specified below, separately from the historical
settings. Figure S8 and **Section S8 are different**: the former is the
numerical cutoff survey; the latter is the measured dense-duration campaign.

**Status:** planning note only; no measurements submitted. “Recorded” means
established by the current source, saved arrays, or acquisition metadata.
“Proposed” denotes a setting for the new campaign. Old frequencies, voltages,
coherence values, and device labels are references, not current calibrations.

## Figure checklist

| Figure or result | What to acquire | Shared acquisition |
| --- | --- | --- |
| Fig. 1 | No required hardware repeat; waveform, Fourier transform, ideal time evolution | Optional interrupted-pulse test below |
| Fig. 2(a,b,e,f) | Root-Lorentzian and echo, each at broad and narrow detuning | Four maps, campaign A |
| Fig. 2(c,d,g,h) | Recalculate simulations using the new device parameters | Same axes as campaign A |
| Fig. 3(a–f) | Currently numerical; proposed matched echo / DRAG / polynomial maps | Three new maps, campaign E |
| Fig. 4(a–d) | Currently numerical; proposed cutoff and duration measurements | Campaign F; shares data with Fig. S8 |
| Fig. S1 | Verify the setup drawing against the actual wiring | No spectroscopy sweep |
| Fig. S2 | Extract five amplitude rows per pulse shape | Reuse the two narrow maps from A |
| Fig. S3 | Echo spectroscopy at five pulse lengths | Five maps, campaign B |
| Fig. S4 | Currently simulated leakage; proposed transition-time comparison | Campaign G, requires resolved three-state readout |
| Fig. S5(a–c) | Currently numerical; optional experimental DRAG optimization | Campaign H |
| Fig. S5(d) | Repeat the two historical pulse settings | Two maps, campaign C |
| Fig. S6 | Save and plot the actual I/Q waveforms for campaign E | No separate population sweep |
| Fig. S7 | Currently numerical; use DRAG and polynomial maps from E for a new experimental comparison | Optional independent SNR acquisition |
| Fig. S8 and Tables S3–S4 | Currently numerical cutoff–duration survey | Campaign F if experimental validation is desired |
| Table S1 and coherence references | Spectroscopy, two-photon spectroscopy, T1, Ramsey, Hahn echo | Calibration block |
| Supplemental Section S8 | Dense echo-duration campaign, including unsuccessful long-duration points | Ten maps, campaign D |

## Common calibration and acquisition block

Use one explicitly identified physical qubit for each matched comparison and
save its complete profile with every run. A new campaign can consolidate onto
one device, but that changes the historical device allocation and requires new
matched simulations and captions.

1. Locate the flux sweet spot and record the bias. Calibrate resonator frequency,
   readout amplitude, duration, integration weights, discrimination, and reset.
2. Measure f01 and f02/2; derive f12 = 2(f02/2) − f01 and
   α/(2π) = f12 − f01. The historical primary-qubit references are
   f01 = 4.267106667 GHz and f02/2 = 4.159106667 GHz. Save the frequency
   reference actually used for each map; resolve the Figure 2 device
   discrepancy below before associating those frequencies with its data.
3. Calibrate the active square π pulse and verify amplitude-to-Rabi conversion:
   **fR = A / (2 Aπ tπ)**. Here A is the nominal envelope scale, not the sampled
   maximum after midpoint smoothing. Save Aπ, tπ, pulse name, and output gain.
4. Measure T1, Ramsey T2*, and Hahn-echo T2 before and after each campaign.
   Recompute Γ = 1/(π T2*) for its normalization. Historical primary-qubit
   values are T1 = 51.24 µs, T2* = 7.31 µs, T2,echo = 16.42 µs and
   α/(2π) = −216 MHz; do not impose these values on a fresh dataset.
5. Save prepared-state readout references. Binary discrimination alone does
   not establish Pf or distinguish Pe from Pe + Pf. For leakage measurements,
   prepare g/e/f and calibrate the complete assignment matrix.

The exact historical calibration sweeps are not fully specified by the paper.
**Proposed starting grids:** T1 delays 0–250 µs in 101 points; Ramsey delays
0–40 µs in 201 points with a 0.5 MHz programmed detuning; Hahn-echo total free
evolution 0–100 µs in 101 points; 2,000 shots per delay. Adapt the delay range
after the first fit to capture the decay and baseline. For Rabi calibration,
use 101 linear amplitudes over 0–2 times the current π-amplitude estimate,
within the configured output range. Use 10,000 readout shots per prepared
state. Frequency-search windows and flux range depend on the current device;
locate the transitions first, then refine around the measured centers.

**Proposed drift control:** acquire compared pulse shapes in interleaved
blocks, with three independently saved passes. Divide the stated shot budget
across passes unless intentionally increasing it. Repeat a fixed reference
spectrum between blocks, and record timestamps. Preserve the original
frequency reference as well as any subsequently corrected coordinate.

Use explicit arrays for all sweeps and save the realized arrays. Frequency
span below always means the **full width**, and all listed linear endpoint
grids are inclusive unless stated otherwise. Save completed shot counts,
raw I/Q or state counts, readout correction settings, I/Q waveform samples,
sample interval, reset settings, and requested versus applied pulse parameters.

## Campaign A — Figure 2 and Figure S2

**Recorded pulse:** `root_lorentzian`, order 1/2, L = 20 µs, template length
20 µs, cutoff c = 0.005. Compare no echo against a midpoint sign inversion.
No AC-Stark correction; κ = 0. The old parameter files do not explicitly
record a DRAG coefficient or midpoint-transition duration. For the intended
uncorrected baseline, explicitly set β = 0 and an abrupt inversion, and save
the realized waveform rather than relying on defaults.

| Run | Pulse | Detuning relative to current f01 | Amplitude | Shots per point |
| --- | --- | --- | --- | --- |
| A1 | Root-Lorentzian, no echo | −25 to +25 MHz, 501 points, 100 kHz spacing | 250 linear factors 0–1 × 0.7 V | 2,000 |
| A2 | Echo-root-Lorentzian | Same broad grid | Same | 2,000 |
| A3 | Root-Lorentzian, no echo | −0.5 to +0.5 MHz, 501 points, 2 kHz spacing | Same | 2,000 |
| A4 | Echo-root-Lorentzian | Same narrow grid | Same | 2,000 |

The saved active π calibration is 40 ns at 0.2058445823 V for A1/A2 and
0.2048095007 V for A3/A4, corresponding to nominal maxima of 42.5078 and
42.7226 MHz. Recalibrate this conversion; 0.7 V does not universally mean
42.7 MHz. The old runs used active reset and binary state discrimination.
Their `use_readout_mitigation` fields differ: 0.4 in the broad files and 1
in the narrow files. Establish and record one consistent treatment for the
new comparison, preserving uncorrected counts.

Each map contains 125,250 points and 250.5 million shots; all four contain
1.002 billion shots before calibration/reset overhead. The historical
manifest shows roughly 3–4 hours per map, not a current runtime guarantee.

Figure S2 needs no additional scan: extract the nearest measured rows to
fR = 3, 10, 20, 30, 40 MHz from A3/A4. Historical selected values round to
2.92, 9.95, 20.07, 30.03, 39.98 MHz. Keep the measured rows and all detunings;
do not shift frequencies or rescale contrast to match the simulation.

The six-run archive also contains c = 0.999 no-echo broad/narrow controls.
These are **optional additional two maps**, not displayed in current Figure 2.
Repeat them only if retaining an experimental near-square reference.

**Device discrepancy to resolve:** every inspected `sweep.npz` in this
six-run archive identifies `q6`; older README text and script descriptions
call it `q1`. Use the run's actual device profile and newly measured coherence
for the new simulation comparison. Do not silently assign the historical
primary-qubit constants to it.

Sources: [Figure 2 caption](../figures/02_spectroscopy_float.tex),
[Figure 2 provenance](../../figures/paper/02_central_spectroscopy_experiment_simulation_variant_provenance.json),
[raw campaign](../../data/experimental/2026-08-25/six_detuning_amplitude_sweeps/),
[loader and Rabi conversion](../../scripts/make_main_central_spectroscopy_experiment.py).

## Campaign B — Figure S3 duration and contrast

**Recorded figure grid:** five echo maps at L = **5, 10, 15, 20, 30 µs**,
c = 0.005; each has 201 detunings from −0.5 to +0.5 MHz (5 kHz spacing)
and 200 measured amplitudes spanning nominal fR = 0–25.076509 MHz.
The five amplitude arrays and detuning arrays survive in the saved figure
NPZ. These are measured arrays; the separate 100-point logarithmic Rabi grid
in that NPZ belongs to the simulation.

**Proposed rerun:** use those 200 saved nominal Rabi values, converted with
the new square-π calibration; explicitly disable DRAG and Stark compensation,
use an abrupt echo and template length equal to L. Acquire **2,000 shots per
point** consistently across all five lengths. This is a new shot budget,
not a recovered historical value. Five maps total 201,000 operating points
and 402 million shots at that budget.

**Incomplete historical reconstruction:** four original duration run bundles
were not found locally. The locally available 20 µs bundle records 1,000
requested shots, 0.2 V base amplitude, factors 0–0.995 in steps of 0.005,
and a `lorentzian_tau_in_ns` field of 8 ns. That record alone does not
establish the applied pulse or shot count of all five saved figure maps.
Resolve this if exact historical reproduction is required; for the new
campaign derive the core width from L and c and save the actual samples.

Fit and save center, width, contrast, quality mask, and uncertainties. Figure
S3 reports FWHM/Γ, contrast A, and Q = AΓ/FWHM. Preserve rejected traces.
This figure's Gaussian estimator differs from the half-height estimator in D;
do not combine their outputs under one unlabeled linewidth definition.

Sources: [duration figure data](../../figures/paper/03_echo_duration_resolution_comparison.npz),
[generator and original run IDs](../../scripts/make_duration_resolution_comparison.py),
[supplement text](../appendices/M_spectrum_slices.tex).

## Campaign C — Figure S5(d) historical DRAG comparison

**Recorded common settings:** second-qubit campaign, root-Lorentzian echo,
L = template = **50 µs**, c = **0.00075**, no Stark correction, κ = 0.
Two maps, each with **200 linear detunings −0.2 to +0.2 MHz**
(2.01005 kHz spacing), **200 linear amplitudes 0–1 V**, and **2,000 shots
per point**. Historical nominal fR spans 0–61.34185 MHz; the display uses
10–60 MHz. Total: 80,000 points, 160 million shots.

| Setting | Recorded β in acquisition convention | Applied midpoint transition |
| --- | --- | --- |
| C1 | 0 | 0 ns, abrupt |
| C2 | −0.22 | 16 ns |

The requested transition field says 16 ns for both, but the **applied**
metadata says 0 ns for C1. Also, requested `frequency_step_in_mhz=2` and
`amp_factor_step=0.03` are inconsistent with the realized 200-point arrays;
use the arrays above, not those stale step fields.

For a controlled new DRAG test, **add C3: β = 0 with the same 16 ns
transition as C2**. C3 versus C2 isolates DRAG; C1 versus C3 isolates
midpoint smoothing. The original two maps alone do not separate those effects.

The acquisition records use **I+iQ** with a positive anharmonicity magnitude;
the current simulations specify **I−iQ**. Do not copy −0.5 from the numerical
figures into the hardware coefficient field without checking the complete
waveform and sign convention. Record the actual derivative and units.

Keep absolute fitted centers and drift references. For reproducing S5(d),
also report the variation after subtracting each run's mean over 10–60 MHz.
Mean-subtracted stability does not establish an absolute frequency offset.
The old “three-state requested” flag did not provide three-state readout:
the metadata explicitly marks it unavailable.

Sources: [experimental arrays and applied settings](../data/experimental/04_main_ac_stark_correction_maps.json),
[S5 provenance](../../figures/paper/supplemental/drag_beta_optimization_provenance.json),
[pulse modification definitions](../appendices/O_pulse_modifications.tex).

## Campaign D — Supplemental Section S8 dense duration study

**Recorded acquisition:** echo-root-Lorentzian, c = 0.005, abrupt midpoint,
β = 0, no Stark correction. Acquire **L = 1, 2, 5, 10, 15, 20, 25, 30,
35, 40 µs**, with template length equal to each L and no stretching.

- Detuning: **400 linear points −1 to +1 MHz**, spacing 5.012531 kHz.
- Nominal amplitude: **400 logarithmic points 0.01–0.70 V**.
- Historical repetitions: **380 per point**.
- Saved π calibration: 40 ns at 0.2037760489 V, giving fR = 0.613419–42.939296 MHz.
- Campaign-specific T1 = 41.71194 µs, T2* = 6.121784 µs,
  α/(2π) = −237.95 MHz. Refresh all three for a new campaign.
- The stored κ = 0.005 MHz⁻¹ is inactive because Stark correction is disabled.

Ten maps contain 1.6 million points and 608 million shots at the historical
budget. The original acquisition took approximately 8.6 hours. Proposed:
retain 380 shots per point for the first full pass, then repeat/interleave
durations and increase averaging where required by the measured contrast.
Report the additional budget explicitly.

Retain all ten lengths, including 30–40 µs: the old estimator accepted no
widths at those lengths and only 32 of 400 at 25 µs. Missing widths are not
zero widths. Save raw traces to separate lost contrast, fit failure, and
drift. Reproduce the old analysis with a nine-point order-2 Savitzky–Golay
filter, linear outer-10% baseline, nearest half-height crossings, SNR ≥ 5,
and width between three bins and 80% of the scan. Also inspect unsmoothed
traces; the old smoothing window spans about 40 kHz.

Sources: [Section S8](../appendices/L_dense_pulse_length.tex),
[dense campaign provenance](../data/experimental/12_dense_pulse_length.json).

## Campaign E — New measurements corresponding to Figure 3 and Figures S6–S7

These would be **new experimental validation** of the numerical comparison.
Use L = **30 µs**, c = **0.001**, a **16 ns sine midpoint transition in
all three cases**, and κ = 0:

| Run | Outer envelope | DRAG in the paper's simulation convention |
| --- | --- | --- |
| E1 | Hard-truncated root-Lorentzian | β = 0 |
| E2 | Same hard-truncated envelope | β = −0.5 |
| E3 | Endpoint-subtracted envelope × polynomial window, p = 5 | β = −0.5 |

The recorded numerical grids are **100 linear nominal fR values 0–100 MHz**
and **201 detunings −0.5 to +0.5 MHz**, spacing 5 kHz. Proposed hardware
budget: 2,000 shots per point, or 120.6 million shots for all three maps.
First establish the available calibrated drive range; 100 MHz is a numerical
target, not an established hardware capability. If the hardware range is
smaller, record that limit and compare simulations on the acquired grid.

For center-stability measurements, add repeated fine scans around the dip
with 1 kHz spacing or finer as justified by uncertainty. The existing
sub-kHz numerical center values are interpolations from 5 kHz samples;
they are not an experimental precision target established by those samples.

For |t| < L/2, define σ = (L/2)c/√(1−c²), f(t) = [1+(t/σ)²]⁻¹/²,
and the corrected envelope
**g(t) = [(f(t)−c)/(1−c)] [1−(2t/L)²]⁵**.
The sign factor is +1 before −T/2, −sin(πt/T) within the transition, and
−1 after +T/2. Use I = Ω0 g sT and Q = −β(dI/dt)/|α| in the paper's
I−iQ convention, with α in angular-frequency units. Differentiate the
**complete signed waveform**, including the midpoint term. Keep g(0) = 1;
do not renormalize area or the smoothed signed peak. Save I/Q samples to
verify the hardware implementation and provide Figure S6.

E2/E3 provide spectra, FWHM, contrast, and centers for an experimental S7.
Its current SNR is an ideal calculation with 1,000 total binary measurements:
500 at the dip and 250 at each ±0.45 MHz shoulder. For **measured** SNR,
acquire repeated independent blocks at those three locations and retain their
variance; label this separately from ideal binomial SNR.

Sources: [Figure 3 configuration](../../figures/paper/03_pulse_improvement_provenance.json),
[pulse equations and S7](../appendices/O_pulse_modifications.tex).

## Campaign F — New measurements corresponding to Figure 4 and Figure S8

Use the corrected polynomial DRAG echo from E3, p = 5, 16 ns sine inversion,
and no added Stark correction. The **recorded simulation grid** is:

- L = **10, 20, 30, 45, 60 µs**.
- c = **100 logarithmic values from 10⁻⁵ to 1**.
- fR = **100 linear values from 0 to 100 MHz**.
- Detuning = **201 linear values from −0.5 to +0.5 MHz**, 5 kHz spacing.

Figure 4(a,b) is the L = 30 µs row. Figure 4(c,d) uses all five durations
at the saved fR = **49.494949 MHz** column. Figure S8 uses every duration.
At c = 1, implement the finite limit **g(t) = [1−(2t/L)²]⁶**; it is not
a square pulse. At fixed c, changing L also changes the core width σ.

A literal hardware copy is 10,050,000 points, or **20.1 billion shots at
2,000 per point**. It is not necessary to reproduce a simulation-only figure.
**Proposed first experimental pass:** L = 30 µs; cutoffs
0.0001, 0.0003, 0.0006579332, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1;
fR = 10, 20, 30, 40, 49.494949 MHz; the same 201 detunings; 2,000 shots.
This is 50 spectra and 20.1 million shots. Next repeat the cutoff slice at
49.494949 MHz for L = 10, 20, 45, 60 µs, adding 40 spectra / 16.08 million
shots. Expand amplitude and cutoff coverage after checking contrast and the
realized waveform, especially for cores comparable to the sample interval.

For analysis, preserve the local-dip quality mask and linear-background
definition. The white contours represent **C = 0.2**, not a linewidth
threshold. A sparse experimental pass tests selected predictions; it cannot
replace the full numerical maps or establish their global optimum.

Sources: [full survey specification](../appendices/N_extended_cutoff_duration.tex),
[Figure 4 provenance](../../figures/paper/04_duration_cutoff_scan_provenance.json).

## Campaign G — New leakage comparison for Figure S4

Hold L = **20 µs**, c = **0.005**, β = 0, κ = 0. Compare midpoint
transitions **T = 0, 4, 8, 16 ns**, using the sine definition above for T > 0.
The numerical grid contains **25 linear drives 0–42.72262747 MHz** and
**201 detunings ±0.5 MHz**, spacing 5 kHz. The figure shows final Pf at the
highest drive and max(Pf) over detuning at each amplitude.

Proposed first pass: measure the highest-drive 201-point spectrum for all
four transitions, with calibrated g/e/f discrimination and 10,000 shots per
point, then expand to all 25 amplitudes. Choose the final repetition count
from measured assignment uncertainty and a pilot: the predicted smallest
leakage is of order 10⁻⁴, so 10,000 shots alone does not guarantee a reliable
measurement. Record the delay from drive completion to readout. This tests
**final leakage**, not maximum transient population during the pulse.

Source: [S4 definition and grid](../appendices/O_pulse_modifications.tex).

## Campaign H — Optional experimental DRAG optimization for Figure S5

For a match to the simulated scan use L = **20 µs**, c = **0.005**,
T = **16 ns for every β**, and κ = 0. Use seven equally spaced fR values
from **10 to 42.72262747 MHz**. The numerical frequency grid is
**−60 to +60 kHz in 1 kHz steps**, 121 points. It is a center-search
window, not a guaranteed full-linewidth measurement; first acquire a broad
pilot so a displaced minimum is not clipped.

The numerical β grid contains 35 values over −1 to 0, with 0.005 spacing
near −0.5; use the saved grid if reproducing it exactly. A practical proposed
hardware first pass is β = −1, −0.75, −0.5, −0.25, 0, followed by a fine
scan near the measured minimum, after mapping the quadrature convention.
Start with 2,000 shots per point and repeat reference spectra between β
blocks. Optimize RMS center offset about the independently tracked zero
detuning; also report mean-subtracted variation and leakage separately.

Sources: [saved β scan](../../outputs/drag_beta_center_scan/provenance.json),
[S5 acquisition and objective](../appendices/O_pulse_modifications.tex).

## Figure 1 and setup verification

Figure 1 uses L = **10 µs**, c = **0.005**, fR = **15 MHz**, and detunings
**0, 0.10, 2.00 MHz** in an ideal unitary two-level model. Its Fourier panel
is computed from the waveform; no spectrum-analyzer measurement is required.
If adding a time-domain experiment, play prefixes of the **same fixed full
waveform**, interrupt at sampled times, and read out. Do not regenerate a
new pulse with a shorter L at each time, since that also changes σ and the
midpoint. Proposed starting sweep: 251 uniformly spaced interruption times
over 0–10 µs, plus extra samples near the midpoint, at 2,000 shots per point.
Compare against a dissipative model using the actual device calibration.

For Figure S1, verify actual line routing, attenuation, filters, circulators,
amplifiers, readout chain, and stage placement before updating the schematic.

Sources: [Figure 1 provenance](../../figures/paper/01_pulse_mechanism_provenance.json),
[setup and Table S1](../appendices/A_measurement_setup.tex).

## Suggested run order and completion criteria

1. Calibrate device, readout, reset, and amplitude conversion; resolve the
   device identity and I/Q convention before transferring old parameters.
2. Run A3/A4 narrow maps first, followed by A1/A2 broad maps. Extract S2
   from A3/A4 without duplicate acquisitions.
3. Run B and C, including the matched-smoothing control C3. Repeat calibration
   references between blocks.
4. Run D with interleaved duration blocks, retaining all raw long-duration
   traces whether or not linewidth extraction succeeds.
5. For new experimental evidence, prioritize E, then the sparse F pilot;
   add G when three-state readout is calibrated, and H for coefficient tuning.

For the existing measured content, A+B+C+D is **21 maps** before repetitions,
the C3 control, optional near-square maps, and calibrations. At the budgets
listed above it totals **2.172 billion shots**; B uses the proposed 2,000-shot
budget, while A/C/D use their recorded budgets. Estimate runtime from a
short completed block with the actual reset and readout overhead.

For each completed run, record a stable data path, device/profile snapshot,
realized grid and waveform, completed averages, calibration references,
timestamps, raw observable definition, and any failed points. Export new
arrays and provenance together with regenerated figures. Update simulations
to the new measured parameters, retain raw center traces and masks, and
keep plot x-limits at the acquired sweep borders. Archive old datasets
separately so that the original and rerun figures remain reproducible.
