# Smooth ensemble control for the single echo

The search changes duration, core width, tail gain and decay, endpoint taper,
transition duration, four smooth log-amplitude coefficients, five even phase
coefficients, three odd phase coefficients, and the central I/Q mixing. The
largest search has 23 active parameters. A single fixed waveform is swept in
amplitude; it is not retuned separately at each amplitude.

The objective is the 50 MHz linewidth, constrained by absolute fitted-background
contrast above 0.4 at 50 MHz and no width or contrast deterioration from the
original polynomial across 5–60 MHz. Additional center, secondary-dip, and
leakage checks are reported separately. The final amplitude grid uses 0.5 MHz
spacing. This finite family and local optimization do not establish a global
optimum or a no-go theorem.

## A smooth reversal does not require an amplitude discontinuity

For x = 2t / T_jump within the central interval, define

    theta(t) = pi * (15x - 10x^3 + 3x^5) / 16
    c(t) = -sin(theta(t)) + i eta cos(theta(t)).

Outside the interval, c is +1 before and -1 after. The joins have two continuous
derivatives. eta = 0 crosses the origin smoothly; eta = +/-1 traverses a unit
semicircle. The raw drive is u_raw = g(t)c(t)exp[i phi(t)]. The completed complex
drive follows the existing I-iQ convention:

    u = u_raw + i beta d(u_raw)/dt / (2 pi |alpha_MHz|).

For a requested peak R MHz, the applied waveform is R*u/max_t|u|, taking the
maximum over both time halves after DRAG. For the repaired higher-contrast
pulse, max_t|u|=0.83956666699, so the coefficient multiplying the unnormalized
complex waveform is 59.55453208 MHz to realize an actual 50 MHz peak.

The completion is continuous with a continuous first derivative, including the
central joins and compact endpoints. A phase angle can change by pi when a
signed real amplitude crosses zero without a jump in the physical I or Q.
Smoothness must be assessed on I and Q themselves.

DRAG can give a nonzero completed drive at t=0 even when eta=0. Conversely,
nonzero eta can cancel the completed central quadrature. The report therefore
shows both eta and the actual completed center amplitude relative to its peak.

## Why a full I/Q turn does not automatically preserve the echo

For an ideal resonant qubit, odd control H_c(-t)=-H_c(t) makes the second half
the time-ordered inverse of the first. With an even envelope and even phase,
eta=0 has this property before DRAG. It holds for every amplitude multiplier.
Continuity and oddness imply H_c(0)=0. A nonzero raw center cannot preserve this
particular automatic cancellation symmetry.

The added eta*cos(theta) quadrature is even. Odd phase functions on the wings
can oppose its integrated complex area. The pilot explicitly balanced that
area for several full and partial rotations, to numerical precision. This
removes only the leading weak-drive term. It does not cancel higher-order
time-ordered rotations, because Hamiltonians at different times need not
commute. Such pulses must be checked over the entire amplitude ensemble.

These statements concern the ideal raw qubit control. Completed DRAG,
transmon drift, relaxation, and dephasing are checked using the numerical
models rather than assumed to inherit the symmetry.

## What the fixed ablations teach

The repaired higher-contrast pulse uses 16.1783 us total duration, a 16.3244 ns
core parameter, and a 36.7776 ns smooth central transition. At the same 50 MHz
completed-drive peak it gives 127.475 kHz FWHM and 0.42020 primary contrast.
The added amplitude and phase basis functions act together with the transition.

| Fixed variant | FWHM at 50 MHz (kHz) | Contrast at 50 MHz | Contrast at 5 MHz |
|---|---:|---:|---:|
| Repaired pulse | 127.475 | 0.42020 | 0.25832 |
| Added amplitude terms removed | 132.637 | 0.38662 | 0.23189 |
| Added phase terms removed | 127.907 | 0.41823 | 0.12486 |
| Transition shortened to 16 ns | 126.639 | 0.43701 | 0.21346 |

Each variant is independently normalized by the peak of the full complex drive.
The phase ablation retains the inherited phase term. These variants are not
reoptimized, so they identify sensitivities of this particular solution, not
necessary ingredients of every possible pulse. Most notably, the shorter
transition looks better at 50 MHz while violating the low-drive requirement.
The richer phase and longer smooth transition help preserve the initial
5 MHz response. See [the reproducible ablation](cleaner_ablation/ablation.json).

## What the linewidth and contrast mean

The primary metric remains unchanged: a linear background fitted at detunings
0.4–0.5 MHz from resonance, a central dip, and its nearest half-depth crossings.
This preserves comparability with the original pulse. A narrow feature inside
a broad pedestal can have primary contrast above 0.4 while its depth relative
to nearby shoulders is much smaller. The report therefore also gives nearest
shoulder depth, an unfitted depth at +/-0.2 MHz, and spectra over +/-2 MHz.

The homogeneous weak-drive reference is 1/(pi T2)=43.54 kHz for T2=7.31 us.
It follows from Fourier transforming exponential coherence decay. It is not
a proven attainable linewidth for the simultaneous transient-echo contrast,
amplitude-robustness, smoothness, and leakage requirements used here.

## Relation to established control methods

Using a small smooth analytic control basis is an established optimal-control
strategy; see [Gradient optimization of analytic controls](https://arxiv.org/abs/1507.04261)
and [Smooth optimal quantum control, supplemental methods](https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.115.190801/smooth_control_paper_suppl_info.pdf).
This implementation uses finite-difference constrained SLSQP and the existing
Lindblad solver, rather than implementing the GOAT sensitivity equations. The
task-specific contributions are the spectroscopy objective, broad-amplitude
constraints, central-path parameterization, and explicit contrast audits.
