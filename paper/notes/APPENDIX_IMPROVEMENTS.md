# Appendix Improvement Plan

I would rebuild the appendix around three goals: define the implemented pulse exactly, derive the mechanism consistently, and make the experiment and simulation reproducible. The present appendix contains useful material, but its organization and several equations obscure the main result.

## Recommended appendix structure

| Section | What it should contain |
|---|---|
| A. Pulse definition and conventions | Exact finite waveform, cutoff definition, duration, units, and phase inversion |
| B. Resonant echo cancellation | Short propagator derivation showing why cancellation occurs at $\Delta=0$ |
| C. Conventional $T_2$ and power-broadening limit | Correct steady-state Bloch solution and FWHM |
| D. Adiabatic power-narrowing mechanism | Correct adiabatic basis and explicit limitation of the published scaling |
| E. Truncation and hardware effects | Endpoint discontinuity, cutoff artifact, phase-switch bandwidth, and sampling |
| F. Numerical model | Hamiltonian, Lindblad equation, levels, parameters, and convergence |
| G. Data analysis | Definition and extraction of linewidth, center frequency, contrast, and uncertainty |
| H. Experimental parameters | Compact device and calibration table rather than generic setup prose |
| I. Limitations | Long pulses, decoherence, strong-drive distortions, and leakage |

## A. Add the exact finite pulse used in the code

The manuscript defines an infinite pulse using $\tau$, but the code specifies total length $L$ and endpoint fraction $c$. Connect them explicitly:

$$
f_{n,c,L}(t)=
\begin{cases}
\left[1+(t/\tau)^2\right]^{-n}, & |t|\le L/2,\\
0, & |t|>L/2,
\end{cases}
$$

with

$$
c=f_{n,c,L}(L/2),\qquad
\tau=\frac{L/2}{\sqrt{c^{-1/n}-1}}.
$$

Then define

$$
\Omega_{\mathrm E}(t)=\Omega_0 f_{n,c,L}(t)s(t),
$$

where $s(t)=+1$ before the pulse midpoint and $-1$ after it.

This matters because the current implementation in `echospec/simulation/pulses.py` changes $\tau$ whenever $c$ changes at fixed $L$. Therefore, the current cutoff sweep is not purely a truncation sweep: it changes both the endpoint amplitude and the underlying Lorentzian width.

I would distinguish two studies:

- **Pure truncation study:** hold $\tau,n,\Omega_0$ fixed and vary $L$.
- **Fixed-duration hardware study:** hold $L$ fixed and vary $c$, explicitly acknowledging that $\tau$ changes.

## B. Add a simple derivation of the echo mechanism

This is the most important missing appendix derivation. For

$$
\frac{H(t)}{\hbar}
=
\frac{1}{2}\left[\Delta\sigma_z+\Omega_{\mathrm E}(t)\sigma_x\right],
$$

at resonance,

$$
U(\Delta=0)
=
\exp\left[-\frac{i\sigma_x}{2}
\int_{-L/2}^{L/2}\Omega_{\mathrm E}(t)\,dt\right]
=I,
$$

because the signed pulse area vanishes.

Off resonance, the detuning term does not reverse. Hamiltonians at different times no longer commute, so the two halves do not cancel. This directly explains why resonance produces a dark feature.

Also state the limitations clearly:

- cancellation is exact only for ideal unitary evolution;
- $T_1$, $T_2$, phase error, and amplitude asymmetry make it imperfect;
- an instantaneous phase jump has infinite formal bandwidth;
- finite AWG sampling and mixer response must be included experimentally.

This argument is cleaner and more directly relevant than calling the sequence a Loschmidt echo without qualification.

## C. Replace the current Bloch derivation

Use one observable throughout. For example,

$$
\langle\sigma_z\rangle_{\rm ss}
=
\frac{1+\Delta^2T_2^2}
{1+\Delta^2T_2^2+\Omega_0^2T_1T_2},
$$

and

$$
P_e(\Delta)
=
\frac{1-\langle\sigma_z\rangle_{\rm ss}}{2}.
$$

The exact angular-frequency FWHM is

$$
\Gamma_{\rm FWHM}
=
\frac{2}{T_2}
\sqrt{1+\Omega_0^2T_1T_2}.
$$

Therefore,

$$
\Gamma_{\rm weak}=\frac{2}{T_2},
\qquad
\Gamma_{\rm strong}\simeq
2\Omega_0\sqrt{\frac{T_1}{T_2}}.
$$

Use the saturation parameter $s=\Omega_0^2T_1T_2$ to define weak and strong driving. The present conditions comparing $\Omega_0$ only with $\Delta$ are incomplete.

Then state the cyclic-frequency result separately:

$$
\Gamma_{\rm FWHM}^{(\mathrm{Hz})}
=
\frac{\Gamma_{\rm FWHM}^{(\mathrm{rad/s})}}{2\pi}.
$$

That will eliminate the current $T_2^{-1}$, $2/T_2$, and $1/(\pi T_2)$ ambiguity.

## D. Rewrite the adiabatic section and isolate $n=1/2$

The published scaling

$$
\Delta_{1/2}\tau
\propto
(\Omega_0\tau)^{-1/(2n-1)}
$$

is derived for Lorentzian powers satisfying $n>1/2$. It cannot be evaluated at $n=1/2$, the root-Lorentzian used here. The underlying theory explicitly excludes the marginal $1/|t|$ tail. The [original 2013 result](https://doi.org/10.1016/j.optcom.2012.09.040) and the [subsequent experimental treatment by Mihov and Vitanov](https://arxiv.org/abs/2308.14187) both state this restriction; the latter uses values approaching but not reaching $1/2$.

Consequently, I would:

- present the known scaling only for $n>1/2$;
- state that the truncated $n=1/2$ echo pulse is a distinct marginal case;
- avoid claiming that the divergent exponent predicts its behavior;
- determine an effective numerical scaling over the experimental amplitude range, for several $c$ and $T_2$, rather than claiming an unsupported asymptotic law.

The corrected adiabatic definitions should use

$$
\epsilon(t)=\sqrt{\Delta^2+\Omega^2(t)},\qquad
\vartheta(t)=\frac12\arctan2[\Omega(t),\Delta],
$$

and

$$
\dot{\vartheta}(t)
=
\frac{\Delta\dot{\Omega}(t)}
{2[\Delta^2+\Omega^2(t)]}.
$$

Keep all Hamiltonians either in energy units or angular-frequency units, never a mixture.

## E. Replace the cutoff argument

The present cutoff inequality should not be retained. A more relevant established estimate for truncation-induced excitation is

$$
P_c(\Delta)\simeq
\frac{\Omega_c^2}{\Omega_c^2+\Delta^2}
\left[1-P_0(\Delta)\right],
\qquad
\Omega_c=c\Omega_0.
$$

This shows directly that the endpoint artifact becomes important when $\Omega_c$ is comparable to the linewidth scale. See [Mihov and Vitanov](https://arxiv.org/abs/2308.14187).

Also correct the Figure 7 interpretation: decreasing $c$, not increasing it, reduces the endpoint jump. The cutoff concerns the beginning and end of the waveform; it does not “smooth the phase inversion point.”

## F. Add the numerical model explicitly

The appendix should reproduce what `echospec/simulation/run.py` actually solves:

$$
\dot\rho=-\frac{i}{\hbar}[H(t),\rho]
+\mathcal D[\sqrt{1/T_1}\,a]\rho
+\mathcal D[\sqrt{2/T_\phi}\,a^\dagger a]\rho,
$$

with

$$
\frac1{T_2}=\frac1{2T_1}+\frac1{T_\phi}.
$$

Include a table of pulse length, cutoff, $\Omega_0$, $T_1$, $T_2$, detuning grid, time steps, and solver tolerances.

Crucially, `echospec/simulation/operators.py` currently sets `N_dim = 2`. Therefore, the anharmonicity term is inactive and this model cannot establish leakage or two-photon-transition claims. Either add a converged three- or four-level simulation or describe the high-amplitude feature cautiously as an observed strong-drive distortion.

## G. Add an analysis-method section

Document:

- whether the response is a peak, dip, or composite feature;
- baseline and contrast definition;
- direct half-depth versus Gaussian-fit FWHM;
- fit window and initial parameters;
- rejection criteria;
- frequency-center estimator;
- bootstrap or covariance-based uncertainty.

For non-Lorentzian, oscillatory responses, a Gaussian fit should be presented as an operational definition—not as the physical lineshape.

## H. Replace generic setup prose with experimental parameters

The refrigerator description can be shortened substantially. Add a compact table containing:

- measured qubit transition frequency;
- anharmonicity;
- readout-resonator frequency;
- $T_1$, $T_2$, and $T_\phi$;
- pulse-calibration date and procedure;
- conversion from programmed amplitude to $\Omega_0$;
- number of shots and frequency points;
- AWG sample rate and phase-switch implementation;
- readout fidelity or assignment error.

The large generic setup schematic can remain in supplemental material if needed.

## I. Consolidate the limitations

Combine the existing “Pulse length” and “Higher amplitudes” sections into a single limitations section covering:

- Fourier broadening for short pulses;
- reduced contrast and relaxation for long pulses;
- cutoff-induced broadening;
- phase-jump bandwidth and waveform distortion;
- AC Stark shifts and multilevel leakage at high drive;
- the range in which a two-level model remains valid.

The current claim of a two-photon transition should be removed or softened unless a multilevel calculation reproduces the feature. The two-level production model cannot test that interpretation.

## What I would remove or consolidate

- Shorten the generic refrigerator description and replace it with a device-parameter table.
- Combine “Pulse length” and “Higher amplitudes” into a limitations section.
- Remove the unsupported “two-photon transition” interpretation unless a multilevel calculation reproduces it.
- Remove phrases such as “$\mathcal O(1)$ agreement”; give a numerical ratio and uncertainty.
- Give every figure and section a unique semantic label.
- Make this a real appendix using `\appendix`, placed before the bibliography, or create a separate Supplemental Material document.

## Recommended order of work

The first three changes should be:

1. Add the exact finite-pulse definition.
2. Replace the Bloch derivation.
3. Add the resonant cancellation proof.

Those changes would give the appendix a much stronger logical spine before adding further figures or technical detail.
