# Why separate the core, tails, and phase?

This note distinguishes an ideal-qubit derivation from the dissipative qutrit
calculations used to evaluate the pulse. It does not establish an optimum or a
no-go result for the requested 5–60 MHz amplitude range.

## Ideal single-echo symmetry

Let the pulse occupy \(t\in[-T/2,T/2]\), and write the ideal qubit Hamiltonian as

\[
\frac{H_A(t;\delta)}{\hbar}
=\frac12\left\{\Omega_A(t)
[\cos\phi(t)\,\sigma_x+\sin\phi(t)\,\sigma_y]
+\delta\sigma_z\right\}.
\]

Here \(\Omega_A(t)\) is the **signed angular Rabi rate**, and
\(\delta=2\pi\Delta f\) is the angular detuning. With time measured in
microseconds, \(\Omega_A\) and \(\delta\) have units rad/µs, and
\(\Delta f\) has units MHz. The amplitude label \(A\) identifies a scaled
waveform; the numerical model separately normalizes the complete complex
waveform, including DRAG, to its requested actual peak.

An odd signed envelope and an even phase obey

\[
\Omega_A(-t)=-\Omega_A(t),\qquad \phi(-t)=\phi(t).
\]

Therefore the resonant control Hamiltonian satisfies
\(H_A(-t;0)=-H_A(t;0)\). The second half reverses the first half's ordered
evolution, giving

\[
U_A(T/2,-T/2;0)=I
\]

for every common amplitude scale. This cancellation does not require the
Hamiltonians at different times to commute. It allows smooth phase shaping
inside one echoed waveform while preserving the ideal resonant cancellation.

## Detuning sensitivity and useful duration

Let \(U_c(t)=U_A(t,-T/2;0)\), and define the real unit vector
\(\boldsymbol n_A(t)\) through

\[
U_c^\dagger(t)\sigma_zU_c(t)
=\boldsymbol n_A(t)\cdot\boldsymbol\sigma.
\]

In the control's interaction frame, the first-order expansion in detuning is

\[
U_A(T/2,-T/2;\delta)
=I-\frac{i\delta}{2}\boldsymbol K_A\cdot\boldsymbol\sigma
+O((\delta T)^2),\qquad
\boldsymbol K_A=\int_{-T/2}^{T/2}\boldsymbol n_A(t)\,dt.
\]

For an initial ground state and an excited-state population measurement,

\[
P_e(\Delta f)
=\frac{\delta^2}{4}(K_{A,x}^2+K_{A,y}^2)
+O((\delta T)^4)
=\pi^2(\Delta f)^2T_{\mathrm{eff},A}^2
+O((2\pi\Delta f T)^4),
\]

where

\[
T_{\mathrm{eff},A}=\sqrt{K_{A,x}^2+K_{A,y}^2}\le T.
\]

The even remainder follows from the ideal symmetry:
\(H_A(t;-\delta)=-H_A(-t;\delta)\), hence
\(U_A(-\delta)=U_A(\delta)^\dagger\); the two transition probabilities of
a two-state unitary have equal magnitude. In particular,
\(P_e''(0)=2\pi^2T_{\mathrm{eff},A}^2\) when differentiating with respect
to frequency \(\Delta f\) in MHz.

For a constant phase chosen as zero,

\[
\theta_A(t)=\int_{-T/2}^{t}\Omega_A(s)\,ds,\qquad
T_{\mathrm{eff},A}
=\left|\int_{-T/2}^{T/2}\sin\theta_A(t)\,dt\right|.
\]

This makes the distinction between nominal duration and effective detuning
encoding explicit. Long intervals with \(\theta_A\) close to zero or π add
little transverse sensitivity. Contributions of opposite sign can cancel.
A longer nominal pulse therefore need not have a larger useful encoding time.
This \(T_{\mathrm{eff},A}\) is a susceptibility diagnostic for the ideal
qubit, not a replacement for the physical coherence time \(T_2\).

Independent core and tail parameters help because they can change different
parts of this integral. The core changes the rapid rotations and orientation
of \(\boldsymbol n_A\); weak tails can extend intervals that contribute to
detuning encoding. Phase shaping steers the direction of those contributions,
and the edge taper controls how the waveform closes. These roles overlap, so
they should be optimized jointly. A single polynomial exponent couples them
and can force a trade between useful tail evolution and the central response.
No Fourier-transform linewidth of an envelope alone describes the general
strongly driven population dip.

## Two independent phase scales

The existing even two-Gaussian phase can separate a broad phase profile from a
compact core profile:

\[
\phi(t)=a_{\mathrm{tail}}e^{-(t/\tau_{\mathrm{tail}})^2}
+a_{\mathrm{core}}e^{-(t/\tau_{\mathrm{core}})^2}.
\]

For one term \(a e^{-(t/\tau)^2}\),

\[
\dot\phi(t)=-\frac{2at}{\tau^2}e^{-(t/\tau)^2},\qquad
\max_t\frac{|\dot\phi(t)|}{2\pi}
=\frac{\sqrt{2/e}}{2\pi}\frac{|a|}{\tau}
\simeq0.13652\frac{|a|}{\tau}\;\mathrm{MHz},
\]

with \(a\) in radians and \(\tau\) in µs. For the Hamiltonian convention
above, transformation to the instantaneous phase frame gives an effective
detuning \(\delta-\dot\phi\). For two terms, the displayed individual
excursions bound the combined excursion by their sum; the actual maximum
should be measured from their combined derivative.

A useful search extension is
\(a_{\mathrm{core}}\in[-\pi,\pi]\) and
\(\tau_{\mathrm{core}}\in[0.01,0.2]\) µs, while retaining the independent
broad phase. Example seeds are ±1.6 or ±2.4 radians at 0.1 µs; their individual
maximum excursions are approximately 2.18 and 3.28 MHz. This is a proposed
control degree of freedom, not a guarantee of better contrast or linewidth.
Even phase preserves the raw symmetry above, but does not make the pulse fully
adiabatic: the odd signed drive and the odd phase derivative both vanish at
the ideal echo center, where the resonant instantaneous gap closes.

## What the T2 reference does and does not mean

An exponential free-coherence envelope has the Fourier transform

\[
\int_{-\infty}^{\infty}e^{-|t|/T_2}e^{-i2\pi\Delta f t}\,dt
=\frac{2T_2}{1+(2\pi\Delta f T_2)^2}.
\]

Its Lorentzian full width at half maximum is

\[
\Delta f_{\mathrm{hom}}=\frac{1}{\pi T_2}
\simeq43.54\;\mathrm{kHz}\qquad(T_2=7.31\;\mathrm{µs}).
\]

This is the homogeneous weak-drive linewidth reference. It is neither a
proved attainable width nor a proved lower bound for this finite-duration,
strongly driven echo dip subject to absolute contrast above 0.4 and amplitude
robustness. The measured dip depends on the off-resonant background, the
dissipative central population, satellites, and half-depth crossings. Its
local ideal curvature does not determine those quantities. Reaching the
reference under the requested constraints requires an actual feasible pulse;
failure of a parameter search does not establish impossibility.

## Numerical and acceptance boundaries

The ideal derivation omits relaxation, dephasing, additional transmon levels,
and the DRAG correction. In particular, the derivative of the odd raw complex
control is even, so adding derivative DRAG does not preserve the exact raw
odd-control symmetry. The transmon anharmonic drift is not time reversed
either. The full dissipative model must determine linewidth, contrast,
frequency shifts, and leakage. Multiplying the ideal sensitivity integrand
by an assumed exponential decay is not an exact replacement for that solver.

The current strict numerical target uses one fixed waveform across actual
peaks 5–60 MHz, absolute contrast above 0.4 throughout that range, and no
width or contrast regression against the original baseline, subject to the
documented numerical tolerances and the additional satellite, center, and
leakage checks. Discrete training points require independent denser validation.
A candidate that improves 50 MHz but fails the full-range conditions is a
tradeoff or fallback candidate, not a successful solution of the strict task.
Any fallback should state exactly which constraints fail and should not
silently replace the requested target.
