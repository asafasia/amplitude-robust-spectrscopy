# Flexible-control pilot and symmetry notes

This note concerns the current target: absolute contrast above 0.4 at an actual
50 MHz peak, with no FWHM or contrast regression relative to the original pulse
over 5–60 MHz. It does not impose a 0.4 contrast floor at every amplitude.

## Continuous rotation and the echo identity

For a continuous control Hamiltonian, exact odd time symmetry requires
\(H_c(0)=-H_c(0)\), hence \(H_c(0)=0\). A nonzero continuous drive at the
echo center cannot retain this particular symmetry. This is a mathematical
obstruction to combining two properties, not a prohibition of robust control
by other mechanisms.

For even amplitude \(g(t)\), even phase \(\phi(t)\), and odd transition
angle \(\theta(t)\), consider the raw complex control

\[
v(t)=g(t)e^{i\phi(t)}[-\sin\theta(t)+i\eta\cos\theta(t)].
\]

The first term is odd and the second is even. At \(\eta=0\), the ideal
resonant qubit evolution of the second half exactly reverses the first half,
for every common amplitude scale. For \(\eta\ne0\),
\(v(0)=i\eta g(0)e^{i\phi(0)}\), and that automatic cancellation is lost.
Both signs of η should be permitted: they describe opposite IQ rotation
directions, and the DRAG correction distinguishes them.

For a peak-normalized dimensionless drive \(u(t)\), in the convention
\(H_c/\hbar=\pi A[\Re u\,\sigma_x+\Im u\,\sigma_y]\), the small-amplitude
expansion from an initial ground state gives

\[
P_e(0;A)=\pi^2 A^2\left|\int u(t)\,dt\right|^2+O(A^3).
\]

Here A is in MHz and time is in µs; the expansion is about A=0, not a claimed
quantitative approximation at 50 MHz. The raw odd contribution integrates to
zero, while the continuous bridge contributes

\[
\int v(t)\,dt=i\eta\int g(t)\cos\theta(t)e^{i\phi(t)}\,dt.
\]

With constant phase, positive g, and a monotone semicircle, this integral is
nonzero. Thus a plain continuous semicircle introduces a leading
amplitude-dependent resonant rotation. Flexible phase can balance its complex
area, and time ordering can support other compensation mechanisms, but both
must be optimized and validated across the amplitude ensemble.

## A useful small-η seed including DRAG

Write the full drive as \(D=v+ibv'\), where
\(b=\beta/(2\pi|\alpha|)\). At the symmetric center,

\[
D(0)=i g(0)e^{i\phi(0)}[\eta-b\theta'(0)].
\]

Consequently \(\eta=b\theta'(0)\) cancels the central DRAG quadrature.
For β=−0.5 and \(|\alpha|=216\) MHz, the old 16 ns linear-angle transition
gives η≈−0.07234. If the new angle uses the quintic smoothstep
\(S(x)=6x^5-15x^4+10x^3\) over total width τ,
\(\theta'(0)=1.875\pi/\tau\); a 30 ns transition has the same center slope
and the same cancellation seed. Matching central slopes therefore suggests
\(\tau_{\mathrm{new}}\approx1.875\tau_{\mathrm{old}}\).

This is only local cancellation. For zero endpoints,
\(\int D\,dt=\int v\,dt\), since the integrated derivative is zero.
The central cancellation does not remove the bridge's net-area contribution
or restore global odd symmetry.

Useful initial η values are 0, ±0.05, ±0.15, and ±0.35, together with the
central-DRAG cancellation seed. Retain ±1 as explicitly different continuous
semicircle trials rather than assuming that a full rotation will be robust.

An optional alternative is an area-balanced even bridge,

\[
q_{\mathrm{even}}(t)=\eta[w_0(t)-\kappa w_1(t)],\qquad
\kappa=\frac{\int g(t)w_0(t)\,dt}{\int g(t)w_1(t)\,dt}.
\]

Here \(w_0\) is centered, \(w_1\) has symmetric shoulder support, and both
close smoothly. For constant phase over their support, this retains a nonzero
center while cancelling the first-order transverse area. With varying phase,
two shoulder coefficients can instead cancel the real and imaginary parts of
the weighted area. Higher-order errors remain. Avoid near-zero denominators,
inspect possible shoulder overshoot, and normalize the complete final waveform.

## Completed bounded pilot

The reproducible script `pilot_transition_phase.py` performed exactly 24 native
qutrit candidate simulations, reusing the saved baseline on the same grid.
It started from the previously validated narrow DecoupledEchoPulse, varied the
old sine-transition width from 4 to 256 ns, and tested compact even Gaussian
phase perturbations. All comparisons used actual complex-waveform peak
normalization. This pilot did not test the new continuous-IQ family.

For zero added core phase:

| Old transition | Contrast at 5 MHz | FWHM at 50 MHz | Contrast at 50 MHz | Feasible on pilot grid? |
|---:|---:|---:|---:|:---:|
| 4 ns | 0.22136 | 135.920 kHz | 0.41615 | No |
| 8 ns | 0.22991 | 136.340 kHz | 0.41428 | No |
| 16 ns | 0.25973 | 137.672 kHz | 0.40378 | Yes |
| 32 ns | 0.34192 | 140.397 kHz | 0.37706 | No |
| 64 ns | 0.47718 | 147.082 kHz | 0.33183 | No |
| 128 ns | 0.52622 | 162.611 kHz | 0.27703 | No |
| 256 ns | 0.48418 | 197.110 kHz | 0.22675 | No |

Only the unchanged 16 ns, zero-added-phase seed met the pilot constraints.
The added phase probes did not provide a standalone improvement. Faster
transitions traded away low-drive contrast; slower ones traded away contrast
at 50 MHz. A 4 ns transition also raised maximum final leakage to about 0.0048
in this model.

A productive untested joint seed is the old 8 ns transition (approximately
15 ns in the quintic-angle convention) plus a modest positive log-amplitude
shoulder bump around 50–150 ns. For example,

\[
\log g_{\mathrm{new}}(t)=\log g_{\mathrm{seed}}(t)
+a e(t/\tau_s)^2e^{-(t/\tau_s)^2},
\quad a\in[0.2,0.6],\quad\tau_s\in[0.05,0.15]\;\mathrm{µs}.
\]

The bump is even, zero at the center, peaks at \(|t|=\tau_s\), and dies away
in the far tails. It may recover low-drive rotation area without increasing
the central peak proportionally. Actual peak normalization can still couple
these changes, so this is a proposed optimization seed, not a validated gain.

Saved JSON contains all metrics, pulse parameters, source hashes, constraints,
and diagnostics. The NPZ contains the complete population grids. Coarse pilot
feasibility is not dense-grid or independent-solver validation, and this finite
pilot establishes no impossibility result.
