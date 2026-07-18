# Paper Review: Amplitude-Robust Qubit Spectroscopy

The core idea is compelling, but I would treat the current manuscript as needing a major revision before submission. The biggest issue is not prose—it is that the theory and the advertised “amplitude robustness” are not yet demonstrated rigorously enough.

## Highest-priority improvements

### 1. Repair and standardize the theory

- The power-narrowing scaling in `sections/introduction.tex` becomes singular for the paper’s chosen \(n=1/2\).
- The Bloch-equation section mixes \(\langle\sigma_z\rangle\), excited-state population, saturation limits, HWHM, and FWHM. Equations 4–8 are not mutually consistent.
- The adiabatic-basis eigenvalues, transformed Hamiltonian, \(\dot\vartheta\), and units require rederivation.
- The cutoff inequality has its direction reversed as written.
- Choose angular or cyclic frequencies throughout and track every \(2\pi\). Likewise, define whether \(\Delta_{1/2}\) is FWHM or HWHM. The figures normalize the “\(T_2\) limit” by \(T_2^{-1}\), while the appendix derives a FWHM of \(2/T_2\).

### 2. Directly demonstrate the claim in the title

The paper currently shows that linewidth is relatively insensitive to amplitude. That is not yet the same as showing robust frequency estimation.

The most valuable new figure would plot:

- extracted resonance-frequency error versus drive amplitude over the claimed three decades;
- repeated-run error bars or confidence intervals;
- sensitivity to deliberate amplitude miscalibration;
- square, root-Lorentzian, and echo-root-Lorentzian results under identical conditions.

This would make “amplitude robust” a measured result rather than an inference from FWHM.

### 3. Use a statistically defensible performance metric

\(\Delta_{1/2}/A_{\rm peak}\) is a useful heuristic, but it is not generally equivalent to inverse Fisher information or the Cramér–Rao bound. Either remove that claim or compute Fisher information using the measured probability slope and noise model. Ideally report frequency-estimation bias and RMSE per shot or per unit acquisition time.

### 4. Make the comparison fair and operational

The abstract claims faster spectroscopy, but the manuscript does not yet compare equal-resource measurements. Benchmark square spectroscopy, the two Lorentzian variants, and Ramsey using the same total shots or wall-clock time. Include pulse duration, number of frequency/time points, estimator, and resulting frequency uncertainty.

### 5. Add enough methods for reproduction

Explicitly document the finite pulse:

- normalization and meaning of \(\tau\);
- total duration and cutoff definition;
- implementation and bandwidth of the phase inversion;
- conversion from programmed amplitude to \(\Omega_0\);
- detuning grid, shot count, readout and fitting procedure;
- confidence intervals and failed-fit policy;
- master equation, number of transmon levels, \(T_1\), \(T_2\), anharmonicity, and all simulation parameters.

The echo cancellation should also be described as exact only for an ideal symmetric, unitary, resonant evolution; relaxation, dephasing, waveform distortion, and phase-switch imperfections prevent exact reversal.

### 6. Reorganize the narrative

Right now `sections/introduction.tex` contains the introduction, protocol, theory, results, discussion, and conclusion. I would use:

1. Introduction and precise contribution
2. Pulse protocol and physical mechanism
3. Experimental and numerical methods
4. Linewidth results
5. Frequency-estimation robustness and speed benchmark
6. Limitations and conclusion

Move the refrigerator schematic, long Bloch derivation, cutoff sweeps, pulse-length sweeps, and high-amplitude behavior to a genuine appendix or supplement.

### 7. Strengthen and temper the claims

- Define the exact endpoints behind “more than three orders of magnitude”; the visible comparison looks closer to exactly \(10^3\).
- “Shorter than standard techniques,” “excellent agreement,” and “reliability” need quantitative measures.
- If the experiment uses only one of the 11 qubits, demonstrate another qubit/device or explicitly limit the generality claim.
- Distinguish the Fourier spectrum of the applied waveform from the nonlinear qubit response.
- The conclusion says operation approaches neighboring transitions, while the appendix identifies two-photon-transition distortions there; present that as a limitation.

### 8. Improve the central figures and submission hygiene

- Make Figure 4 the visual centerpiece: larger panels, experimental uncertainties, a \(T_2\)-limit uncertainty band, and an amplitude-independence fit.
- Increase labels in Figures 2–4 and distinguish data from simulation more clearly.
- Remove `showkeys`, hyperlink boxes, `[999]` placeholders, duplicate labels, and two unresolved citations.
- Put the bibliography last. `main.tex` currently places it before six ordinary numbered sections because `\appendix` is commented out.
- Correct the PDF metadata (`Article`, `Author`), duplicated reference labels, typography, inconsistent pulse naming, and extensive grammatical errors.

## Recommended order of work

If I prioritized ruthlessly: first correct the equations and linewidth conventions, then add the frequency-error-versus-amplitude benchmark. Those two changes would do more for the paper than extensive stylistic polishing.

## Line-specific technical findings

### Scaling is singular for the pulse used

**File:** `sections/introduction.tex`, lines 76–83

The exponent \(-1/(2n-1)\) is undefined at \(n=1/2\), which is precisely the root-Lorentzian studied throughout the paper. Derive the applicable \(n=1/2\) result separately or change the scaling argument and its claimed connection to the experiment.

### Bloch limits are internally inconsistent

**File:** `sections/appendix1.tex`, lines 35–81

The steady-state expression, its strong-drive approximation, and the subsequent linewidth formulas mix sigma-z and excitation-population conventions. In particular, the stated strong-drive approximation tends to \(-1\) on resonance although the preceding equation tends to zero under saturation. Re-derive the section from one observable and define FWHM versus HWHM explicitly.

### Re-derive the adiabatic-basis equations

**File:** `sections/appendix1.tex`, lines 141–163

The eigenvalues do not follow from the displayed Hamiltonian, the transformed Hamiltonian is not manifestly Hermitian and mixes energy and frequency units, and the displayed derivative of the mixing angle has the wrong denominator. These equations should be replaced by a dimensionally consistent derivation.

### Cutoff inequality direction is reversed

**File:** `sections/appendix1.tex`, lines 95–109

From the two displayed linewidths, \(\Delta_{T_2}<\Delta_{\Omega}\) implies \(\Omega_c>1/\sqrt{T_1T_2}\), not less than. If the intended requirement is that cutoff broadening stay below the coherence linewidth, reverse the premise and clarify whether \(\Omega\) values are angular or cyclic frequencies.

### Duplicate label changes cross-references

**File:** `sections/introduction.tex`, line 149

This repeats the `fig:fig4` label used by the preceding linewidth figure, so references can resolve to the wrong result. Give every figure and section a unique semantic label; the appendix also repeats `sec:develop` and `fig:placeholder`.
