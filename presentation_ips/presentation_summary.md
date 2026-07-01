# 10-Minute Presentation Plan: Echo-Root-Lorentzian Qubit Spectroscopy

## One-Sentence Thesis

This work introduces an echo-root-Lorentzian pulse that enables near-T2-limited qubit spectroscopy at drive amplitudes more than three orders of magnitude larger than conventional spectroscopy, by cancelling resonant Rabi oscillations while preserving a useful off-resonant spectral response.

## Recommended Timing

Use **7 slides total**. Aim for about **1 minute per slide**, with extra time for the main result and conclusion.

| Slide | Topic | Time |
|---|---:|---:|
| 1 | Motivation and problem | 1:00 |
| 2 | Standard spectroscopy trade-off | 1:15 |
| 3 | Pulse idea: root-Lorentzian plus echo phase jump | 1:30 |
| 4 | Physical mechanism | 1:15 |
| 5 | Main spectroscopy comparison | 1:45 |
| 6 | Quantitative linewidth and robustness | 1:30 |
| 7 | Practical limits and conclusion | 1:45 |

Total: **10:00**

## Slide 1 - Motivation: Why Qubit Frequency Calibration Matters

**Time:** 1:00

**Purpose:** Quickly establish why this problem matters.

**Show:**

- Title: *Amplitude-Robust Qubit Spectroscopy Using Echo-Root-Lorentzian Pulse Shapes*
- Authors and affiliation.
- One simple visual: qubit energy-level diagram or a resonance peak.

**Say:**

- Accurate qubit frequency characterization is required for high-fidelity control.
- If the drive frequency is detuned from the qubit transition, gate fidelity drops.
- The goal is to estimate the transition frequency precisely, quickly, and robustly.

**Key takeaway:**

- Spectroscopy is not just diagnostic; it is part of the calibration stack for high-fidelity quantum gates.

## Slide 2 - The Problem: Resolution Versus Robustness

**Time:** 1:15

**Purpose:** Explain the central limitation of conventional spectroscopy.

**Show:**

- A schematic comparison: weak drive = narrow but low signal; strong drive = high signal but broadened line.
- Optional: use the square-pulse part of Figure 2 or Figure 4.

**Say:**

- In conventional driven spectroscopy, increasing the drive amplitude improves contrast and robustness to measurement noise.
- But stronger driving causes power broadening: the linewidth grows with the Rabi frequency.
- To approach the T2-limited linewidth, standard spectroscopy must use weak drive amplitudes.
- That makes precision and robustness compete with each other.
- Ramsey spectroscopy can reach the T2 limit, but it needs long interrogation times and dense time sampling, and it can be biased by pulse errors, phase fluctuations, and AC Stark shifts.

**Key takeaway:**

- The challenge is to keep T2-limited resolution without being forced into a fragile weak-drive measurement.

## Slide 3 - The Proposed Pulse

**Time:** 1:30

**Purpose:** Introduce the root-Lorentzian and echo-root-Lorentzian pulse.

**Show:**

- Figure 1(a): time-domain pulse envelopes.
- If there is room, include Figure 1(b): spectral response with the resonance null.

**Say:**

- The base pulse is a Lorentzian-derived envelope:

```text
L^(n)(t) = [1 / (1 + (t/tau)^2)]^n
```

- The work focuses on the root-Lorentzian case, `n = 1/2`.
- Lorentzian pulses are interesting because, unlike square pulses, they can show power narrowing.
- The new ingredient is the echo version:

```text
L_echo^(1/2)(t) = L^(1/2)(t) * sgn(t)
```

- This means the pulse has a pi-phase inversion at the peak.
- The first half and second half have opposite phase.

**Key takeaway:**

- The method changes the spectroscopy problem by using pulse shape and phase symmetry, not just drive amplitude.

## Slide 4 - Mechanism: Resonant Cancellation, Off-Resonant Response

**Time:** 1:15

**Purpose:** Give the physical intuition without spending too much time on derivations.

**Show:**

- Figure 3: simulation comparing root-Lorentzian and echo-root-Lorentzian dynamics.

**Say:**

- On resonance, the two halves of the echo pulse approximately undo each other.
- This cancels coherent Rabi oscillations at zero detuning.
- Away from resonance, detuning breaks the exact cancellation.
- The final excited-state population therefore remains sensitive to frequency.
- The resonance appears as a sharp dip or depletion feature, instead of simply a maximum excitation peak.
- This is similar in spirit to a Loschmidt echo: reversibility is strongest at resonance and imperfect away from it.

**Key takeaway:**

- The protocol detects resonance through phase-sensitive cancellation, not direct population transfer.

## Slide 5 - Main Result: Comparing Square, Root-Lorentzian, and Echo Pulses

**Time:** 1:45

**Purpose:** Show the main qualitative experimental result.

**Show:**

- Figure 2 as the central slide figure.

**Say:**

- The square pulse shows strong power broadening even at relatively small drive amplitudes.
- The root-Lorentzian pulse maintains much narrower spectral features because of power narrowing.
- However, the non-echo root-Lorentzian still has coherent oscillatory artifacts as the amplitude changes.
- The echo-root-Lorentzian suppresses these vertical oscillations.
- The result is a smoother and more stable spectroscopic feature over a broad amplitude range.
- For the measured coherence time, `T2 approximately 14 us`, the T2-limited linewidth is about `20 kHz`.

**Key takeaway:**

- The echo-root-Lorentzian pulse keeps the response close to the T2-limited linewidth while remaining usable at much larger amplitudes.

## Slide 6 - Quantitative Robustness

**Time:** 1:30

**Purpose:** State the numerical result clearly.

**Show:**

- Figure 4: linewidth versus peak Rabi frequency and linewidth-to-signal ratio.
- Optionally mention Figure 5 if discussing parameter robustness.

**Say:**

- Square pulses broaden approximately with drive amplitude, as expected from conventional power broadening.
- Lorentzian and echo-Lorentzian pulses show power narrowing instead.
- The echo pulse gives a better linewidth-to-signal trade-off and smoother behavior across amplitude.
- The key claim is that the echo protocol broadens the usable drive-amplitude window by approximately three orders of magnitude while maintaining high spectral resolution.
- This makes the method more tolerant to amplitude calibration errors and experimental drift.

**Key takeaway:**

- The method turns strong drive from a source of linewidth degradation into a usable, robust operating regime.

## Slide 7 - Practical Constraints and Conclusion

**Time:** 1:45

**Purpose:** Close with what matters experimentally and the final contribution.

**Show:**

- Use Figure 5 if you want to emphasize the robust operating island.
- Use Figure 6 only if the audience needs the hardware setup.
- Do not spend time on all appendix figures in a 10-minute talk.

**Say:**

- The ideal Lorentzian pulse has infinite support, so experiments require a temporal cutoff.
- If the cutoff amplitude is too large, the cutoff behaves partly like a square-pulse edge and reintroduces broadening.
- The paper finds that relative cutoffs around `10^-4` to `10^-5` are appropriate depending on pulse parameters, and less than `10^-3` is often sufficient empirically.
- Pulse duration must also be chosen carefully: long enough to avoid Fourier broadening, but not so long that T1 decay destroys contrast.
- At very high amplitudes, above roughly `50 MHz` in this experiment, two-photon transitions and higher-level effects begin to distort the response.
- The final result is a spectroscopy protocol that is fast, amplitude robust, and near the T2 limit.

**Final sentence:**

- Echo-root-Lorentzian spectroscopy uses phase-sensitive off-resonant response rather than direct population transfer, allowing high-resolution qubit frequency calibration under strong driving.

## What to Skip in a 10-Minute Talk

- Full Bloch-equation derivation of the T2 limit.
- Detailed adiabatic-basis derivation.
- Full measurement-chain explanation, unless the audience is experimental and expects it.
- All supplementary cutoff and pulse-length figures; mention the constraints verbally instead.
- Detailed reference discussion.

## Figures to Use

Use at most **4 figures**:

1. **Figure 1:** Pulse shape and spectrum.
2. **Figure 2:** Main spectroscopy maps.
3. **Figure 3:** Mechanism simulation.
4. **Figure 4:** Quantitative linewidth comparison.

Optional if replacing another figure:

- **Figure 5:** Parameter robustness.
- **Figure 6:** Experimental setup.

## 30-Second Backup Explanation

If someone asks for the mechanism in one sentence:

> The pi phase jump makes the resonant evolution reverse itself, so on resonance the pulse cancels its own excitation, while detuning breaks this symmetry and leaves a measurable final population difference.

## 60-Second Backup Explanation of the T2 Limit

The coherence time T2 sets the natural homogeneous linewidth of the qubit transition. In weak-drive spectroscopy, the narrowest resolvable linewidth is therefore approximately set by `1/T2`. Conventional strong-drive spectroscopy exceeds this limit because the drive itself broadens the line through power broadening. The point of the echo-root-Lorentzian pulse is not to beat the T2 limit, but to remain near it while using much stronger drive amplitudes.

## Short Abstract for the Talk

This 10-minute presentation introduces an amplitude-robust qubit spectroscopy method based on echo-root-Lorentzian pulse shaping. Standard driven spectroscopy suffers from a trade-off between signal strength and spectral resolution because stronger drive amplitudes cause power broadening. The proposed pulse uses a root-Lorentzian envelope with a pi-phase inversion at its peak, producing an echo-like cancellation of resonant Rabi oscillations while preserving an off-resonant spectral response. Experiments on superconducting qubits show near-T2-limited linewidths across drive amplitudes more than three orders of magnitude larger than those usable in standard spectroscopy, enabling faster and more robust qubit frequency calibration.
