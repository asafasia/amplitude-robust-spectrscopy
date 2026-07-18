# Plot Ideas for the Paper

This note collects candidate figures for the amplitude-robust spectroscopy
paper. The central visual story should be that the echo-root-Lorentzian pulse
creates a narrow resonant depletion feature whose position and width remain
stable over a broad range of drive amplitudes.

## Core plot ideas

### 1. One-dimensional spectroscopy: simulation and experiment

Overlay experimental data and numerical simulation for representative
echo-root-Lorentzian spectroscopy traces.

- Show at least one low-amplitude, narrow trace and one high-amplitude, broad
  or distorted trace.
- Center the horizontal axis at the fitted resonance, using detuning rather
  than absolute frequency when possible.
- Clearly mark the central dip, its fitted center, and its FWHM or half-depth
  width.
- Use markers with uncertainty bars for experiment and solid lines for
  simulation.
- If the background changes strongly with amplitude, normalize only in an
  auxiliary panel; retain the raw signal in the main comparison.

**Purpose:** establish experiment--simulation agreement while making the
narrow dip and the transition to the high-amplitude regime immediately
visible.

### 2. Two-dimensional echo spectroscopy versus amplitude

Plot the measured qubit response as a function of drive detuning and peak Rabi
frequency (or calibrated drive amplitude).

- Use a logarithmic amplitude axis if the scan spans several decades.
- Choose a perceptually uniform color map and show the raw response rather
  than independently normalizing every row.
- Overlay the extracted dip center and its width as lines or markers.
- Mark the high-amplitude region where the dip broadens, splits, loses
  contrast, or becomes distorted.
- Add a small number of horizontal guide lines that correspond to the slices
  shown in Plot 1.

**Purpose:** show the full amplitude-robust operating window and its eventual
high-drive limitation in one panel.

### 3. Echo waveform in time and frequency

Use paired panels for the signed echo-root-Lorentzian waveform
$\Omega(t)$ and its Fourier transform $|\widetilde{\Omega}(f)|$.

- In the time-domain panel, mark the pulse duration, endpoint cutoff, and
  $\pi$ phase inversion at the midpoint.
- In the frequency-domain panel, plot frequency relative to the carrier and
  show the central spectral null and adjacent lobes.
- State the FFT normalization, sampling interval, zero-padding convention,
  and whether the plotted quantity is amplitude or power spectral density.

**Purpose:** connect the phase-inverted time-domain pulse to its notch-like
frequency-domain structure.

### 4. Fourier comparison of square and echo pulses

Compare the spectra of a square pulse and the echo-root-Lorentzian pulse using
the same duration and a clearly stated normalization.

- Show both linear and decibel views if the dynamic range is large.
- Mark the carrier, central null, main-lobe width, and relevant qubit
  transition frequencies.
- A second set of curves may show how increasing peak amplitude scales the
  available spectral weight.
- Avoid describing this panel alone as proof that a wider spectroscopy range
  is "opened": the FFT describes the applied waveform, whereas the measured
  spectroscopy also contains nonlinear qubit dynamics, relaxation, and
  dephasing.

**Purpose:** explain which frequencies are present in the applied drive and
why the echo pulse suppresses the response at the carrier while retaining
off-resonant spectral weight.

### 5. Two-dimensional FFT map versus cutoff and amplitude

The proposed variables form a three-dimensional data set
$(f, c, \Omega_0)$, so this idea should be implemented as small multiples
rather than an ambiguous single heat map.

- Option A: plot frequency versus cutoff as a heat map for several selected
  amplitudes.
- Option B: plot frequency versus amplitude for several selected cutoff
  values.
- Option C: reduce each FFT to meaningful metrics--central-null depth,
  main-lobe width, side-lobe power, or integrated spectral weight in a target
  band--and plot those metrics as cutoff-versus-amplitude heat maps.
- Distinguish a pure truncation sweep (fixed Lorentzian time scale, varying
  duration) from a fixed-duration sweep in which changing the endpoint cutoff
  also changes the pulse time scale.

**Purpose:** identify which combinations of cutoff and amplitude preserve the
desired spectral notch without excessive endpoint leakage or side lobes.

## Additional ideas from Codex

### 6. Resonance-frequency error versus drive amplitude

Plot the fitted frequency offset
$\hat f_0-f_{0,\mathrm{ref}}$ versus drive amplitude for square,
root-Lorentzian, and echo-root-Lorentzian pulses.

- Include repeated-run confidence intervals or bootstrap uncertainties.
- Use the same reference frequency and fitting policy for all pulse types.
- Add a horizontal acceptance band representing the desired calibration
  accuracy.

**Why this is especially valuable:** it directly tests the paper's
"amplitude-robust" claim. A stable linewidth is useful, but a stable and
accurate frequency estimate is the operational result.

### 7. Resolution, contrast, and estimation precision versus amplitude

Use aligned panels with a shared amplitude axis for:

1. dip width;
2. dip depth or contrast;
3. frequency-estimation uncertainty or RMSE per shot.

The third metric can be obtained from repeated measurements, bootstrapping, or
a Fisher-information calculation with an explicit measurement-noise model.
Plotting these quantities separately is more defensible than treating
linewidth divided by peak signal as automatically equivalent to inverse
Fisher information.

**Purpose:** reveal the trade-off between a narrow feature, usable signal, and
actual frequency precision.

### 8. Deliberate amplitude-miscalibration test

Choose a nominal operating point, apply controlled amplitude errors, and plot
the resulting frequency bias and linewidth change.

- Compare the three pulse families under identical conditions.
- Report both the programmed amplitude error and the calibrated Rabi-frequency
  error if available.

**Purpose:** turn robustness into a practical calibration test rather than an
inference from a parameter sweep.

### 9. Time-resolved mechanism at resonance and off resonance

Show the pulse envelope together with simulated excited-state population for
$\Delta=0$ and one or two nonzero detunings.

- At resonance, highlight the forward evolution and its cancellation after
  the phase inversion.
- Off resonance, show the incomplete cancellation that produces the
  spectroscopic signal.
- A Bloch-sphere inset is optional; population-versus-time should remain the
  quantitative panel.

**Purpose:** explain why the resonant response becomes a dip rather than a
peak.

### 10. Robust operating-region map

Plot cutoff versus peak amplitude and classify each point using simultaneous
requirements on:

- maximum allowed linewidth;
- minimum contrast;
- maximum frequency bias;
- successful and stable fit.

Show the resulting valid region as a contiguous operating island, with failed
or unreliable fits explicitly marked rather than assigned placeholder values.

**Purpose:** provide a practical design rule for selecting pulse parameters.

### 11. Experiment--simulation residual map

Alongside the two-dimensional spectroscopy comparison, show
$P_{\mathrm{exp}}-P_{\mathrm{sim}}$ using a symmetric color scale.

**Purpose:** make agreement quantitative and expose systematic discrepancies
such as amplitude-dependent offsets, imperfect phase inversion, leakage, or
readout drift.

### 12. Equal-resource comparison

Compare square, root-Lorentzian, echo-root-Lorentzian, and Ramsey frequency
estimation at equal total shots or equal wall-clock acquisition time.

**Purpose:** support any claim that the echo protocol is faster or more
efficient than standard spectroscopy or Ramsey interferometry.

## Suggested figure hierarchy

### Main text

1. **Pulse mechanism:** time-domain waveform, FFT, and time-resolved resonant
   cancellation.
2. **Central spectroscopy result:** the two-dimensional experimental map plus
   selected one-dimensional experiment--simulation slices.
3. **Amplitude robustness:** frequency error, linewidth, contrast, and
   estimation precision versus amplitude.
4. **Operating window:** cutoff-versus-amplitude robustness map.

### Supplemental Material

- full square-versus-echo FFT comparison;
- detailed FFT sweeps versus cutoff and amplitude;
- **fit-derived resonance stability:** a two-panel plot versus drive amplitude
  showing (a) the fitted resonance offset
  $\hat f_0-f_{0,\mathrm{ref}}$ and (b) the fitted FWHM. Use the same fit model
  and fit window for all amplitudes, include fit uncertainties, and explicitly
  mark rejected or unreliable fits;
- **FWHM extraction method:** explain the complete fitting procedure used to
  obtain the resonance offset and FWHM from every one-dimensional amplitude
  slice. Include the fit function, preprocessing, initial values, parameter
  bounds, conversion from fitted width to FWHM, and fit-rejection rules;
- experiment--simulation residual maps;
- high-amplitude failure modes;
- fit diagnostics and alternative linewidth definitions;
- amplitude-miscalibration and equal-resource benchmarks if they do not fit in
  the Letter.

## Supplemental explanation of the fit-derived FWHM

The Supplemental Material should include a short analysis-method subsection
next to the resonance-offset and FWHM plot. The following description matches
the current extraction implemented in `figures/utils.py`.

### Procedure used in the analysis

For each fixed drive-amplitude slice, the measured spectroscopy signal was
fitted as a function of detuning. The first and last 40 frequency points were
discarded to reduce sensitivity to the boundaries of the scan, and the
remaining trace was smoothed with a one-point Gaussian filter before fitting.

The line was described operationally by a Gaussian with a constant offset,

$$
S(\Delta)=d+sA\exp\left[-\frac{(\Delta-\mu)^2}{2\sigma^2}\right],
$$

where $s=+1$ for a peak and $s=-1$ for the echo depletion dip. Here $d$ is the
background, $A$ is the feature amplitude, $\mu$ is the fitted resonance offset,
and $\sigma$ is the fitted Gaussian standard deviation. The Gaussian is an
operational model used to extract a consistent center and width; it is not a
claim that the physical spectroscopy line shape is exactly Gaussian.

The nonlinear least-squares fit was initialized using

$$
A_0=\max(S)-\min(S),\qquad
\mu_0=0,\qquad
\sigma_0=40~\mathrm{kHz},\qquad
d_0=\langle S\rangle.
$$

The amplitude was constrained to $0\le A\le1$, the center to the measured
detuning interval, the width to $\sigma>0$ and no larger than the positive
edge of the scan, and the offset to $-1\le d\le1$. The fitted full width at
half maximum was then calculated from

$$
\mathrm{FWHM}=2\sqrt{2\ln2}\,|\sigma|,
$$

and the fitted resonance offset was taken directly as $\mu$. The present code
also computes the signal metric $|A|/|d|$.

### Fit validation and reporting

In the current implementation, a fit is rejected if

$$
\mathrm{FWHM}>0.12~\mathrm{MHz}
\quad\text{or}\quad
|\mu|>\frac{1}{3}\max|\Delta|,
$$

or if the nonlinear fit raises an exception. The implementation currently
returns zeros for rejected fits. For the paper plots and reported statistics,
these zeros must be converted to missing values and shown as rejected points;
otherwise they would be mistaken for zero linewidth and zero frequency
offset.

The Supplemental Material should also state how many fits were rejected and
why. Fit uncertainties should be obtained from the covariance matrix returned
by the nonlinear least-squares fit, or preferably from bootstrap resampling of
the measured shots. If $u_\sigma$ is the standard uncertainty of the fitted
$\sigma$, covariance propagation gives

$$
u_{\mathrm{FWHM}}=2\sqrt{2\ln2}\,u_\sigma.
$$

One representative narrow trace and one high-amplitude trace should be shown
with the raw data, smoothed data, fitted curve, fitted center, and half-maximum
boundaries. This makes the extraction procedure visually reproducible and
shows where the Gaussian description begins to fail.

## Plotting and analysis checks

- Use one frequency convention throughout: angular frequency or cyclic
  frequency, with every factor of $2\pi$ explicit.
- Define FWHM versus HWHM and use the same convention in every panel.
- Calibrate the horizontal amplitude axis to peak Rabi frequency whenever
  possible.
- Keep the raw two-dimensional color scale comparable across amplitudes.
- Show uncertainty on extracted quantities and mark rejected fits.
- Keep failed points out of best-point selection and operating-region maps.
- State whether each plot comes from experiment, simulation, or waveform-only
  Fourier analysis.
- Do not interpret a two-level simulation as evidence for leakage or
  multiphoton transitions at high amplitude.
