Fresh three-level simulation survey, not hardware measurements.
Polynomial p=5 -> 2; 100 linear 0–100 MHz amplitudes -> 100 geometric 1–100 MHz amplitudes (same displayed range). No zero-drive point on the logarithmic grid. Other settings unchanged.
Durations 10, 20, 30, 45, 60 us; 100 geometric cutoffs 1e-5–1;
201 detunings -0.5–0.5 MHz; beta=-0.5; 16 ns sine transition;
T1=51.24 us, T2=7.31 us; alpha=-216 MHz; maximum RK4 step 0.25 ns.
Amplitude is the nominal pre-smoothing scale; no peak normalization.
status.json records the current stage. Each cutoff_###.npz is resumable.
After simulation: 45 representative finer-grid traces, physicality checks,
CSV/NPZ metrics and a separate paper-style PDF/PNG with provenance.
Resume from the repository root using:
.venv/bin/python scripts/repeat_extended_cutoff_poly2.py --workers 6 --output-dir /Users/asafsolonnikov/Developer/amplitude-robust-spectroscopy/outputs/polynomial_duration_cutoff_poly2_log/n100
