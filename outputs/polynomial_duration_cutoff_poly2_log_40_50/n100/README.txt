Matched three-level simulation survey for Figure 4 and Figure S8.
Durations: 10, 20, 30, 40, 50 us. The 10/20/30 us rows and their
fine-grid checkpoints are checksum-verified copies of the original
power-2 logarithmic survey; 40/50 us are newly integrated.
100 geometric cutoffs 1e-5 to 1; 100 geometric drives 1 to 100 MHz;
201 detunings -0.5 to 0.5 MHz; beta=-0.5; 16 ns sine transition;
T1=51.24 us, T2=7.31 us; alpha=-216 MHz; RK4 step <=0.25 ns.
Drive is the nominal pre-smoothing scale, without peak normalization.
Plot linewidths relative to Gamma=1/(pi*7.6 us); keep drive in MHz.
Validation halves the time step and doubles the detuning intervals
for 45 representative traces; it is not a full-grid error bound.
Resume or reproduce from the repository root:
.venv/bin/python scripts/extend_cutoff_duration_40_50.py --workers 6 --output-dir /Users/asafsolonnikov/Developer/amplitude-robust-spectroscopy/outputs/polynomial_duration_cutoff_poly2_log_40_50/n100
Render the supplemental figure before the main figure:
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_supplemental_extended_cutoff.py
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_main_duration_cutoff_scan.py
