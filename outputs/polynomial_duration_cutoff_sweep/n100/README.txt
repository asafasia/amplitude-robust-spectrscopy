POLYNOMIAL ECHO — DURATION / CUTOFF COMPARISON

Available pulse lengths (us): 10, 20, 30, 45, 60
Pending lengths (us): []

Open report.html directly in a browser. All charts, the Plotly runtime, and
display data are embedded; no internet or server is required. Local download
links refer to the companion files in this folder.
Detuning-versus-amplitude maps show every saved Pe sample at the shared cutoff
for each pulse length. All 100 amplitudes are also available as line spectra.
Vertical dashed markers at +/-Gamma2/2 about zero detuning bracket one T2
reference FWHM. They stay fixed when cutoff, amplitude or display units change.
FWHM defaults to units of Gamma2 = 1/(pi*T2), with a kHz display switch.
For T2 = 7.31 us, Gamma2 = 43.544444 kHz.
The default PNG/PDF comparison uses FWHM/Gamma2; *_khz exports use kHz.

SETTINGS
100 logarithmic cutoffs from 1e-5 to 1, 100 drive scales from 0 to 100 MHz,
201 detunings over +/-0.5 MHz; polynomial p=5, cutoff subtraction, full 16 ns
sine echo transition, corrected DRAG beta=-0.5, I-iQ, alpha=-216 MHz,
T1=51.24 us, T2=7.31 us, three levels, maximum RK4 step 0.25 ns.
SNR uses 1000 ideal binary-readout shots; it is not measured hardware SNR.

DATA LAYOUT
- baseline_30us/: independent copy of the previously validated 30 us study.
- simulations/L...us/: new 10, 20, 45, and 60 us runs. Each cutoff_###.npz
  is an atomic checkpoint, with full configuration and physicality diagnostics.
- Each scan.npz: populations[cutoff, level=(g,e,f), amplitude, detuning], float64;
  metrics[cutoff, amplitude], plus all axes, duration and configuration.
- comparison_metrics.npz: metrics[duration, cutoff, amplitude], axes and config.
- metrics_all.csv: all duration/cutoff/amplitude rows, including rejected dips.
  'accepted' and 'status' must be respected; invalid values are not zeros.
- comparison_summary.json: physicality checks and a fixed-cutoff robustness
  comparison across sampled amplitudes >=10 MHz.
- source_snapshot/: exact simulation and report sources, with the same layout
  as the repository. dataset_manifest.json records their hashes and provenance.
- checksums.sha256: SHA-256 for every saved file except itself (final export).
- validation/: representative finer-time/frequency runs and their raw arrays
  (populations[level, amplitude, detuning]), plus software-check records.

READ ARRAYS
import numpy as np
with np.load('comparison_metrics.npz', allow_pickle=False) as data:
    lengths = data['durations_us']
    fwhm_khz = data['fwhm_hz'] / 1000
    fwhm_t2_units = data['fwhm_hz'] * np.pi * 7.31e-6
    ideal_snr = data['contrast_snr']
    accepted = data['accepted']

REPRODUCE / RESUME THE NEW SIMULATIONS (from repository or source_snapshot)
python scripts/scan_polynomial_cutoffs.py --cutoff-count 100 \
  --durations-us 10 20 45 60 --drag-beta -0.5 --workers 8 \
  --output-dir /absolute/path/to/this/dataset/simulations
The 30 us baseline is reused only after its settings/source hashes match.

REBUILD THE REPORT
python scripts/validate_polynomial_durations.py --input-dir /absolute/path/to/this/dataset
python scripts/report_polynomial_durations.py --input-dir /absolute/path/to/this/dataset
Use --allow-partial for a clearly labeled preview while integrations run.
Report dependencies: numpy, scipy, matplotlib, plotly.
Install from the repository with: python -m pip install -e '.[reports]'

RECREATE THE BASELINE IF NEEDED
python scripts/scan_polynomial_cutoffs.py --cutoff-count 100 \
  --durations-us 30 --drag-beta -0.5 --workers 8 \
  --output-dir /absolute/path/to/this/dataset/baseline_30us

PRECISION AND INTERPRETATION
Metric extraction uses float64 originals. HTML maps and spectra share the full
Pe cube as float32 display copies, ordered [cutoff, amplitude, detuning]. All
three-level float64 populations are retained in scan.npz. No sweep data are
interpolated to invent missing runs. Exported fwhm_hz metrics remain in Hz;
the T2 normalization only changes the presentation, not extraction or SNR.
FWHM uses a fixed +/-0.5 MHz window; inspect spectra for satellite dips and
finite-window effects. At fixed cutoff, sigma scales with pulse duration.
