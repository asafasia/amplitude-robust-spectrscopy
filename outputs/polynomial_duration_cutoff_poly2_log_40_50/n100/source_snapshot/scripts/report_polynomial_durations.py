"""Build an offline HTML comparison and archive metadata for duration sweeps.

Requires NumPy, SciPy, Matplotlib, and Plotly (for bundled browser JavaScript).
Run with --allow-partial while simulations are still running; an incomplete
report is explicitly labeled and never marks the dataset complete.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import os
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")

from echospec.analysis.duration_cutoffs import METRIC_KEYS, load_study  # noqa: E402
from scripts.scan_polynomial_cutoffs import clean_json  # noqa: E402

REPORT_SOURCES = [
    "scripts/report_polynomial_durations.py",
    "scripts/templates/polynomial_duration_report.html",
    "echospec/analysis/duration_cutoffs.py",
    "scripts/validate_polynomial_durations.py",
    "tests/analysis/test_duration_cutoffs.py",
    "tests/check_polynomial_report.cjs",
]
COLORS = {10: "#007c91", 20: "#bd642b", 30: "#6057a4", 45: "#268358", 60: "#b94f76"}


def atomic_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def json_text(value):
    return json.dumps(clean_json(value), allow_nan=False, separators=(",", ":"))


def export_tables(root, study):
    rows = [row for case in study["cases"] for row in case["rows"]]
    with (root / "metrics_all.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    arrays = {key: np.stack([c["metrics"][key] for c in study["cases"]])
              for key in METRIC_KEYS}
    np.savez_compressed(
        root / "comparison_metrics.npz", **arrays,
        durations_us=[c["duration_us"] for c in study["cases"]],
        cutoffs=study["cutoffs"], rabi_mhz=study["amplitudes"],
        configuration=json.dumps(study["common"], sort_keys=True),
        pending_durations_us=study["pending"],
    )
    ci = int(np.argmin(abs(np.log(study["cutoffs"] / .001))))
    ai = int(np.argmin(abs(study["amplitudes"] - 50)))
    matched = dict(
        cutoff=float(study["cutoffs"][ci]), amplitude_mhz=float(study["amplitudes"][ai]),
        selection="Nearest sampled cutoff to 0.001 and drive scale to 50 MHz; report default.",
        results=[dict(duration_us=c["duration_us"],
                      accepted=bool(c["metrics"]["accepted"][ci, ai]),
                      fwhm_khz=float(c["metrics"]["fwhm_hz"][ci, ai] / 1000),
                      ideal_snr=float(c["metrics"]["contrast_snr"][ci, ai]),
                      snr_per_mhz=float(c["metrics"]["snr_per_mhz"][ci, ai]))
                 for c in study["cases"]],
    )
    summary = dict(
        complete=not study["pending"], pending_durations_us=study["pending"],
        matched_operating_point=matched,
        results=[dict(duration_us=c["duration_us"], validation=c["validation"],
                      best_fixed_cutoff_over_10_to_100mhz=c["robust"])
                 for c in study["cases"]],
        selection="Among cutoffs with resolved dips at every sampled amplitude >=10 MHz, maximize the minimum SNR/FWHM. No extrapolation between samples.",
    )
    atomic_text(root / "comparison_summary.json", json.dumps(clean_json(summary), indent=2) + "\n")


def plot_comparison(root, study, *, normalized=True):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    cases = study["cases"]
    cutoffs, amps = study["cutoffs"], study["amplitudes"]
    logc = np.log(cutoffs)
    c_edges = np.exp(np.r_[logc[0] - (logc[1] - logc[0]) / 2,
                          (logc[1:] + logc[:-1]) / 2,
                          logc[-1] + (logc[-1] - logc[-2]) / 2])
    a_edges = np.r_[amps[0] - (amps[1] - amps[0]) / 2,
                    (amps[1:] + amps[:-1]) / 2,
                    amps[-1] + (amps[-1] - amps[-2]) / 2]
    fig, axes = plt.subplots(2, len(cases), figsize=(3.5 * len(cases) + 1, 7),
                             squeeze=False, layout="constrained", sharex=True, sharey=True)
    reference_hz = 1 / (np.pi * study["common"]["t2_us"] * 1e-6)
    width_factor = 1 / reference_hz if normalized else .001
    width_label = "Dip FWHM / Γ₂  [Γ₂ = 1/(πT₂)]" if normalized else "Dip FWHM (kHz)"
    # One scale per quantity across all durations, with no clipped data.
    for row, (key, factor, label, upper, cmap) in enumerate([
        ("fwhm_hz", width_factor, width_label, 600_000, "magma"),
        ("contrast_snr", 1, "Ideal contrast SNR", 25, "viridis"),
    ]):
        rounding_step = 5000 if key == "fwhm_hz" else 5
        upper = factor * max(upper, np.ceil(max(
            float(np.nanmax(c["metrics"][key][c["metrics"]["accepted"]]))
            if np.any(c["metrics"]["accepted"]) else 0 for c in cases
        ) / rounding_step) * rounding_step)
        for col, case in enumerate(cases):
            m = case["metrics"]
            values = np.where(m["accepted"], m[key] * factor, np.nan)
            color = plt.get_cmap(cmap).copy()
            color.set_bad("#e2e7e9")
            artist = axes[row, col].pcolormesh(a_edges, c_edges, np.ma.masked_invalid(values),
                                              cmap=color, vmin=0, vmax=upper, rasterized=True)
            axes[row, col].set(yscale="log", ylim=(cutoffs[0], cutoffs[-1]),
                               xlim=(amps[0], amps[-1]), xlabel="Drive scale A (MHz)")
            if row == 0:
                axes[row, col].set_title(f"{case['duration_us']:g} µs", fontweight="bold")
            if col == 0:
                axes[row, col].set_ylabel("Cutoff")
        fig.colorbar(artist, ax=list(axes[row]), label=label)
    fig.suptitle("Pulse length at fixed polynomial shaping and DRAG\n"
                 "p = 5 · β = −0.5 · 16 ns echo transition · 1,000 ideal shots · gray = rejected dip",
                 fontsize=14)
    suffix = "" if normalized else "_khz"
    for ext in ("png", "pdf"):
        fig.savefig(root / f"duration_comparison{suffix}.{ext}", dpi=180)
    plt.close(fig)


def payload(study):
    cases = []
    for case in study["cases"]:
        cases.append(dict(
            duration=case["duration_us"], color=COLORS.get(case["duration_us"], "#293f50"),
            metrics={k: v.tolist() for k, v in case["metrics"].items()},
            population_base64=base64.b64encode(
                case["population_maps"].tobytes(order="C")).decode("ascii"),
            robust=case["robust"], validation=case["validation"], directory=case["directory"],
        ))
    return dict(cases=cases, cutoffs=study["cutoffs"].tolist(),
                amps=study["amplitudes"].tolist(), detuning=study["detuning"].tolist(),
                population_layout="Pe[cutoff, amplitude, detuning], little-endian float32, C order",
                pending=study["pending"], configuration=study["common"])


def make_data_links(study):
    rows = []
    for c in study["cases"]:
        folder = html.escape(c["directory"], quote=True)
        label = f"{c['duration_us']:g} µs"
        rows.append(f'<tr><th>{label}</th><td><a href="{folder}/scan.npz">Full populations + metrics</a></td>'
                    f'<td><a href="{folder}/metrics.csv">CSV</a></td>'
                    f'<td><a href="{folder}/summary.json">Summary</a></td>'
                    f'<td><a href="{folder}/metric_maps.pdf">Metric-map PDF</a></td></tr>')
    return "\n".join(rows)


def convergence_description(result):
    if result is None:
        return "Representative finer-grid comparisons are pending."
    maximum = result["maximum_changes"]
    if not maximum:
        return "Finer-grid checkpoints are saved; comparison to main-grid data is pending."
    return (
        f"Representative checks: {result['compared_traces']} traces compared"
        f"{' (partial)' if not result['complete'] else ''}. "
        f"Halving the time step changes populations by at most "
        f"{maximum['max_population_time_step_change']:.3g}, FWHM by "
        f"{maximum['time_fwhm_hz_absolute_change'] / 1000:.3f} kHz, "
        f"and ideal SNR by {maximum['time_contrast_snr_absolute_change']:.4f}. "
        f"Doubling detuning intervals changes FWHM by at most "
        f"{maximum['frequency_fwhm_hz_absolute_change'] / 1000:.3f} kHz "
        f"and ideal SNR by {maximum['frequency_contrast_snr_absolute_change']:.4f}. "
        f"Dip-status changes: {result['time_status_changes']} for time refinement, "
        f"{result['frequency_status_changes']} for frequency refinement. "
        "Metric differences compare traces accepted at both resolutions."
    )


def write_readme(root, study):
    current = ", ".join(f"{c['duration_us']:g}" for c in study["cases"])
    readme = f"""POLYNOMIAL ECHO — DURATION / CUTOFF COMPARISON

Available pulse lengths (us): {current}
Pending lengths (us): {study['pending']}

Open report.html directly in a browser. All charts, the Plotly runtime, and
display data are embedded; no internet or server is required. Local download
links refer to the companion files in this folder.
Detuning-versus-amplitude maps show every saved Pe sample at the shared cutoff
for each pulse length. All 100 amplitudes are also available as line spectra.
Vertical dashed markers at +/-Gamma2/2 about zero detuning bracket one T2
reference FWHM. They stay fixed when cutoff, amplitude or display units change.
FWHM defaults to units of Gamma2 = 1/(pi*T2), with a kHz display switch.
For T2 = {study['common']['t2_us']:g} us, Gamma2 = {1000 / (np.pi * study['common']['t2_us']):.6f} kHz.
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
    fwhm_t2_units = data['fwhm_hz'] * np.pi * {study['common']['t2_us']:g}e-6
    ideal_snr = data['contrast_snr']
    accepted = data['accepted']

REPRODUCE / RESUME THE NEW SIMULATIONS (from repository or source_snapshot)
python scripts/scan_polynomial_cutoffs.py --cutoff-count 100 \\
  --durations-us 10 20 45 60 --drag-beta -0.5 --workers 8 \\
  --output-dir /absolute/path/to/this/dataset/simulations
The 30 us baseline is reused only after its settings/source hashes match.

REBUILD THE REPORT
python scripts/validate_polynomial_durations.py --input-dir /absolute/path/to/this/dataset
python scripts/report_polynomial_durations.py --input-dir /absolute/path/to/this/dataset
Use --allow-partial for a clearly labeled preview while integrations run.
Report dependencies: numpy, scipy, matplotlib, plotly.
Install from the repository with: python -m pip install -e '.[reports]'

RECREATE THE BASELINE IF NEEDED
python scripts/scan_polynomial_cutoffs.py --cutoff-count 100 \\
  --durations-us 30 --drag-beta -0.5 --workers 8 \\
  --output-dir /absolute/path/to/this/dataset/baseline_30us

PRECISION AND INTERPRETATION
Metric extraction uses float64 originals. HTML maps and spectra share the full
Pe cube as float32 display copies, ordered [cutoff, amplitude, detuning]. All
three-level float64 populations are retained in scan.npz. No sweep data are
interpolated to invent missing runs. Exported fwhm_hz metrics remain in Hz;
the T2 normalization only changes the presentation, not extraction or SNR.
FWHM uses a fixed +/-0.5 MHz window; inspect spectra for satellite dips and
finite-window effects. At fixed cutoff, sigma scales with pulse duration.
"""
    atomic_text(root / "README.txt", readme)


def build_report(root, *, allow_partial=False):
    import matplotlib
    import plotly
    from plotly.offline import get_plotlyjs

    root = Path(root).resolve()
    study = load_study(root, allow_partial=allow_partial)
    if not study["pending"] and not (root / study["manifest"]["new_runs"] / "provenance.json").exists():
        raise RuntimeError("Wait for the simulation exporter to finish before finalizing the report")
    convergence_path = root / "validation/convergence.json"
    convergence = json.loads(convergence_path.read_text()) if convergence_path.exists() else None
    if not study["pending"] and (convergence is None or not convergence["complete"]):
        raise RuntimeError("Run validate_polynomial_durations.py before finalizing the report")
    export_tables(root, study)
    plot_comparison(root, study)
    plot_comparison(root, study, normalized=False)
    template = (ROOT / REPORT_SOURCES[1]).read_text()
    template = re.sub(r"<!-- TEMPLATE_NOTICE_START -->.*?<!-- TEMPLATE_NOTICE_END -->",
                      "", template, flags=re.DOTALL)
    template = template.replace('<main id="reportContent" hidden>',
                                '<main id="reportContent">')
    figure = base64.b64encode((root / "duration_comparison.png").read_bytes()).decode("ascii")
    replacements = {
        "/*__PLOTLY_JS__*/": get_plotlyjs(),
        "__PAYLOAD__": json_text(payload(study)).replace("</", "<\\/"),
        "__COMPARISON_PNG__": figure,
        "__COMPARISON_KHZ_PNG__": base64.b64encode(
            (root / "duration_comparison_khz.png").read_bytes()).decode("ascii"),
        "__POPULATION_PANELS__": "\n".join(
            f'<div id="populationMap{i}" class="plot tall" '
            f'aria-label="Detuning versus amplitude, {c["duration_us"]:g} microseconds"></div>'
            for i, c in enumerate(study["cases"])),
        "__DATA_ROWS__": make_data_links(study),
        "__CONVERGENCE_SUMMARY__": html.escape(convergence_description(convergence)),
        "__COVERAGE_SUMMARY__": html.escape("Accepted local-dip spectra: " + "; ".join(
            f"{c['duration_us']:g} µs: {c['validation']['accepted_traces']:,}/{c['validation']['total_traces']:,}"
            for c in study["cases"]) + ". Every rejected spectrum is retained in the data."),
        "__GENERATED__": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    if study["pending"]:
        template = template.replace('<a href="checksums.sha256">SHA-256 checksums</a>',
                                    '<span class="caption">Checksums will be indexed after completion.</span>')
    if not (root / "validation/convergence.json").exists():
        template = template.replace('<a href="validation/convergence.json">validation/convergence.json</a>',
                                    '<code>validation/convergence.json</code> (checks pending)')
    atomic_text(root / "report.html", template)
    write_readme(root, study)
    manifest = study["manifest"]
    manifest.update(
        status="partial" if study["pending"] else "complete",
        completed_durations_us=[c["duration_us"] for c in study["cases"]],
        pending_durations_us=study["pending"],
        report_generated_utc=datetime.now(timezone.utc).isoformat(),
        report_source_sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                              for f in REPORT_SOURCES},
        report_runtime=dict(python=platform.python_version(), numpy=np.__version__,
                            matplotlib=matplotlib.__version__, plotly=plotly.__version__),
        report_display=dict(
            default_fwhm_units="FWHM / Gamma2, Gamma2 = 1/(pi*T2)",
            reference_linewidth_hz=1 / (np.pi * study["common"]["t2_us"] * 1e-6),
            population_dtype="little-endian float32",
            population_layout="Pe[cutoff, amplitude, detuning]",
            all_amplitudes_and_cutoffs_embedded=True,
            population_color_range=[0, 1],
            population_t2_markers=dict(
                center_detuning_khz=0,
                positions_khz=[sign / (2 * np.pi * study["common"]["t2_us"] * .001)
                               for sign in (-1, 1)],
                definition="+/-Gamma2/2; marker separation = Gamma2 = 1/(pi*T2)",
            ),
        ),
        data_layout="metrics[duration, cutoff, amplitude]; per-duration populations[cutoff, level, amplitude, detuning]",
        validation=[dict(duration_us=c["duration_us"], **c["validation"]) for c in study["cases"]],
        representative_convergence=convergence,
    )
    for source in REPORT_SOURCES:
        destination = root / "source_snapshot" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, destination)
    atomic_text(root / "dataset_manifest.json", json.dumps(manifest, indent=2) + "\n")
    if not study["pending"]:
        checksums = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "checksums.sha256" and not path.name.endswith(".tmp"):
                checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
        atomic_text(root / "checksums.sha256", "\n".join(checksums) + "\n")
    print(f"Saved {'partial' if study['pending'] else 'complete'} report: {root / 'report.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "outputs/polynomial_duration_cutoff_sweep/n100")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    build_report(args.input_dir, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
