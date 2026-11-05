"""Compare 30 us extensions and full time dilation of frozen flexible echoes."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.analysis.dip_metrics import dip_metrics  # noqa: E402
from echospec.simulation.flexible_echo import FlexibleEchoPulse  # noqa: E402
from echospec.simulation.polynomial_echo import simulate_polynomial_echo  # noqa: E402
from echospec.simulation.time_scaled_echo import TimeScaledEchoPulse  # noqa: E402
from scripts.optimize_50mhz_echo import metrics, peak_scale  # noqa: E402
from scripts.report_decoupled_50mhz_echo import (  # noqa: E402
    clean,
    local_shoulder_metrics,
)

SOURCE = ROOT / "outputs/flexible_echo_50mhz_20261101"
OUT = ROOT / "outputs/flexible_echo_30us_20261101"
FAMILIES = ("cleaner_smooth", "flexible_phase")
LABELS = {"cleaner_smooth": "Higher-contrast smooth pulse",
          "flexible_phase": "Narrow phase-shaped pulse"}
AMPLITUDES = np.arange(5., 60.25, .5)
DETUNING = np.linspace(-.5, .5, 401)
FINE_DETUNING = np.linspace(-.5, .5, 801)
T1, T2 = 51.24, 7.31
MODEL_PATHS = [ROOT / "echospec/simulation" / n for n in (
    "time_scaled_echo.py", "flexible_echo.py", "decoupled_echo.py",
    "polynomial_echo.py", "qutrit_native.py", "qutrit_rk4.cpp")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(clean(data), indent=2) + "\n")


def number(value):
    return np.nan if value is None else float(value)


def fmt(value, scale=1, digits=2):
    value = number(value)
    return "unresolved" if not np.isfinite(value) else f"{value / scale:.{digits}f}"


def simulate_cached(pulse, amplitudes, detuning, dt_ns, t1=T1, t2=T2):
    identity = {"pulse_type": type(pulse).__name__, "pulse": asdict(pulse),
                "amplitudes_mhz": list(amplitudes), "detuning_mhz": list(detuning),
                "dt_ns": dt_ns, "t1_us": t1, "t2_us": t2,
                "model_sha256": {p.name: sha(p) for p in MODEL_PATHS}}
    encoded = json.dumps(identity, sort_keys=True)
    path = OUT / "evaluations" / (hashlib.sha256(encoded.encode()).hexdigest() + ".npz")
    if path.exists():
        with np.load(path) as data:
            assert json.loads(str(data["identity"])) == json.loads(encoded)
            return data["populations"].copy(), json.loads(str(data["diagnostics"])), path.name
    pop, diag = simulate_polynomial_echo(pulse, amplitudes, detuning,
                                         t1_us=t1, t2_us=t2, dt_ns=dt_ns)
    np.savez_compressed(path, populations=pop, diagnostics=json.dumps(diag),
                        identity=encoded)
    return pop, diag, path.name


def comparison(rows, reference):
    width = np.array([r["fwhm_hz"] - b["fwhm_hz"]
                      for r, b in zip(rows, reference, strict=True)]) / 1000
    contrast = np.array([r["contrast"] - b["contrast"]
                         for r, b in zip(rows, reference, strict=True)])
    valid = np.array([r["accepted"] for r in rows])
    return {"all_metrics_accepted": bool(valid.all()),
            "width_regression_amplitudes_mhz": AMPLITUDES[(width > 0) | ~valid].tolist(),
            "contrast_regression_amplitudes_mhz": AMPLITUDES[(contrast < 0) | ~valid].tolist(),
            "largest_width_change_khz": float(np.nanmax(width)),
            "smallest_contrast_change": float(np.nanmin(contrast)),
            "no_regression_on_sampled_grid": bool(valid.all() and (width <= 0).all()
                                                  and (contrast >= 0).all())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / "evaluations").mkdir(exist_ok=True)
    if args.render_only:
        render()
        return
    source = json.loads((SOURCE / "results.json").read_text())
    with np.load(SOURCE / "validated_spectra.npz") as old:
        np.testing.assert_array_equal(old["rabi_mhz"], AMPLITUDES)
        np.testing.assert_array_equal(old["detuning_mhz"], DETUNING)
        arrays = {"amplitudes_mhz": AMPLITUDES, "detuning_mhz": DETUNING,
                  "fine_detuning_mhz": FINE_DETUNING,
                  "baseline": old["baseline"].copy()}
        for family in FAMILIES:
            arrays[family] = old[family].copy()
            arrays[f"{family}_fine50"] = old[f"{family}_fine50"].copy()
    records, pulses, variants = {}, {}, {}
    for family in FAMILIES:
        previous = source["cases"][family]
        base = FlexibleEchoPulse(**previous["pulse"])
        pulses[family] = base
        records[family] = {
            "family": family, "mode": "original", "label": LABELS[family],
            "duration_us": base.duration_us, "transition_us": base.transition_us,
            "pulse_type": type(base).__name__, "pulse": asdict(base),
            "unit_waveform_peak": previous["unit_waveform_peak"],
            "rows": previous["rows"],
            "fine50": previous["validation"]["fine_grid_50mhz"],
            "source": str(SOURCE / "results.json"),
        }
        variants[f"{family}_duration30"] = (family, "duration_only",
                                             replace(base, duration_us=30.))
        variants[f"{family}_stretch30"] = (family, "full_stretch",
                                           TimeScaledEchoPulse(base, 30 / base.duration_us))

    def run_case(name, family, mode, pulse):
        normalization = peak_scale(pulse)
        print(f"{name}: 111 x 401 map", flush=True)
        pop, diag, cache = simulate_cached(pulse, AMPLITUDES / normalization,
                                           DETUNING, .25)
        print(f"{name}: refined 50 MHz and coherent comparison", flush=True)
        fine, fine_diag, fine_cache = simulate_cached(
            pulse, [50 / normalization], FINE_DETUNING, .125)
        coherent, coherent_diag, coherent_cache = simulate_cached(
            pulse, [50 / normalization], FINE_DETUNING, .0625, 1e9, 1e9)
        rows = metrics(pop, AMPLITUDES, DETUNING)
        fm = dip_metrics(FINE_DETUNING, fine[1, 0])
        fm["local_shoulders"] = local_shoulder_metrics(FINE_DETUNING, fine[1, 0])
        baseline_check = comparison(rows, source["cases"]["baseline"]["rows"])
        short_check = comparison(rows, source["cases"][family]["rows"])
        record = {
            "family": family, "mode": mode, "label": LABELS[family],
            "duration_us": pulse.duration_us, "transition_us": pulse.transition_us,
            "pulse_type": type(pulse).__name__, "pulse": asdict(pulse),
            "unit_waveform_peak": normalization, "actual_peak_mhz": 50,
            "rows": rows, "fine50": fm,
            "coherent50": dip_metrics(FINE_DETUNING, coherent[1, 0]),
            "versus_original_polynomial": baseline_check,
            "versus_short_source_pulse": short_check,
            "passes_previous_user_constraints": bool(fm["accepted"] and fm["contrast"] > .4
                and baseline_check["no_regression_on_sampled_grid"]),
            "diagnostics": {"map": diag, "fine50": fine_diag, "coherent50": coherent_diag},
            "cache_files": [cache, fine_cache, coherent_cache],
        }
        print(name, json.dumps({"fine50": fm, "coherent50": record["coherent50"],
                               "passes": record["passes_previous_user_constraints"]}), flush=True)
        save_json(OUT / f"{name}.json", record)
        return name, record, pop, fine[:, 0], coherent[:, 0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(run_case, name, *spec) for name, spec in variants.items()]
        for future in as_completed(jobs):
            name, record, pop, fine, coherent = future.result()
            records[name] = record
            arrays[name], arrays[f"{name}_fine50"] = pop, fine
            arrays[f"{name}_coherent50"] = coherent
            pulses[name] = variants[name][2]
    for name, pulse in pulses.items():
        time = np.unique(np.r_[np.linspace(-pulse.duration_us / 2, pulse.duration_us / 2, 30001),
                               np.linspace(-.2, .2, 8001)])
        drive = 50 * pulse.drive(time) / records[name]["unit_waveform_peak"]
        # Include the resolved maximum in the display/export grid rather than
        # confusing a finite sampling miss with a peak-normalization error.
        peak_index = int(np.argmax(abs(drive)))
        refined_peak = minimize_scalar(lambda t, pulse=pulse: -abs(pulse.drive(t)),
                                       bounds=(time[peak_index - 1], time[peak_index + 1]),
                                       method="bounded", options={"xatol": 1e-13})
        time = np.unique(np.r_[time, refined_peak.x])
        drive = 50 * pulse.drive(time) / records[name]["unit_waveform_peak"]
        records[name]["peak_time_us"] = float(refined_peak.x)
        records[name]["sampled_peak_mhz"] = float(abs(drive).max())
        assert abs(abs(drive).max() - 50) < .0001
        arrays[f"{name}_time_us"], arrays[f"{name}_drive_50mhz"] = time, drive
    result = {
        "metadata": {
            "t1_us": T1, "t2_us": T2, "actual_peak_definition": "max over both time halves of |I-iQ| after DRAG",
            "grid": "5 to 60 MHz, 0.5 MHz steps; -0.5 to +0.5 MHz detuning, 2.5 kHz steps",
            "duration_only": "Set duration_us=30; retain core, tail, taper-width and phase scales. Move endpoints outward.",
            "full_stretch": "Raw u(t) becomes u(t/s), s=30/L. Every envelope, phase, transition and taper time stretches. DRAG derivative is divided by s before peak normalization.",
            "optimization": "None. Frozen coefficients and beta; duration changes only.",
            "comparison_scope": "Exact signs on sampled maps, not a continuous interval proof; spectral guardrails are not new optimization constraints.",
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
                Path(__file__), *MODEL_PATHS, SOURCE / "results.json", SOURCE / "validated_spectra.npz")},
        }, "cases": records,
    }
    save_json(OUT / "results.json", result)
    np.savez_compressed(OUT / "spectra.npz", **arrays)
    render()


def render():
    result = json.loads((OUT / "results.json").read_text())
    cases = result["cases"]
    with np.load(OUT / "spectra.npz") as data:
        arrays = {k: data[k].copy() for k in data.files}
    order = [[family, f"{family}_duration30", f"{family}_stretch30"] for family in FAMILIES]
    mode_labels = {"original": "Original optimized duration", "duration_only": "30 us: extend duration only",
                   "full_stretch": "30 us: stretch full shape"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True,
                             layout="constrained")
    for row, family in enumerate(FAMILIES):
        for col, name in enumerate(order[row]):
            case, ax = cases[name], axes[row, col]
            m = case["fine50"]
            im = ax.pcolormesh(DETUNING * 1000, AMPLITUDES, arrays[name][1],
                               shading="nearest", vmin=0, vmax=1, cmap="viridis",
                               rasterized=True)
            ax.axhline(50, color="white", ls="--", lw=1)
            ax.set_title(f"{mode_labels[case['mode']]} ({case['duration_us']:.2f} us)\n"
                         f"50 MHz: {fmt(m['fwhm_hz'], 1000, 1)} kHz, C={fmt(m['contrast'], 1, 3)}", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{LABELS[family]}\nPeak Rabi rate (MHz)")
            if row == 1:
                ax.set_xlabel("Detuning (kHz)")
    fig.colorbar(im, ax=axes, label="Excited-state probability", shrink=.9)
    fig.suptitle("Making the new single-echo pulses 30 us long — coefficients unchanged")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"maps.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), layout="constrained")
    colors = ("C0", "C1", "C2")
    styles = ("-", "--", "-.")
    for row, family in enumerate(FAMILIES):
        for name, color, style in zip(order[row], colors, styles, strict=True):
            case = cases[name]
            label = mode_labels[case["mode"]]
            axes[row, 0].plot(FINE_DETUNING * 1000, arrays[f"{name}_fine50"][1],
                              color=color, ls=style, label=label)
            axes[row, 1].plot(AMPLITUDES, [number(r["fwhm_hz"]) / 1000 for r in case["rows"]],
                              color=color, ls=style)
            axes[row, 2].plot(AMPLITUDES, [number(r["contrast"]) for r in case["rows"]],
                              color=color, ls=style)
        axes[row, 0].set(xlim=(-250, 250), ylim=(0, 1), xlabel="Detuning (kHz)",
                         ylabel=f"{LABELS[family]}\nExcited-state probability", title="50 MHz slices")
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].axhline(1000 / (np.pi * T2), color="gray", ls=":", label="T2 reference")
        axes[row, 1].set(xlabel="Peak Rabi rate (MHz)", ylabel="FWHM (kHz)")
        axes[row, 1].legend(fontsize=8)
        axes[row, 2].axhline(.4, color="gray", ls=":", label="50 MHz contrast target")
        axes[row, 2].axvline(50, color="gray", ls=":")
        axes[row, 2].set(xlabel="Peak Rabi rate (MHz)", ylabel="Fitted-background contrast")
        for ax in axes[row]:
            ax.grid(alpha=.2)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"comparison.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), layout="constrained")
    for row, family in enumerate(FAMILIES):
        for name, color, style in zip(order[row], colors, styles, strict=True):
            time, drive = arrays[f"{name}_time_us"], arrays[f"{name}_drive_50mhz"]
            axes[row, 0].semilogy(time, np.maximum(abs(drive), 1e-7), color=color,
                                  ls=style, label=mode_labels[cases[name]["mode"]])
            axes[row, 1].plot(time * 1000, drive.real, color=color, ls=style)
            axes[row, 2].plot(time * 1000, -drive.imag, color=color, ls=style)
        axes[row, 0].set(xlim=(-15, 15), ylim=(.001, 60), xlabel="Time (us)",
                         ylabel=f"{LABELS[family]}\n|I-iQ| (MHz)")
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].set(xlim=(-100, 100), xlabel="Time (ns)", ylabel="I (MHz)")
        axes[row, 2].set(xlim=(-100, 100), xlabel="Time (ns)", ylabel="Q (MHz)")
        for ax in axes[row]:
            ax.grid(alpha=.2)
    fig.savefig(OUT / "waveforms.png", dpi=180)
    plt.close(fig)

    table = []
    for row in order:
        for name in row:
            case, m = cases[name], cases[name]["fine50"]
            comp = case.get("versus_original_polynomial")
            status = "Reference" if comp is None else (
                "PASS" if case["passes_previous_user_constraints"] else "FAIL")
            regressions = "—" if comp is None else (
                f"{len(comp['width_regression_amplitudes_mhz'])} width / "
                f"{len(comp['contrast_regression_amplitudes_mhz'])} contrast")
            local = m["local_shoulders"]["contrast"]
            coherent = case.get("coherent50", {}).get("fwhm_hz")
            table.append(f"<tr><td>{html.escape(LABELS[case['family']])}<br>{mode_labels[case['mode']]}</td>"
                         f"<td>{case['duration_us']:.3f}</td><td>{case['transition_us']*1000:.2f}</td>"
                         f"<td>{fmt(m['fwhm_hz'], 1000)}</td><td>{fmt(m['contrast'], 1, 5)}</td><td>{fmt(local, 1, 5)}</td>"
                         f"<td>{'—' if coherent is None else fmt(coherent, 1000)}</td>"
                         f"<td>{status}</td><td>{regressions}</td></tr>")
    details = "".join(f"<details><summary>{html.escape(n)}: exact parameters and diagnostics</summary>"
                      f"<pre>{html.escape(json.dumps(c, indent=2))}</pre></details>" for n, c in cases.items())
    new_cases = [c for c in cases.values() if c["mode"] != "original"]
    passing = sum(c["passes_previous_user_constraints"] for c in new_cases)
    outcome = (f"{passing} of {len(new_cases)} unoptimized 30 us variants meet the prior contrast and amplitude-preservation requirements."
               if passing else "None of the four unoptimized 30 us variants meets the prior contrast and amplitude-preservation requirements.")
    validation = ""
    validation_path = OUT / "implementation_validation.json"
    if validation_path.exists():
        validation = ("<h2>Implementation checks</h2><pre>"
                      + html.escape(validation_path.read_text())
                      + '</pre><p><a href="implementation_validation.json">Validation record</a></p>')
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>30 us flexible echo comparison</title>
<style>body{{max-width:1500px;margin:32px auto;padding:0 20px;font:16px system-ui;line-height:1.5;color:#172435}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #ccd5dd;text-align:left}}
.note{{padding:18px;background:#edf4fa;border-left:4px solid #316c9d}}pre{{white-space:pre-wrap;max-height:500px;overflow:auto;font-size:12px}}</style></head><body>
<h1>Making the new single-echo pulses 30 us long</h1>
<p><b>{outcome}</b></p>
<p>Both recent candidates are tested because the higher-contrast pulse was 16.18 us and the narrower phase-shaped pulse was 17.52 us.
The same waveform is scaled in amplitude over 5–60 MHz; its coefficients are never retuned per amplitude. These are local simulations.</p>
<div class="note"><b>Two distinct changes:</b> “Extend duration only” keeps core, tail, phase, transition and taper-width parameters fixed and moves the outer endpoints.
“Stretch full shape” rescales every raw-envelope and phase time by 30/L, including the transition and taper; DRAG is recomputed with the rescaled derivative.
Both are normalized by the maximum of the completed complex drive over both time halves. The nominal actual peak is 50 MHz. No optimization is performed.</div>
<p>Model: T1={T1} us, T2={T2} us, alpha/2pi=−216 MHz, beta=−0.5. Maps contain 111 amplitudes × 401 detunings (0.5 MHz and 2.5 kHz steps).
The 50 MHz entries use 801 detunings and a halved timestep. The T2 reference is {1000/(np.pi*T2):.2f} kHz.</p>
<table><tr><th>Pulse</th><th>Duration (us)</th><th>Transition (ns)</th><th>FWHM50 (kHz)</th><th>Contrast50</th><th>Local depth50</th><th>Coherent FWHM50 (kHz)</th><th>Prior requirements</th><th>Regressing samples vs original polynomial</th></tr>{''.join(table)}</table>
<p>“Prior requirements” means contrast &gt;0.4 at 50 MHz and no increase in linewidth or decrease in contrast versus the original polynomial at any sampled amplitude.
Reference rows are the previously validated short pulses. FWHM and primary contrast use the same fitted-background metric as before; local shoulder depth is separate.
Exact no-regression checks are sampled tests, not interval proofs. Coherent widths remove dissipation while keeping the same pulse; widths are not additive.</p>
<h2>2D maps</h2><img src="maps.png" alt="Amplitude versus detuning maps for both pulse families and duration changes">
<h2>Linewidth, contrast, and 50 MHz spectra</h2><img src="comparison.png" alt="Slices and amplitude-dependent widths and contrasts">
<h2>Actual completed waveforms</h2><img src="waveforms.png" alt="Envelope and central I and Q waveforms">
<p><a href="results.json">Results and provenance</a> · <a href="spectra.npz">Raw derived arrays</a> · <a href="maps.pdf">Maps PDF</a> · <a href="maps.svg">Maps SVG</a> ·
<a href="../../scripts/sweep_flexible_echo_30us.py">Reproducible generator</a></p>
{validation}{details}</body></html>"""
    (OUT / "report.html").write_text(page)
    save_json(OUT / "render_provenance.json", {
        "generator_sha256": sha(Path(__file__)),
        "inputs": {n: sha(OUT / n) for n in ("results.json", "spectra.npz")},
        "outputs": {n: sha(OUT / n) for n in ("report.html", "maps.png", "maps.pdf", "maps.svg", "comparison.png", "waveforms.png")},
        "validation_sha256": sha(validation_path) if validation_path.exists() else None,
    })
    print(OUT / "report.html", flush=True)


if __name__ == "__main__":
    main()
