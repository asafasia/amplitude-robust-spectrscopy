"""Render the endpoint-ramp simulation report from saved, reproducible arrays."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.make_echo_edge_ramp_comparison import (  # noqa: E402
    AMPS,
    COMMON,
    CUTOFFS,
    OUT,
    WINDOWS,
    add_experiment_arguments,
    case_name,
    configure_experiment,
    export_metrics,
    waveform,
)

LABELS = {"hard": "Hard endpoints", "ramp1": "1 µs ramps",
          "ramp3": "3 µs ramps", "ramp6": "6 µs ramps",
          "polynomial": "Polynomial window"}
COLORS = {"hard": "#536173", "ramp1": "#147db3", "ramp3": "#2a9d8f",
          "ramp6": "#e28724", "polynomial": "#b34c8a"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def load(cutoff, window, beta=-.5, suffix=""):
    with np.load(OUT / f"{case_name(cutoff, window, beta)}{suffix}.npz") as data:
        return {key: data[key].copy() for key in data.files}


def grouped(rows):
    return {case_name(c, w): [row for row in rows if row["case"] == case_name(c, w)]
            for c in CUTOFFS for w, _, _ in WINDOWS}


def waveforms():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    areas = []
    half = COMMON["duration_us"] / 2
    for col, c in enumerate(CUTOFFS):
        time = np.unique(np.r_[np.linspace(-half, half, 20001),
                               np.linspace(-.1, .1, 20001)])
        for w, kind, ramp in WINDOWS:
            iq = np.array([waveform(t, c, kind, ramp) for t in time])
            base = np.array([waveform(t, c, "none", 0)[0] for t in time])
            area = np.trapezoid(abs(iq[:, 0]), time)
            ratio = area / np.trapezoid(abs(base), time)
            areas.append(dict(cutoff=c, window=w, unsigned_area_us=float(area),
                              area_fraction=float(ratio),
                              peak_i_scale=float(abs(iq[:, 0]).max())))
            # Divide out nominal cutoff to expose the weak endpoint tails.
            mask = time <= -1
            axes[0, col].plot(time[mask], iq[mask, 0]/c, color=COLORS[w], label=LABELS[w])
            axes[1, col].plot(time[mask], iq[mask, 1]/c, color=COLORS[w])
        axes[0, col].set(title=f"Base cutoff {c:g}", ylabel="I / (A × cutoff)",
                         ylim=(-.02, 5), xlim=(-half, -half/5))
        axes[1, col].set(xlabel="Time relative to midpoint (µs)",
                         ylabel="Q / (A × cutoff)", xlim=(-half, -half/5))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Smooth outer edges; central width and 16 ns echo held fixed\n"
                 "Turn-on shown; turn-off follows echo symmetry. β = −0.5")
    save(fig, "waveforms")
    (OUT / "waveform_metrics.json").write_text(json.dumps(areas, indent=2))
    return areas


def populations():
    for c in CUTOFFS:
        datasets = [load(c, w) for w, _, _ in WINDOWS]
        fig, axes = plt.subplots(3, 5, figsize=(19, 10), layout="constrained",
                                 sharex=True, sharey=True)
        pfmax = max(data["populations"][2].max() for data in datasets)
        for col, ((w, _, _), data) in enumerate(zip(WINDOWS, datasets, strict=True)):
            for state in range(3):
                scale = 100 if state == 2 else 1
                mesh = axes[state, col].pcolormesh(
                    data["detuning_mhz"]*1000, data["rabi_mhz"],
                    data["populations"][state]*scale, shading="auto",
                    vmin=0, vmax=pfmax*100 if state == 2 else 1,
                    cmap="viridis" if state != 2 else "magma", rasterized=True)
                if col == 4:
                    fig.colorbar(mesh, ax=axes[state, :], label=["Pg", "Pe", "Pf (%)"][state],
                                 shrink=.85, pad=.012)
                if col == 0:
                    axes[state, col].set_ylabel("Nominal amplitude (MHz)")
                if state == 2:
                    axes[state, col].set_xlabel("Detuning (kHz)")
            axes[0, col].set_title(LABELS[w])
        fig.suptitle(f"Three final-state populations | base cutoff {c:g} | β = −0.5\n"
                     f"{COMMON['duration_us']:g} µs duration · 16 ns echo · 100 amplitude points")
        save(fig, f"populations_c{c:g}")


def quality_curves(groups):
    specs = [("fwhm_hz", .001, "Dip FWHM (kHz)"),
             ("shoulder_contrast", 1, "Dip contrast (population)"),
             ("contrast_snr", 1, "Ideal contrast SNR · 1000 total shots"),
             ("center_hz", .001, "Resonance center (kHz)"),
             ("contrast_per_mhz", 1, "Contrast / FWHM (MHz⁻¹)"),
             ("max_final_pf", 100, "Maximum final Pf in sweep (%)")]
    for c in CUTOFFS:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for ax, (key, scale, ylabel) in zip(axes.flat, specs, strict=True):
            for w, _, _ in WINDOWS:
                group = groups[case_name(c, w)]
                ax.plot([r["amplitude_mhz"] for r in group],
                        [r[key]*scale for r in group], color=COLORS[w], label=LABELS[w])
            ax.axvspan(50, 100, color="#8892a0", alpha=.07)
            ax.set(xlabel="Nominal amplitude (MHz)", ylabel=ylabel, xlim=(0, 100))
            if key in {"shoulder_contrast", "contrast_snr", "fwhm_hz", "contrast_per_mhz"}:
                ax.set_ylim(bottom=0)
            if key == "contrast_snr":
                ax.axhline(10, color="black", ls=":", lw=.8)
            if key == "fwhm_hz":
                ax.axhline(200, color="black", ls=":", lw=.8)
            ax.grid(alpha=.18)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Narrowness and visibility | base cutoff {c:g} | β = −0.5 | "
                     f"{COMMON['duration_us']:g} µs\n"
                     "Shading: 50–100 MHz. Illustrative guides: 200 kHz FWHM and ideal SNR 10.")
        save(fig, f"quality_c{c:g}")


def slices():
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), layout="constrained", sharex=True)
    for row, c in enumerate(CUTOFFS):
        for col, nominal in enumerate([50, 75, 100]):
            index = np.argmin(abs(AMPS - nominal))
            for w, _, _ in WINDOWS:
                d = load(c, w)
                axes[row, col].plot(d["detuning_mhz"]*1000, d["populations"][1, index],
                                    color=COLORS[w], label=LABELS[w])
            axes[row, col].set(title=f"c = {c:g} · A = {AMPS[index]:.1f} MHz",
                               ylabel="Final Pe", ylim=(0, 1), xlim=(-500, 500))
            axes[row, col].grid(alpha=.15)
            if row == 2:
                axes[row, col].set_xlabel("Detuning (kHz)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Actual dip profiles at high drive · β = −0.5 · {COMMON['duration_us']:g} µs")
    save(fig, "high_amplitude_slices")


def equal_area_curves(groups, areas):
    """Relabel each sampled amplitude by its exact unsigned-I-area equivalent."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for col, c in enumerate(CUTOFFS):
        for w, _, _ in WINDOWS:
            ratio = next(a["area_fraction"] for a in areas
                         if a["cutoff"] == c and a["window"] == w)
            group = groups[case_name(c, w)]
            x = np.array([r["amplitude_mhz"] for r in group]) * ratio
            axes[0, col].plot(x, [r["fwhm_hz"]/1000 for r in group],
                              color=COLORS[w], label=LABELS[w])
            axes[1, col].plot(x, [r["contrast_snr"] for r in group], color=COLORS[w])
        axes[0, col].set(title=f"Base cutoff {c:g}", ylabel="Dip FWHM (kHz)",
                         ylim=(0, None), xlim=(0, 100))
        axes[1, col].set(ylabel="Ideal contrast SNR · 1000 shots", ylim=(0, None),
                         xlabel="Hard-pulse amplitude with equal unsigned I-area (MHz)",
                         xlim=(0, 100))
        for ax in axes[:, col]:
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Does improvement survive matching pulse area?\n"
                 "Same simulations, horizontal coordinate scaled by unsigned I-area; "
                 "comparison only within overlapping ranges")
    save(fig, "equal_area_quality")


def headline(groups):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    for w in ["hard", "ramp6", "polynomial"]:
        group = groups[case_name(.001, w)]
        x = [r["amplitude_mhz"] for r in group]
        axes[0].plot(x, [r["fwhm_hz"]/1000 for r in group], color=COLORS[w], label=LABELS[w])
        axes[1].plot(x, [r["contrast_snr"] for r in group], color=COLORS[w])
        data = load(.001, w)
        axes[2].plot(data["detuning_mhz"]*1000, data["populations"][1, -1], color=COLORS[w])
    axes[0].set(xlabel="Nominal amplitude (MHz)", ylabel="Dip FWHM (kHz)", xlim=(50, 100))
    axes[1].set(xlabel="Nominal amplitude (MHz)", ylabel="Ideal contrast SNR", xlim=(50, 100))
    axes[2].set(xlabel="Detuning (kHz)", ylabel="Final Pe at 100 MHz", xlim=(-500, 500))
    axes[0].legend(fontsize=9)
    axes[1].axhline(10, ls=":", lw=.8, color="black")
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle("Smooth edges: narrowness and visibility at high amplitude\n"
                 f"Base cutoff 0.001 · fixed {COMMON['duration_us']:g} µs duration · "
                 "16 ns echo · β = −0.5 · ideal SNR: 1000 total shots")
    save(fig, "headline_comparison")


def compare_duration(groups, reference_dir):
    """Compare saved 20 us data at the same nominal amplitudes and base cutoffs."""
    if COMMON["duration_us"] == 20:
        return
    source = Path(reference_dir)
    if not (source / "metrics.csv").exists():
        raise FileNotFoundError(f"Missing duration reference: {source}")
    with (source / "metrics.csv").open() as file:
        previous = list(csv.DictReader(file))
    result = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for col, c in enumerate(CUTOFFS):
        for w in ["hard", "ramp6", "polynomial"]:
            name = case_name(c, w)
            old = [r for r in previous if r["case"] == name]
            current = groups[name]
            with np.load(source / f"{name}.npz") as data:
                old_config = json.loads(str(data["config"]))
                assert old_config["duration_us"] == 20
                for key, value in COMMON.items():
                    if key != "duration_us":
                        assert old_config[key] == value, key
                with np.load(OUT / f"{name}.npz") as new_data:
                    new_config = json.loads(str(new_data["config"]))
                    for key, value in old_config.items():
                        if key not in {"duration_us", "num_steps_per_half"}:
                            assert new_config[key] == value, key
                    np.testing.assert_array_equal(data["detuning_mhz"],
                                                  new_data["detuning_mhz"])
                np.testing.assert_array_equal(data["rabi_mhz"], AMPS)
            for duration, group, style in [(20., old, "--"),
                                            (COMMON["duration_us"], current, "-")]:
                x = [float(r["amplitude_mhz"]) for r in group]
                axes[0, col].plot(x, [float(r["fwhm_hz"])/1000 for r in group],
                                  color=COLORS[w], ls=style,
                                  label=f"{LABELS[w]} · {duration:g} µs")
                axes[1, col].plot(x, [float(r["contrast_snr"]) for r in group],
                                  color=COLORS[w], ls=style)
            result.append(dict(
                cutoff=c, window=w, previous_duration_us=20.,
                duration_us=COMMON["duration_us"],
                previous_at_100={k: float(old[-1][k]) for k in
                                ["fwhm_hz", "shoulder_contrast", "contrast_snr", "center_hz"]},
                current_at_100={k: current[-1][k] for k in
                               ["fwhm_hz", "shoulder_contrast", "contrast_snr", "center_hz"]},
                improves_both_high_amplitude_points=sum(
                    n["fwhm_hz"] < float(o["fwhm_hz"])
                    and n["contrast_snr"] > float(o["contrast_snr"])
                    for n, o in zip(current, old, strict=True)
                    if n["amplitude_mhz"] >= 50)))
        axes[0, col].set(title=f"Base cutoff {c:g}", ylabel="Dip FWHM (kHz)",
                         xlim=(50, 100))
        axes[1, col].set(xlabel="Nominal amplitude (MHz)", ylabel="Ideal contrast SNR",
                         xlim=(50, 100))
        for ax in axes[:, col]:
            ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"20 versus {COMMON['duration_us']:g} µs · same base cutoff and nominal amplitude\n"
                 "Core width scales with duration; 1/3/6 µs ramps and 16 ns midpoint stay fixed")
    save(fig, "duration_comparison")
    (OUT / "duration_comparison.json").write_text(json.dumps(result, indent=2))


def summary(groups):
    result = []
    reference = groups[case_name(.001, "hard")]
    for c in CUTOFFS:
        for w, _, _ in WINDOWS:
            group = groups[case_name(c, w)]
            high = [r for r in group if r["amplitude_mhz"] >= 50]
            baseline = groups[case_name(c, "hard")]
            dominate = [r["fwhm_hz"] < b["fwhm_hz"] and r["contrast_snr"] >= b["contrast_snr"]
                        for r, b in zip(group, baseline, strict=True) if r["amplitude_mhz"] >= 50]
            goal = [r for r in high if r["fwhm_hz"] <= 200000 and r["contrast_snr"] >= 10]
            result.append(dict(
                case=case_name(c, w), cutoff=c, window=w,
                points_passing_goal=len(goal), high_amplitude_points=len(high),
                dominates_same_core_points=sum(dominate),
                median_width_khz=float(np.nanmedian([r["fwhm_hz"] for r in high])/1000),
                worst_width_khz=float(np.nanmax([r["fwhm_hz"] for r in high])/1000),
                minimum_snr=float(np.nanmin([r["contrast_snr"] for r in high])),
                median_snr=float(np.nanmedian([r["contrast_snr"] for r in high])),
                minimum_contrast_per_mhz=float(np.nanmin([r["contrast_per_mhz"] for r in high])),
                at_100_mhz=group[-1],
                dominates_previous_reference_points=sum(
                    r["fwhm_hz"] < b["fwhm_hz"] and r["contrast_snr"] >= b["contrast_snr"]
                    for r, b in zip(group, reference, strict=True) if r["amplitude_mhz"] >= 50),
            ))
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def report(rows, summaries, areas):
    reference = next(r for r in summaries if r["case"] == case_name(.001, "hard"))["at_100_mhz"]
    candidate = next(r for r in summaries if r["case"] == case_name(.001, "polynomial"))["at_100_mhz"]
    finding = (f"At 100 MHz and base cutoff 0.001, the polynomial window gives "
               f"{candidate['fwhm_hz']/1000:.1f} kHz FWHM versus "
               f"{reference['fwhm_hz']/1000:.1f} kHz for hard endpoints. "
               f"Its contrast is {candidate['shoulder_contrast']:.3f} versus "
               f"{reference['shoulder_contrast']:.3f}; ideal SNR is "
               f"{candidate['contrast_snr']:.1f} versus {reference['contrast_snr']:.1f}.")
    table = "".join(f"<tr><td>{r['cutoff']:g}</td><td>{LABELS[r['window']]}</td>"
                    f"<td>{r['at_100_mhz']['fwhm_hz']/1000:.1f}</td>"
                    f"<td>{r['at_100_mhz']['shoulder_contrast']:.3f}</td>"
                    f"<td>{r['at_100_mhz']['contrast_snr']:.1f}</td>"
                    f"<td>{r['points_passing_goal']}/50</td></tr>" for r in summaries)
    areatable = "".join(f"<tr><td>{r['cutoff']:g}</td><td>{LABELS[r['window']]}</td>"
                        f"<td>{100*r['area_fraction']:.2f}%</td></tr>" for r in areas)
    sections = "".join(f"<section><h2>Base cutoff {c:g}</h2>"
                       f"<img src='quality_c{c:g}.png' alt='Quality curves for cutoff {c:g}'>"
                       f"<details><summary>Show all three populations</summary>"
                       f"<img src='populations_c{c:g}.png' alt='Final populations'></details></section>"
                       for c in CUTOFFS)
    payload = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in row.items()} for row in rows]
    text = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DURATION µs: smooth endpoints in echo spectroscopy</title><style>
body{font:16px/1.55 system-ui,sans-serif;background:#f1f4f7;color:#172635;margin:0}
main{max-width:1400px;margin:auto;padding:32px}h1{font-size:34px;line-height:1.2}
h2{font-size:23px}section,.card{background:white;padding:24px;margin:24px 0;border-radius:12px}
img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
td,th{text-align:left;border-bottom:1px solid #e0e5eb;padding:9px}th{background:#edf2f6}
.tag{color:#35677f;font-weight:650;letter-spacing:.08em;font-size:13px}
.note{background:#eef5fa;border-left:4px solid #377da0;padding:14px}a{color:#146991}
label{display:inline-block;margin:0 22px 12px 0}input{width:85px;padding:6px}summary{cursor:pointer;font-weight:650}
</style><main><div class="tag">SIMULATION STUDY · 100 AMPLITUDE POINTS · THREE LEVELS</div>
<h1>DURATION µs: can smooth endpoints improve echo spectroscopy?</h1>
<p>Goal: a narrow resonance dip with good visibility across high drive amplitudes.
Each cutoff fixes the underlying Lorentzian width. Only the endpoint window changes within each group.</p>
<section><p><b>FINDING</b></p><img src="headline_comparison.png" alt="High amplitude comparison">
<p><a href="headline_comparison.pdf">Download this comparison as PDF</a></p></section>
<div class="card"><b>Matched physical settings</b><p>DURATION µs duration; 16 ns sine echo transition;
β = −0.5; α = −216 MHz; T₁ = 51.24 µs; T₂ = 7.31 µs. Nominal amplitude 0–100 MHz,
100 points; detuning −500 to +500 kHz, 201 points. Lindblad RK4, dt = 0.5 ns.
Ramps occupy the first and last 1, 3 or 6 µs of the existing DURATION µs pulse.</p>
<p class="note"><b>SNR is an ideal projection-noise estimate, not experimental SNR.</b>
1000 total shots: 500 at the dip center, 250 at each of ±450 kHz. Signal is the mean shoulder
Pe minus the center Pe. Noise includes independent Bernoulli variance at those three points.
Perfect Pe classification is assumed; technical noise, drift and center-search overhead are absent.</p></div>
<section><h2>Set your definition of useful</h2><p>The defaults below are illustrative design targets,
not measured requirements. Counts apply to the 50 sampled amplitudes between 50.5 and 100 MHz.</p>
<label>Maximum FWHM (kHz) <input id="width" type="number" value="200" min="10" step="10"></label>
<label>Minimum ideal SNR <input id="snr" type="number" value="10" min="0" step="1"></label>
<table><thead><tr><th>Base cutoff</th><th>Window</th><th>Passing / 50</th><th>Longest passing sampled interval (MHz)</th></tr></thead>
<tbody id="operating"></tbody></table></section>
<section><h2>At 100 MHz</h2><table><thead><tr><th>Base cutoff</th><th>Window</th><th>FWHM (kHz)</th>
<th>Contrast</th><th>Ideal SNR</th><th>High-drive points passing 200 kHz / SNR 10</th></tr></thead><tbody>TABLE</tbody></table></section>
SECTIONS
<section><h2>Check the dip itself</h2><img src="high_amplitude_slices.png" alt="Dip profiles"></section>
<section><h2>Separate shaping from loss of pulse area</h2><img src="equal_area_quality.png" alt="Equal area comparison">
<p>Each curve is plotted against A × (windowed unsigned I-area / hard unsigned I-area).
Equal horizontal coordinates therefore mean equal unsigned I-area within a given cutoff.
This is a re-expression of the simulated grid, with lines joining samples. Compare only the overlapping
horizontal range; the windowed pulse requires a higher nominal amplitude for the same area.</p></section>
<section><h2>What changed in the waveform?</h2><img src="waveforms.png" alt="Turn-on waveform">
<p>Quintic ramps use w(u) = 10u³ − 15u⁴ + 6u⁵, with u from zero at the endpoint to one at the ramp join.
The polynomial uses w(t) = [1 − (2t/L)²]³. Both have zero value, slope and curvature at the outer endpoints.
DRAG is recomputed from the complete windowed I envelope: Q = −β I′/(2π|α|).
No amplitude or area normalization is applied in the primary comparison.</p>
<details><summary>Unsigned I-area remaining relative to hard endpoints</summary><table>
<thead><tr><th>Base cutoff</th><th>Window</th><th>Area remaining</th></tr></thead><tbody>AREAS</tbody></table></details></section>
<section><h2>Interpretation and reproducibility</h2>
<p>FWHM uses the nearest half-depth crossings after a linear background fit at |detuning| ≥ 400 kHz.
Center is the raw central minimum refined by three-point quadratic interpolation. No Gaussian fit or smoothing.
Contrast/SNR uses the actual ±450 kHz shoulders. Width and contrast depend on this measurement window.</p>
<p>The high-drive range is defined here as 50–100 MHz. A narrow feature with negligible contrast is not useful.
Final Pf is not maximum transient leakage. The three-level model omits higher transmon states, transfer-function
distortion and readout errors. Results concern this finite pulse and these coherence parameters.</p>
<p><a href="metrics.csv">All metrics CSV</a> · <a href="summary.json">Summary</a> ·
<a href="provenance.json">Parameters and source hashes</a> · <a href="waveform_metrics.json">Waveform areas</a> ·
<a href="backend_agreement.json">Independent RK4 implementation agreement</a></p>
<div id="checks"></div></section></main><script>
const rows=PAYLOAD; const names=NAMES;
function update(){const w=+document.getElementById('width').value*1000,s=+document.getElementById('snr').value;
let out='';for(const key of [...new Set(rows.map(r=>r.case))]){const rr=rows.filter(r=>r.case===key&&r.amplitude_mhz>=50);
let count=0,start=null,last=null,best=null;for(const r of rr){const pass=r.accepted&&r.fwhm_hz!==null&&r.fwhm_hz<=w&&r.contrast_snr>=s;
if(pass){count++;if(start===null)start=r.amplitude_mhz;last=r.amplitude_mhz;if(best===null||last-start>best[1]-best[0])best=[start,last];}
else{start=null;last=null;}}const interval=best?best.map(x=>x.toFixed(1)).join('–'):'—';
out+=`<tr><td>${rr[0].cutoff}</td><td>${names[rr[0].window]}</td><td>${count}/50</td><td>${interval}</td></tr>`;}
document.getElementById('operating').innerHTML=out;}
document.getElementById('width').addEventListener('input',update);document.getElementById('snr').addEventListener('input',update);update();
</script></html>"""
    text = text.replace("DURATION", f"{COMMON['duration_us']:g}")
    text = text.replace("TABLE", table).replace("SECTIONS", sections).replace("AREAS", areatable)
    text = text.replace("FINDING", finding)
    text = text.replace("PAYLOAD", json.dumps(payload, allow_nan=False)).replace("NAMES", json.dumps(LABELS))
    duration_file = OUT / "duration_comparison.json"
    if COMMON["duration_us"] != 20 and duration_file.exists():
        comparisons = json.loads(duration_file.read_text())
        selected = next(r for r in comparisons
                        if r["cutoff"] == .001 and r["window"] == "polynomial")
        old, new = selected["previous_at_100"], selected["current_at_100"]
        duration_finding = (f"For the polynomial window at cutoff 0.001 and 100 MHz, "
                            f"FWHM changes from {old['fwhm_hz']/1000:.1f} to "
                            f"{new['fwhm_hz']/1000:.1f} kHz and ideal SNR changes from "
                            f"{old['contrast_snr']:.2f} to {new['contrast_snr']:.2f}.")
        if new["fwhm_hz"] < old["fwhm_hz"] and new["contrast_snr"] < old["contrast_snr"]:
            duration_finding += " The longer pulse gives a narrower dip with lower visibility."
        comparison_table = "".join(
            f"<tr><td>{r['cutoff']:g}</td><td>{LABELS[r['window']]}</td>"
            f"<td>{r['previous_at_100']['fwhm_hz']/1000:.1f} → "
            f"{r['current_at_100']['fwhm_hz']/1000:.1f}</td>"
            f"<td>{r['previous_at_100']['contrast_snr']:.2f} → "
            f"{r['current_at_100']['contrast_snr']:.2f}</td>"
            f"<td>{r['improves_both_high_amplitude_points']}/50</td></tr>"
            for r in comparisons)
        extra = (f"<section><h2>Compared with 20 µs</h2>"
                 f"<p class='note'>{duration_finding}</p>"
                 f"<p>Holding the base cutoff fixed scales the core width σ by "
                 f"{COMMON['duration_us']/20:g}. Here σ is "
                 + ", ".join(f"{COMMON['duration_us']*500/np.sqrt(c**-2-1):.2f} ns"
                             for c in CUTOFFS)
                 + " for cutoffs 0.001, 0.003, 0.005. This compares the requested settings; "
                 "it does not isolate duration at fixed σ. The coherence times, echo transition, "
                 "nominal amplitudes and physical ramp times are unchanged.</p>"
                 '<img src="duration_comparison.png" alt="Duration comparison">'
                 '<table><thead><tr><th>Cutoff</th><th>Window</th>'
                 f'<th>FWHM at 100 MHz: 20 → {COMMON["duration_us"]:g} µs (kHz)</th>'
                 '<th>Ideal SNR at 100 MHz</th><th>Points improving both / 50</th>'
                 '</tr></thead><tbody>' + comparison_table + '</tbody></table>'
                 '<p><a href="duration_comparison.json">Duration comparison data</a></p></section>')
        text = text.replace('<section><h2>Set your definition', extra + '<section><h2>Set your definition')
    checks = OUT / "validation.json"
    if checks.exists():
        check_rows = json.loads(checks.read_text())
        max_pop = max(r["max_population_difference"] for r in check_rows)
        max_time = max(r["time_refinement_max_fwhm_hz_change"] for r in check_rows)
        max_grid = max(r["grid_refinement_max_fwhm_hz_change"] for r in check_rows)
        text = text.replace('<div id="checks"></div>', '<h3>Numerical checks</h3>'
                            f'<p>Checked {len(check_rows)} window/core combinations at three high amplitudes. '
                            f'Maximum population difference: {max_pop:.2g}; maximum FWHM change from '
                            f'dt = 0.5 to 0.25 ns: {max_time:.1f} Hz; from 5 to 2.5 kHz detuning steps: '
                            f'{max_grid:.1f} Hz. <a href="validation.json">Full numerical and equal-area checks</a>.</p>')
    if (OUT / "software_checks.json").exists():
        extra = ('<p><a href="software_checks.json">Software validation and repository limitations</a>'
                 + (' · <a href="array_checks.json">Population and parameter checks</a>'
                    if (OUT / "array_checks.json").exists() else '') + '</p>')
        text = text.replace('</main>', extra + '</main>')
    if (OUT/"beta_controls.png").exists():
        controls = json.loads((OUT/"beta_controls.json").read_text())
        shoulder_data = json.loads((OUT/"shoulder_sensitivity.json").read_text())
        shoulder_ranges = []
        for window in ["hard", "polynomial"]:
            subset = [r for r in shoulder_data if r["window"] == window
                      and r["amplitude_mhz"] == 100]
            if subset:
                shoulder_ranges.append(
                    f"{LABELS[window]}: {min(r['fwhm_khz'] for r in subset):.1f}–"
                    f"{max(r['fwhm_khz'] for r in subset):.1f} kHz")
        shoulder_text = ("At 100 MHz, moving the reference shoulders from ±0.3 to ±1.0 MHz gives "
                         + "; ".join(shoulder_ranges)
                         + ". The narrower, more visible dip survives every tested shoulder placement.")
        control_rows = "".join(
            f"<tr><td>{r['beta']:g}</td><td>{html.escape(r['window'])}</td>"
            f"<td>{r['center_hz']/1000:.3f}</td><td>{r['fwhm_hz']/1000:.1f}</td>"
            f"<td>{r['contrast_snr']:.1f}</td></tr>"
            for r in controls if r["amplitude_mhz"] == 100)
        extra = ('<section><h2>DRAG and wider-frequency controls</h2>'
                 '<p>Additional β = 0 and −1 checks use three high amplitudes, rather than the full 100-point grid. '
                 'The line profiles and table below show 100 MHz.</p>'
                 '<img src="beta_controls.png" alt="DRAG controls"><table><thead><tr>'
                 '<th>β</th><th>Window</th><th>Center (kHz)</th><th>FWHM (kHz)</th><th>Ideal SNR</th>'
                 '</tr></thead><tbody>' + control_rows + '</tbody></table>'
                 '<img src="wide_frequency_check.png" alt="Wide frequency check">'
                 '<p>' + shoulder_text + '</p>'
                 '<p><a href="beta_controls.json">All DRAG controls</a> · '
                 '<a href="shoulder_sensitivity.json">Sensitivity to shoulder placement</a></p></section>')
        text = text.replace('</main>', extra + '</main>')
    (OUT / "report.html").write_text(text)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.add_argument("--reference-dir", type=Path,
                        default=ROOT / "outputs/echo_edge_ramp_comparison")
    args = parser.parse_args()
    OUT = configure_experiment(args.duration_us, args.output_dir)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlepad": 9})
    rows = export_metrics()
    groups = grouped(rows)
    if not all(len(group) == 100 for group in groups.values()):
        raise RuntimeError("Wait for all 15 simulation maps before rendering the report")
    areas = waveforms()
    populations()
    quality_curves(groups)
    slices()
    equal_area_curves(groups, areas)
    headline(groups)
    compare_duration(groups, args.reference_dir)
    summaries = summary(groups)
    report(rows, summaries, areas)
    with (OUT / "summary.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=[k for k in summaries[0] if k != "at_100_mhz"])
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != "at_100_mhz"} for row in summaries)
    print(OUT / "report.html")


if __name__ == "__main__":
    main()
