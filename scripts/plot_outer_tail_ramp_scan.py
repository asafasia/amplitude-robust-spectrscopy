"""Render ramp-time optimization, selected populations, and a local HTML report."""

from __future__ import annotations

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
from scripts import make_outer_tail_ramp_scan as scan  # noqa: E402

OUT = scan.OUT
COLORS = ["#536173", "#cf8125", "#087e8b", "#4968ba", "#b34c8a"]


def label(s):
    if s["kind"] == "none":
        return "Hard endpoints"
    if s["kind"] == "polynomial":
        return "Whole-pulse polynomial"
    return f"{s['ramp_us']:g} µs per-edge taper"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def selected(summaries, cutoff):
    group = [s for s in summaries if s["cutoff"] == cutoff]
    short = max((s for s in group if s["kind"] == "quintic" and s["ramp_us"] <= 6),
                key=lambda s: s["minimum_snr_per_mhz"])
    best = max((s for s in group if s["kind"] == "quintic"),
               key=lambda s: s["minimum_snr_per_mhz"])
    strict = max((s for s in group if s["kind"] == "quintic" and s["ramp_us"] < 15),
                 key=lambda s: s["minimum_snr_per_mhz"])
    choices = [next(s for s in group if s["kind"] == "none"), short, strict, best,
               next(s for s in group if s["kind"] == "polynomial")]
    return list({s["case"]: s for s in choices}.values())


def load(s):
    with np.load(OUT / f"{s['case']}.npz") as data:
        return {key: data[key].copy() for key in data.files}


def ramp_curves(summaries):
    specs = [("worst_fwhm_khz", "Worst FWHM over 50–100 MHz (kHz)"),
             ("minimum_snr", "Minimum ideal SNR over 50–100 MHz"),
             ("minimum_snr_per_mhz", "Minimum SNR / FWHM (MHz⁻¹)")]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), layout="constrained")
    for col, c in enumerate(scan.base.CUTOFFS):
        group = [s for s in summaries if s["cutoff"] == c]
        ramps = sorted((s for s in group if s["kind"] == "quintic"),
                       key=lambda s: s["ramp_us"])
        best = max(ramps, key=lambda s: s["minimum_snr_per_mhz"])
        for row, (key, ylabel) in enumerate(specs):
            ax = axes[row, col]
            ax.plot([s["ramp_us"] for s in ramps], [s[key] for s in ramps],
                    "o-", color=COLORS[2], ms=3, label="Outer-tail taper")
            for kind, color in [("none", COLORS[0]), ("polynomial", COLORS[-1])]:
                reference = next(s for s in group if s["kind"] == kind)
                ax.axhline(reference[key], color=color, ls="--", label=label(reference))
            ax.axvline(best["ramp_us"], color="black", ls=":", lw=.8)
            ax.set(xlim=(0, 15.2), ylabel=ylabel, xlabel="Ramp time at each end (µs)")
            ax.grid(alpha=.16)
        axes[0, col].set_title(f"Base cutoff {c:g} · selected R = {best['ramp_us']:g} µs")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Choose one taper for the full high-amplitude range\n"
                 "30 µs total · β = −0.5 · 16 ns echo · dots are simulated ramp times")
    save(fig, "ramp_tradeoffs")


def scan_heatmaps(rows, summaries):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained", sharey=True)
    for col, c in enumerate(scan.base.CUTOFFS):
        ss = sorted((s for s in summaries if s["cutoff"] == c and s["kind"] == "quintic"),
                    key=lambda s: s["ramp_us"])
        groups = [[r for r in rows if r["case"] == s["case"]] for s in ss]
        for row, (metric, scale, vmax, title) in enumerate([
                ("fwhm_hz", .001, 550, "FWHM (kHz)"),
                ("contrast_snr", 1., 18, "Ideal SNR")]):
            z = np.array([[r[metric]*scale for r in g] for g in groups])
            mesh = axes[row, col].pcolormesh(
                scan.base.AMPS, [s["ramp_us"] for s in ss], z,
                shading="nearest", cmap="viridis", vmin=0, vmax=vmax, rasterized=True)
            axes[row, col].axvline(50, ls=":", color="white", lw=1)
            axes[row, col].set(xlim=(0, 100), ylim=(0, 15),
                               xlabel="Nominal amplitude (MHz)",
                               ylabel="Ramp time at each end (µs)",
                               title=f"c = {c:g} · {title}")
            fig.colorbar(mesh, ax=axes[row, col], shrink=.8)
    fig.suptitle("Full ramp-time and amplitude scan · 100 amplitude samples per ramp\n"
                 "Cell colors represent saved samples; white line marks the high-drive range")
    save(fig, "scan_heatmaps")


def comparison_figures(rows, summaries):
    for c in scan.base.CUTOFFS:
        choices = selected(summaries, c)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        specs = [("fwhm_hz", .001, "FWHM (kHz)"),
                 ("contrast_snr", 1, "Ideal SNR · 1000 total shots"),
                 ("shoulder_contrast", 1, "Dip contrast"),
                 ("center_hz", .001, "Resonance center (kHz)"),
                 ("max_final_pf", 100, "Maximum final Pf in sweep (%)")]
        datasets = []
        for s, color in zip(choices, COLORS, strict=False):
            group = [r for r in rows if r["case"] == s["case"]]
            data = load(s)
            datasets.append(data)
            for ax, (key, scale, ylabel) in zip(axes.flat, specs, strict=False):
                ax.plot(scan.base.AMPS, [r[key]*scale for r in group],
                        color=color, label=label(s))
                ax.set(xlabel="Nominal amplitude (MHz)", ylabel=ylabel, xlim=(0, 100))
                ax.axvspan(50, 100, alpha=.025, color="black")
            axes[1, 2].plot(data["detuning_mhz"]*1000, data["populations"][1, -1],
                            color=color, label=label(s))
        axes[1, 2].set(xlabel="Detuning (kHz)", ylabel="Final Pe at 100 MHz")
        axes[0, 0].axhline(43.544444, color="black", ls=":", lw=.8)
        axes[0, 1].axhline(10, color="black", ls=":", lw=.8)
        axes[0, 0].legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(alpha=.15)
        fig.suptitle(f"Selected 30 µs pulses · base cutoff {c:g} · β = −0.5\n"
                     "Dotted guides: CW T₂ reference 43.5 kHz and ideal SNR 10")
        save(fig, f"comparison_c{c:g}")

        fig, axes = plt.subplots(3, len(choices), figsize=(4*len(choices), 10),
                                 layout="constrained", sharex=True, sharey=True)
        pfmax = max(d["populations"][2].max() for d in datasets)*100
        for col, (s, d) in enumerate(zip(choices, datasets, strict=True)):
            for state in range(3):
                mesh = axes[state, col].pcolormesh(
                    d["detuning_mhz"]*1000, d["rabi_mhz"],
                    d["populations"][state]*(100 if state == 2 else 1),
                    vmin=0, vmax=pfmax if state == 2 else 1,
                    cmap="magma" if state == 2 else "viridis", shading="auto",
                    rasterized=True)
                if col == len(choices)-1:
                    fig.colorbar(mesh, ax=axes[state, :], shrink=.8,
                                 label=["Pg", "Pe", "Pf (%)"][state])
                if col == 0:
                    axes[state, col].set_ylabel("Nominal amplitude (MHz)")
                if state == 2:
                    axes[state, col].set_xlabel("Detuning (kHz)")
            axes[0, col].set_title(label(s), fontsize=10)
        fig.suptitle(f"Three final populations · 30 µs · base cutoff {c:g} · β = −0.5")
        save(fig, f"populations_c{c:g}")


def waveform_figure(summaries):
    choices = selected(summaries, .001)
    t = np.unique(np.r_[np.linspace(-15, 15, 10001), np.linspace(-.1, .1, 2001)])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for s, color in zip(choices, COLORS, strict=False):
        w = [scan.base.edge_window(x, 30., kind=s["kind"], ramp_us=s["ramp_us"])[0]
             for x in t]
        iq = np.array([scan.base.waveform(x, .001, s["kind"], s["ramp_us"]) for x in t])
        axes[0].plot(t, w, color=color, label=label(s))
        axes[1].plot(t, iq[:, 0]/.001, color=color)
        axes[2].plot(t*1000, iq[:, 0], color=color)
    axes[0].set(xlabel="Time relative to midpoint (µs)", ylabel="Window multiplier",
                title="Which part of the pulse changes?", xlim=(-15, 15))
    axes[1].set(xlabel="Time relative to midpoint (µs)", ylabel="I / (A × cutoff)",
                title="Outer tail at cutoff 0.001", xlim=(-15, -3), ylim=(-.05, 5))
    axes[2].set(xlabel="Time relative to midpoint (ns)", ylabel="I / A",
                title="Central 16 ns echo transition", xlim=(-100, 100))
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle("Ramps fit inside the original 30 µs; no peak or area renormalization")
    save(fig, "waveforms")


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def report(rows, summaries):
    selections = scan.choose_ramps(summaries)
    first = next(s for s in summaries if s["case"] == selections[0]["best_tradeoff_case"])
    poly = next(s for s in summaries if s["cutoff"] == .001 and s["kind"] == "polynomial")
    a, b = first["at_100_mhz"], poly["at_100_mhz"]
    intro = (f"The best sampled taper by the stated high-drive score at cutoff 0.001 "
             f"uses {first['ramp_us']:g} µs at each end. At 100 MHz it gives "
             f"{a['fwhm_hz']/1000:.1f} kHz FWHM and ideal SNR {a['contrast_snr']:.2f}; "
             f"the whole-pulse polynomial gives {b['fwhm_hz']/1000:.1f} kHz and "
             f"SNR {b['contrast_snr']:.2f}.")
    if first["ramp_us"] >= 12:
        intro += (f" This taper leaves only {30-2*first['ramp_us']:g} µs of the central "
                  "envelope untouched, so the preferred ramp is not confined to a short outer tail.")
    strict_ramp = selections[0]["best_with_untouched_center_ramp_us"]
    near_best = ", ".join(f"{r:g}" for r in
                          selections[0]["sampled_ramps_within_5percent_of_best"])
    intro += (f" The best tested taper with a strictly untouched central segment uses "
              f"{strict_ramp:g} µs per edge, leaving {30-2*strict_ramp:g} µs unchanged. "
              f"The sampled {near_best} µs ramps lie within 5% of the best taper score.")
    winners = [next(s for s in summaries if s["case"] == r["best_tradeoff_case"])
               for r in selections]
    references = [next(s for s in summaries if s["cutoff"] == r["cutoff"]
                       and s["kind"] == "polynomial") for r in selections]
    if all(p["worst_fwhm_khz"] < w["worst_fwhm_khz"]
           and p["minimum_snr"] > w["minimum_snr"]
           for w, p in zip(winners, references, strict=True)):
        intro += (" Across all three cutoffs, the polynomial has both a smaller worst-case "
                  "linewidth and a higher minimum SNR than the best taper over 50–100 MHz. "
                  "The proposed short outer-tail taper does not improve on it in this scan.")
    tr = []
    for c in scan.base.CUTOFFS:
        for s in selected(summaries, c):
            m = s["at_100_mhz"]
            tr.append(f"<tr><td>{c:g}</td><td><a href='{s['case']}.npz'>{label(s)}</a></td>"
                      f"<td>{m['fwhm_hz']/1000:.1f}</td><td>{m['shoulder_contrast']:.3f}</td>"
                      f"<td>{m['contrast_snr']:.2f}</td><td>{m['center_hz']/1000:.3f}</td>"
                      f"<td>{s['worst_fwhm_khz']:.1f}</td><td>{s['minimum_snr']:.2f}</td>"
                      f"<td>{s['points_passing_goal']}/50</td></tr>")
    sections = "".join(
        f"<section><h2>Cutoff {c:g}: selected pulses</h2>"
        f"<img src='comparison_c{c:g}.png' alt='Linewidth, SNR, center and leakage'>"
        f"<details open><summary>Three population maps</summary>"
        f"<img src='populations_c{c:g}.png' alt='Pg Pe and Pf'></details></section>"
        for c in scan.base.CUTOFFS)
    text = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>30 µs outer-tail taper optimization</title><style>
body{font:16px/1.55 system-ui,sans-serif;background:#f1f4f7;color:#172635;margin:0}
main{max-width:1450px;margin:auto;padding:30px}h1{font-size:34px;line-height:1.2}
section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}
td,th{text-align:left;border-bottom:1px solid #dee5eb;padding:8px}th{background:#edf2f6}
.scroll{overflow-x:auto}.note{background:#edf5f8;border-left:4px solid #087e8b;padding:15px}
input{width:70px;padding:5px}label{margin-right:20px}a{color:#087e8b}summary{cursor:pointer}
</style><main><p>THREE-LEVEL SIMULATION · 30 µs · 100 AMPLITUDES TO 100 MHz</p>
<h1>How much outer-tail taper is useful?</h1><p class="note">INTRO</p>
<section><h2>What was scanned</h2><p>Quintic turn-on and turn-off ramps inside the same 30 µs pulse:
w(u)=10u³−15u⁴+6u⁵, with u from zero at the endpoint to one at the ramp join.
The central 30−2R µs retains the original root-Lorentzian envelope. Value, first derivative
and second derivative vanish at the outer endpoints. R=15 µs is a whole-pulse limiting taper.</p>
<p>Base cutoffs 0.001, 0.003, 0.005; β=−0.5; sine midpoint transition 16 ns;
α=−216 MHz; T₁=51.24 µs; T₂=7.31 µs. Nominal amplitude: 100 uniformly spaced samples
from 0 to 100 MHz. Detuning: −0.5 to +0.5 MHz, 201 points. RK4 timestep 0.5 ns.
DRAG uses the derivative of the complete windowed I envelope. No renormalization.</p>
<p>All ramp times are per edge. Hard endpoints and the previous whole-pulse polynomial
w(t)=[1−(2t/L)²]³ are controls. Matched prior arrays were reused after checking parameters,
grids and simulation source hashes.</p><img src="waveforms.png" alt="Selected waveforms"></section>
<section><h2>How “best” is defined</h2><p>Choose one fixed ramp for the entire high-drive range:
maximize the minimum SNR/FWHM over the 50 sampled amplitudes from 50.5 to 100 MHz.
An invalid dip at any of these points disqualifies the candidate. This is an explicitly chosen
sharpness heuristic, not a fitted frequency uncertainty or a universal optimum.
We also report the narrowest ramp, the highest-SNR ramp and non-dominated width/SNR choices.
The full 0–100 MHz response is retained; the zero-drive point has no resonance signal.</p>
<p class="note">SNR assumes 1000 independent ideal binary Pe shots: 500 at the dip center,
250 at each of ±450 kHz. Technical noise, readout errors and center-search overhead are absent.
The 200 kHz/SNR 10 targets are illustrative, not user-specified hard requirements.</p>
<img src="ramp_tradeoffs.png" alt="Ramp-time optimization">
<p><a href="selection.json">Selected ramp by each criterion</a> ·
<a href="summary.json">Every ramp and Pareto flags</a></p></section>
<section><h2>At 100 MHz and across 50–100 MHz</h2><div class="scroll"><table><thead><tr>
<th>Cutoff</th><th>Pulse</th><th>100 MHz width (kHz)</th><th>Contrast</th><th>Ideal SNR</th>
<th>Center (kHz)</th><th>Worst width (kHz)</th><th>Minimum SNR</th><th>Passing / 50</th>
</tr></thead><tbody>TABLE</tbody></table></div></section>
<section><h2>Test your own thresholds</h2><label>Maximum width (kHz)
<input id="width" type="number" value="200" min="1"></label><label>Minimum ideal SNR
<input id="snr" type="number" value="10" min="0" step="0.5"></label>
<div class="scroll"><table><thead><tr><th>Cutoff</th><th>Pulse</th><th>Passing / 50</th>
<th>Longest passing sampled interval (MHz)</th></tr></thead><tbody id="operating"></tbody>
</table></div></section><section><h2>The entire scan</h2>
<img src="scan_heatmaps.png" alt="Full ramp and amplitude scan"></section>SECTIONS
<section><h2>Interpretation and reproducibility</h2>
<p>Widths are nearest half-depth crossings after a linear background fit at |detuning|≥400 kHz.
Centers use quadratic interpolation of the raw central minimum. No Gaussian fit or smoothing
is used. The finite measurement window affects the reported widths. The 43.5 kHz line is the
weak-drive CW reference 1/(πT₂), not a universal lower bound on echo-dip FWHM.</p>
<p>Final Pf is not maximum transient leakage. The model omits higher transmon levels,
drive-line distortion and experimental noise. A best sampled quintic ramp does not establish
an optimum over all possible pulse shapes. Ramps were not optimized separately at each amplitude.</p>
<p><a href="metrics.csv">All metrics</a> · <a href="provenance.json">Parameters and source hashes</a>
 · <a href="waveform_metrics.json">Pulse areas</a></p>CHECKS</section></main><script>
const rows=PAYLOAD;
function update(){const width=+document.getElementById('width').value*1000;
const snr=+document.getElementById('snr').value;let out='';
for(const key of [...new Set(rows.map(r=>r.case))]){
const rr=rows.filter(r=>r.case===key&&r.amplitude_mhz>=50);let count=0,start=null,last=null,best=null;
for(const r of rr){const ok=r.accepted&&r.fwhm_hz!==null&&r.fwhm_hz<=width&&r.contrast_snr>=snr;
if(ok){count++;if(start===null)start=r.amplitude_mhz;last=r.amplitude_mhz;
if(best===null||last-start>best[1]-best[0])best=[start,last];}else{start=null;last=null;}}
out+=`<tr><td>${rr[0].cutoff}</td><td>${rr[0].window}</td><td>${count}/50</td><td>${best?best.map(x=>x.toFixed(1)).join('–'):'—'}</td></tr>`;}
document.getElementById('operating').innerHTML=out;}
document.getElementById('width').addEventListener('input',update);
document.getElementById('snr').addEventListener('input',update);update();
</script></html>"""
    checks = ""
    if (OUT / "validation.json").exists():
        validation = json.loads((OUT / "validation.json").read_text())
        max_time = max(r["time_max_fwhm_change_hz"] for r in validation)
        max_grid = max(r["grid_max_fwhm_change_hz"] for r in validation)
        checks += (f"<p>Refined selected ramps at three high amplitudes: maximum FWHM "
                   f"change {max_time:.1f} Hz on halving dt and {max_grid:.1f} Hz on "
                   f"halving the detuning step. <a href='validation.json'>Numerical and area controls</a>.</p>")
    if (OUT / "shoulder_checks.json").exists():
        shoulders = json.loads((OUT / "shoulder_checks.json").read_text())
        comparisons = []
        for winner in winners:
            for delta in [.3, .45, .6, .8, 1.]:
                subset = [r for r in shoulders if r["cutoff"] == winner["cutoff"]
                          and r["amplitude_mhz"] == 100 and r["shoulder_mhz"] == delta]
                w = next(r for r in subset if r["case"] == winner["case"])
                p = next(r for r in subset if r["window"] == "polynomial")
                comparisons.append(p["fwhm_khz"] < w["fwhm_khz"] and p["snr"] > w["snr"])
        if all(comparisons):
            checks += ("<p>At 100 MHz the polynomial is narrower and higher-SNR than the "
                       "selected taper for all three cutoffs and all five tested reference "
                       "shoulders (±0.3, ±0.45, ±0.6, ±0.8, ±1.0 MHz).</p>")
        checks += "<p><a href='shoulder_checks.json'>Wider-frequency and shoulder-placement checks</a></p>"
    if (OUT / "array_checks.json").exists():
        checks += "<p><a href='array_checks.json'>All population arrays and parameter checks</a></p>"
    if (OUT / "software_checks.json").exists():
        checks += "<p><a href='software_checks.json'>Software and data checks, with limitations</a></p>"
    for token, replacement in [("INTRO", html.escape(intro)), ("TABLE", "".join(tr)),
                               ("SECTIONS", sections), ("CHECKS", checks),
                               ("PAYLOAD", json.dumps(clean(rows), allow_nan=False))]:
        text = text.replace(token, replacement)
    (OUT / "report.html").write_text(text)


def waveform_metrics(summaries):
    t = np.unique(np.r_[np.linspace(-15, 15, 15001), np.linspace(-.1, .1, 10001)])
    results = []
    for c in scan.base.CUTOFFS:
        base_i = np.array([scan.base.waveform(x, c, "none", 0)[0] for x in t])
        area = np.trapezoid(abs(base_i), t)
        for s in (s for s in summaries if s["cutoff"] == c):
            # I is exactly the base I times the window, including the echo.
            w = np.array([scan.base.edge_window(x, 30., kind=s["kind"],
                                                ramp_us=s["ramp_us"])[0] for x in t])
            i = base_i*w
            results.append(dict(case=s["case"], cutoff=c, window=s["window"],
                                ramp_us=s["ramp_us"],
                                area_fraction=float(np.trapezoid(abs(i), t)/area),
                                peak_i_scale=float(abs(i).max())))
    (OUT / "waveform_metrics.json").write_text(json.dumps(results, indent=2))


def main():
    scan.configure()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    rows, summaries = scan.export_metrics()
    if len(summaries) < 45:
        raise RuntimeError("Complete the 13-ramp, 3-cutoff coarse scan first")
    waveform_metrics(summaries)
    ramp_curves(summaries)
    scan_heatmaps(rows, summaries)
    comparison_figures(rows, summaries)
    waveform_figure(summaries)
    report(rows, summaries)
    print(OUT / "report.html")


if __name__ == "__main__":
    main()
