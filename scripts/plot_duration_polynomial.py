"""Render the duration/exponent study from saved, parameter-checked simulations."""

from __future__ import annotations

import hashlib
import html
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts import scan_duration_polynomial as scan  # noqa: E402

OUT = scan.OUT
REFERENCE = 1000 / (np.pi * scan.T2_US)


def label(row):
    family = "subtract" if row["family"] == "subtracted" else "original"
    scaling = ", fixed core" if row["scaling"] == "core" else ""
    return f"{row['duration_us']:g} µs, p={row['power']:g}, {family}{scaling}"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def screen_figure(summary):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for column, family in enumerate(["raw", "subtracted"]):
        for row, (key, title, lo, hi, cmap) in enumerate([
                ("worst_fwhm_khz", "Worst FWHM (kHz)", REFERENCE, 350, "magma_r"),
                ("minimum_snr", "Minimum ideal SNR", 0, 20, "viridis")]):
            values = np.full((len(scan.DURATIONS), len(scan.POWERS)), np.nan)
            for s in summary:
                if (s["family"] == family and s["scaling"] == "cutoff"
                        and s["duration_us"] in scan.DURATIONS
                        and s["power"] in scan.POWERS):
                    i = scan.DURATIONS.index(s["duration_us"])
                    j = scan.POWERS.index(s["power"])
                    values[i, j] = np.nan if s[key] is None else s[key]
            ax = axes[row, column]
            palette = plt.get_cmap(cmap).copy()
            palette.set_bad("#dddddd")
            image = ax.imshow(values, vmin=lo, vmax=hi, cmap=palette, aspect="auto")
            ax.set_xticks(range(len(scan.POWERS)), [f"{p:g}" for p in scan.POWERS])
            ax.set_yticks(range(len(scan.DURATIONS)),
                          [f"{d:g}" for d in scan.DURATIONS])
            ax.set(xlabel="Polynomial exponent p", ylabel="Total duration (µs)",
                   title=f"{'Original' if family == 'raw' else 'Cutoff-subtracted'}: {title}")
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    value = values[i, j]
                    text = "invalid" if not np.isfinite(value) else (
                        f"{value:.0f}" if row == 0 else f"{value:.2f}")
                    color = "black" if not np.isfinite(value) else "white"
                    ax.text(j, i, text, ha="center", va="center", color=color,
                            fontsize=10, bbox=dict(facecolor="black", alpha=.35,
                                                  edgecolor="none", pad=2))
            fig.colorbar(image, ax=ax, label=title, extend="max")
    fig.suptitle("Longer duration × stronger polynomial taper\n"
                 "Three-amplitude screen: 50.5, 74.7, 100 MHz • cutoff = 0.001",
                 fontsize=16)
    save(fig, "screen")


def selected_figures(summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), layout="constrained")
    details, dax = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    slices, sax = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")
    map_figures = [plt.subplots(3, min(3, len(summary) - start),
                               figsize=(12, 9), layout="constrained", squeeze=False)
                   for start in range(0, len(summary), 3)]
    colors = plt.get_cmap("tab10")(np.arange(len(summary)))
    for col, (s, color) in enumerate(zip(summary, colors, strict=True)):
        maps, maxes = map_figures[col // 3]
        map_col = col % 3
        path = OUT / f"{s['case']}_full.npz"
        rows = scan.metrics(path)
        x = np.array([r["amplitude_mhz"] for r in rows])
        widths = np.array([r["fwhm_hz"] / 1000 for r in rows])
        snr = np.array([r["contrast_snr"] for r in rows])
        high = x >= 50
        axes[0].plot(x[high], widths[high], color=color, label=label(s), lw=2)
        axes[1].plot(x[high], snr[high], color=color, label=label(s), lw=2)
        quantities = [widths, snr, [r["center_hz"] / 1000 for r in rows],
                      [100 * r["max_final_pf"] for r in rows]]
        for ax, quantity in zip(dax.flat, quantities, strict=True):
            ax.plot(x, quantity, color=color, label=label(s))
        with np.load(path) as data:
            p, det = data["populations"], data["detuning_mhz"] * 1000
            for level in range(3):
                factor = 100 if level == 2 else 1
                sax[level].plot(det, factor * p[level, -1], color=color,
                                label=label(s))
                image = maxes[level, map_col].pcolormesh(
                    det, x, factor * p[level], shading="auto", cmap="viridis",
                    vmin=0, vmax=.04 if level == 2 else 1, rasterized=True)
                if map_col == 0:
                    maxes[level, map_col].set_ylabel(
                        f"P{'gef'[level]}{' (%)' if level == 2 else ''}\nAmplitude (MHz)")
                if level == 0:
                    maxes[level, map_col].set_title(label(s).replace(", ", "\n"),
                                                  fontsize=11)
                if level == 2:
                    maxes[level, map_col].set_xlabel("Detuning (kHz)")
                if map_col == maxes.shape[1] - 1:
                    maps.colorbar(image, ax=list(maxes[level]), shrink=.8)
    axes[0].axhline(REFERENCE, color="black", ls="--", label="T₂ reference: 43.54 kHz")
    axes[0].set(xlabel="Nominal amplitude (MHz)", ylabel="FWHM (kHz)",
                ylim=(0, 310), title="Narrower long-pulse dips lose visibility")
    axes[1].axhline(10, color="black", ls="--", label="SNR = 10")
    axes[1].set(xlabel="Nominal amplitude (MHz)", ylabel="Ideal contrast SNR",
                ylim=(0, 19), title="Same 1,000-shot budget for every pulse")
    for ax in axes:
        ax.grid(alpha=.2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc="outside lower center", ncol=3)
    axes[0].text(51, REFERENCE + 4, "T₂ reference", fontsize=10)
    fig.suptitle("Selected pulses: full 100-amplitude maps; high-amplitude region shown",
                 fontsize=14)
    save(fig, "selected")
    for ax, ylabel in zip(dax.flat, ["FWHM (kHz)", "Ideal contrast SNR",
                                    "Resonance center (kHz)", "Maximum final Pf (%)"],
                         strict=True):
        ax.set(xlabel="Nominal amplitude (MHz)", ylabel=ylabel)
        ax.axvspan(50, 100, alpha=.04, color="black")
        ax.grid(alpha=.2)
    dax[0, 0].axhline(REFERENCE, color="black", ls="--")
    dax[0, 1].axhline(10, color="black", ls="--")
    dax[0, 1].legend(fontsize=8)
    details.suptitle("Full amplitude range • final Pf does not measure peak transient leakage")
    save(details, "details")
    for ax, ylabel in zip(sax, ["Pg", "Pe", "Pf (%)"], strict=True):
        ax.set(xlabel="Detuning (kHz)", ylabel=ylabel)
        ax.grid(alpha=.2)
    sax[1].legend(fontsize=8)
    slices.suptitle("All three final populations at nominal amplitude 100 MHz")
    save(slices, "spectra_100mhz")
    for index, (maps, _) in enumerate(map_figures, start=1):
        maps.suptitle("Three-level populations • 100 amplitudes × 201 detunings per pulse")
        save(maps, f"populations_{index}")


def report(screen, full):
    rows = "".join(
        f"<tr><td>{html.escape(label(s))}</td><td>{s['worst_fwhm_khz']:.2f}</td>"
        f"<td>{s['worst_fwhm_khz'] / REFERENCE:.2f}×</td>"
        f"<td>{s['minimum_snr']:.2f}</td><td>{s['minimum_contrast']:.3f}</td>"
        f"<td>{s['max_abs_center_khz']:.2f}</td></tr>" for s in full)
    valid = [s for s in screen if s["all_accepted"]]
    feasible = [s for s in valid if s["minimum_snr"] >= 10]
    best = min(feasible, key=lambda s: s["worst_fwhm_khz"])
    controls = [s for s in screen if s["scaling"] == "core"]
    local = [s for s in screen if s["duration_us"] not in scan.DURATIONS
             or s["power"] not in scan.POWERS]
    local_rows = "".join(
        f"<tr><td>{html.escape(label(s))}</td>"
        f"<td>{s['worst_fwhm_khz']:.2f}</td><td>{s['minimum_snr']:.2f}</td></tr>"
        for s in sorted(local, key=lambda s: (s["duration_us"], s["power"]))
        if s["all_accepted"])
    control_rows = "".join(
        f"<tr><td>{s['duration_us']:g}</td><td>{s['power']:g}</td>"
        f"<td>{s['worst_fwhm_khz']:.2f}</td><td>{s['minimum_snr']:.2f}</td></tr>"
        for s in sorted(controls, key=lambda s: (s["duration_us"], s["power"]))
        if s["all_accepted"])
    convergence = json.loads((OUT / "checks_refine.json").read_text())
    largest_pop = max(c["max_population_dt_change"] for c in convergence)
    largest_dt = max(c["max_width_dt_change_hz"] for c in convergence)
    largest_grid = max(c["max_width_grid_change_hz"] for c in convergence)
    goals = sum(s["near_t2_and_snr10_points"] for s in screen)
    long_example = next(s for s in full if s["case"] == "L120_p32_raw_cutoff")
    long_at_100 = long_example["at_100_mhz"]
    population_images = "\n".join(
        f'<img src="populations_{i}.png" alt="Three populations versus amplitude '
        f'and detuning, group {i}">' for i in range(1, (len(full) + 2) // 3 + 1))
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Longer pulses and higher polynomial powers</title>
<style>body{{font:17px/1.6 system-ui;max-width:1400px;margin:auto;padding:32px;
color:#172636;background:#fafbfd}}img{{width:100%;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:15px}}
td,th{{padding:9px;border-bottom:1px solid #d5dce2;text-align:left}}
.lead{{background:#eaf2f8;padding:18px;border-left:4px solid #176da1}}
code{{background:#eef1f5;padding:3px}}a{{color:#176da1}}</style>
<h1>Can longer pulses and stronger tapering reach the T₂ linewidth?</h1>
<p class="lead"><b>The tested family still misses the joint target.</b>
At fixed T₂ = 7.31 µs the coherence reference is {REFERENCE:.2f} kHz.
Across {len(screen)} duration/power/shape settings screened at three high amplitudes,
{goals} sampled points meet both FWHM ≤ 50 kHz and ideal SNR ≥ 10.
Longer duration can narrow a visible dip, but the signal falls; increasing the
polynomial exponent restores visibility while broadening the dip again.</p>
<p>The lowest worst-case width among the screened cases with minimum SNR ≥ 10 is
<b>{best['worst_fwhm_khz']:.2f} kHz</b>, for {html.escape(label(best))}.
This is a finite search, not a proof that every possible pulse fails.
The full grids below check {len(full)} selected cases with <b>100 amplitudes from 0 to
100 MHz</b>. High-amplitude statistics use all 50 points from 50.505 to 100 MHz.</p>
<p>At 100 MHz specifically, the 120 µs, power-32 original pulse reaches
<b>{long_at_100['fwhm_hz'] / 1000:.2f} kHz</b>, but its ideal SNR is only
<b>{long_at_100['contrast_snr']:.2f}</b> and its population contrast is
{100 * long_at_100['shoulder_contrast']:.2f} percentage points.
It is much closer to the linewidth reference, with too little signal at this shot budget.</p>
<p>This study varies polynomial windows, with and without subtracting the endpoint
value. The previous C2 endpoint-subtraction construction is a different family and
was not varied here; this is not a ranking of every earlier pulse design.</p>
<h2>Full-grid comparison</h2>
<p>Each column reports a conservative worst case across the high-amplitude range;
the extrema need not occur at the same amplitude. Contrast is a population difference;
SNR is that difference divided by its ideal shot noise.</p>
<table><tr><th>Pulse</th><th>Worst FWHM (kHz)</th><th>Width / T₂ reference</th>
<th>Minimum SNR</th><th>Minimum Pe contrast</th><th>Max |center| (kHz)</th></tr>
{rows}</table>
<img src="selected.png" alt="High-amplitude linewidth and SNR for selected pulses">
<h2>Why higher power does not solve the duration tradeoff</h2>
<p>The pulse is I(t) = A s(t) g(t) [1 − (2t/L)²]<sup>p</sup> on |t| &lt; L/2.
For the original family, g = [1 + (t/σ)²]<sup>−1/2</sup>.
For cutoff subtraction, g = (f − c)/(1 − c). Both have g(0) = 1 before the
16 ns sine echo inversion. The full product derivative supplies DRAG Q.</p>
<p>The window alone has temporal half-height width
<code>W = L √(1 − 2<sup>−1/p</sup>) ≈ L √(ln 2 / p)</code>.
This is the width of the polynomial window, not the Lorentzian core or an exact
spectral-width formula. For large p, doubling L and quadrupling p nearly preserves
the taper around the center. It adds little useful tail duration while moving
readout farther from the center. At fixed L, raising p suppresses more of the
slow Lorentzian wings. The simulation includes relaxation and dephasing throughout;
the loss of visibility cannot be described by a single exp(−L/T₂) factor.</p>
<p>Thus endpoint smoothness and amplitude robustness do not by themselves produce
the narrowest line. The additional width is protocol- and drive-dependent;
1/(πT₂) is the weak-drive Lorentzian reference, not a universal lower bound on a
finite pulse's interference dip.</p>
<p>Literature context: truncating Lorentzian tails creates amplitude-dependent
broadening in coherent excitation spectroscopy
(<a href="https://arxiv.org/html/2308.14187v2">Mihov and Vitanov</a>).
That result motivates smooth tails, but it does not establish the linewidth or
visibility of this three-level dissipative echo sequence.</p>
<h2>Screen across duration and power</h2>
<p>Durations: {', '.join(f'{x:g}' for x in scan.DURATIONS)} µs.
Powers: {', '.join(f'{x:g}' for x in scan.POWERS)}.
These heatmaps use only 50.505, 74.747 and 100 MHz.
Gray means at least one trace failed the central-dip quality criteria.
Values above the color scale are printed explicitly.</p>
<img src="screen.png" alt="Screen of duration versus polynomial exponent">
<h2>Resolve the SNR = 10 boundary</h2>
<p>After the broad screen, {len(local)} extra settings resolve intermediate powers
and 35–40 µs durations near the most promising short-pulse region.
These rows also use three screening amplitudes.</p>
<table><tr><th>Pulse</th><th>Worst FWHM (kHz)</th><th>Minimum SNR</th></tr>
{local_rows}</table>
<h2>Separate control: keep the Lorentzian core fixed</h2>
<p>At fixed c = 0.001, σ scales with duration: 15 ns at 30 µs, 30 ns at 60 µs,
and 60 ns at 120 µs. We also keep σ = {scan.SIGMA_REFERENCE * 1000:.6f} ns
while increasing L, using c = [1 + (L/(2σ))²]<sup>−1/2</sup>.
This extends the tails without widening the core. These controls use the original
envelope and the same three screening amplitudes; selected controls also have a full grid.</p>
<table><tr><th>L (µs)</th><th>p</th><th>Worst FWHM (kHz)</th><th>Minimum SNR</th></tr>
{control_rows}</table>
<h2>Center, leakage, and population spectra</h2>
<img src="details.png" alt="Full-range linewidth, SNR, center and final leakage">
<img src="spectra_100mhz.png" alt="Three populations versus detuning at 100 MHz">
{population_images}
<h2>Parameters and measurement definition</h2>
<p>β = −0.5; 16 ns sine echo transition for every pulse; α = −216 MHz;
T₁ = 51.24 µs; T₂ = 7.31 µs; Tφ = 7.871480508563624 µs.
Three-level RWA Lindblad simulation with adjacent-level relaxation and number
dephasing; no extra phenomenological Stark correction. Q = −β I′/(2π|α|),
with time in µs and α in MHz. No area or signed-envelope peak renormalization.</p>
<p>Detuning spans ±0.5 MHz with 201 points. FWHM uses nearest half-depth crossings
above the central Pe minimum, relative to a linear background fitted at
|detuning| ≥ 0.4 MHz. This is a fixed finite-window definition, not a Lorentzian fit.
The SNR budget is 500 shots at the fitted center and 250 at each ±0.45 MHz shoulder.
It assumes perfect binary Pe readout and omits technical noise and center-search
cost. A longer pulse also costs more time per shot, which this equal-shot comparison
does not reward or penalize. Final Pf is not peak transient leakage.</p>
<h2>Validation and reproducibility</h2>
<p>Original 30 µs powers 3, 8 and 16 reproduce the prior populations within 10⁻¹⁰.
Envelope positivity, symmetry and analytic derivatives were checked across the screen.
All output populations are finite, within [0,1], and sum to one within 10⁻¹⁰.
For {len(convergence)} selected cases at the three screen amplitudes, halving the
time step from 0.5 to 0.25 ns changes populations by at most {largest_pop:.3g}
and widths by {largest_dt:.2f} Hz. Refining the detuning grid from 5 to 2.5 kHz
changes widths by at most {largest_grid:.2f} Hz.</p>
<p>43 focused model/window/metric tests passed. Full-suite collection is blocked
by the environment's missing Torch package. Repository-wide Ruff reports 692
existing findings; the two new scripts pass scoped Ruff. All results here are simulations.</p>
<p><a href="provenance.json">Parameters and numerical provenance</a> ·
<a href="source_hashes.json">Source hashes</a> ·
<a href="summary_screen.json">Screen summaries</a> ·
<a href="summary_full.json">Full-grid summaries</a> ·
<a href="metrics_full.csv">Full metrics CSV</a> ·
<a href="checks_refine.json">Numerical convergence</a> ·
<a href="selected.pdf">Comparison PDF</a> · <a href="screen.pdf">Screen PDF</a></p>
<p>Reproduce: <code>.venv/bin/python scripts/scan_duration_polynomial.py --stage screen</code>,
then <code>--stage full --cases {' '.join(s['case'] for s in full)}</code>,
then <code>--stage refine --cases {' '.join(c['case'] for c in convergence)}</code>,
and <code>.venv/bin/python scripts/plot_duration_polynomial.py</code>.</p>
</html>"""
    (OUT / "report.html").write_text(body)


def main():
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    screen = json.loads((OUT / "summary_screen.json").read_text())
    full = json.loads((OUT / "summary_full.json").read_text())
    full.sort(key=lambda s: (s["duration_us"], s["power"], s["scaling"]))
    screen_figure(screen)
    selected_figures(full)
    report(screen, full)
    sources = list(scan.previous.MODEL_FILES) + [
        "scripts/scan_duration_polynomial.py", "scripts/plot_duration_polynomial.py",
        "echospec/analysis/dip_metrics.py", "echospec/analysis/spectroscopy_quality.py",
        "scripts/make_polynomial_power_scan.py",
        "scripts/make_echo_edge_ramp_comparison.py"]
    (OUT / "source_hashes.json").write_text(json.dumps({
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources
    }, indent=2))
    print(OUT / "report.html")


if __name__ == "__main__":
    main()
