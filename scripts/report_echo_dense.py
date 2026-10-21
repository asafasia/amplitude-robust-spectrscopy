"""Refine, validate and report the 100-amplitude spectrum optimization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts import optimize_echo_dense as opt  # noqa: E402
from scripts.report_optimized_echo_tail import alternate_width  # noqa: E402

OUT = opt.OUT
LABELS = {"p4": "Polynomial p=4", "p5": "Polynomial p=5",
          "previous": "Previous three-amplitude optimization",
          "spectrum": "Spectrum-optimized tail", "narrow": "Narrowest feasible trial"}
COLORS = {"p4": "#65748b", "p5": "#aaaeb8", "previous": "#d89034",
          "spectrum": "#007956", "narrow": "#5f51b7"}


def load(path):
    with np.load(path) as d:
        data = {k: d[k].copy() for k in d.files}
    if "populations" in data:
        p = data["populations"]
        assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
        np.testing.assert_allclose(p.sum(axis=0), 1., atol=1e-10, rtol=0)
    return data


def simulate_chunk(args):
    parameters, amplitudes, detuning, dt = args
    return opt.model.simulate(parameters, amplitudes=amplitudes,
                              detuning=detuning, dt_ns=dt)


def validation_map(tag, stage, parameters, amplitudes, detuning, dt):
    path = OUT / f"{tag}_{stage}.npz"
    config = json.dumps(dict(parameters=parameters, dt_ns=dt,
                             computation_sha256=opt.computation_hashes()), sort_keys=True)
    if path.exists():
        d = load(path)
        assert str(d["config"]) == config
        np.testing.assert_array_equal(d["rabi_mhz"], amplitudes)
        np.testing.assert_array_equal(d["detuning_mhz"], detuning)
        return d
    args = [(parameters, chunk, detuning, dt) for chunk in np.array_split(amplitudes, 4)]
    with Pool(4) as pool:
        p = np.concatenate(pool.map(simulate_chunk, args), axis=1)
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    np.testing.assert_allclose(p.sum(axis=0), 1., atol=1e-10, rtol=0)
    d = dict(populations=p, rabi_mhz=amplitudes, detuning_mhz=detuning, config=config)
    np.savez_compressed(path, **d)
    return d


def prepare():
    data = {"p4": load(opt.model.prior.OUT / "L30_p4_subtracted_cutoff_full.npz"),
            "p5": load(opt.model.OUT / "polynomial5_full.npz"),
            "previous": load(opt.model.OUT / "clean_full.npz")}
    assert json.loads(str(data["p4"]["config"]))["computation_sha256"] == (
        opt.model.prior.computation_hashes())
    for tag in ["p5", "previous"]:
        assert json.loads(str(data[tag]["config"]))["computation_sha256"] == opt.model.hashes()
    selections = {tag: json.loads((OUT / name).read_text()) for tag, name in
                  [("spectrum", "selected.json"), ("narrow", "selected_narrow.json")]}
    checks = []
    wide_data, refined_data = {}, {}
    for tag, selected in selections.items():
        filename = (selected["full_map_filename"] if "full_map_filename" in selected else
                    f"candidate_{selected['candidate_index']}_full.npz")
        data[tag] = load(OUT / filename)
        d = data[tag]
        fine = validation_map(tag, "refine", selected["parameters"], opt.model.AMPS,
                              np.linspace(-.5, .5, 401), .25)
        refined_data[tag] = fine
        coarse_metrics = opt.spectrum_metrics(d["populations"], d["rabi_mhz"], d["detuning_mhz"])
        refined_metrics = opt.spectrum_metrics(fine["populations"], fine["rabi_mhz"],
                                               fine["detuning_mhz"])
        active = range(10, 100)
        error = float(abs(fine["populations"][:, :, ::2] - d["populations"]).max())
        width_error = max(abs(refined_metrics["rows"][i]["fwhm_hz"] -
                              coarse_metrics["rows"][i]["fwhm_hz"]) for i in active)
        snr_error = max(abs(refined_metrics["rows"][i]["contrast_snr"] -
                            coarse_metrics["rows"][i]["contrast_snr"]) for i in active)
        assert error < 1e-4 and width_error < 1000
        checks.append(dict(tag=tag, amplitude_points=100, max_population_change=error,
                           max_fwhm_change_hz=width_error, max_snr_change=snr_error,
                           refined_minimum_snr=refined_metrics["minimum_snr"],
                           refined_worst_fwhm_khz=refined_metrics["worst_fwhm_khz"],
                           refined_max_satellite_prominence=refined_metrics[
                               "max_satellite_prominence"]))
        wide = validation_map(tag, "wide", selected["parameters"], opt.model.AMPS,
                              np.linspace(-1., 1., 401), .5)
        wide_data[tag] = wide
        # New amplitudes between every training pair; independent of search.
        shifted = (opt.model.AMPS[:-1] + opt.model.AMPS[1:]) / 2
        hold = validation_map(tag, "midpoints", selected["parameters"], shifted,
                              opt.model.DETUNING, .5)
        hold_summary = opt.spectrum_metrics(hold["populations"], shifted,
                                            opt.model.DETUNING)
        (OUT / f"{tag}_midpoints.json").write_text(json.dumps(
            opt.model.prior.clean(hold_summary), indent=2, allow_nan=False))
        endpoint_path = OUT / f"{tag}_10mhz.json"
        if not endpoint_path.exists():
            exact = opt.model.simulate(selected["parameters"], amplitudes=np.array([10.]),
                                       detuning=np.linspace(-.5, .5, 401), dt_ns=.25)
            endpoint = opt.spectrum_metrics(exact, np.array([10.]), np.linspace(-.5, .5, 401))
            endpoint_path.write_text(json.dumps(opt.model.prior.clean(endpoint), indent=2))
        t = np.linspace(-15, 15, 60001)
        g, dg = opt.model.envelope(t, selected["parameters"])
        z = opt.model.drive(t, selected["parameters"])
        assert g.min() >= 0 and g.max() <= 1.000001
        np.testing.assert_allclose([g[0], g[-1], dg[0], dg[-1]], 0., atol=1e-14)
        np.testing.assert_allclose(g, g[::-1], atol=1e-12)
        np.savez_compressed(OUT / f"{tag}_waveform.npz", time_us=t,
                            i_per_nominal_amplitude=z.real, q_per_nominal_amplitude=-z.imag)
        print(tag, "refinement and midpoint checks complete", flush=True)
    summaries = {}
    for tag, d in data.items():
        p = d["populations"]
        assert p.shape == (3, 100, 201) and np.isfinite(p).all()
        assert p.min() >= 0 and p.max() <= 1
        np.testing.assert_allclose(p.sum(axis=0), 1., atol=1e-10, rtol=0)
        summaries[tag] = opt.spectrum_metrics(p, d["rabi_mhz"], d["detuning_mhz"])
    (OUT / "summary.json").write_text(json.dumps(opt.model.prior.clean(summaries), indent=2))
    (OUT / "validation.json").write_text(json.dumps(checks, indent=2))
    rows = [dict(pulse=tag, **r) for tag, summary in summaries.items() for r in summary["rows"]]
    with (OUT / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return data, summaries, selections, checks, wide_data, refined_data


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=165, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_comparison(summaries):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    panels = [("fwhm_hz", .001, "FWHM (kHz)"),
              ("contrast_snr", 1., "Ideal contrast SNR"),
              ("satellite_prominence", 100., "Competing-dip prominence (pp)"),
              ("center_hz", .001, "Resonance center (kHz)"),
              ("shoulder_contrast", 100., "Pe contrast (percentage points)"),
              ("max_final_pf", 100., "Maximum final Pf (%)")]
    for ax, (key, scale, label) in zip(axes.flat, panels, strict=True):
        for tag, summary in summaries.items():
            ax.plot(opt.model.AMPS, [r[key] * scale for r in summary["rows"]],
                    color=COLORS[tag], label=LABELS[tag], lw=1.8)
        ax.set(xlabel="Nominal amplitude (MHz)", ylabel=label, xlim=(0, 100))
        ax.grid(alpha=.2)
    axes[0, 0].axhline(opt.T2_WIDTH_MHZ * 1000, color="black", ls="--")
    axes[0, 0].set_ylim(0, 240)
    axes[0, 1].axhline(10, color="black", ls="--")
    axes[0, 2].axhline(1.5, color="black", ls="--")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.055),
               ncols=3, fontsize=9)
    fig.suptitle("100 amplitude points • same 30 µs pulse and physical parameters")
    save_path = OUT / "full_range.png"
    fig.savefig(save_path, dpi=165, bbox_inches="tight")
    fig.savefig(OUT / "full_range.pdf", bbox_inches="tight")
    for ax in axes.flat:
        ax.set_xlim(10, 100)
    fig.suptitle("Required operating range: 10–100 MHz • T₂ reference = 43.54 kHz")
    save(fig, "comparison")


def plot_populations(data):
    tags = ["p4", "previous", "spectrum", "narrow"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), layout="constrained")
    max_pf = max(data[tag]["populations"][2].max() for tag in tags) * 100
    for col, tag in enumerate(tags):
        d = data[tag]
        for level in range(3):
            im = axes[level, col].pcolormesh(d["detuning_mhz"] * 1000, d["rabi_mhz"],
                d["populations"][level] * (100 if level == 2 else 1), shading="auto",
                vmin=0, vmax=max_pf if level == 2 else 1, rasterized=True)
            if col == 0:
                axes[level, col].set_ylabel(f"P{'gef'[level]}{' (%)' if level == 2 else ''}\nAmplitude (MHz)")
            if level == 0:
                axes[level, col].set_title(LABELS[tag], fontsize=10)
            if level == 2:
                axes[level, col].set_xlabel("Detuning (kHz)")
            if col == 3:
                fig.colorbar(im, ax=list(axes[level]), shrink=.75)
    fig.suptitle("All three final populations; common color scale within each row")
    save(fig, "populations")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for ax, i in zip(axes.flat, [10, 25, 40, 60, 80, 99], strict=True):
        for tag in tags:
            d = data[tag]
            ax.plot(d["detuning_mhz"] * 1000, d["populations"][1, i],
                    color=COLORS[tag], label=LABELS[tag])
        ax.set(title=f"A = {opt.model.AMPS[i]:.2f} MHz", xlabel="Detuning (kHz)", ylabel="Pe")
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.045), ncols=4,
               fontsize=9)
    fig.suptitle("The spectrum throughout the operating range")
    save(fig, "spectra")


def plot_waveforms(selections):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    times = np.linspace(-.07, .07, 3001)
    tail = np.geomspace(.02, 15, 3001)
    for tag in ["previous", "spectrum", "narrow"]:
        params = (json.loads((opt.model.OUT / "selected_clean.json").read_text())["parameters"]
                  if tag == "previous" else selections[tag]["parameters"])
        z = opt.model.drive(times, params)
        axes[0].plot(times * 1000, 100 * z.real, color=COLORS[tag], label=LABELS[tag])
        axes[1].plot(times * 1000, -100 * z.imag, color=COLORS[tag])
        axes[2].loglog(tail, 100 * opt.model.envelope(tail, params)[0], color=COLORS[tag])
    axes[0].set(xlabel="Time from midpoint (ns)", ylabel="I at A=100 MHz (MHz)")
    axes[1].set(xlabel="Time from midpoint (ns)", ylabel="Q at A=100 MHz (MHz)")
    axes[2].set(xlabel="Time from midpoint (µs)", ylabel="Unsigned envelope × 100 (MHz)",
                ylim=(1e-5, 100))
    for ax in axes:
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.065), ncols=3,
               fontsize=9)
    fig.suptitle("Fixed Lorentzian core; one optimized tail shared by every amplitude")
    save(fig, "waveforms")


def plot_wide(wide):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    indices = [10]
    for d in wide.values():
        metrics = opt.spectrum_metrics(d["populations"], d["rabi_mhz"], d["detuning_mhz"])
        indices.append(10 + int(np.argmax([r["satellite_prominence"]
                                           for r in metrics["rows"][10:]])))
    indices.append(99)
    indices = list(dict.fromkeys(indices))
    for i in [40, 70]:
        if len(indices) < 4 and i not in indices:
            indices.append(i)
    indices.sort()
    for ax, i in zip(axes.flat, indices, strict=True):
        for tag, d in wide.items():
            ax.plot(d["detuning_mhz"] * 1000, d["populations"][1, i],
                    color=COLORS[tag], label=LABELS[tag], lw=2)
        for lo, hi in [(-500, -400), (400, 500)]:
            ax.axvspan(lo, hi, color="gray", alpha=.12)
        for lo, hi in [(-1000, -800), (800, 1000)]:
            ax.axvspan(lo, hi, color="#007956", alpha=.08)
        ax.set(title=f"A = {opt.model.AMPS[i]:.2f} MHz", xlabel="Detuning (kHz)", ylabel="Pe")
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.045), ncols=2)
    fig.suptitle("Wider spectra; shaded bands show the two background definitions")
    save(fig, "wide_spectra")


def plot_search(summaries):
    valid = [r for r in opt.records() if r["all_accepted"]]
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for accepted, color, label in [(False, "#c5c8ce", "Extra-dip/center criteria fail"),
                                    (True, "#8eafad", "Extra-dip/center criteria pass")]:
        rows = [r for r in valid if (r["max_satellite_prominence"] <= .015 and
                                    r["max_center_khz"] <= 2.) == accepted]
        ax.scatter([r["worst_fwhm_khz"] for r in rows],
                   [r["minimum_snr"] for r in rows], s=18, alpha=.65,
                   color=color, label=label)
    for tag in ["p4", "p5", "spectrum", "narrow"]:
        s = summaries[tag]
        ax.scatter(s["worst_fwhm_khz"], s["minimum_snr"], s=100,
                   color=COLORS[tag], edgecolors="black", label=LABELS[tag], zorder=5)
    ax.axvline(opt.T2_WIDTH_MHZ * 1000, color="black", ls="--", label="T₂ reference")
    ax.axhline(10., color="black", ls=":", label="SNR = 10")
    ax.set(xlabel="Worst FWHM across 10–100 MHz (kHz)",
           ylabel="Minimum SNR across 10–100 MHz", xscale="log",
           title="All trials with valid dip detection throughout the operating range")
    ax.set_xticks([43.54, 100, 150, 200, 300, 500, 800],
                  labels=["43.5", "100", "150", "200", "300", "500", "800"])
    ax.grid(alpha=.2)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.15), ncols=2)
    save(fig, "search_tradeoff")


def write_report(data, summaries, selections, checks, wide, refined):
    selected = summaries["spectrum"]
    narrow = summaries["narrow"]
    table = ""
    for tag, s in summaries.items():
        table += (f"<tr><td>{LABELS[tag]}</td><td>{s['worst_fwhm_khz']:.2f}</td>"
                  f"<td>{s['minimum_snr']:.3f}</td><td>{100 * s['max_satellite_prominence']:.3f}</td>"
                  f"<td>{s['max_center_khz']:.3f}</td></tr>")
    midpoint_rows, background_rows = "", ""
    background_records = []
    for tag in selections:
        hold = json.loads((OUT / f"{tag}_midpoints.json").read_text())
        midpoint_rows += (f"<tr><td>{LABELS[tag]}</td><td>{hold['worst_fwhm_khz']:.2f}</td>"
                         f"<td>{hold['minimum_snr']:.3f}</td>"
                         f"<td>{100 * hold['max_satellite_prominence']:.3f}</td></tr>")
        d = wide[tag]
        widths = [alternate_width(d["detuning_mhz"], p) for p in d["populations"][1, 10:]]
        wide_metrics = opt.spectrum_metrics(d["populations"], d["rabi_mhz"], d["detuning_mhz"])
        worst_wide_row = max(wide_metrics["rows"][10:], key=lambda r: r["satellite_prominence"])
        background_rows += (f"<tr><td>{LABELS[tag]}</td><td>{min(widths):.2f}–{max(widths):.2f}</td>"
                            f"<td>{100 * wide_metrics['max_satellite_prominence']:.3f}</td></tr>")
        background_records.append(dict(tag=tag, outer_background_widths_khz=widths,
                                       wide_satellite_prominence=wide_metrics[
                                           "max_satellite_prominence"], worst_row=worst_wide_row))
    (OUT / "background_checks.json").write_text(json.dumps(background_records, indent=2))
    n = len(opt.records())
    branch = json.loads((OUT / "narrow_refinement.json").read_text())
    simulated_count = len(list((OUT / "evaluations").glob("*.npz")))
    knot_rows = "".join(
        f"<tr><td>{t:.3g}</td><td>{a:.10g}</td><td>{b:.10g}</td></tr>"
        for t, a, b in zip(opt.model.KNOTS, np.r_[0., selections["spectrum"]["parameters"]],
                          np.r_[0., selections["narrow"]["parameters"]], strict=True))
    wide_caveat = "; ".join(f"{LABELS[r['tag']]}: {100 * r['wide_satellite_prominence']:.2f} pp"
                             for r in background_records)
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Whole-spectrum echo optimization at 100 amplitudes</title>
<style>body{{font:17px/1.6 system-ui;max-width:1350px;margin:auto;padding:32px;color:#172636;
background:#fafbfd}}img{{width:100%;margin:16px 0}}table{{border-collapse:collapse;width:100%;
font-size:15px}}th,td{{padding:9px;border-bottom:1px solid #d5dce2;text-align:left}}
.lead{{background:#e8f2ed;border-left:4px solid #007956;padding:18px}}a{{color:#007956}}</style>
<h1>One pulse optimized against all 100 amplitude points</h1>
<p class="lead">The narrowest candidate satisfying the original frequency-window
constraints gives worst FWHM <b>{narrow['worst_fwhm_khz']:.2f} kHz</b>, minimum ideal SNR
<b>{narrow['minimum_snr']:.3f}</b>, and maximum competing-dip prominence
<b>{100 * narrow['max_satellite_prominence']:.3f} percentage points</b> across
10.101–100 MHz. Its worst width is {narrow['worst_fwhm_khz'] / (opt.T2_WIDTH_MHZ * 1000):.2f}×
the 43.54 kHz T₂ reference. <b>The joint near-T₂ target was not achieved.</b></p>
<p>The more conservative whole-spectrum objective, including a penalty on broad
shoulder curvature, gives {selected['worst_fwhm_khz']:.2f} kHz and SNR
{selected['minimum_snr']:.3f}. Both fixed shapes are shown. These are finite numerical searches within
one six-parameter family, not evidence of a fundamental physical lower bound.</p>
<p><b>Frequency-window limitation:</b> optimization and primary constraints use the
complete original ±500 kHz spectrum. The independent ±1 MHz scan finds larger
extra-dip prominences: {wide_caveat}. Suppression within the optimization window
does not establish suppression at arbitrary detuning. In particular, the narrower
shape does not meet the 1.5-point selectivity cap over ±1 MHz; its smaller central
linewidth should not be interpreted as a clean isolated resonance over that window.</p>
<h2>Parameters and operating range</h2>
<p>Every simulated candidate uses <b>all 100 amplitudes from 0 to 100 MHz</b>.
The requested visibility/selectivity requirement is enforced at the 90 grid points
from 10.101 to 100 MHz. All 100 spectra contribute to the line-shape objective;
SNR and linewidth are undefined at zero drive. The unchanged parameters are
30 µs duration, cutoff 0.001, σ = 15.0000075 ns, 16 ns sine echo inversion,
DRAG β = −0.5, α = −216 MHz, T₁ = 51.24 µs, T₂ = 7.31 µs,
Tφ = 7.871480508563624 µs. Three-level RWA Lindblad dynamics, the same relaxation
and dephasing operators, no extra Stark term, and no per-amplitude waveform rescaling.</p>
<h2>Comparison over 10–100 MHz</h2>
<p>Extrema can occur at different amplitudes. FWHM retains the previous central-dip,
nearest-half-depth-crossing definition using the linear background at
|detuning| = 0.4–0.5 MHz. These values are not Lorentzian-fit widths.</p>
<table><tr><th>Pulse</th><th>Worst FWHM (kHz)</th><th>Minimum SNR</th>
<th>Largest extra dip (pp)</th><th>Max |center| (kHz)</th></tr>{table}</table>
<img src="comparison.png" alt="Six metrics across the required amplitude range">
<h2>Populations and line shapes</h2>
<img src="populations.png" alt="Pg, Pe and Pf at all 100 amplitudes">
<img src="spectra.png" alt="Population spectra at six representative amplitudes">
<h2>What was optimized</h2>
<p>The same six-parameter tail family is retained: g(t) = f(t) exp[h(ln|t|)] w(t),
where f is the original root-Lorentzian, h = 0 for |t| ≤ 50 ns, and h is a quintic
spline through |t| = 0.05, 0.15, 0.5, 1.5, 4, 9, 15 µs. Its first two derivatives
vanish at both spline endpoints. A fixed quintic window w closes the last 2 µs of
each edge. The pulse and its first two derivatives vanish at ±15 µs.
DRAG uses the complete analytic derivative of the signed envelope,
Q = −β I′/(2π|α|), and the complex drive is I − iQ.</p>
<table><tr><th>|t| (µs)</th><th>h: spectrum objective</th><th>h: narrower candidate</th></tr>
{knot_rows}</table>
<img src="waveforms.png" alt="I Q and optimized tails">
<p>The objective combines worst FWHM; mean and worst residuals relative to a
43.54 kHz Lorentzian template, fitting only its depth; an SNR penalty below 10.15;
extra-dip prominence above 1 percentage point; outward spectral reversals above
2.5 percentage points total; and center displacement above 2 kHz. The template
residual uses every frequency sample and all 100 amplitudes. Additional minima are
identified after excluding the central basin within ±20 kHz, including dips closer
than the previous 50 kHz exclusion. Feasible candidates require minimum SNR ≥ 10,
extra-dip prominence ≤ 1.5 percentage points, and |center| ≤ 2 kHz.</p>
<p>Initial search: {n - branch['additional_trials']} distinct trials, 24 initial shapes,
12 differential-evolution generations, followed by six parallel coordinate sweeps,
seed 20261013.
Each trial used 100 × 101 frequency/amplitude points and a 1 ns RK4 step.
Four distinct finalists spanning lowest objective and narrowest feasible width
were evaluated at 100 × 201 points and 0.5 ns. Primary selection uses the lowest
whole-spectrum objective among feasible finalists; the second uses lowest worst
FWHM. The same 100 amplitudes were used for optimization and final resolution
checks, so the latter are numerical validation, not unseen-amplitude validation.</p>
<p>A final width-oriented branch added {branch['additional_trials']} trials, still using
all 100 amplitudes and the complete frequency spectrum. It retained the template,
SNR, extra-dip and center penalties while removing only the separate penalty on
broad outward spectral curvature. Eight coordinate sweeps tested steps of
0.15, 0.075, 0.035 and 0.015 in the six spline ordinates. The new narrow candidate
was accepted only after satisfying the original SNR/selectivity constraints at
0.5 ns / 5 kHz resolution. Total distinct trials: {n}.</p>
<p>The narrowest screening result slightly exceeded the 1.5-point extra-dip cap
when recomputed on the finer grid. It was rejected, and the next narrow candidate
was checked before selection. This boundary check is retained in
<a href="narrow_finalist_checks.json">the finalist records</a>.</p>
<p>{simulated_count} trials passed the waveform peak check and were propagated at
all 100 amplitudes; the remaining trials were rejected before propagation because
the unsigned envelope exceeded the original normalized peak of one.</p>
<img src="search_tradeoff.png" alt="Linewidth versus SNR for all valid search trials">
<h2>Numerical resolution and amplitudes between search points</h2>
<p>Both selected waveforms were rerun at all 100 amplitudes using 401 detunings
and a 0.25 ns step. Maximum probability change is
{max(c['max_population_change'] for c in checks):.3g}; maximum linewidth change is
{max(c['max_fwhm_change_hz'] for c in checks):.2f} Hz; maximum SNR change is
{max(c['max_snr_change'] for c in checks):.4f}.</p>
<p>Independent checks used 99 amplitudes halfway between every pair of search
amplitudes, with 201 detunings and a 0.5 ns step. Statistics below use midpoint
amplitudes ≥ 10 MHz; no optimization was performed on them. The exact 10 MHz
endpoint was checked separately with 401 detunings and a 0.25 ns step.</p>
<table><tr><th>Pulse</th><th>Worst FWHM (kHz)</th><th>Minimum SNR</th>
<th>Largest extra dip (pp)</th></tr>{midpoint_rows}</table>
<h2>Wider frequency window</h2>
<p>Both pulses were also simulated at all 100 amplitudes over ±1 MHz. The following
widths use a different background band, 0.8–1 MHz, and should not be mixed with
the primary definition. Extra-dip prominence is checked across that wider spectrum.</p>
<table><tr><th>Pulse</th><th>Outer-background FWHM range (kHz)</th>
<th>Largest extra dip over ±1 MHz (pp)</th></tr>{background_rows}</table>
<img src="wide_spectra.png" alt="Wider spectra and alternative background bands">
<h2>Full amplitude range</h2><img src="full_range.png" alt="Results from zero to 100 MHz">
<h2>Interpretation and reproducibility</h2>
<p>SNR assumes 500 ideal Bernoulli shots at the center and 250 at each ±0.45 MHz
shoulder. It excludes readout error, drift and center-search cost. It is not
percentage contrast. Final Pf does not bound transient leakage. The 1/(πT₂)
value is a weak-drive Lorentzian reference, not a universal bound on finite-pulse
interference features. No hardware measurements were performed.</p>
<p>48 focused objective/model/metric tests passed. The full suite cannot collect
because Torch is missing; repository-wide Ruff has 692 pre-existing findings.
The new scripts and focused test pass scoped Ruff. All maps are normalized and
physical, and the reused controls were checked against their source hashes.</p>
<p><a href="selected.json">Selected shape</a> ·
<a href="selected_narrow.json">Narrowest feasible shape</a> ·
<a href="spectrum_waveform.npz">Selected normalized I/Q</a> ·
<a href="narrow_waveform.npz">Narrow candidate normalized I/Q</a> ·
<a href="summary.json">All metrics</a> · <a href="metrics.csv">Metrics CSV</a> ·
<a href="validation.json">Resolution checks</a> ·
<a href="search_provenance.json">Objective and grid provenance</a> ·
<a href="provenance.json">Sources and reproduction</a> ·
<a href="comparison.pdf">Comparison PDF</a> · <a href="populations.pdf">Populations PDF</a></p>
</html>"""
    (OUT / "report.html").write_text(body)
    source_paths = ["scripts/optimize_echo_dense.py", "scripts/report_echo_dense.py",
                    "scripts/optimize_echo_tail.py", "scripts/report_optimized_echo_tail.py",
                    "tests/analysis/test_echo_dense_objective.py",
                    "echospec/analysis/dip_metrics.py", "echospec/analysis/spectroscopy_quality.py"]
    source_paths.extend(opt.model.prior.previous.MODEL_FILES)
    provenance = dict(source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                    for p in source_paths},
                      computation_sha256=opt.computation_hashes(),
                      reproduction=[
                          ".venv/bin/python scripts/optimize_echo_dense.py --iterations 12 --workers 4",
                          ".venv/bin/python scripts/optimize_echo_dense.py --stage validate",
                          ".venv/bin/python scripts/optimize_echo_dense.py --stage narrow-refine --workers 4",
                          ".venv/bin/python scripts/optimize_echo_dense.py --stage validate-narrow",
                          ".venv/bin/python scripts/report_echo_dense.py"])
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(OUT / "report.html", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-spectrum", action="store_true")
    args = parser.parse_args()
    if args.prepare_spectrum:
        selected = json.loads((OUT / "selected.json").read_text())
        shifted = (opt.model.AMPS[:-1] + opt.model.AMPS[1:]) / 2
        for stage, amplitudes, detuning, dt in [
            ("refine", opt.model.AMPS, np.linspace(-.5, .5, 401), .25),
            ("wide", opt.model.AMPS, np.linspace(-1, 1, 401), .5),
            ("midpoints", shifted, opt.model.DETUNING, .5),
        ]:
            validation_map("spectrum", stage, selected["parameters"], amplitudes, detuning, dt)
            print("Primary", stage, "ready", flush=True)
        return
    result = prepare()
    data, summaries, selections, checks, wide, refined = result
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    plot_comparison(summaries)
    plot_populations(data)
    plot_waveforms(selections)
    plot_wide(wide)
    plot_search(summaries)
    write_report(*result)


if __name__ == "__main__":
    main()
