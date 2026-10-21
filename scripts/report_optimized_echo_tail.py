"""Validate and render the fixed-core, optimized-tail spectroscopy experiment."""

from __future__ import annotations

import csv
import hashlib
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

from scripts import optimize_echo_tail as opt  # noqa: E402
from scripts.refine_echo_tail_selectivity import satellite_metrics  # noqa: E402

OUT = opt.OUT
FAMILIES = ["polynomial5", "c2", "spline", "margin", "clean"]
LABELS = {"polynomial5": "Cutoff-subtracted p=5", "c2": "C2 subtraction",
          "spline": "Fringed tail: narrower", "margin": "Fringed tail: SNR margin",
          "clean": "Tail with suppressed satellites"}
COLORS = {"polynomial5": "#7b8495", "c2": "#d58826", "spline": "#64a7b1",
          "margin": "#a285b7", "clean": "#007c53"}
REFERENCE = 1000 / (np.pi * 7.31)


def load(family, stage):
    with np.load(OUT / f"{family}_{stage}.npz") as data:
        result = {k: data[k].copy() for k in data.files}
    assert json.loads(str(result["config"]))["computation_sha256"] == opt.hashes()
    p = result["populations"]
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    assert abs(p.sum(axis=0) - 1).max() < 1e-10
    return result


def alternate_width(x, pe, band=(.8, 1.)):
    """Half-depth width with a different explicitly specified background band."""
    center = np.flatnonzero(abs(x) <= .25)
    edges = (abs(x) >= band[0]) & (abs(x) <= band[1])
    background = np.polyval(np.polyfit(x[edges], pe[edges], 1), x)
    depth = background - pe
    peak = center[np.argmax(depth[center])]
    half = depth[peak] / 2
    left = np.flatnonzero(depth[:peak] <= half)
    right = np.flatnonzero(depth[peak + 1:] <= half) + peak + 1
    if depth[peak] < .001 or not left.size or not right.size:
        return None
    li, ri = left[-1], right[0]
    xl = np.interp(half, depth[li:li + 2], x[li:li + 2])
    xr = np.interp(half, depth[ri - 1:ri + 1][::-1], x[ri - 1:ri + 1][::-1])
    return float((xr - xl) * 1000)


def validate():
    results, summaries, metrics_rows, backgrounds = [], {}, [], []
    for family in FAMILIES:
        data, fine, wide = [load(family, stage) for stage in ["full", "refine", "wide"]]
        assert data["populations"].shape == (3, 100, 201)
        np.testing.assert_array_equal(data["rabi_mhz"], opt.AMPS)
        summary = opt.quality(data["populations"], data["rabi_mhz"], data["detuning_mhz"])
        summary.update(satellite_metrics(data["populations"][:, 50:], data["detuning_mhz"]))
        summaries[family] = summary
        metrics_rows.extend([dict(family=family, **row) for row in summary["rows"]])
        numerical = float(abs(fine["populations"][:, :, ::2] -
                              data["populations"][:, opt.TRAIN_INDICES]).max())
        coarse_rows = [summary["rows"][i] for i in opt.TRAIN_INDICES]
        refined_rows = opt.quality(fine["populations"], fine["rabi_mhz"],
                                   fine["detuning_mhz"])["rows"]
        subsampled = opt.quality(fine["populations"][:, :, ::2], fine["rabi_mhz"],
                                fine["detuning_mhz"][::2])["rows"]
        dt_width = max(abs(a["fwhm_hz"] - b["fwhm_hz"])
                       for a, b in zip(coarse_rows, subsampled, strict=True))
        freq_width = max(abs(a["fwhm_hz"] - b["fwhm_hz"])
                         for a, b in zip(subsampled, refined_rows, strict=True))
        dt_center = max(abs(a["center_hz"] - b["center_hz"])
                        for a, b in zip(coarse_rows, subsampled, strict=True))
        snr_change = max(abs(a["contrast_snr"] - b["contrast_snr"])
                         for a, b in zip(coarse_rows, refined_rows, strict=True))
        assert numerical < 1e-4 and dt_width < 1000 and freq_width < 1000
        results.append(dict(family=family, max_population_change=numerical,
                            max_time_step_width_change_hz=dt_width,
                            max_frequency_grid_width_change_hz=freq_width,
                            max_time_step_center_change_hz=dt_center,
                            max_snr_change=snr_change))
        if family == "polynomial5":
            with np.load(opt.prior.OUT / "L30_p5_subtracted_cutoff_full.npz") as old:
                err = float(abs(data["populations"] - old["populations"]).max())
                assert err < 1e-10
                results[-1]["previous_full_map_difference"] = err
        elif family == "c2":
            with np.load(ROOT / "outputs/cutoff_subtraction_screen/c0.001_subtract_c2.npz") as old:
                err = float(abs(data["populations"][:, opt.TRAIN_INDICES] -
                                old["populations"]).max())
                assert err < 1e-10
                results[-1]["previous_three_amplitude_difference"] = err
        for i, amp in enumerate(wide["rabi_mhz"]):
            backgrounds.append(dict(
                family=family, amplitude_mhz=float(amp),
                original_background_width_khz=coarse_rows[i]["fwhm_hz"] / 1000,
                outer_background_width_khz=alternate_width(
                    wide["detuning_mhz"], wide["populations"][1, i])))
    parameters = {}
    for tag, filename in [("spline", "selected.json"), ("margin", "selected_margin.json"),
                          ("clean", "selected_clean.json")]:
        parameters[tag] = json.loads((OUT / filename).read_text())["parameters"]
        pars = parameters[tag]
        times = np.linspace(-15, 15, 60001)
        f, df = opt.envelope(times, pars)
        assert np.max(f) <= 1.000001 and np.min(f) >= 0
        np.testing.assert_allclose(f, f[::-1], atol=1e-12)
        np.testing.assert_allclose(df, -df[::-1], atol=1e-10)
        for t in [.003, .049, .05, .051, .15, .5, 1.5, 4., 9., 14.]:
            numeric = (opt.envelope(np.array([t + 1e-6]), pars)[0] -
                       opt.envelope(np.array([t - 1e-6]), pars)[0]) / 2e-6
            np.testing.assert_allclose(opt.envelope(np.array([t]), pars)[1], numeric,
                                       atol=1e-8, rtol=1e-5)
        z = opt.drive(times, pars)
        np.savez_compressed(OUT / f"optimized_{tag}_waveform.npz", time_us=times,
                            i_per_nominal_amplitude=z.real, q_per_nominal_amplitude=-z.imag,
                            envelope=f, derivative_per_us=df)
    (OUT / "validation.json").write_text(json.dumps(results, indent=2))
    (OUT / "background_sensitivity.json").write_text(json.dumps(backgrounds, indent=2))
    (OUT / "summary.json").write_text(json.dumps(opt.prior.clean(summaries), indent=2))
    with (OUT / "metrics.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=metrics_rows[0])
        writer.writeheader()
        writer.writerows(metrics_rows)
    return summaries, results, backgrounds, parameters


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def validate_scan_extrema(summary, parameters):
    """Refine the three different worst cases discovered in the full scan."""
    high = summary["rows"][50:]
    indices = sorted({
        50 + int(np.argmin([r["contrast_snr"] for r in high])),
        50 + int(np.argmax([r["fwhm_hz"] for r in high])),
        50 + int(np.argmax([r["max_prominence"] for r in summary["satellites"]])),
    })
    amplitudes = opt.AMPS[indices]
    detuning = np.linspace(-.5, .5, 401)
    config = json.dumps(dict(parameters=parameters, amplitude_indices=indices,
                             dt_ns=.25, computation_sha256=opt.hashes()), sort_keys=True)
    path = OUT / "clean_scan_extrema_refine.npz"
    if path.exists():
        with np.load(path) as cached:
            assert str(cached["config"]) == config
            populations = cached["populations"]
    else:
        populations = opt.simulate(parameters, amplitudes=amplitudes,
                                   detuning=detuning, dt_ns=.25)
        np.savez_compressed(path, populations=populations, rabi_mhz=amplitudes,
                            detuning_mhz=detuning, config=config)
    baseline = load("clean", "full")["populations"][:, indices]
    error = float(abs(populations[:, :, ::2] - baseline).max())
    refined = opt.quality(populations, amplitudes, detuning)
    width_change = max(abs(refined["rows"][j]["fwhm_hz"] -
                           summary["rows"][i]["fwhm_hz"])
                       for j, i in enumerate(indices))
    assert error < 1e-4 and width_change < 1000
    record = dict(amplitude_indices=indices, amplitudes_mhz=amplitudes.tolist(),
                  max_population_change=error, max_width_change_hz=width_change,
                  refined_metrics=refined,
                  refined_satellites=satellite_metrics(populations, detuning))
    (OUT / "scan_extrema_validation.json").write_text(json.dumps(record, indent=2))
    return record


def comparison(summaries):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for family, summary in summaries.items():
        rows = summary["rows"]
        x = [r["amplitude_mhz"] for r in rows]
        for ax, key, scale in [(axes[0, 0], "fwhm_hz", .001),
                               (axes[0, 1], "contrast_snr", 1.),
                               (axes[1, 0], "center_hz", .001),
                               (axes[1, 1], "shoulder_contrast", 100.)]:
            ax.plot(x, [r[key] * scale for r in rows], color=COLORS[family],
                    label=LABELS[family], lw=2)
    axes[0, 0].axhline(REFERENCE, color="black", ls="--", label="T₂ reference")
    axes[0, 1].axhline(10, color="black", ls="--", label="SNR = 10")
    axes[0, 0].set(ylabel="FWHM (kHz)", ylim=(0, 210))
    axes[0, 1].set(ylabel="Ideal contrast SNR", ylim=(0, 23))
    axes[1, 0].set(ylabel="Resonance center (kHz)")
    axes[1, 1].set(ylabel="Pe contrast (percentage points)")
    for ax in axes.flat:
        ax.axvspan(50, 100, color="black", alpha=.035)
        ax.set(xlabel="Nominal amplitude (MHz)", xlim=(0, 100))
        ax.grid(alpha=.2)
    axes[0, 1].legend(loc="lower center", fontsize=9)
    reference_text = axes[0, 0].text(4, REFERENCE + 4, "43.54 kHz reference", fontsize=10)
    axes[0, 0].set_ylim(0, 610)
    fig.suptitle("Each shape fixed across amplitudes • 30 µs • full 100-point range", fontsize=15)
    fig.savefig(OUT / "full_range.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "full_range.pdf", bbox_inches="tight")
    for ax in axes.flat:
        ax.set_xlim(50, 100)
    axes[0, 0].set_ylim(0, 190)
    axes[0, 1].set_ylim(0, 17)
    reference_text.set_position((51, REFERENCE + 4))
    fig.suptitle("High-amplitude range • 50 of the 100 simulated amplitude points", fontsize=15)
    save(fig, "comparison")


def pulse_shapes(parameters):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    full = np.linspace(-15, 15, 10001)
    core = np.linspace(-.07, .07, 3001)
    tail = np.geomspace(.01, 15., 2001)
    for family in FAMILIES:
        pars = parameters.get(family)
        shape = "spline" if family in ["margin", "clean"] else family
        z = opt.drive(full, pars, shape)
        zc = opt.drive(core, pars, shape)
        f, df = opt.envelope(tail, pars, shape)
        args = dict(color=COLORS[family], label=LABELS[family], lw=2)
        axes[0, 0].plot(core * 1000, zc.real * 100, **args)
        axes[0, 1].plot(core * 1000, -zc.imag * 100, **args)
        axes[1, 0].loglog(tail, f * 100, **args)
        axes[1, 1].plot(full, z.real * 100, **args)
    axes[0, 0].set(xlabel="Time from midpoint (ns)", ylabel="I at A=100 MHz (MHz)",
                    title="Central Lorentzian core and sine inversion")
    axes[0, 1].set(xlabel="Time from midpoint (ns)", ylabel="Q at A=100 MHz (MHz)",
                    title="Same β = −0.5; full envelope derivative")
    axes[1, 0].set(xlabel="Time from midpoint (µs)", ylabel="Unsigned envelope × 100 (MHz)",
                    ylim=(1e-5, 100), title="Independently reshaped weak tails")
    axes[1, 1].set(xlabel="Time from midpoint (µs)", ylabel="I at A=100 MHz (MHz)",
                    ylim=(-.5, .5), title="Weak-drive region; central peak clipped for scale")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    axes[1, 0].legend(fontsize=9)
    fig.suptitle("Fixed core, smooth free tail • no amplitude or area renormalization")
    save(fig, "waveforms")


def population_figures():
    fig, axes = plt.subplots(3, 5, figsize=(18, 10), layout="constrained")
    slices, sax = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
    max_pf = max(float(load(f, "full")["populations"][2].max()) * 100 for f in FAMILIES)
    for column, family in enumerate(FAMILIES):
        d = load(family, "full")
        p, det = d["populations"], d["detuning_mhz"] * 1000
        for level in range(3):
            scale = 100 if level == 2 else 1
            image = axes[level, column].pcolormesh(
                det, d["rabi_mhz"], p[level] * scale, shading="auto", vmin=0,
                vmax=max_pf if level == 2 else 1, cmap="viridis", rasterized=True)
            if column == 0:
                axes[level, column].set_ylabel(
                    f"P{'gef'[level]}{' (%)' if level == 2 else ''}\nAmplitude (MHz)")
            if level == 0:
                axes[level, column].set_title(LABELS[family].replace(": ", ":\n"),
                                              fontsize=10)
            if level == 2:
                axes[level, column].set_xlabel("Detuning (kHz)")
            if column == len(FAMILIES) - 1:
                fig.colorbar(image, ax=list(axes[level]), shrink=.8)
            sax[level].plot(det, p[level, -1] * scale, color=COLORS[family],
                            label=LABELS[family], lw=2)
    for ax, ylabel in zip(sax, ["Pg", "Pe", "Pf (%)"], strict=True):
        ax.set(xlabel="Detuning (kHz)", ylabel=ylabel)
        ax.grid(alpha=.2)
    handles, labels = sax[1].get_legend_handles_labels()
    slices.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.1),
                  ncols=3, fontsize=9)
    fig.suptitle("Three final populations • 100 amplitudes × 201 detunings")
    slices.suptitle("Population spectra at 100 MHz")
    save(fig, "populations")
    save(slices, "spectra_100mhz")


def wide_spectra():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")
    for family in ["polynomial5", "c2", "margin", "clean"]:
        data = load(family, "wide")
        for i, ax in enumerate(axes):
            ax.plot(data["detuning_mhz"] * 1000, data["populations"][1, i],
                    color=COLORS[family], label=LABELS[family], lw=2)
            ax.set(title=f"A = {data['rabi_mhz'][i]:.3f} MHz",
                   xlabel="Detuning (kHz)", ylabel="Pe")
    for ax in axes:
        for sign in [-1, 1]:
            ax.axvspan(sign * 400, sign * 500, color="gray", alpha=.12)
            ax.axvspan(sign * 800, sign * 1000, color="#007c91", alpha=.07)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Wider spectra • shaded bands mark the two background definitions")
    save(fig, "wide_spectra")


def intermediate_spectra(summaries, checks):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for family in ["polynomial5", "c2", "clean"]:
        data = load(family, "full")
        for ax, i in zip(axes, checks["amplitude_indices"], strict=True):
            ax.plot(data["detuning_mhz"] * 1000, data["populations"][1, i],
                    color=COLORS[family], label=LABELS[family], lw=2)
            row = summaries["clean"]["rows"][i]
            ax.set(title=f"A = {data['rabi_mhz'][i]:.2f} MHz\n"
                   f"New tail: {row['fwhm_hz'] / 1000:.1f} kHz, SNR {row['contrast_snr']:.2f}",
                   xlabel="Detuning (kHz)", ylabel="Pe")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.07),
               ncols=3, fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("Cases found between the three optimization amplitudes")
    save(fig, "intermediate_spectra")


def main():
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    summaries, checks, backgrounds, parameters = validate()
    extrema = validate_scan_extrema(summaries["clean"], parameters["clean"])
    comparison(summaries)
    pulse_shapes(parameters)
    population_figures()
    wide_spectra()
    intermediate_spectra(summaries, extrema)
    write_report(summaries, checks, backgrounds, parameters, extrema)


def write_report(summaries, checks, backgrounds, parameters, extrema):
    selected = json.loads((OUT / "selected.json").read_text())
    clean_search = json.loads((OUT / "selected_clean.json").read_text())
    best = summaries["clean"]
    p5 = summaries["polynomial5"]
    improvement = 100 * (1 - best["worst_fwhm_khz"] / p5["worst_fwhm_khz"])
    meets_goal = best["worst_fwhm_khz"] <= 50 and best["minimum_snr"] >= 10
    conclusion = ("The sampled high-amplitude range meets both targets." if meets_goal else
                  "The joint near-T₂ / SNR-10 target is not achieved across the full range.")
    rows = ""
    for family, summary in summaries.items():
        high = [r for r in summary["rows"] if r["amplitude_mhz"] >= 50]
        rows += (
            f"<tr><td>{LABELS[family]}</td><td>{summary['worst_fwhm_khz']:.2f}</td>"
            f"<td>{summary['minimum_snr']:.3f}</td>"
            f"<td>{min(r['shoulder_contrast'] for r in high):.3f}</td>"
            f"<td>{summary['max_center_khz']:.3f}</td>"
            f"<td>{100 * max(r['max_final_pf'] for r in high):.4f}</td>"
            f"<td>{100 * summary['max_satellite_prominence']:.2f}</td></tr>")
    knot_rows = "".join(
        f"<tr><td>{t:.3g}</td><td>{a:.10g}</td><td>{b:.10g}</td><td>{c:.10g}</td></tr>"
        for t, a, b, c in zip(opt.KNOTS, np.r_[0., parameters["spline"]],
                              np.r_[0., parameters["margin"]],
                              np.r_[0., parameters["clean"]], strict=True))
    background_rows = "".join(
        f"<tr><td>{LABELS[b['family']]}</td><td>{b['amplitude_mhz']:.3f}</td>"
        f"<td>{b['original_background_width_khz']:.2f}</td>"
        f"<td>{b['outer_background_width_khz']:.2f}</td></tr>" for b in backgrounds)
    worst_pop = max(c["max_population_change"] for c in checks)
    worst_dt = max(c["max_time_step_width_change_hz"] for c in checks)
    worst_grid = max(c["max_frequency_grid_width_change_hz"] for c in checks)
    high = best["rows"][50:]
    snr_worst = min(high, key=lambda r: r["contrast_snr"])
    width_worst = max(high, key=lambda r: r["fwhm_hz"])
    satellite_index = 50 + int(np.argmax(
        [r["max_prominence"] for r in best["satellites"]]))
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Optimized Lorentzian echo tails: linewidth and selectivity</title>
<style>body{{font:17px/1.6 system-ui;max-width:1300px;margin:auto;padding:32px;
color:#172636;background:#fafbfd}}img{{width:100%;margin:15px 0}}
table{{border-collapse:collapse;width:100%;font-size:15px}}
td,th{{padding:9px;border-bottom:1px solid #d5dce2;text-align:left}}
.lead{{background:#e6f2f4;padding:18px;border-left:4px solid #007c91}}
code{{background:#eef1f5;padding:3px}}a{{color:#007c91}}</style>
<h1>A fixed Lorentzian core with a separately optimized tail</h1>
<p class="lead">Candidate selected to suppress competing dips, applied unchanged at all amplitudes:
<b>worst FWHM {best['worst_fwhm_khz']:.2f} kHz</b> and
<b>minimum ideal SNR {best['minimum_snr']:.3f}</b> over all 50 sampled amplitudes
from 50.505 to 100 MHz. The width is {best['worst_fwhm_khz'] / REFERENCE:.2f} times
the {REFERENCE:.2f} kHz T₂ reference, and {improvement:.1f}% below the p=5 control.
<b>{conclusion}</b></p>
<p><b>The narrowest central feature was not the cleanest resonance.</b> Optimizing
only linewidth and SNR produced central widths of 75–88 kHz with SNR above 10,
but also two side dips nearly as deep as the center. These are interference
fringes, not a single isolated narrow line. We retained those results and repeated
the search with a penalty for competing dips. The table distinguishes both outcomes.</p>
<p>Five full maps, each <b>100 amplitudes from 0 to 100 MHz × 201 detunings ×
three populations</b>. All use 30 µs duration, the same 15.0000075 ns root-Lorentzian
core, β = −0.5, the same 16 ns sine echo inversion, α = −216 MHz,
T₁ = 51.24 µs and T₂ = 7.31 µs. No waveform rescaling at individual amplitudes.</p>
<h2>Full amplitude test</h2>
<p>Statistics below are conservative extrema over 50.505–100 MHz, and need not
occur at the same amplitude. SNR assumes the same ideal 1,000-shot measurement
for every pulse. "Near T₂" is operationally FWHM ≤ 50 kHz.</p>
<table><tr><th>Pulse</th><th>Worst FWHM (kHz)</th><th>Minimum SNR</th>
<th>Minimum Pe contrast</th><th>Max |center| (kHz)</th><th>Max final Pf (%)</th>
<th>Max competing-dip prominence (pp)</th></tr>
{rows}</table>
<img src="comparison.png" alt="Linewidth, SNR, center and contrast versus amplitude">
<h2>The pulse shape</h2>
<p>Let f(t) = [1 + (t/σ)²]<sup>−1/2</sup>, σ = {opt.SIGMA:.12g} µs.
The positive envelope is <code>g(t) = f(t) exp[h(ln |t|)] w(t)</code>.
The correction h is exactly zero for |t| ≤ 0.05 µs, so the strong central core
is preserved. Outside that region it is a quintic interpolating spline through
the table below, with its first two derivatives with respect to ln|t| zero at
the first and last knots. Values shown are log multipliers; the selected*.json files
retain full floating-point precision.</p>
<table><tr><th>|t| (µs)</th><th>h: narrower candidate</th>
<th>h: SNR-margin candidate</th><th>h: suppressed-satellite candidate</th></tr>{knot_rows}</table>
<p>The endpoint closure is w = 1 for |t| ≤ 13 µs,
w = 10u³ − 15u⁴ + 6u⁵ for 13 &lt; |t| &lt; 15 µs, where u = (15 − |t|)/2,
and w = 0 outside. Thus g, g′ and g″ vanish at the support endpoints.
The signed pulse is I = A s(t)g(t), with the original 16 ns sine inversion s.
Q = −β I′/(2π|α|), using the complete analytic product derivative, µs and MHz.
The simulated complex envelope is I − iQ.</p>
<p>The envelope is positive and never exceeds its original normalized midpoint
value of one. There is no peak or area renormalization. The cutoff c = 0.001
still sets the retained Lorentzian core width; the shaped endpoints are exactly
zero. The new tail curve replaces the original single polynomial window.</p>
<img src="waveforms.png" alt="Central I and Q waveforms and independently shaped tails">
<h2>How the candidate was selected</h2>
<p>The physical idea is to control the weak-drive tails independently of the fast
core, rather than changing the full duration or one global exponent.
The general idea of redistributing ramp speed according to the dynamics is
motivated by <a href="https://arxiv.org/abs/1411.5783">fast quasiadiabatic control</a>.
This is our numerical pulse-design test, not an implementation of that paper's
FAQUAD protocol or a literature claim that the target is attainable here.</p>
<p>The six free spline ordinates were selected using amplitudes
50.505, 74.747 and 100 MHz, with 201 detunings and a 0.5 ns RK4 step.
The deterministic search used seed 20261011, 32 initial candidates, 15 generations
of differential evolution, then up to 200 Powell evaluations; it recorded
{selected['evaluations']} distinct evaluated shapes. The optimization objective
penalizes SNR below 10.2, resonance centers beyond ±10 kHz, and envelopes above
the original peak. Among evaluated candidates with minimum SNR ≥ 10, the lowest
worst-case linewidth was selected. This is a finite search, not a global optimum.</p>
<p>The first candidate's three-point result was {selected['worst_fwhm_khz']:.2f} kHz
and SNR {selected['minimum_snr']:.3f}. Its remaining 47 high-amplitude points were
not used in that selection; the full grid revealed minimum SNR
{summaries['spline']['minimum_snr']:.3f}. We then selected the narrowest existing
trial whose three-point SNR was at least 10.5, without rerunning the optimizer,
and tested this second fixed waveform on the full grid. This is an adaptive
follow-up, not a claim that the second selection was blind to the first validation.</p>
<p>Inspecting the full spectra then revealed strong satellite dips. The additional
search used the same six-parameter family and the same training amplitudes, but
penalized the prominence of any extra Pe minimum at |detuning| ≥ 50 kHz. The
optimization target was at most 0.015 population prominence, with a final
selection cap of 0.025 (2.5 percentage points). This is measured relative to the
neighboring spectral maxima, not the far-off-resonance background. A synthetic
single-dip and two-dip check verified the detector.</p>
<p>This second search evaluated {clean_search['evaluations']} distinct shapes over
{clean_search['generations']} differential-evolution generations with seed 20261012.
The selected three-point width/SNR were {clean_search['worst_fwhm_khz']:.2f} kHz and
{clean_search['minimum_snr']:.3f}. The full 100-point map has maximum competing-dip
prominence {100 * best['max_satellite_prominence']:.2f} percentage points over 50–100 MHz.
Both search stages are documented; the unsuccessful fringed designs are not discarded
from the comparison.</p>
<p><b>The sparse optimization did not guarantee the full amplitude range.</b>
The full scan's minimum SNR is {snr_worst['contrast_snr']:.3f} at
{snr_worst['amplitude_mhz']:.2f} MHz, its maximum width is
{width_worst['fwhm_hz'] / 1000:.2f} kHz at {width_worst['amplitude_mhz']:.2f} MHz,
and competing-dip prominence reaches {100 * best['max_satellite_prominence']:.2f}
percentage points at {opt.AMPS[satellite_index]:.2f} MHz. This exceeds the 2.5-point
selection cap used at the three optimization amplitudes. The pulse is a partial
improvement, not a fully robust solution. All three newly identified cases were
also rerun with a 0.25 ns timestep and 2.5 kHz frequency spacing.</p>
<img src="intermediate_spectra.png" alt="Intermediate amplitudes with the weakest SNR,
strongest competing dip, and broadest central dip">
<h2>Populations</h2>
<img src="populations.png" alt="Pg Pe and Pf maps for the five compared pulses">
<img src="spectra_100mhz.png" alt="Three population spectra at 100 MHz">
<p>The full amplitude range also matters: the fringed candidates lose contrast
and broaden substantially below their high-amplitude operating range.</p>
<img src="full_range.png" alt="Full amplitude range including low-amplitude limitations">
<h2>Background sensitivity</h2>
<p>The primary FWHM definition remains exactly the previous one: a central Pe dip
and nearest half-depth crossings relative to a linear background fitted at
|detuning| = 0.4–0.5 MHz. It is not a Lorentzian fit and is not independent of
background choice. We also simulated ±1 MHz at the three selection amplitudes
and recomputed the half-depth width using the 0.8–1.0 MHz outer background.</p>
<img src="wide_spectra.png" alt="Wider population spectra with background bands">
<table><tr><th>Pulse</th><th>Amplitude (MHz)</th><th>Original background (kHz)</th>
<th>Outer background (kHz)</th></tr>{background_rows}</table>
<h2>Model and validation</h2>
<p>Same three-level RWA Lindblad equations, adjacent-level relaxation and number
dephasing, Tφ = 7.871480508563624 µs, and no extra phenomenological Stark correction.
The new vectorized waveform preparation feeds the existing native RK4 kernel.
The p=5 full map reproduces the prior full map within 10⁻¹⁰, and the C2 control
reproduces the prior three-amplitude screen within 10⁻¹⁰.
Selected waveform derivatives, symmetry, peak and endpoint conditions were checked.</p>
<p>All reported probabilities are finite, within [0,1], and normalized within
10⁻¹⁰. Refinement at the three selection amplitudes gives maximum population
change {worst_pop:.3g}, maximum linewidth change {worst_dt:.2f} Hz when halving
the timestep to 0.25 ns, and {worst_grid:.2f} Hz when halving frequency spacing
to 2.5 kHz. This checks numerical resolution, not hardware performance.</p>
<p>The extra checks at the worst intermediate amplitudes give maximum population
change {extrema['max_population_change']:.3g} and combined timestep/grid linewidth
change {extrema['max_width_change_hz']:.2f} Hz.</p>
<p>SNR is shoulder-minus-center Pe divided by ideal Bernoulli noise: 500 shots
at the fitted center, 250 each at ±0.45 MHz. It excludes assignment errors,
technical noise, drift and the cost of locating the center. It is not a percent
contrast. Final Pf does not quantify maximum transient leakage.
1/(πT₂) is the weak-drive Lorentzian reference, not a universal bound on every
finite-duration interference feature.</p>
<p>43 focused model/window/metric tests passed and the new scripts pass scoped
Ruff. The full test suite stops at collection because Torch is absent; repository-wide
Ruff reports 692 existing findings. These environment and repository issues do not
replace the numerical and shape checks documented above.</p>
<p><a href="selected_clean.json">Shape with suppressed satellites</a> ·
<a href="selected_margin.json">Fringed shape with SNR margin</a> ·
<a href="selected.json">Narrower shape and search result</a> ·
<a href="optimized_clean_waveform.npz">Sampled I/Q with suppressed satellites</a> ·
<a href="optimized_margin_waveform.npz">Fringed sampled I/Q with SNR margin</a> ·
<a href="optimized_spline_waveform.npz">Narrower sampled I/Q waveform</a> ·
<a href="metrics.csv">Full metrics CSV</a> · <a href="summary.json">Full summaries</a> ·
<a href="validation.json">Numerical checks</a> ·
<a href="scan_extrema_validation.json">Intermediate-amplitude checks</a> ·
<a href="background_sensitivity.json">Background checks</a> ·
<a href="provenance.json">Source hashes and reproduction</a> ·
<a href="comparison.pdf">Comparison PDF</a> · <a href="waveforms.pdf">Waveforms PDF</a></p>
</html>"""
    (OUT / "report.html").write_text(body)
    sources = list(opt.prior.previous.MODEL_FILES) + [
        "scripts/optimize_echo_tail.py", "scripts/report_optimized_echo_tail.py",
        "scripts/refine_echo_tail_selectivity.py",
        "echospec/analysis/dip_metrics.py", "echospec/analysis/spectroscopy_quality.py",
        "scripts/scan_duration_polynomial.py"]
    provenance = dict(
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                       for p in sources},
        computation_sha256=opt.hashes(), parameters=parameters,
        knot_times_us=opt.KNOTS.tolist(), bounds=opt.BOUNDS,
        training_amplitudes_mhz=opt.AMPS[opt.TRAIN_INDICES].tolist(),
        full_amplitude_points=100, full_amplitude_range_mhz=[0, 100],
        cutoff_for_retained_core=.001, sigma_us=opt.SIGMA, duration_us=30.,
        beta=-.5, echo_transition_ns=16, t1_us=51.24, t2_us=7.31,
        anharmonicity_mhz=-216, dt_ns=.5, detuning_step_khz=5,
        refined_dt_ns=.25, refined_detuning_step_khz=2.5,
        goal=dict(maximum_fwhm_khz=50, minimum_snr=10),
        numpy_version=np.__version__, python_version=sys.version,
        reproduction=[
            ".venv/bin/python scripts/optimize_echo_tail.py --stage validate",
            ".venv/bin/python scripts/optimize_echo_tail.py --stage optimize --iterations 15 --workers 2",
            ".venv/bin/python scripts/optimize_echo_tail.py --stage full",
            ".venv/bin/python scripts/optimize_echo_tail.py --stage refine",
            ".venv/bin/python scripts/optimize_echo_tail.py --stage wide",
            ".venv/bin/python scripts/optimize_echo_tail.py --stage select-margin",
            ".venv/bin/python scripts/optimize_echo_tail.py --stage full --margin",
            ".venv/bin/python scripts/optimize_echo_tail.py --stage refine --margin",
            ".venv/bin/python scripts/optimize_echo_tail.py --stage wide --margin",
            ".venv/bin/python scripts/refine_echo_tail_selectivity.py --iterations 18",
            ".venv/bin/python scripts/optimize_echo_tail.py --stage full --clean",
            ".venv/bin/python scripts/optimize_echo_tail.py --stage refine --clean",
            ".venv/bin/python scripts/optimize_echo_tail.py --stage wide --clean",
            ".venv/bin/python scripts/report_optimized_echo_tail.py"])
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(OUT / "report.html")


if __name__ == "__main__":
    main()
