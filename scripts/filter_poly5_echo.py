"""Apply Gaussian smoothing to the original poly5 complete I/Q waveform."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.simulation.polynomial_echo import PolynomialEchoPulse  # noqa: E402
from scripts.optimize_50mhz_echo import metrics, peak_scale  # noqa: E402
from scripts.report_flexible_50mhz_echo import (  # noqa: E402
    SampledPulse,
    assess,
    local_shoulder_metrics,
    save_json,
    sha,
)
from scripts.report_hybrid_echo import SOURCE, HybridCache  # noqa: E402

OUT = ROOT / "outputs/poly5_gaussian_filter_20261101"


def filtered_pulse(base, sigma_ns, sample_dt_ns=.125):
    count = int(round(base.duration_us * 1000 / sample_dt_ns))
    time = np.linspace(-base.duration_us / 2, base.duration_us / 2, count + 1)
    raw = base.drive(time) / peak_scale(base)
    filtered = gaussian_filter1d(raw, sigma_ns / sample_dt_ns,
                                mode="constant", cval=0., truncate=6.)
    retention = float(np.max(abs(filtered)))
    pulse = SampledPulse(base, time, filtered / retention, sigma_ns)
    return pulse, retention


def main():
    source_hash = sha(Path(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("scan", "maps", "render", "all"), default="scan")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[.5, 1., 2., 4., 8., 16., 32.])
    parser.add_argument("--map-sigmas", type=float, nargs="+", default=[2., 8., 32.])
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    previous = json.loads((SOURCE / "results.json").read_text())
    base = PolynomialEchoPulse(**previous["cases"]["baseline"]["pulse"])
    cache, detuning = HybridCache(OUT), np.linspace(-.5, .5, 801)
    if args.stage in ("scan", "all"):
        baseline, diagnostics, source = cache.native(base, [50 / peak_scale(base)], detuning, dt_ns=.125, allow_old=True)
        arrays = dict(detuning_mhz=detuning, baseline=baseline[:, 0])
        scan = dict(pulse=previous["cases"]["baseline"]["pulse"], t1_us=51.24, t2_us=7.31,
                    generator_sha256=source_hash, baseline_source_sha256=sha(SOURCE / "results.json"),
                    filtering="Gaussian convolution of completed u=I-iQ, sigma in ns, six-sigma kernel, zero extension, 0.125 ns sample spacing.",
                    peak_convention="Peak-fixed: rescale after filtering to actual 50 MHz. Command-fixed: no postfilter rescaling, so actual peak is 50 times retention.",
                    baseline=metrics(baseline, np.array([50.]), detuning)[0],
                    baseline_diagnostics=diagnostics, baseline_cache=source, cases=[])

        def evaluate(sigma):
            pulse, retention = filtered_pulse(base, sigma)
            amps = np.array([50., 50. * retention])
            pop, diag, source = cache.native(pulse, amps, detuning, dt_ns=.125)
            rows = metrics(pop, amps, detuning)
            record = dict(sigma_ns=sigma, amplitude_3db_mhz=np.sqrt(np.log(2)) / (2 * np.pi * sigma / 1000),
                          peak_retention=retention, unrescaled_peak_mhz=50 * retention,
                          fixed_peak=rows[0], fixed_command=rows[1],
                          local50=local_shoulder_metrics(detuning, pop[1, 0]),
                          endpoint_magnitude_mhz=float(50 * max(abs(pulse.values[[0, -1]]))),
                          diagnostics=diag, cache=source)
            print(json.dumps(record), flush=True)
            return record, pop

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            evaluated = list(pool.map(evaluate, args.sigmas))
        for record, pop in evaluated:
            scan["cases"].append(record)
            arrays[f"sigma_{record['sigma_ns']:g}"] = pop
        # Independently refine both sampled waveform and integration timestep.
        checks = []
        for sigma in (min(args.sigmas), args.sigmas[len(args.sigmas) // 2], max(args.sigmas)):
            pulse, retention = filtered_pulse(base, sigma, sample_dt_ns=.0625)
            pop, diag, source = cache.native(pulse, [50.], detuning, dt_ns=.0625)
            nominal = arrays[f"sigma_{sigma:g}"][:, 0]
            coarse = next(r for r in scan["cases"] if r["sigma_ns"] == sigma)
            fine = metrics(pop, np.array([50.]), detuning)[0]
            check = dict(sigma_ns=sigma, waveform_step_ns=.0625, integration_dt_ns=.0625,
                         max_population_difference=float(abs(pop[:, 0] - nominal).max()),
                         fwhm_difference_khz=abs(fine["fwhm_hz"] - coarse["fixed_peak"]["fwhm_hz"]) / 1000,
                         contrast_difference=abs(fine["contrast"] - coarse["fixed_peak"]["contrast"]),
                         refined_metrics=fine, refined_retention=retention, diagnostics=diag, cache=source)
            checks.append(check)
            print("REFINEMENT", json.dumps(check), flush=True)
        scan["refinement"] = checks
        save_json(OUT / "scan.json", scan)
        np.savez_compressed(OUT / "scan_arrays.npz", **arrays)
    if args.stage in ("maps", "all"):
        amps, x = np.arange(5., 60.01, .5), np.linspace(-.5, .5, 401)
        with np.load(SOURCE / "validated_spectra.npz") as old:
            np.testing.assert_array_equal(old["rabi_mhz"], amps)
            np.testing.assert_array_equal(old["detuning_mhz"], x)
            baseline = old["baseline"].copy()
        baseline_rows = metrics(baseline, amps, x)
        data = dict(rabi_mhz=amps, detuning_mhz=x, baseline=baseline)
        summary = dict(amplitude_step_mhz=.5, integration_dt_ns=.25, cases=[], generator_sha256=source_hash,
                       baseline_arrays_sha256=sha(SOURCE / "validated_spectra.npz"))
        settings = dict(amplitude_min_mhz=5., amplitude_max_mhz=60., contrast_min_at50=.4,
                        no_regression_width_khz_tolerance=0., no_regression_contrast_tolerance=0.,
                        max_center_khz=2., max_satellite_prominence=.01, max_final_leakage=.005)
        scan = json.loads((OUT / "scan.json").read_text())
        for sigma in args.map_sigmas:
            pulse, retention = filtered_pulse(base, sigma)
            chunks = [amps[i:i + 12] for i in range(0, len(amps), 12)]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                parts = list(pool.map(lambda a, pulse=pulse: cache.native(pulse, a, x), chunks))
            pop = np.concatenate([p[0] for p in parts], axis=1)
            rows = metrics(pop, amps, x)
            fine50 = next(c for c in scan["cases"] if c["sigma_ns"] == sigma)["fixed_peak"]
            quality = assess(rows, baseline_rows, settings, fine50)
            record = dict(sigma_ns=sigma, rows=rows, constraints=quality,
                          cache_components=[p[2] for p in parts], diagnostics=[p[1] for p in parts])
            data[f"sigma_{sigma:g}"] = pop
            summary["cases"].append(record)
            print("MAP", json.dumps(dict(sigma_ns=sigma, constraints=quality)), flush=True)
            save_json(OUT / "maps.json", summary)
        np.savez_compressed(OUT / "map_arrays.npz", **data)
    render(base)


def render(base):
    scan = json.loads((OUT / "scan.json").read_text())
    with np.load(OUT / "scan_arrays.npz") as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    cases = sorted(scan["cases"], key=lambda r: r["sigma_ns"])
    sigmas = np.array([0.] + [r["sigma_ns"] for r in cases])
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for mode, label in (("fixed_peak", "Actual peak held at 50 MHz"), ("fixed_command", "Original command amplitude; peak drops")):
        ax[0, 0].plot(sigmas, [scan["baseline"]["fwhm_hz"] / 1000] + [r[mode]["fwhm_hz"] / 1000 for r in cases], "o-", label=label)
        ax[0, 1].plot(sigmas, [scan["baseline"]["contrast"]] + [r[mode]["contrast"] for r in cases], "o-", label=label)
    ax[0, 0].axhline(1000 / (np.pi * 7.31), color="gray", ls=":", label="T₂ reference")
    ax[0, 0].set(xlabel="Gaussian sigma (ns)", ylabel="FWHM (kHz)")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].axhline(.4, color="black", ls=":", label="Contrast target 0.4")
    ax[0, 1].set(xlabel="Gaussian sigma (ns)", ylabel="Primary dip contrast")
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(sigmas, [50] + [r["unrescaled_peak_mhz"] for r in cases], "o-")
    ax[1, 0].set(xlabel="Gaussian sigma (ns)", ylabel="Actual peak before rescaling (MHz)")
    selected = [r for r in cases if r["sigma_ns"] in (2, 8, 32)]
    x = arrays["detuning_mhz"] * 1000
    ax[1, 1].plot(x, arrays["baseline"][1], label="Original")
    for r in selected:
        ax[1, 1].plot(x, arrays[f"sigma_{r['sigma_ns']:g}"][1, 0], label=f"σ={r['sigma_ns']:g} ns")
    ax[1, 1].set(xlabel="Detuning (kHz)", ylabel="Pₑ", xlim=(-300, 300), title="All curves have actual 50 MHz peak")
    ax[1, 1].legend(fontsize=9)
    fig.suptitle("Gaussian smoothing of the original 30 µs poly5 echo")
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"filter_effects.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    t = np.linspace(-.15, .15, 6001)
    long_t = np.linspace(-15, 15, 12001)
    for sigma in (0., *[r["sigma_ns"] for r in selected]):
        if sigma == 0:
            wave = base.drive(t) * 50 / peak_scale(base)
            long_wave = base.drive(long_t) * 50 / peak_scale(base)
        else:
            pulse, _ = filtered_pulse(base, sigma)
            wave, long_wave = pulse.drive(t) * 50, pulse.drive(long_t) * 50
        label = "Original" if sigma == 0 else f"σ={sigma:g} ns"
        axes[0].plot(t * 1000, wave.real, label=label)
        axes[1].plot(t * 1000, -wave.imag, label=label)
        axes[2].semilogy(long_t, np.maximum(abs(long_wave), 1e-12), label=label)
    axes[0].set(xlabel="Time (ns)", ylabel="I (MHz)", title="Central in-phase drive")
    axes[1].set(xlabel="Time (ns)", ylabel="Q (MHz)", title="Central quadrature drive")
    axes[2].set(xlabel="Time (µs)", ylabel="|I−iQ| (MHz)", ylim=(.001, 70), title="The full pulse remains 30 µs")
    axes[0].legend(fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"waveforms.{suffix}", dpi=180)
    plt.close(fig)
    maps_html = ""
    if (OUT / "maps.json").exists():
        maps = json.loads((OUT / "maps.json").read_text())
        with np.load(OUT / "map_arrays.npz") as saved:
            data = {k: saved[k].copy() for k in saved.files}
        keys = ["baseline"] + [f"sigma_{r['sigma_ns']:g}" for r in maps["cases"]]
        fig, axes = plt.subplots(1, len(keys), figsize=(3.5 * len(keys), 4.5), layout="constrained", squeeze=False)
        for ax, key in zip(axes[0], keys, strict=True):
            mesh = ax.pcolormesh(data["detuning_mhz"] * 1000, data["rabi_mhz"], data[key][1], vmin=0, vmax=1, cmap="viridis", shading="nearest")
            ax.axhline(50, color="white", ls="--", lw=1)
            ax.set(xlabel="Detuning (kHz)", ylabel="Actual peak Rabi (MHz)", title="Original" if key == "baseline" else key.replace("sigma_", "σ=") + " ns")
        fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Pₑ")
        for suffix in ("png", "pdf"):
            fig.savefig(OUT / f"maps.{suffix}", dpi=180)
        plt.close(fig)
        band_rows = ""
        for record in maps["cases"]:
            q = record["constraints"]
            band_rows += (f'<tr><td>{record["sigma_ns"]:g}</td>'
                          f'<td>{len(q["linewidth_regression_amplitudes_mhz"])}/111</td>'
                          f'<td>{len(q["contrast_regression_amplitudes_mhz"])}/111</td>'
                          f'<td>{q["worst_width_change_khz"]:.3f}</td>'
                          f'<td>{q["worst_contrast_change"]:.6f}</td></tr>')
        maps_html = ('<h2>Amplitude–detuning maps</h2><img src="maps.png">'
                     '<table><tr><th>Filter σ (ns)</th><th>Linewidth regressions</th>'
                     '<th>Contrast regressions</th><th>Worst broadening (kHz)</th>'
                     '<th>Smallest contrast change</th></tr>' + band_rows + '</table>'
                     '<p>Direct simulations at 111 amplitudes, 5–60 MHz in 0.5 MHz steps; '
                     'zero allowed regression tolerance. Each filtered waveform is scaled to '
                     'the actual peak on the vertical axis. '
                     '<a href="maps.json">Complete band comparisons</a> · '
                     '<a href="map_arrays.npz">Map population arrays</a></p>')
    rows = f'<tr><td>None</td><td>50</td><td>{scan["baseline"]["fwhm_hz"]/1000:.3f}</td><td>{scan["baseline"]["contrast"]:.6f}</td><td>—</td><td>—</td></tr>'
    for r in cases:
        rows += f'<tr><td>{r["sigma_ns"]:g}</td><td>{r["unrescaled_peak_mhz"]:.3f}</td><td>{r["fixed_peak"]["fwhm_hz"]/1000:.3f}</td><td>{r["fixed_peak"]["contrast"]:.6f}</td><td>{r["fixed_command"]["fwhm_hz"]/1000:.3f}</td><td>{r["fixed_command"]["contrast"]:.6f}</td></tr>'
    outcome = ""
    if all(r["fixed_peak"]["fwhm_hz"] > scan["baseline"]["fwhm_hz"]
           and r["fixed_peak"]["contrast"] < scan["baseline"]["contrast"] for r in cases):
        outcome = (f'<p><b>At an actual 50 MHz peak, every tested filter broadened the line '
                   f'and reduced contrast.</b> The tested Gaussian σ range was '
                   f'{cases[0]["sigma_ns"]:g}–{cases[-1]["sigma_ns"]:g} ns.</p>')
    example = next((r for r in cases if r["sigma_ns"] == 4), cases[0])
    explanation = (f'<h2>Why smoothing does not help here</h2><p>The filter attenuates the sharp '
                   f'central part more than the slowly varying tails. For σ={example["sigma_ns"]:g} ns, '
                   f'the peak retains {100*example["peak_retention"]:.2f}% of its original height. '
                   f'Restoring 50 MHz multiplies the whole filtered waveform by '
                   f'{1/example["peak_retention"]:.4f}, also increasing the tail amplitude. '
                   'The comparison without rescaling shows a nearly unchanged spectroscopy line '
                   'even though the central peak drops considerably. This points to the core/tail '
                   'balance as an important mechanism; it does not isolate every dynamical contribution. '
                   'Smoothing a nanosecond feature does not extend the 30 µs pulse or remove dephasing. '
                   'A narrower Fourier spectrum of the drive alone would not establish a narrower '
                   'dip in this driven echo simulation.</p>')
    page = f'''<!doctype html><meta charset="utf-8"><title>Filtering the original poly5 echo</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:35px auto;padding:0 20px;color:#182637}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ccd;text-align:left}}pre{{white-space:pre-wrap}}</style>
<h1>What happens if we simply smooth the poly5 echo?</h1>
{outcome}
<p>The original pulse is unchanged except for identical Gaussian convolution of its completed I and Q waveforms. Duration remains 30 µs. The existing 16 ns sine reversal is smoothed together with the amplitude and derivative correction. A Gaussian sigma describes the filter width, not a new pulse duration.</p>
<p>The main comparison restores the <b>actual completed peak to 50 MHz</b>. A separate control leaves the original command amplitude unchanged: filtering then lowers the physical peak. This separates pulse-shape effects from simply using a weaker drive.</p>
<table><tr><th>Filter σ (ns)</th><th>Peak without rescaling (MHz)</th><th>FWHM at restored 50 MHz (kHz)</th><th>Contrast at restored 50 MHz</th><th>FWHM without rescaling (kHz)</th><th>Contrast without rescaling</th></tr>{rows}</table>
<img src="filter_effects.png">{explanation}<img src="waveforms.png">{maps_html}
<p>Contrast uses the same linear-wing background as the preceding reports. Filtering does not remove T₁/T₂ loss. This is a three-level Lindblad simulation with T₁=51.24 µs, T₂=7.31 µs, and anharmonicity −216 MHz. Filters are illustrative, not measured transfer functions; no hardware run was performed.</p>
<p>The sampling and integration timestep are both halved for representative filter widths. The tiny postfilter values at the original endpoints are recorded; pulse support is retained at 30 µs. The full 50 MHz scan uses 801 detunings from −0.5 to +0.5 MHz.</p>
<p><a href="scan.json">Metrics, conventions, and convergence</a> · <a href="scan_arrays.npz">50 MHz spectra</a> · <a href="cache_imports.json">Historical cache provenance</a> · <a href="filter_effects.pdf">Figure PDF</a></p>
<h2>Original pulse parameters</h2><pre>{html.escape(json.dumps(scan['pulse'], indent=2))}</pre>'''
    (OUT / "report.html").write_text(page)
    for link in re.findall(r'(?:href|src)="([^"]+)"', page):
        assert (OUT / link).is_file(), link
    names = ["scan.json", "scan_arrays.npz", "maps.json", "map_arrays.npz"]
    output_names = ["report.html", "filter_effects.png", "waveforms.png"]
    if (OUT / "maps.png").exists():
        output_names.append("maps.png")
    save_json(OUT / "render_provenance.json", dict(generator_sha256=sha(Path(__file__)),
              inputs={n: sha(OUT / n) for n in names if (OUT / n).exists()},
              outputs={n: sha(OUT / n) for n in output_names}))


if __name__ == "__main__":
    main()
