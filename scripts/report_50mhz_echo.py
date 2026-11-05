"""Refine, independently validate, and report the constrained 50 MHz search."""

from __future__ import annotations

import hashlib
import html
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.analysis.dip_metrics import dip_metrics  # noqa: E402
from echospec.simulation.polynomial_echo import (  # noqa: E402
    PolynomialEchoPulse,
    simulate_polynomial_echo,
)
from echospec.simulation.selective_echo import simulate_echo_split  # noqa: E402
from scripts.optimize_50mhz_echo import (  # noqa: E402
    OUT,
    T1,
    T2,
    make_pulse,
    metrics,
    peak_scale,
)


def native_cached(pulse, amplitudes, detuning, **kwargs):
    """Cache complete validation arrays with waveform, model, and grid identity."""
    source_files = [ROOT / "echospec/simulation/selective_echo.py",
                    ROOT / "echospec/simulation/polynomial_echo.py",
                    ROOT / "echospec/simulation/qutrit_rk4.cpp"]
    identity = dict(pulse=asdict(pulse), family=type(pulse).__name__,
                    amplitudes=np.asarray(amplitudes).tolist(),
                    detuning=np.asarray(detuning).tolist(), kwargs=kwargs,
                    sources=[hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files])
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = OUT / "evaluations" / f"report_{key}.npz"
    if path.exists():
        with np.load(path) as saved:
            return saved["populations"].copy(), json.loads(str(saved["diagnostics"]))
    pop, diagnostics = simulate_polynomial_echo(pulse, amplitudes, detuning, **kwargs)
    np.savez_compressed(path, populations=pop, diagnostics=json.dumps(diagnostics))
    return pop, diagnostics


def select(family, peak=True):
    directory = OUT / "peak" if peak else OUT
    rows = [json.loads(p.read_text()) for p in
            (directory / "evaluations").glob(f"{family}_*.json")]
    eligible = [r for r in rows if all(
        m["accepted"] and m["contrast"] >= .4 and m["satellite_prominence"] <= .01
        and abs(m["center_hz"]) <= 2000 for m in r["rows"])]
    if not eligible:
        raise RuntimeError(f"No feasible {family} candidate")
    best = min(eligible, key=lambda r: max(m["fwhm_hz"] for m in r["rows"]))
    return make_pulse(best["parameters"], family), best, len(rows)


def main():
    x = np.linspace(-.5, .5, 401)
    amps = np.unique(np.r_[np.arange(10., 101., 5.), np.arange(40., 61., 2.5)])
    poly, poly_record, poly_count = select("polynomial")
    phase, phase_record, phase_count = select("phase")
    local, local_record, local_count = select("selective", peak=False)
    cases = {
        "baseline": PolynomialEchoPulse(),
        "retuned": poly,
        "phase": phase,
        "local_adiabatic": local,
    }
    labels = {"baseline": "Original polynomial p=5", "retuned": "Retuned polynomial",
              "phase": "Symmetric phase shaping", "local_adiabatic": "Local-adiabatic candidate"}
    summary, arrays, validation = {}, {}, {}
    for name, pulse in cases.items():
        norm = peak_scale(pulse)
        pop, diagnostics = native_cached(pulse, amps / norm, x, dt_ns=.25)
        rows = metrics(pop, amps, x)
        at50 = int(np.flatnonzero(amps == 50)[0])
        finer, finer_diag = native_cached(pulse, [50 / norm], x, dt_ns=.125)
        m50 = rows[at50]
        fine_metric = dip_metrics(x, finer[1, 0])
        grid_metric = dip_metrics(x[::2], pop[1, at50, ::2])
        validation[name] = dict(
            max_population_change_dt_halving=float(abs(finer[:, 0] - pop[:, at50]).max()),
            fwhm_change_dt_halving_khz=abs(fine_metric["fwhm_hz"] - m50["fwhm_hz"]) / 1000,
            fwhm_change_grid_halving_khz=abs(grid_metric["fwhm_hz"] - m50["fwhm_hz"]) / 1000,
            native_diagnostics=diagnostics, finer_diagnostics=finer_diag,
        )
        operating = [r for r in rows if 40 <= r["rabi_mhz"] <= 60]
        summary[name] = dict(label=labels[name], pulse=asdict(pulse),
                             unit_waveform_peak=norm, nominal_scale_for_50_peak_mhz=50 / norm,
                             at50=m50, rows=rows,
                             robust40_60=dict(min_contrast=min(r["contrast"] for r in operating),
                                             max_fwhm_khz=max(r["fwhm_hz"] for r in operating) / 1000,
                                             max_satellite=max(r["satellite_prominence"] for r in operating)))
        arrays[name] = pop
        wide_x = np.linspace(-2, 2, 801)
        wide, _ = native_cached(pulse, [50 / norm], wide_x, dt_ns=.25)
        arrays[f"{name}_wide"] = wide[:, 0]
        print(name, m50, flush=True)

    # The historical convention is recorded separately, not conflated with peak=50.
    for name in ("baseline", "retuned", "phase"):
        pulse = cases[name]
        if name != "baseline":
            pulse, _, _ = select("polynomial" if name == "retuned" else "phase", peak=False)
        pop, _ = native_cached(pulse, [50], x, dt_ns=.25)
        summary[name]["historical_nominal50"] = dict(
            pulse=asdict(pulse), actual_peak_mhz=50 * peak_scale(pulse),
            **dip_metrics(x, pop[1, 0]))

    # Independent propagation and three-/four-/five-level truncation checks.
    check_x = np.linspace(-.5, .5, 201)
    for name in ("baseline", "retuned", "phase"):
        pulse = cases[name]
        scale = 50 / peak_scale(pulse)
        native, _ = native_cached(pulse, [scale], check_x, dt_ns=.125)
        split = {}
        for levels in (3, 4, 5):
            split[levels] = simulate_echo_split(pulse, scale, check_x, levels=levels,
                                                refinement=2)
            arrays[f"{name}_levels{levels}"] = split[levels]
        split_fine = simulate_echo_split(pulse, scale, check_x, levels=4, refinement=4)
        validation[name].update(
            split3_vs_native_max_pe=float(abs(split[3][1] - native[1, 0]).max()),
            levels3_vs4_max_pe=float(abs(split[3][1] - split[4][1]).max()),
            levels4_vs5_max_pe=float(abs(split[4][1] - split[5][1]).max()),
            split4_time_refinement_max_pe=float(abs(split[4][1] - split_fine[1]).max()),
            levels4_metrics=dip_metrics(check_x, split_fine[1]),
            levels5_metrics=dip_metrics(check_x, split[5][1]),
        )
        coherent, _ = native_cached(pulse, [scale], x, t1_us=1e9,
                                    t2_us=1e9, dt_ns=.0625)
        summary[name]["coherent_control"] = dip_metrics(x, coherent[1, 0])
        arrays[f"{name}_coherent"] = coherent[:, 0]
        print("validated", name, validation[name], flush=True)

    phase_off = replace(phase, phase_rad=0)
    off_pop, _ = native_cached(phase_off, [50 / peak_scale(phase_off)], x, dt_ns=.125)
    summary["phase"]["same_envelope_phase_off"] = dip_metrics(x, off_pop[1, 0])
    arrays["phase_off_ablation"] = off_pop[:, 0]

    reference = 1000 / (np.pi * T2)
    metadata = dict(t1_us=T1, t2_us=T2, anharmonicity_mhz=-216,
                    coherence_reference_khz=reference, peak_rabi_mhz=50,
                    constraint="one echo reversal; no free interval; fixed shape over 40-60 MHz",
                    amplitude_definition="maximum absolute complex I-iQ drive, including derivative correction",
                    contrast_definition="absolute background-minus-Pe; background fitted in 0.4-0.5 MHz",
                    training_amplitudes_mhz=[40, 45, 50, 55, 60],
                    validation_amplitudes_mhz=amps.tolist(), detuning_step_khz=2.5,
                    target_achieved=any(v["at50"]["fwhm_hz"] <= reference * 1000 and
                                        v["robust40_60"]["min_contrast"] >= .4 for v in summary.values()),
                    search_evaluations=dict(polynomial_peak=poly_count, phase_peak=phase_count,
                                            local_adiabatic_nominal=local_count),
                    selected_records=dict(polynomial=poly_record, phase=phase_record,
                                           local_adiabatic=local_record))
    source_paths = [Path(__file__), ROOT / "scripts/optimize_50mhz_echo.py",
                    ROOT / "scripts/check_50mhz_bandwidth.py",
                    ROOT / "echospec/simulation/selective_echo.py",
                    ROOT / "echospec/simulation/polynomial_echo.py",
                    ROOT / "echospec/simulation/qutrit_rk4.cpp",
                    ROOT / "echospec/analysis/dip_metrics.py"]
    metadata["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in source_paths}
    (OUT / "results.json").write_text(json.dumps(dict(metadata=metadata, cases=summary,
                                                     validation=validation), indent=2))
    np.savez_compressed(OUT / "validated_spectra.npz", detuning_mhz=x, rabi_mhz=amps,
                        wide_detuning_mhz=wide_x, level_check_detuning_mhz=check_x, **arrays)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"baseline": "#777777", "retuned": "#136f9c", "phase": "#c35a2b",
              "local_adiabatic": "#9b79b4"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    at50 = int(np.flatnonzero(amps == 50)[0])
    for name in cases:
        rows = summary[name]["rows"]
        kwargs = dict(color=colors[name], label=labels[name])
        axes[0, 0].plot(x * 1000, arrays[name][1, at50], **kwargs)
        axes[0, 1].plot(amps, np.array([r["fwhm_hz"] for r in rows]) / 1000, **kwargs)
        axes[1, 0].plot(amps, [r["contrast"] for r in rows], **kwargs)
        axes[1, 1].plot(wide_x, arrays[f"{name}_wide"][1], **kwargs)
    axes[0, 0].set(xlim=(-250, 250), xlabel="Detuning (kHz)", ylabel="Excited-state probability",
                   title="50 MHz actual waveform peak")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axhline(reference, color="#222222", ls="--", label="1 / (π T₂)")
    axes[0, 1].set(xlabel="Actual peak Rabi rate (MHz)", ylabel="Dip FWHM (kHz)", ylim=(0, 230))
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].axhline(.4, color="#222222", ls="--")
    axes[1, 0].set(xlabel="Actual peak Rabi rate (MHz)", ylabel="Absolute dip contrast", ylim=(0, .65))
    for axis in (axes[0, 1], axes[1, 0]):
        axis.axvspan(40, 60, color="#e7ecef", alpha=.6, zorder=-10)
    axes[1, 1].set(xlabel="Detuning (MHz)", ylabel="Excited-state probability", title="Wider scan: inspect competing structure")
    fig.suptitle("Single-echo optimization: fixed 50 MHz peak, T₂ = 7.31 µs")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"comparison.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for name in ("baseline", "retuned", "phase"):
        pulse = cases[name]
        t = np.unique(np.r_[np.linspace(-pulse.duration_us/2, pulse.duration_us/2, 3001),
                             np.linspace(-.05, .05, 2001)])
        waveform = pulse.drive(t) * 50 / peak_scale(pulse)
        axes[0].semilogy(t, abs(waveform), color=colors[name], label=labels[name])
        axes[1].plot(t * 1000, waveform.real, color=colors[name], label=labels[name] + " I")
        axes[1].plot(t * 1000, -waveform.imag, color=colors[name], ls="--")
    axes[0].set(ylim=(1e-4, 70), xlabel="Time (µs)", ylabel="|I − iQ| (MHz)", title="Weak tails and strong core")
    axes[0].legend(fontsize=8)
    axes[1].set(xlim=(-40, 40), xlabel="Time (ns)", ylabel="I / Q (MHz)", title="Echo reversal: I solid, Q dashed")
    fig.savefig(OUT / "waveforms.png", dpi=180)
    plt.close(fig)

    table = "".join(f"<tr><td>{labels[k]}</td><td>{v['at50']['fwhm_hz']/1000:.2f}</td>"
                    f"<td>{v['at50']['contrast']:.3f}</td><td>{v['robust40_60']['min_contrast']:.3f}</td>"
                    f"<td>{v['robust40_60']['max_fwhm_khz']:.2f}</td></tr>" for k, v in summary.items())
    parameters = html.escape(json.dumps({k: v["pulse"] for k, v in summary.items()}, indent=2))
    bandwidth = json.loads((OUT / "bandwidth_check.json").read_text())
    bwtable = "".join(f"<tr><td>{r['gaussian_sigma_ns']:.0f}</td>"
                      f"<td>{r['metrics'][1]['fwhm_hz']/1000:.2f}</td>"
                      f"<td>{r['metrics'][1]['contrast']:.3f}</td></tr>" for r in bandwidth["rows"])
    report = f"""<!doctype html><meta charset="utf-8"><title>50 MHz single-echo study</title>
<style>body{{font:17px/1.55 system-ui;max-width:1080px;margin:40px auto;color:#172b3a;padding:20px}}h1,h2{{line-height:1.2}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ccd5dc;text-align:left}}pre{{font-size:13px;background:#f2f5f7;padding:20px;overflow:auto}}.callout{{background:#edf4f8;padding:20px;border-left:4px solid #136f9c}}a{{color:#136f9c}}</style>
<h1>50 MHz single-echo linewidth study</h1><p>1 November 2026. Local simulations only.</p>
<div class="callout">The searches found a narrower single echo while maintaining approximately 0.4 absolute contrast near 50 MHz. They did <b>not</b> reach the {reference:.2f} kHz coherence reference. This is a finite search, not a proof of an unattainable target or a global optimum.</div>
<p>Every main comparison has an actual maximum |I−iQ| of 50 MHz, one echo reversal, and no inserted free-evolution interval. Robustness was optimized on 40, 45, 50, 55, 60 MHz and checked at 2.5 MHz spacing. The wider amplitude scan explicitly shows the limits outside that band. Model: three-level Duffing Lindblad, T₁={T1} µs, T₂={T2} µs, α/2π=−216 MHz; independent four-/five-level checks are in results.json.</p>
<table><tr><th>Shape</th><th>FWHM at 50 (kHz)</th><th>Contrast at 50</th><th>Min C, 40–60</th><th>Max width, 40–60 (kHz)</th></tr>{table}</table>
<img src="comparison.png" alt="Spectra, linewidth, contrast, and wider spectra"><h2>Why the polynomial does not remove every source of broadening</h2>
<p>The endpoint correction removes a finite drive jump, but a global (1−(2t/L)²)ᵖ taper also redistributes the useful weak tail. Pulse length and pulse-envelope width are not the detuning-sensitive interrogation time. For a real two-level echo, θ(t)=∫Ω<sub>I</sub>(s)ds and S=∫sinθ(t)dt give Pₑ(Δ)=Δ²S²/4+O(Δ⁴). Rapid rotations contribute with alternating signs. Increasing L at fixed cutoff also increases the root-Lorentzian core width σ=Lc/(2√(1−c²)); it is not simply extending an unchanged tail.</p>
<p>A polynomial edge behaves as Ω≈b s<sup>p+1</sup>, where s is distance from the endpoint. Rescaling the local Schrödinger equation gives a characteristic edge scale Δ<sub>edge</sub>∝b<sup>1/(p+2)</sup>. A high p can weaken amplitude dependence while still having a larger absolute linewidth. This scaling is not a full dissipative FWHM formula.</p>
<h2>New control direction: time-reversed phase shaping</h2>
<p>Let the complex control be u(t)=A g(t)s(t)exp[iφ(t)], with g and φ even and the echo sign s odd. Then u(−t)=−u(t). In an ideal resonant two-level system, the second half is the inverse of the first: U(0,A)=identity for every global A, even when the instantaneous control axes do not commute. This preserves the amplitude-independent resonant cancellation and permits optimization of the off-resonant response through φ(t). It does not guarantee constant linewidth or contrast, and multilevel terms, DRAG, and dissipation break the ideal cancellation.</p>
<p>This study tests φ(t)=φ₀ exp[−(t/τφ)²], with the derivative correction applied to the complete complex waveform. It is a concrete tested extension of the scalar envelope. The limited search does not establish that phase shaping reaches the coherence reference.</p>
<p>Controlled comparison: keeping this candidate's envelope parameters fixed and switching off the phase modulation gives {summary['phase']['same_envelope_phase_off']['fwhm_hz']/1000:.2f} kHz and contrast {summary['phase']['same_envelope_phase_off']['contrast']:.3f}, with the actual peak again normalized to 50 MHz.</p>
<p>A separate local-adiabatic candidate uses |dΩ/dt|∝(Ω²+Δ*²)<sup>3/2</sup>/|Δ*|, inspired by <a href="https://arxiv.org/abs/1411.5783">fast quasiadiabatic dynamics</a>. Its regularized tail did not outperform the selected retuned polynomial in this search. That method controls the time spent near Ω≈Δ*, but it does not remove the dephasing cost of making the qubit sensitive to detuning. <a href="https://arxiv.org/abs/2308.14187">Mihov and Vitanov</a> analyze power narrowing and cutoff broadening; their ideal asymptotic result is not a dissipative linewidth guarantee for this echo.</p>
<h2>What the T₂ reference means</h2>
<p>1/(πT₂) is the weak-drive homogeneous Lorentzian FWHM. It is a reference, not a universal bound on the FWHM of a transient interference dip. A very narrow dip with tiny contrast or large competing fringes does not meet the present objective. The model treats the supplied T₂ as Markovian; it does not justify replacing it by the measured echo coherence time. Slow frequency noise and a DC detuning signal act through the same operator, so protecting against one may also suppress the desired frequency response.</p>
<h2>Amplitude convention matters</h2>
<p>The original 50 MHz nominal scale peaks at {summary['baseline']['historical_nominal50']['actual_peak_mhz']:.2f} MHz after echo smoothing. Unnormalized optimization can reduce that peak further by shrinking the core. Those historical-convention comparisons are recorded separately in results.json; they are not used to claim a fixed-peak improvement.</p>
<img src="waveforms.png" alt="Waveform tails and central I/Q reversal"><h2>Exact candidate parameters</h2><pre>{parameters}</pre>
<h2>Bandwidth sensitivity</h2><p>The phase candidate has a root-Lorentzian core σ={phase.sigma_us*1000:.3f} ns. Gaussian convolution of the complete I/Q waveform, followed by renormalization to a 50 MHz actual peak, gives the following illustrative results. These are not measured transfer functions. A 2 ns Gaussian standard deviation corresponds to a 66.3 MHz amplitude −3 dB bandwidth; 4 ns corresponds to 33.1 MHz. A sufficiently narrow hardware passband can erase the contrast margin.</p>
<table><tr><th>Filter σ (ns)</th><th>FWHM at 50 (kHz)</th><th>Contrast at 50</th></tr>{bwtable}</table><p><a href="bandwidth_check.json">Filter details and 40/50/60 MHz checks</a>. Full hardware transfer-function calibration remains necessary.</p>
<h2>Reproduction and limits</h2><p>Run scripts/optimize_50mhz_echo.py with --family polynomial or phase --normalization peak, then scripts/report_50mhz_echo.py. The report also uses the nominal selective and phase searches; use --normalization nominal to reproduce those. The random seed is 20261101. The objective penalizes contrast below 0.405, additional minima with prominence above 0.01, and center offsets above 2 kHz. Selection then requires C≥0.4 at all five training amplitudes. The peak local-adiabatic row rescales the best nominal candidate; it is not a separate peak-constrained optimization.</p>
<p>Focused validation: 32 tests passed and all added Python files pass Ruff. Full pytest is blocked by missing torch; excluding that module gives 217 passes and one unrelated existing Figure 3 plotting failure. Repository-wide lint has 692 existing findings. See <a href="validation_checks.json">validation commands and limitations</a>.</p>
<p><a href="results.json">Exact metrics, numerical checks, and source hashes</a> · <a href="validated_spectra.npz">Population arrays</a> · <a href="comparison.pdf">Vector figure</a></p>
"""
    (OUT / "report.html").write_text(report)
    print("Report:", OUT / "report.html", flush=True)


if __name__ == "__main__":
    main()
