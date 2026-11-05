#!/usr/bin/env python3
"""Fit saved purple rotary-echo spectra and recover an injected resonance shift.

Run: PYTHONPATH=. .venv/bin/python scripts/fit_rotary_echo_results.py
No raw acquisitions, manuscript figures, or existing comparison files are edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from echospec.analysis.rotary_echo_fit import fit_frequency_shift, fit_fringe_spacing
from echospec.simulation.config import SimulationConfig
from echospec.simulation.rotary_echo import rotary_echo_states

ROOT = Path(__file__).resolve().parents[1]


def scalar_summary(result):
    return {key: value if np.isfinite(value) else None for key, value in result.items()
            if np.isscalar(value) and not isinstance(value, np.ndarray)}


def make_figures(x, original, fringe, observed, variance, recovered, injected, output):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 175})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), layout="constrained",
                             height_ratios=(2.3, 1))
    ax = axes[0, 0]
    ax.plot(x*1000, original, color="#8b61a8", lw=2.3,
            label="Original three-level simulation")
    ax.plot(x*1000, fringe["prediction"], "--", color="#333333", lw=1.2,
            label="Sinusoid + linear background fit")
    ax.axvline(0, color="#208056", ls=":", label="Known bare resonance")
    ax.set(title=f"Fringe spacing fit: {fringe['spacing_mhz']*1000:.2f} kHz",
           ylabel="Excited-state probability", ylim=(.33, .63))
    ax.legend(fontsize=8, loc="upper left")
    axes[1, 0].plot(x*1000, fringe["residual"], color="#8b61a8", lw=1)
    axes[1, 0].axhline(0, color="gray", lw=.8)
    axes[1, 0].set(title=f"Sinusoid residual RMS = {fringe['rms_probability']:.4f}",
                   ylabel="Data − fit", xlabel="Drive − original bare f01 (kHz)")
    ax = axes[0, 1]
    ax.errorbar(x[::4]*1000, observed[::4], yerr=np.sqrt(variance[::4]),
                 fmt="o", color="#8b61a8", ms=2.4, alpha=.7, lw=.6,
                 label="Synthetic shots (every 4th point shown)")
    ax.plot(x*1000, recovered["prediction"], color="#222222", lw=1.2,
            label="Full transmon-model fit (all points)")
    ax.axvline(injected*1000, color="#208056", ls="--", lw=1.3,
                label=f"Injected resonance: +{injected*1000:.1f} kHz")
    ax.set(title=(f"Recovered resonance: +{recovered['resonance_shift_mhz']*1000:.2f}"
                  f" ± {recovered['local_standard_error_mhz']*1000:.2f} kHz"),
            ylabel="Excited-state probability", ylim=(.33, .63))
    ax.legend(fontsize=8, loc="upper left")
    axes[1, 1].plot(x*1000, recovered["residual"], ".", color="#8b61a8", ms=2.5)
    axes[1, 1].fill_between(x*1000, -np.sqrt(variance), np.sqrt(variance),
                            color="gray", alpha=.16, label="Expected ±1σ shot noise")
    axes[1, 1].axhline(0, color="gray", lw=.8)
    axes[1, 1].set(title=f"Reduced χ² = {recovered['reduced_chi_squared']:.2f}",
                   ylabel="Data − fit", xlabel="Drive − frequency reference (kHz)")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlim(-500, 500)
        ax.grid(alpha=.15)
    fig.suptitle("Fitting the purple repeated-square-echo response\n"
                 "20 μs · peak Rabi 10.05 MHz · 201 cycles · 4,000 synthetic shots/point")
    fig.savefig(output / "frequency_fit.png")
    fig.savefig(output / "frequency_fit.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    axes[0].plot(x*1000, original, color="#8b61a8", lw=1.6)
    for index, alias in enumerate(fringe["candidate_troughs_mhz"]):
        axes[0].axvline(alias*1000, color="#555555", ls=":", alpha=.45,
                        label="Equivalent sinusoidal troughs" if index == 0 else None)
    axes[0].axvline(0, color="#208056", lw=1.5, label="Known bare resonance")
    axes[0].set(title="Sinusoidal phase repeats every fringe",
                xlabel="Drive − original bare f01 (kHz)", ylabel="Pe",
                xlim=(-300, 300))
    axes[0].legend(fontsize=8)
    shift, cost = recovered["search_shift_mhz"], recovered["search_cost"]
    minima = recovered["minima"]
    axes[1].plot(shift*1000, cost-recovered["cost"], color="#333333", lw=1)
    axes[1].scatter(minima[:, 0]*1000, minima[:, 1]-recovered["cost"],
                    color="#8b61a8", s=20, zorder=4, label="Refined candidate minima")
    axes[1].axvline(injected*1000, color="#208056", ls="--", label="Injected shift")
    axes[1].axhline(1, color="gray", ls=":", lw=.8)
    axes[1].set(title="Fit searches the full ±400 kHz interval",
                xlabel="Trial bare-resonance shift (kHz)", ylabel="Profile Δχ²",
                yscale="symlog", ylim=(-.2, None))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.16)
    fig.suptitle("Fringe aliases: spacing alone does not identify the bare resonance")
    fig.savefig(output / "alias_search.png")
    plt.close(fig)


def write_report(summary, output):
    fringe, fitted = summary["sinusoid"], summary["injected_noisy_fit"]
    source_fit = summary["original_noiseless_fit"]
    injected = summary["injected_shift_mhz"]*1000
    recovered = fitted["resonance_shift_mhz"]*1000
    sigma = fitted["local_standard_error_mhz"]*1000
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fit the purple rotary-echo response</title>
<style>body{{font:17px/1.6 system-ui;color:#223246;max-width:1100px;margin:32px auto;padding:0 24px}}
img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:10px;text-align:left}}
.math{{background:#eef3f8;padding:14px;font-family:monospace}}a{{color:#176b9c}}</style>
<h1>Recovering frequency from the purple rotary-echo response</h1>
<p>The simple sinusoidal fit finds a fringe spacing of
<b>{fringe['spacing_mhz']*1000:.3f} kHz</b>. A full calibrated transmon-model fit
finds the original simulation's bare resonance at
<b>{source_fit['resonance_shift_mhz']*1000:.6f} kHz</b> relative to its input reference.
The independent synthetic shot-noise example recovers
<b>{recovered:.3f} ± {sigma:.3f} kHz</b> for an injected shift of
<b>{injected:.3f} kHz</b>. The quoted uncertainty is a local 1σ statistical
estimate under the assumed model.</p>
<img src="frequency_fit.png" alt="Data, sinusoid fit, full model fit, and residuals">
<p>The source is the previously saved three-level purple response at peak
Rabi 10.05 MHz, duration 20 μs, and 201 complete square-echo cycles.
The original grid has 401 microwave frequencies spanning ±500 kHz. The exact
same pulse timings remain fixed throughout every frequency scan and every trial fit.
The model includes T1=43.8 μs, T2=7.6 μs, and anharmonicity −216 MHz.</p>
<p>The descriptive fringe fit is</p>
<div class="math">P(x) = a + b x + c cos(2π q x) + d sin(2π q x)</div>
<p>Here x is in MHz, q is in μs, and the fringe spacing is 1/q MHz.
For each trial q in 8–18 μs, the four linear coefficients are optimized;
the lowest squared residual is then refined. The fit gives
q={fringe['effective_time_us']:.6f} μs, compared with the ideal small-detuning
prediction 2T/π={2*20/np.pi:.6f} μs. The predicted ideal spacing is
{np.pi/(2*20)*1000:.3f} kHz. These are frequency-axis fringes; q is not
a temporal oscillation frequency.</p>
<p>Its fitted troughs occur at
{fringe['phase_center_modulo_spacing_mhz']*1000:.3f} kHz + n ×
{fringe['spacing_mhz']*1000:.3f} kHz. Those repeating phase locations
do not directly give the bare resonance. Systematic residuals of RMS
{fringe['rms_probability']:.5f} show that a single cosine does not capture the
full three-level response, including drive-induced shifts and its varying envelope.</p>
<p>To estimate the bare frequency, the second fit uses</p>
<div class="math">Pfit(x) = offset + gain × Ptransmon(x − s); &nbsp; f01 = fref + s</div>
<p>The response is recomputed using exact Lindblad matrix exponentials for each
trial s. Readout offset and positive gain are fitted too; the calibrated drive,
timing, anharmonicity and coherence parameters are held fixed. The fit searches
241 trial shifts from −400 to +400 kHz and refines every sampled local minimum.
It is not initialized at the injected answer. All candidate minima are saved.</p>
<img src="alias_search.png" alt="Periodic candidate troughs and global profile fit">
<p>The shot-noise data are generated by translating the simulated bare resonance
by +{injected:g} kHz, then drawing binomial counts with 4,000 shots per frequency
using seed {summary['random_seed']}. This is a total of 1,604,000 synthetic shots,
not hardware data. The fit receives the frequency axis, observed probabilities,
estimated variances, and the calibrated response function; it does not receive
the injected shift. It uses weighted least squares with a positive fitted gain.</p>
<table><tr><th>Quantity</th><th>Result</th></tr>
<tr><td>Injected bare-resonance shift</td><td>+{injected:.3f} kHz</td></tr>
<tr><td>Recovered shift</td><td>+{recovered:.3f} kHz</td></tr>
<tr><td>Error from input</td><td>{recovered-injected:+.3f} kHz</td></tr>
<tr><td>Local statistical standard error</td><td>{sigma:.3f} kHz</td></tr>
<tr><td>Reduced χ²</td><td>{fitted['reduced_chi_squared']:.3f}</td></tr>
<tr><td>Fitted gain / offset</td><td>{fitted['gain']:.5f} / {fitted['offset']:+.5f}</td></tr>
<tr><td>Next candidate minimum Δχ²</td><td>{summary['next_minimum_delta_chi_squared']:.2f}</td></tr></table>
<p>The frequency uncertainty comes from the local weighted Jacobian with respect
to shift, gain and offset. It does not include calibration uncertainty, transmon
truncation, other noise processes, or model error. The same three-level model
generates and fits this demonstration, so it validates parameter recovery under
that model; it does not independently validate the model against hardware.</p>
<p>In particular, the prior comparison found that adding a fourth level changes
some repeated-echo populations at this amplitude by about 0.0015. Such mismatch
must be considered before interpreting a small statistical error as physical
frequency accuracy. A pure sinusoid retains exact frequency aliases; the full
model uses its envelope and deviations from exact periodicity to distinguish them
in this synthetic example.</p>
<p><a href="fit_summary.json">Fit parameters and diagnostics</a> ·
<a href="fit_arrays.npz">Data, fitted curves, residuals and search profiles</a> ·
<a href="data_and_fits.csv">Inspectable CSV</a> ·
<a href="provenance.json">Configuration and source hashes</a></p>
<pre>PYTHONPATH=. .venv/bin/python scripts/fit_rotary_echo_results.py</pre>
</html>"""
    (output / "report.html").write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "outputs/square_root_echo_comparison_20us")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/rotary_echo_frequency_fit")
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = args.source / "configuration.json"
    array_path = args.source / "responses.npz"
    cfg = SimulationConfig(**json.loads(config_path.read_text())["configuration"])
    with np.load(array_path) as source:
        index = int(np.argmin(abs(source["amplitude_mhz"]-10.05)))
        amplitude = float(source["amplitude_mhz"][index])
        x = source["detuning_mhz"].copy()
        original = source["dissipative3_repeated"][index, :, 1].copy()
        cycles = int(source["repeated_cycles"][index])
    cfg = replace(cfg, amplitude_mhz=(amplitude,), levels=3)

    def response(detuning):
        return rotary_echo_states(replace(cfg, detuning_mhz=tuple(detuning)),
                                   cycles=cycles)[0, :, 1, 1].real

    print("Fit sinusoidal spacing of saved purple response", flush=True)
    fringe = fit_fringe_spacing(x, original)
    print("Fit original bare resonance with calibrated full model", flush=True)
    original_fit = fit_frequency_shift(x, original, response)
    injected = .123
    random_seed, shots = 1729, 4000
    noiseless = response(x-injected)
    counts = np.random.default_rng(random_seed).binomial(shots, noiseless)
    observed = counts/shots
    regularized = (counts+.5)/(shots+1)
    variance = regularized*(1-regularized)/shots
    print("Search all resonance-shift candidates for synthetic shot-noise data", flush=True)
    recovered = fit_frequency_shift(x, observed, response, variance=variance)
    summary = dict(
        sinusoid=scalar_summary(fringe), original_noiseless_fit=scalar_summary(original_fit),
        injected_noisy_fit=scalar_summary(recovered), injected_shift_mhz=injected,
        shots_per_frequency=shots, total_shots=shots*len(x), random_seed=random_seed,
        next_minimum_delta_chi_squared=float(recovered["minima"][1, 1]-recovered["cost"]),
        fitted_original_candidates_mhz_cost=original_fit["minima"].tolist(),
        fitted_noisy_candidates_mhz_chi_squared=recovered["minima"].tolist(),
        conditional_fit_parameters=["bare resonance shift", "readout offset", "readout gain"],
        fixed_parameters=["Rabi amplitude", "cycle timing", "anharmonicity", "T1", "T2", "levels"],
    )
    arrays = dict(frequency_minus_reference_mhz=x, original_probability=original,
                  injected_noiseless_probability=noiseless, counts=counts,
                  observed_probability=observed, observation_variance=variance)
    for label, result in [("fringe", fringe), ("original", original_fit),
                           ("noisy", recovered)]:
        arrays.update({f"{label}_{k}": v for k, v in result.items()
                       if isinstance(v, np.ndarray)})
    np.savez_compressed(output / "fit_arrays.npz", **arrays)
    np.savetxt(output / "data_and_fits.csv", np.column_stack([
        x, original, fringe["prediction"], original_fit["prediction"], noiseless,
        counts, observed, recovered["prediction"], recovered["residual"],
    ]), delimiter=",", comments="", header=(
        "frequency_minus_reference_mhz,original_probability,sinusoid_fit,"
        "original_full_model_fit,injected_noiseless_probability,synthetic_counts,"
        "observed_probability,shifted_full_model_fit,residual"
    ))
    make_figures(x, original, fringe, observed, variance, recovered, injected, output)
    (output / "fit_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    write_report(summary, output)
    paths = [Path(__file__), ROOT / "echospec/analysis/rotary_echo_fit.py",
             ROOT / "echospec/simulation/rotary_echo.py",
             ROOT / "echospec/simulation/piecewise.py", config_path, array_path]
    provenance = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p.resolve().relative_to(ROOT)): hashlib.sha256(
            p.read_bytes()).hexdigest() for p in paths},
        configuration=cfg.__dict__, cycles=cycles,
        python_version=platform.python_version(), numpy_version=np.__version__,
        scipy_version=scipy.__version__, elapsed_seconds=time.perf_counter()-started,
        synthetic_only=True, model_validation="same-model recovery; no hardware validation",
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps({key: summary[key] for key in
                      ("sinusoid", "original_noiseless_fit", "injected_noisy_fit",
                       "next_minimum_delta_chi_squared")}, indent=2), flush=True)
    print(f"Saved fits in {output}", flush=True)


if __name__ == "__main__":
    main()
