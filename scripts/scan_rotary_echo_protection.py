#!/usr/bin/env python3
"""Separate pulse-angle changes from hypothetical reductions in dephasing.

PYTHONPATH=. .venv/bin/python scripts/scan_rotary_echo_protection.py
No hardware work is performed. The T2 sweep is a requirement study, not a
demonstration that a control sequence has achieved those coherence times.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from echospec.analysis.fringe_metrics import local_fringe_metrics
from echospec.simulation.config import SimulationConfig
from echospec.simulation.rotary_echo import rotary_echo_states

ROOT = Path(__file__).resolve().parents[1]


def evaluate(config, cycles):
    rho = rotary_echo_states(config, cycles=cycles)[0]
    pop = np.diagonal(rho, axis1=-2, axis2=-1).real
    assert np.max(abs(pop.sum(axis=1)-1)) < 1e-9
    assert np.linalg.eigvalsh(rho).min() > -1e-9
    records = local_fringe_metrics(config.detuning_mhz, pop[:, 1])
    if not records:
        return pop, None
    return pop, dict(median_fwhm_khz=float(np.median([r["fwhm_khz"] for r in records])),
                     median_depth=float(np.median([r["depth"] for r in records])),
                     resolved_fringes=len(records))


def save_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    output = ROOT / "outputs/rotary_echo_protection_20khz"
    output.mkdir(parents=True, exist_ok=True)
    drive = 10.05
    x = np.linspace(-.25, .25, 1001)
    cfg = SimulationConfig(
        levels=4, amplitude_mhz=(drive,), detuning_mhz=tuple(x),
        pulse_type="square", duration_us=395/drive,
        t1_us=43.8, t2_us=7.6, anharmonicity_mhz=-216.,
    )
    arrays, coherence, angles = dict(detuning_mhz=x), [], []
    for t2 in [7.6, 10., 12., 15., 16., 20., 30., 60.]:
        current = replace(cfg, t2_us=t2)
        pop, metrics = evaluate(current, 395)
        if metrics is None:
            raise RuntimeError("Missing fringe in the coherence requirement scan")
        arrays[f"coherence_T2_{t2:g}"] = pop
        coherence.append(dict(t2_us=t2, t_phi_us=current.t_phi_us, **metrics))
        print("coherence", coherence[-1], flush=True)
    # Keep the peak drive fixed and vary the half-cycle angle. Cycle count is
    # adjusted to match the actual multilevel width, rather than assuming the
    # weak-detuning two-level formula remains exact.
    for fraction in np.arange(.4, 1.101, .05):
        fraction = float(np.round(fraction, 8))
        cycles = int(round(np.pi*drive/(4*.020*np.sin(np.pi*fraction/2))))
        for _ in range(4):
            duration = cycles*fraction/drive
            current = replace(cfg, duration_us=duration)
            pop, metrics = evaluate(current, cycles)
            if metrics is None:
                break
            if abs(metrics["median_fwhm_khz"]-20.) < .08:
                break
            cycles = max(1, int(round(cycles*metrics["median_fwhm_khz"]/20.)))
        if metrics is None:
            continue
        # Record the actually evaluated cycle count if the final proposed
        # adjustment would otherwise be outside this iteration budget.
        cycles = int(round(duration*drive/fraction))
        arrays[f"angle_{fraction:g}_pi"] = pop
        angles.append(dict(half_angle_over_pi=fraction, cycles=cycles,
                           duration_us=duration, **metrics))
        print("angle", angles[-1], flush=True)
    eligible = [r for r in angles if abs(r["median_fwhm_khz"]-20) < .2]
    best = max(eligible, key=lambda row: row["median_depth"])

    thresholds = {}
    for depth in [.1, .122, .2]:
        def residual(t2, target=depth):
            return evaluate(replace(cfg, t2_us=t2), 395)[1]["median_depth"]-target

        thresholds[str(depth)] = float(brentq(residual, 7.6, 40., xtol=1e-4))
    validation = {}
    selected = [("baseline", cfg, 395), ("T2_15", replace(cfg, t2_us=15.), 395),
                ("T2_30", replace(cfg, t2_us=30.), 395),
                ("best_angle", replace(cfg, duration_us=best["duration_us"]),
                 best["cycles"])]
    for name, current, cycles in selected:
        p4, m4 = evaluate(current, cycles)
        p5, m5 = evaluate(replace(current, levels=5), cycles)
        fine_x = np.linspace(-.25, .25, 2001)
        _, fine = evaluate(replace(current, detuning_mhz=tuple(fine_x)), cycles)
        validation[name] = dict(
            max_population_change_4_to_5=float(
                np.max(abs(np.pad(p4, ((0, 0), (0, 1)))-p5))
            ),
            width_change_4_to_5_khz=abs(m4["median_fwhm_khz"]-m5["median_fwhm_khz"]),
            depth_change_4_to_5=abs(m4["median_depth"]-m5["median_depth"]),
            half_grid_step_width_change_khz=abs(m4["median_fwhm_khz"]-fine["median_fwhm_khz"]),
        )
        arrays[f"validation_{name}_five_level"] = p5
    save_csv(output / "coherence_requirements.csv", coherence)
    save_csv(output / "angle_scan.csv", angles)
    np.savez_compressed(output / "responses.npz", **arrays)
    summary = dict(coherence_scan=coherence, angle_scan=angles,
                    best_tested_angle=best, required_model_t2_us=thresholds,
                    validation=validation,
                    interpretation="T2 sweep changes the assumed noise; it is not achieved by the angle scan")
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for t2, color in [(7.6, "#8b61a8"), (15., "#13968d"), (20., "#cc7d25"),
                       (30., "#3456a8")]:
        pe = arrays[f"coherence_T2_{t2:g}"][:, 1]
        axes[0].plot(x*1000, pe, color=color,
                      label=(f"T2 = {t2:g} μs"+ (" (current model)" if t2 == 7.6 else "")))
    axes[0].set(xlim=(-80, 80), xlabel="Drive − bare resonance (kHz)", ylabel="Pe",
                title="Same ≈20 kHz fringes; less assumed dephasing")
    axes[0].legend(fontsize=9)
    axes[1].plot([r["t2_us"] for r in coherence],
                  [r["median_depth"] for r in coherence], "o-", color="#13968d")
    axes[1].axhline(best["median_depth"], color="#8b61a8", ls="--",
                    label="Best tested angle at current T2")
    axes[1].axhline(.122, color="gray", ls=":",
                    label="Original 39 kHz-fringe depth")
    axes[1].set(xlabel="Assumed Markovian T2 (μs)", ylabel="Fringe depth ΔPe",
                title="Required reduction in dephasing", xlim=(5, 62))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.16)
    fig.suptitle("20 kHz rotary-echo feasibility · 10.05 MHz drive · T1=43.8 μs\n"
                 "Coherence sweep is hypothetical; pulse-angle changes alone do not produce it")
    fig.savefig(output / "coherence_requirement.png", dpi=175)
    fig.savefig(output / "coherence_requirement.svg")
    plt.close(fig)

    table = "".join(f"<tr><td>{r['t2_us']:g}</td><td>{r['median_fwhm_khz']:.2f}</td>"
                    f"<td>{r['median_depth']:.3f}</td></tr>" for r in coherence)
    (output / "report.html").write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>20 kHz coherence requirements</title>
<style>body{{font:17px/1.6 system-ui;max-width:1100px;margin:32px auto;padding:0 24px;color:#203044}}
img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px 24px;border-bottom:1px solid #ddd}}</style>
<h1>Keeping contrast at a 20 kHz fringe width</h1>
<p>The practical limitation in the present model is dephasing. At fixed 39.30 μs,
395 π rotary-echo cycles, peak Rabi 10.05 MHz, T1=43.8 μs, and anharmonicity
−216 MHz, reducing the modeled dephasing increases contrast while retaining
approximately 20 kHz local fringe widths. All primary calculations use four
transmon levels and exact segment exponentials.</p>
<img src="coherence_requirement.png" alt="Contrast improvements require reduced dephasing">
<table><tr><th>Assumed T2 (μs)</th><th>FWHM (kHz)</th><th>Depth ΔPe</th></tr>{table}</table>
<p>Depth 0.10 requires model T2≈{thresholds['0.1']:.2f} μs; preserving the old
0.122 depth requires T2≈{thresholds['0.122']:.2f} μs; depth 0.20 requires
T2≈{thresholds['0.2']:.2f} μs. These are hypothetical reductions of the
Markovian dephasing rate, not measured improvements or values achieved by a pulse sequence.</p>
<p>A separate angle scan holds T2=7.6 μs, T1, and the drive fixed while adjusting
duration to approximately 20 kHz. Of the tested half-angles from 0.4π to 1.1π,
the best accepted depth is {best['median_depth']:.4f}, at half-angle
{best['half_angle_over_pi']:.2f}π, {best['cycles']} cycles and
{best['duration_us']:.3f} μs; its width is {best['median_fwhm_khz']:.3f} kHz.
This is a limited family search, not proof of a globally optimal pulse. It
does not deliver the contrast of the improved-coherence scenarios.</p>
<p>Widths are medians over resolved troughs within ±150 kHz, using the average
of the two adjacent maxima as background. Asymmetric troughs without both
half-depth crossings are rejected. The full grid spans ±250 kHz in 0.5 kHz steps.
Selected cases halve the grid step and add a fifth transmon level. Phase flips
are instantaneous, the drive is perfectly calibrated and noiseless, and no
hardware bandwidth, drift, readout errors or colored noise spectrum is included.</p>
<p>The noise model is constant Lindblad relaxation and dephasing. Ramsey T2*,
spin-echo T2, driven T1ρ and actual rotary-echo fringe-decay times are different
observables; T1ρ must not simply replace T2 in this model. Measure these decays
under the intended drive and determine the relevant noise spectrum before
predicting benefits of a protection sequence. The simulations do not establish
whether 7.6 μs describes the driven device.</p>
<p>For a DC detuning signal and noise coupling through the same σz operator,
ordinary refocusing also removes the desired signal. Useful control must exploit
a difference in noise/signal spectrum, operator, or independently accessible
reference. This constraint is discussed in
<a href="https://www.nature.com/articles/s41467-017-01895-5">the sensing/error-correction literature</a>.
Rotary echoes reject drive errors, while their detuning response and coherence
must be assessed together; see
<a href="https://arxiv.org/abs/1207.5868">Aiello et al.</a>.</p>
<p>For this device, a concrete diagnostic is a matched-condition set of Ramsey,
Hahn-echo, spin-locking and rotary-echo contrast-versus-duration scans, with
interleaved short frequency references to identify slow drift. If drive noise
dominates, optimize phase/amplitude control and echo angle while measuring the
retained detuning slope. If resonator photon noise dominates, readout depletion,
thermalization and filtering are candidate interventions. Identifying and
suppressing thermal-photon dephasing has been demonstrated by
<a href="https://arxiv.org/abs/1801.00467">Yan et al.</a>; it has not been
established as the limiting noise source here. No hardware work was performed.</p>
<p><a href="summary.json">All results and validation</a> ·
<a href="coherence_requirements.csv">Coherence sweep CSV</a> ·
<a href="angle_scan.csv">Angle sweep CSV</a> ·
<a href="responses.npz">Population arrays</a> · <a href="provenance.json">Provenance</a></p>
<pre>PYTHONPATH=. .venv/bin/python scripts/scan_rotary_echo_protection.py</pre></html>""")
    paths = [Path(__file__), ROOT / "echospec/analysis/fringe_metrics.py",
             ROOT / "echospec/simulation/rotary_echo.py",
             ROOT / "echospec/simulation/piecewise.py"]
    (output / "provenance.json").write_text(json.dumps(dict(
        configuration=cfg.__dict__, cycles=395,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in paths},
        no_hardware=True, hypothetical_noise_changes=True,
    ), indent=2)+"\n")
    print("Best tested angle", best, flush=True)
    print("Required model T2 for target depths", thresholds, flush=True)
    print(f"Saved requirements in {output}", flush=True)


if __name__ == "__main__":
    main()
