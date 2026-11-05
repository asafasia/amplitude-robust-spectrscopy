#!/usr/bin/env python3
"""Check 30/20 kHz rotary-echo fringes and the associated loss of contrast.

Run: PYTHONPATH=. .venv/bin/python scripts/scan_rotary_echo_linewidth.py
This numerical comparison preserves the previous drive and coherence settings.
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
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.signal import find_peaks

from echospec.simulation.config import SimulationConfig
from echospec.simulation.rotary_echo import frequency_information, rotary_echo_states

ROOT = Path(__file__).resolve().parents[1]


def local_widths(x, pe):
    """Half-depth between neighboring maxima; keep troughs within ±150 kHz."""
    minima, maxima = find_peaks(-pe)[0], find_peaks(pe)[0]
    interp = PchipInterpolator(x, pe)
    records = []
    for j in minima:
        if abs(x[j]) > .15:
            continue
        left, right = maxima[maxima < j], maxima[maxima > j]
        if not len(left) or not len(right):
            continue
        left, right = int(left[-1]), int(right[0])
        background = (pe[left]+pe[right])/2
        depth = background-pe[j]
        half = (background+pe[j])/2
        xl = brentq(lambda z, half=half: interp(z)-half, x[left], x[j])
        xr = brentq(lambda z, half=half: interp(z)-half, x[j], x[right])
        records.append(dict(center_mhz=float(x[j]), fwhm_khz=float((xr-xl)*1000),
                            depth=float(depth), background=float(background),
                            left=left, right=right))
    if not records:
        raise RuntimeError("No complete local fringe in the specified window")
    return records


def main():
    output = ROOT / "outputs/rotary_echo_linewidth_targets"
    output.mkdir(parents=True, exist_ok=True)
    amplitude = 10.05
    cases = [
        ("baseline", "20 μs · π", 20., 201),
        ("target_30", "26.17 μs · π", 263/amplitude, 263),
        ("target_20", "39.30 μs · π", 395/amplitude, 395),
        ("shorter_half_angle", "20 μs · π/2", 20., 402),
    ]
    x = np.linspace(-.5, .5, 2001)
    cfg = SimulationConfig(
        levels=3, amplitude_mhz=(amplitude,), detuning_mhz=tuple(x),
        pulse_type="square", t1_us=43.8, t2_us=7.6, anharmonicity_mhz=-216.,
    )
    rows, validation, arrays, fringes = [], {}, dict(detuning_mhz=x), {}
    for name, label, duration, cycles in cases:
        current = replace(cfg, duration_us=duration)
        populations = {}
        for levels in (3, 4, 5):
            rho = rotary_echo_states(replace(current, levels=levels), cycles=cycles)
            populations[levels] = np.diagonal(rho[0], axis1=-2, axis2=-1).real
            assert np.max(abs(populations[levels].sum(axis=1)-1)) < 1e-9
            assert np.linalg.eigvalsh(rho).min() > -1e-9
            arrays[f"{name}_populations_{levels}"] = populations[levels]
        pe = populations[3][:, 1]
        widths = local_widths(x, pe)
        coarse = local_widths(x[::2], pe[::2])
        fringes[name] = min(widths, key=lambda row: abs(row["center_mhz"]))
        near = abs(x) <= .15
        information = frequency_information(x[near], pe[near])
        angle = np.pi*amplitude*duration/cycles
        ideal_width = angle/(4*duration*abs(np.sin(angle/2)))*1000
        row = dict(
            case=name, label=label, duration_us=duration, cycles=cycles,
            half_rotation_over_pi=angle/np.pi,
            ideal_two_level_width_khz=ideal_width,
            median_fwhm_khz=float(np.median([r["fwhm_khz"] for r in widths])),
            min_fwhm_khz=min(r["fwhm_khz"] for r in widths),
            max_fwhm_khz=max(r["fwhm_khz"] for r in widths),
            median_fringe_depth=float(np.median([r["depth"] for r in widths])),
            local_sigma_khz_sqrt_shots=information["sigma_f_khz_sqrt_shots"],
            frequency_noise_hz_sqrt_seconds=(
                information["sigma_f_khz_sqrt_shots"]*1000*np.sqrt(duration*1e-6)
            ),
        )
        checks = dict(
            detuning_step_refinement_median_width_change_khz=abs(
                row["median_fwhm_khz"]-np.median([r["fwhm_khz"] for r in coarse])
            ),
            three_to_four_max_pe_change=float(np.max(abs(
                populations[3][:, 1]-populations[4][:, 1]
            ))),
            four_to_five_max_pe_change=float(np.max(abs(
                populations[4][:, 1]-populations[5][:, 1]
            ))),
        )
        for levels in (4, 5):
            level_widths = local_widths(x, populations[levels][:, 1])
            checks[f"{levels}_level_median_width_khz"] = float(np.median(
                [r["fwhm_khz"] for r in level_widths]
            ))
            checks[f"{levels}_level_median_depth"] = float(np.median(
                [r["depth"] for r in level_widths]
            ))
        validation[name] = checks
        rows.append(row)
        print(json.dumps(row), flush=True)
    reference = rows[0]["frequency_noise_hz_sqrt_seconds"]
    for row in rows:
        row["local_equal_time_precision_ratio_to_baseline"] = (
            row["frequency_noise_hz_sqrt_seconds"]/reference
        )
    np.savez_compressed(output / "responses.npz", **arrays)
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "metrics.json").write_text(json.dumps(rows, indent=2)+"\n")
    (output / "validation.json").write_text(json.dumps(validation, indent=2)+"\n")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    colors = ["#8b61a8", "#148a90", "#cf6836", "#2869ad"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    for row, color in zip(rows, colors, strict=True):
        fringe = fringes[row["case"]]
        indices = slice(fringe["left"], fringe["right"]+1)
        pe = arrays[f"{row['case']}_populations_3"][:, 1]
        normalized = (fringe["background"]-pe[indices])/fringe["depth"]
        axes[0].plot((x[indices]-fringe["center_mhz"])*1000, normalized,
                      color=color, label=row["label"], lw=2)
    axes[0].axhline(.5, color="gray", ls="--", lw=1)
    axes[0].set(xlabel="Offset from selected fringe minimum (kHz)",
                ylabel="Normalized dip depth", title="Individual fringes get narrower",
                xlim=(-45, 45), ylim=(-.05, 1.05))
    axes[0].legend(fontsize=9)
    bars = axes[1].bar(np.arange(len(rows)),
                       [r["median_fringe_depth"] for r in rows], color=colors)
    axes[1].bar_label(bars, fmt="%.3f", padding=4)
    axes[1].set(xticks=np.arange(len(rows)),
                xticklabels=[r["label"].replace(" · ", "\n") for r in rows],
                ylabel="Actual fringe depth ΔPe", ylim=(0, .145),
                title="The surviving signal gets weaker")
    fig.suptitle("Repeated square echo · peak Rabi 10.05 MHz · T1=43.8 μs, T2=7.6 μs\n"
                 "Three-level simulation; left traces aligned and normalized only to compare widths")
    fig.savefig(output / "linewidth_tradeoff.png", dpi=175)
    fig.savefig(output / "linewidth_tradeoff.svg")
    plt.close(fig)
    paths = [Path(__file__), ROOT / "echospec/simulation/rotary_echo.py",
             ROOT / "echospec/simulation/piecewise.py"]
    (output / "provenance.json").write_text(json.dumps(dict(
        configuration=cfg.__dict__, cases=cases,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in paths},
        width_definition="Median local dip FWHM for minima within ±150 kHz; baseline is mean of neighboring maxima",
        sensitivity_definition="Best local ideal binary Pe readout within ±150 kHz; calibrated working point and drive; interrogation time only, zero reset/readout overhead",
        scope="Simulation with Markovian T1/T2 and instantaneous phase flips; no hardware validation",
    ), indent=2)+"\n")
    print(f"Saved results in {output}", flush=True)


if __name__ == "__main__":
    main()
