#!/usr/bin/env python3
"""Matched-duration/peak-drive echo benchmark; never submits hardware jobs.

Run: PYTHONPATH=. .venv/bin/python scripts/compare_square_root_echo.py
Outputs are isolated from manuscript figures. --reuse replots saved arrays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from echospec.analysis.dip_metrics import dip_metrics
from echospec.simulation.config import SimulationConfig
from echospec.simulation.echo_theory import (
    EchoTheoryPulse,
    bloch_population,
    root_curvature_times,
)
from echospec.simulation.piecewise import lindblad_generator, simulate_pulse_pair
from echospec.simulation.qutrit import simulate_qutrit_map
from echospec.simulation.rotary_echo import (
    frequency_information,
    rotary_echo_states,
    square_echo_probability,
)

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("square", "root", "repeated")
LABELS = ("Square: one inversion", "Root-Lorentzian: one inversion",
          "Square: repeated, timing tuned")
COLORS = ("#ce6b24", "#126aa2", "#8b61a8")
SLICES = np.array([1.05, 3.05, 10.05, 20.05, 40.05, 59.95])


def qutrit(cfg, detuning, amplitude, *, cutoff, steps):
    result = simulate_qutrit_map(
        duration_us=cfg.duration_us, t1_us=cfg.t1_us, t_phi_us=cfg.t_phi_us,
        anharmonicity_mhz=cfg.anharmonicity_mhz, cutoff=cutoff, order=.5,
        echo=True, detuning_mhz=np.asarray(detuning), rabi_mhz=np.asarray(amplitude),
        num_steps_per_half=steps, integration_backend="native",
    )
    return np.stack([result.ground, result.excited, result.second_excited], axis=-1)


def build_data(cfg, output):
    started = time.perf_counter()
    amp, det = np.asarray(cfg.amplitude_mhz), np.asarray(cfg.detuning_mhz)
    counts = np.maximum(1, np.floor(amp * cfg.duration_us + .5)).astype(int)
    pulse = EchoTheoryPulse(duration_us=cfg.duration_us, cutoff=cfg.cutoff)
    data = dict(amplitude_mhz=amp, detuning_mhz=det, repeated_cycles=counts)
    validation = {}
    for levels, model in [(2, "dissipative2"), (3, "dissipative3"), (2, "ideal2")]:
        for name, cycles in [("square", 1), ("repeated", counts)]:
            print(f"Exact exponential: {model} {name}", flush=True)
            rho = rotary_echo_states(replace(cfg, levels=levels), cycles=cycles,
                                     unitary=model == "ideal2")
            populations = np.diagonal(rho, axis1=-2, axis2=-1).real
            data[f"{model}_{name}"] = populations
            validation[f"{model}_{name}_trace_error"] = float(
                np.max(abs(populations.sum(axis=-1) - 1))
            )
            validation[f"{model}_{name}_min_eigenvalue"] = float(
                np.linalg.eigvalsh((rho + rho.swapaxes(-1, -2).conj()) / 2).min()
            )
    expected = square_echo_probability(amp[:, None], det, cfg.duration_us)
    validation["square_unitary_max_analytic_error"] = float(
        np.max(abs(expected - data["ideal2_square"][..., 1]))
    )
    for model, g1, gp in [("ideal2", 0., 0.),
                          ("dissipative2", 1/cfg.t1_us, 1/cfg.t_phi_us)]:
        print(f"Adaptive Bloch integration: {model} root", flush=True)
        probabilities = []
        for i, a in enumerate(amp):
            probabilities.append(bloch_population(pulse, a, det, g1, gp))
            if i % 30 == 0:
                print(f"  {i+1}/{len(amp)} amplitudes", flush=True)
        pe = np.asarray(probabilities)
        data[f"{model}_root"] = np.stack([1-pe, pe], axis=-1)
    print("Native RK4: dissipative3 root (0.5 ns step)", flush=True)
    data["dissipative3_root"] = qutrit(
        cfg, det, amp, cutoff=cfg.cutoff, steps=cfg.num_steps_per_half
    )
    np.savez_compressed(output / "responses.npz", **data)
    print("Validation: root half-step convergence, square independent solver", flush=True)
    selected = np.array([np.argmin(abs(amp-a)) for a in SLICES])
    fine = qutrit(cfg, det, amp[selected], cutoff=cfg.cutoff,
                  steps=2*cfg.num_steps_per_half)
    validation["root_half_step_max_population_error"] = float(
        np.max(abs(fine-data["dissipative3_root"][selected]))
    )
    sparse_det = np.array([-.5, -.137, 0., .117, .5])
    check_cfg = replace(cfg, amplitude_mhz=tuple(SLICES),
                        detuning_mhz=tuple(sparse_det), levels=3)
    square_check = qutrit(cfg, sparse_det, SLICES, cutoff=None,
                          steps=2*cfg.num_steps_per_half)
    square_exact = rotary_echo_states(check_cfg)
    validation["square_native_vs_exact_max_population_error"] = float(np.max(abs(
        square_check-np.diagonal(square_exact, axis1=-2, axis2=-1).real
    )))
    # More detuning samples check derivatives and extracted widths, not just Pe.
    fine_det = np.linspace(det[0], det[-1], 2*len(det)-1)
    fine_grid = qutrit(cfg, fine_det, SLICES, cutoff=cfg.cutoff,
                       steps=2*cfg.num_steps_per_half)
    data["validation_detuning_mhz"] = fine_det
    data["validation_amplitude_mhz"] = SLICES
    data["validation_root_fine"] = fine_grid
    for name in ("square", "repeated"):
        fine_cfg = replace(cfg, amplitude_mhz=tuple(SLICES),
                           detuning_mhz=tuple(fine_det), levels=3)
        nc = (1 if name == "square" else
              np.maximum(1, np.floor(SLICES*cfg.duration_us+.5)).astype(int))
        rho = rotary_echo_states(fine_cfg, cycles=nc)
        data[f"validation_{name}_fine"] = np.diagonal(
            rho, axis1=-2, axis2=-1
        ).real
    changes = {}
    for name in NAMES:
        coarse = data[f"dissipative3_{name}"][selected, :, 1]
        refined = data[f"validation_{name}_fine"][..., 1]
        info_changes, width_changes = [], []
        for p, q in zip(coarse, refined, strict=True):
            a = frequency_information(det, p)["max_fisher_per_mhz2"]
            b = frequency_information(fine_det, q)["max_fisher_per_mhz2"]
            info_changes.append(abs(a-b)/b if b > 1e-12 else 0.)
            m1, m2 = dip_metrics(det, p), dip_metrics(fine_det, q)
            if m1["accepted"] and m2["accepted"]:
                width_changes.append(abs(m1["fwhm_hz"]-m2["fwhm_hz"]))
        changes[name] = dict(max_relative_fisher_change=max(info_changes),
                             max_width_change_hz=max(width_changes, default=None))
    validation["half_detuning_step"] = changes
    # Four-level check of the shaped envelope by an independent exponential
    # method; convergence of its envelope segmentation is recorded separately.
    print("Validation: fourth-level truncation and independent envelope solver", flush=True)
    # Transform-uniform integration is preferable for narrow root peaks, but
    # this uniform midpoint method provides a separate, conservatively checked
    # diagnostic at a small number of representative points.
    trunc_amp = (3.05, 20.05, 59.95)
    trunc_det = (-.5, -.1, 0., .1, .5)
    cfg3 = replace(cfg, levels=3, pulse_type="lorentzian",
                   amplitude_mhz=trunc_amp, detuning_mhz=trunc_det)
    trunc3 = qutrit(cfg3, trunc_det, trunc_amp, cutoff=cfg.cutoff,
                   steps=2*cfg.num_steps_per_half)
    fourth = []
    for segments in (2000, 4000):
        rho = simulate_pulse_pair(replace(cfg3, levels=4),
                                   segments_per_half=segments)["echo"]
        fourth.append(np.diagonal(rho, axis1=-2, axis2=-1).real)
    validation["four_level_envelope_refinement_max_population_change"] = float(
        np.max(abs(fourth[0]-fourth[1]))
    )
    validation["root_four_vs_three_max_Pe_change"] = float(
        np.max(abs(fourth[1][..., 1]-trunc3[..., 1]))
    )
    validation["root_max_final_P3_four_level"] = float(fourth[1][..., 3].max())
    data["truncation_amplitude_mhz"] = np.array(trunc_amp)
    data["truncation_detuning_mhz"] = np.array(trunc_det)
    data["truncation_root_three"] = trunc3
    data["truncation_root_four"] = fourth[1]
    for name in ("square", "repeated"):
        trunc_cfg = replace(cfg3, pulse_type="square")
        nc = (1 if name == "square" else
              np.floor(np.asarray(trunc_amp)*cfg.duration_us+.5).astype(int))
        r3 = rotary_echo_states(trunc_cfg, cycles=nc)
        r4 = rotary_echo_states(replace(trunc_cfg, levels=4), cycles=nc)
        p3 = np.diagonal(r3, axis1=-2, axis2=-1).real
        p4 = np.diagonal(r4, axis1=-2, axis2=-1).real
        validation[f"{name}_four_vs_three_max_Pe_change"] = float(
            np.max(abs(p4[..., 1]-p3[..., 1]))
        )
        data[f"truncation_{name}_three"] = p3
        data[f"truncation_{name}_four"] = p4
    validation["elapsed_seconds"] = time.perf_counter()-started
    np.savez_compressed(output / "responses.npz", **data)
    (output / "validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    return data


def metrics(data, output):
    rows = []
    x, amps = data["detuning_mhz"], data["amplitude_mhz"]
    for model in ("ideal2", "dissipative2", "dissipative3"):
        for name in NAMES:
            for amp, pop in zip(amps, data[f"{model}_{name}"], strict=True):
                pe = pop[:, 1]
                row = dict(model=model, pulse=name, amplitude_mhz=float(amp),
                           peak_to_peak=float(np.ptp(pe)),
                           Pe_at_bare_resonance=float(pe[np.argmin(abs(x))]),
                           max_final_leakage=float(pop[:, 2:].sum(axis=1).max()))
                row.update(dip_metrics(x, pe))
                row["resolved_visible_dip"] = bool(
                    row["accepted"] and row["contrast"] >= .02
                    and row["fwhm_hz"] >= 6*np.min(np.diff(x))*1e6
                )
                row.update(frequency_information(x, pe))
                row["non_ground_max_fisher_per_mhz2"] = frequency_information(
                    x, 1-pop[:, 0]
                )["max_fisher_per_mhz2"]
                rows.append(row)
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = [r for r in rows if r["model"] == "dissipative3"
                and np.any(np.isclose(r["amplitude_mhz"], SLICES, atol=1e-10))]
    (output / "selected_metrics.json").write_text(json.dumps(selected, indent=2)+"\n")
    return rows


def root_exponential_check(cfg, segments):
    """Independent Lindblad solver with exact segment-averaged root envelope.

    Segments are uniform in asinh(t/tau), resolving the narrow peak. The two
    matrix products reconstruct the reverse-ordered second half without
    storing all segment propagators. No RK4 or Bloch code is shared.
    """
    pulse = EchoTheoryPulse(duration_us=cfg.duration_us, cutoff=cfg.cutoff)
    edges = pulse.tau_us*np.sinh(np.linspace(-pulse.u_limit, 0., segments+1))
    amp, det = np.meshgrid(cfg.amplitude_mhz, cfg.detuning_mhz, indexing="ij")
    amp, det = amp.ravel(), det.ravel()
    n = cfg.levels
    forward = np.broadcast_to(np.eye(n*n), (len(amp), n*n, n*n)).copy()
    reverse = forward.copy()
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        dt = right-left
        scale = pulse.tau_us*(np.arcsinh(right/pulse.tau_us)
                              - np.arcsinh(left/pulse.tau_us))/dt
        step = expm(dt*lindblad_generator(cfg, det, amp*scale))
        forward = step @ forward
        reverse = reverse @ step
    parity = ((-1.)**(np.arange(n)[:, None]+np.arange(n))).ravel()
    rho = parity[None, :]*np.einsum(
        "bij,bj->bi", reverse, parity[None, :]*forward[..., 0]
    )
    return np.diagonal(rho.reshape(*np.meshgrid(cfg.amplitude_mhz,
                        cfg.detuning_mhz, indexing="ij")[0].shape, n, n),
                       axis1=-2, axis2=-1).real


def extra_controls(cfg, data, output, *, force=False):
    """Wide frequency window, fixed repetition timing, converged level checks."""
    path = output / "additional_controls.npz"
    if path.exists() and not force:
        return
    print("Additional controls: wide spectra and fixed repetition timing", flush=True)
    wide_det = np.linspace(-10., 10., 1601)
    wide_cfg = replace(cfg, amplitude_mhz=tuple(SLICES), detuning_mhz=tuple(wide_det))
    controls = dict(wide_detuning_mhz=wide_det, amplitude_mhz=SLICES)
    controls["wide_root"] = qutrit(cfg, wide_det, SLICES, cutoff=cfg.cutoff,
                                   steps=cfg.num_steps_per_half*2)
    rho = rotary_echo_states(wide_cfg)
    controls["wide_square"] = np.diagonal(rho, axis1=-2, axis2=-1).real
    fixed_cycles = 201
    rho = rotary_echo_states(cfg, cycles=fixed_cycles)
    controls["fixed_cycles"] = np.array(fixed_cycles)
    controls["fixed_repeated"] = np.diagonal(rho, axis1=-2, axis2=-1).real
    det = (-.5, -.1, 0., .1, .5)
    check_cfg = replace(cfg, amplitude_mhz=tuple(SLICES), detuning_mhz=det)
    fine3 = qutrit(cfg, det, SLICES, cutoff=cfg.cutoff,
                   steps=2*cfg.num_steps_per_half)
    validation = {}
    previous = None
    for segments in (256, 512, 1024, 2048, 4096, 8192):
        print(f"Independent root exponential solver: {segments} segments/half", flush=True)
        current = root_exponential_check(replace(check_cfg, levels=4), segments)
        change = (float(np.max(abs(current-previous))) if previous is not None
                  else np.inf)
        previous = current
        if change < 5e-5:
            break
    validation["root_four_level_segments_per_half"] = segments
    validation["root_four_level_refinement_max_population_change"] = change
    validation["root_four_vs_three_max_Pe_change_by_amplitude"] = (
        abs(current[..., 1]-fine3[..., 1]).max(axis=1).tolist()
    )
    controls["check_detuning_mhz"] = np.asarray(det)
    controls["check_root_3"] = fine3
    controls["check_root_4"] = current
    for name in ("square", "repeated"):
        cycles = (1 if name == "square" else
                  np.floor(SLICES*cfg.duration_us+.5).astype(int))
        for levels in (3, 4, 5):
            rho = rotary_echo_states(replace(check_cfg, levels=levels), cycles=cycles)
            controls[f"check_{name}_{levels}"] = np.diagonal(
                rho, axis1=-2, axis2=-1
            ).real
        for n1, n2 in [(3, 4), (4, 5)]:
            validation[f"{name}_{n1}_vs_{n2}_max_Pe_change_by_amplitude"] = abs(
                controls[f"check_{name}_{n1}"][..., 1]
                - controls[f"check_{name}_{n2}"][..., 1]
            ).max(axis=1).tolist()
    np.savez_compressed(path, **controls)
    (output / "additional_validation.json").write_text(
        json.dumps(validation, indent=2)+"\n"
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), layout="constrained")
    for ax, i, amp in zip(axes.flat, range(len(SLICES)), SLICES, strict=True):
        for name, color in zip(("square", "root"), COLORS[:2], strict=True):
            ax.plot(wide_det, controls[f"wide_{name}"][i, :, 1], color=color,
                    label=name, lw=1.1)
        ax.axvspan(-.5, .5, color="gray", alpha=.15, label="Main comparison window")
        ax.set(xlabel="Drive − bare f01 (MHz)", ylabel="Pe", ylim=(0, .85),
                title=f"Peak Rabi {amp:g} MHz")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Wide-window control · same duration and peak drive · one inversion")
    fig.savefig(output / "wide_window_control.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    for ax, p, title in zip(
        axes, (data["dissipative3_root"], controls["fixed_repeated"],
               data["dissipative3_repeated"]),
        ("Root: fixed envelope", "Square: fixed 201 cycles",
         "Square: retuned cycle count"), strict=True,
    ):
        im = ax.pcolormesh(data["detuning_mhz"]*1000, data["amplitude_mhz"],
                           p[..., 1], vmin=0, vmax=.8, cmap="viridis", shading="nearest")
        ax.set(title=title, ylabel="Actual peak Rabi (MHz)", xlabel="Detuning (kHz)",
                yscale="log")
    fig.colorbar(im, ax=axes, label="Pe", shrink=.8)
    fig.suptitle("Timing calibration control · three-level model\n"
                 "201 cycles corresponds to π half-rotations at 10.05 MHz")
    fig.savefig(output / "timing_control.png")
    plt.close(fig)


def plot(data, rows, output, cfg):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 170})
    det, amps = data["detuning_mhz"], data["amplitude_mhz"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), layout="constrained")
    for ax, a in zip(axes.flat, SLICES, strict=True):
        idx = np.argmin(abs(amps-a))
        for name, label, color in zip(NAMES, LABELS, COLORS, strict=True):
            ax.plot(det*1000, data[f"dissipative3_{name}"][idx, :, 1],
                    color=color, label=label, lw=1.8 if name != "repeated" else 1.1,
                    alpha=1 if name != "repeated" else .7)
        ax.set(title=f"Peak Rabi {a:g} MHz", xlabel="Drive − bare f01 (kHz)",
               ylabel="Excited-state probability", xlim=(-500, 500), ylim=(0, .85))
        ax.grid(alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle("Same 20 μs duration and peak drive · three-level Lindblad model\n"
                 "No DRAG, edge window, or finite phase-transition time")
    fig.savefig(output / "matched_spectra.png")
    fig.savefig(output / "matched_spectra.svg")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4), layout="constrained")
    for ax, name, label in zip(axes[0], NAMES, LABELS, strict=True):
        im = ax.pcolormesh(det*1000, amps, data[f"dissipative3_{name}"][..., 1],
                           shading="nearest", cmap="viridis", vmin=0, vmax=.8,
                           rasterized=True)
        ax.set(title=label, xlabel="Detuning (kHz)", ylabel="Peak Rabi (MHz)",
               yscale="log", ylim=(amps[0], amps[-1]))
    fig.colorbar(im, ax=axes[0], label="Pe", shrink=.8)
    for name, label, color in zip(NAMES, LABELS, COLORS, strict=True):
        subset = [r for r in rows if r["model"] == "dissipative3"
                  and r["pulse"] == name]
        visible = [r for r in subset if r["resolved_visible_dip"]]
        # Repeated echoes have multiple fringes; don't label an arbitrary
        # fringe width as the width of one isolated spectroscopy dip.
        if name != "repeated":
            axes[1, 0].plot([r["amplitude_mhz"] for r in visible],
                            [r["fwhm_hz"]/1000 for r in visible],
                            color=color, label=label)
        axes[1, 1].plot(amps, [r["peak_to_peak"] for r in subset],
                        color=color, label=label)
        axes[1, 2].plot(amps, [r["sigma_f_khz_sqrt_shots"] for r in subset],
                        color=color, label=label)
    for ax in axes[1]:
        ax.set(xscale="log", xlabel="Peak Rabi (MHz)")
        ax.grid(alpha=.2)
    axes[1, 0].set(ylabel="Finite-window dip FWHM (kHz)",
                   title="Only dips with contrast ≥ 0.02")
    axes[1, 1].set(ylabel="Max Pe − min Pe in ±500 kHz", title="Spectral variation")
    axes[1, 2].set(ylabel="kHz × √shots (lower is better)", yscale="log",
                   title="Best local ideal-readout frequency scale")
    axes[1, 2].legend(fontsize=8)
    fig.suptitle("Matched peak drive · dissipative transmon · 20 μs\n"
                 "Repeated reference retunes cycle count at each nominal amplitude")
    fig.savefig(output / "comparison_summary.png")
    fig.savefig(output / "comparison_summary.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), layout="constrained")
    idx = np.argmin(abs(amps-10.05))
    for ax, model, title in zip(axes[:2], ("ideal2", "dissipative2"),
                                ("Ideal two-level", "Dissipative two-level"),
                                strict=True):
        for name, label, color in zip(NAMES, LABELS, COLORS, strict=True):
            ax.plot(det*1000, data[f"{model}_{name}"][idx, :, 1],
                    color=color, label=label, lw=1.3)
        ax.set(title=title+" · 10.05 MHz", xlabel="Detuning (kHz)", ylabel="Pe",
               xlim=(-500, 500), ylim=(0, 1.02))
    fine_amp = np.linspace(9.9, 10.1, 1001)
    pulse = EchoTheoryPulse(duration_us=cfg.duration_us, cutoff=cfg.cutoff)
    root_s, _ = root_curvature_times(pulse, fine_amp)
    square_s = 4*np.sin(np.pi*fine_amp*cfg.duration_us/2)**2/(2*np.pi*fine_amp)
    axes[2].semilogy(fine_amp, root_s**2, color=COLORS[1], label="Root")
    axes[2].semilogy(fine_amp, np.maximum(square_s**2, 1e-10),
                    color=COLORS[0], label="Square")
    axes[2].set(title="Ideal small-detuning response", xlabel="Peak Rabi (MHz)",
                ylabel="S² (μs²),  Pe ≈ (π δf S)²", ylim=(1e-8, 100))
    axes[2].legend()
    fig.suptitle("Controls: pulse-area blind spots are resolved analytically")
    fig.savefig(output / "physics_controls.png")
    plt.close(fig)


def write_report(cfg, data, rows, output):
    """Save the interpretation with definitions and validation boundaries."""
    pulse = EchoTheoryPulse(duration_us=cfg.duration_us, cutoff=cfg.cutoff)
    energy_ratio = (2*pulse.tau_us*np.arctan(cfg.duration_us/(2*pulse.tau_us))
                    / cfg.duration_us)
    scaled = SLICES*np.sqrt(energy_ratio)
    energy_cfg = replace(cfg, amplitude_mhz=tuple(scaled))
    energy_rho = rotary_echo_states(energy_cfg)
    energy_pop = np.diagonal(energy_rho, axis1=-2, axis2=-1).real
    np.savez_compressed(output / "equal_energy_control.npz",
                        root_amplitude_mhz=SLICES, square_amplitude_mhz=scaled,
                        detuning_mhz=data["detuning_mhz"], square_population=energy_pop)
    table = []
    for a in SLICES:
        selected = {r["pulse"]: r for r in rows if r["model"] == "dissipative3"
                    and abs(r["amplitude_mhz"]-a) < 1e-8}
        root, square = selected["root"], selected["square"]
        table.append(f"<tr><td>{a:g}</td><td>{root['fwhm_hz']/1000:.1f}</td>"
                     f"<td>{root['center_hz']/1000:.2f}</td>"
                     f"<td>{root['contrast']:.3f}</td>"
                     f"<td>{root['peak_to_peak']:.4f}</td>"
                     f"<td>{square['peak_to_peak']:.4f}</td></tr>")
    level = json.loads((output / "additional_validation.json").read_text())
    validation = json.loads((output / "validation.json").read_text())
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Square versus root-Lorentzian echo</title>
<style>body{{max-width:1150px;margin:32px auto;padding:0 24px;font:17px/1.6 system-ui;color:#203043}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:15px}}
td,th{{padding:8px;text-align:right;border-bottom:1px solid #ddd}}
th{{background:#edf3f8}}a{{color:#126aa2}}code{{font-size:14px}}details{{margin:20px 0}}
.lead{{font-size:21px}}.note{{background:#f2f5f8;padding:16px;border-radius:8px}}</style>
<h1>Square versus root-Lorentzian echo</h1>
<p class="lead">At matched duration and peak drive, the root-Lorentzian provides
a pronounced near-resonant dip over a broad drive range where a single square
echo becomes nearly flat. This supports an advantage for locating the resonance
without pulse-area tuning. It does not establish universally superior frequency sensitivity.</p>
<p>Both primary pulses last <b>20 μs</b> and have one abrupt midpoint π phase
inversion. The root envelope is 1/√(1+(t/τ)²), τ={pulse.tau_us:.8f} μs,
cutoff 0.005, order 1/2. No DRAG, edge window, finite sign-transition time, or
Stark compensation is applied. The 132 matched peak drives span 0.05–60 MHz.
The primary grid is ±0.5 MHz with 2.5 kHz spacing.</p>
<p>The three-level Duffing/Lindblad model uses anharmonicity −216 MHz,
T1=43.8 μs, and a phenomenological Markovian T2=7.6 μs
(Tφ={cfg.t_phi_us:.4f} μs). These are explicit comparison parameters,
not a new fit or a measurement. Ideal and dissipative two-level controls are saved too.</p>
<img src="matched_spectra.png" alt="Six matched-amplitude spectra">
<table><tr><th>Peak Rabi MHz</th><th>Root FWHM kHz</th><th>Root center kHz</th>
<th>Root dip depth</th><th>Root max−min Pe</th><th>Square max−min Pe</th></tr>
{''.join(table)}</table>
<p>At 3.05 MHz, the root dip is approximately 74 kHz wide and its center is
within 0.3 kHz of bare f01. The total signal variation in ±500 kHz is 0.399,
versus 0.0048 for the single square echo (about 83× larger). At 10.05 MHz the
root dip is about 191 kHz wide and its variation is about 160× larger.
At 1.05 MHz, both have similar total variation, but the root dip is approximately
18× narrower than the square dip under the stated finite-window definition.</p>
<p>The root pulse is not amplitude-independent: its width grows, contrast falls,
and center moves at large drive. At 59.95 MHz its center displacement is about
−38 kHz and the finite-window width is about 458 kHz. At small peak drive
(below roughly 1 MHz here), the root receives too little pulse area and can be
less useful than the square. The useful operating range depends on cutoff,
duration, and coherence.</p>
<p class="note">Widths and dip depths use the repository's finite-window
definition: raw minimum within ±250 kHz, a linear background fitted outside
±400 kHz, and nearest half-depth crossings. A plotted “visible” width additionally
requires depth ≥0.02 and at least six grid intervals. These are not universal
Lorentzian-fit linewidths. Max−min Pe is a separate, unambiguous signal-variation
measure over ±500 kHz, not a sensitivity or information gain.</p>
<img src="comparison_summary.png" alt="Population maps and metrics">
<p>Both unitary single echoes cancel exactly on resonance. Their difference is
the response to detuning. For a square echo,
<code>Pe = 4 Δ² Ω²/(Δ²+Ω²)² · sin⁴[T√(Δ²+Ω²)/4]</code>.
Its small-detuning response oscillates with pulse area and can vanish.
The root envelope replaces this strongly oscillating response with a much
smoother drive dependence. Integer MHz drives at 20 μs happen to be ideal-square
blind spots. The displayed slices instead use odd half-rotation multiples of π,
which maximize the square echo's ideal small-detuning curvature. The benefit
therefore does not come from choosing its coherent blind spots.</p>
<img src="physics_controls.png" alt="Two-level and analytical curvature controls">
<p>The extra purple reference is a repeated square rotary echo. It has
N=max(1, floor(fR T+0.5)) complete cycles, each containing equal positive and
negative segments. This preserves duration and peak drive but retunes the
cycle count at every nominal amplitude. At the displayed amplitudes, each
half-rotation is exactly π. It can provide steeper local fringes and better
local ideal-readout sensitivity than the root pulse; the root's simpler central
dip and amplitude range are different advantages. Repeated high-drive curves
are a three-level illustration, not converged performance predictions.</p>
<p>The local information metric is max[(dPe/df)²/(Pe(1−Pe))], excluding two
endpoint samples and probabilities within 10⁻⁴ of 0 or 1. Its inverse square
root is the plotted kHz × √shots scale. It assumes known amplitude and an
already located operating point with perfect binary Pe readout. Unknown-amplitude
nuisance parameters, readout errors, reset overhead, drift, and the cost of
finding the appropriate fringe are excluded. No experimental sensitivity gain is claimed.</p>
<img src="timing_control.png" alt="Fixed and retuned repeated echo timing">
<p>The fixed-201-cycle control holds its timing while amplitude varies; its π
half-rotation calibration is at 10.05 MHz. This distinguishes waveform robustness
from a comparator retuned separately for every drive. All maps sample a discrete
amplitude grid; analytical curvature, rather than that coarse grid, resolves the
single square echo's rapid pulse-area oscillations.</p>
<img src="wide_window_control.png" alt="Wide frequency control">
<p>The ±10 MHz control shows the broad structure and fringes of the square echo
outside the main near-resonance window. The square response is not globally flat.
At equal peak drive, the ratio of integrated Ω² is
<b>{energy_ratio:.6f}</b>: the root uses {100*energy_ratio:.3f}% of the square's
drive-energy proxy. Matching peak amplitude therefore does not match energy,
RMS drive, or pulse area. An additional equal-energy square control, with peak
amplitude multiplied by {np.sqrt(energy_ratio):.6f}, is saved in
<a href="equal_energy_control.npz">equal_energy_control.npz</a>.</p>
<details open><summary>Numerical verification and limits</summary>
<p>Exact matrix exponentials propagate all square segments. The unitary square
formula agrees to {validation['square_unitary_max_analytic_error']:.2g} in Pe.
Halving the root RK4 step from 0.5 to 0.25 ns changes any tested population by at
most {validation['root_half_step_max_population_error']:.2g}. Halving the detuning
spacing changes tested root widths by at most 165 Hz and local Fisher information
by 0.57%. A separate segment-exponential root calculation resolves the envelope
in asinh(t/τ); its final four-level refinement changes populations by
{level['root_four_level_refinement_max_population_change']:.2g}.</p>
<p>At tested detunings −500, −100, 0, 100, and 500 kHz, adding a fourth level
changes root Pe by at most 3.0×10⁻⁵ up to 20.05 MHz and 0.00111 at 59.95 MHz.
The square reference has stronger high-drive truncation effects: the single
square 3→4-level change reaches 0.0129, and the repeated echo reaches 0.116;
its 4→5-level change can still reach 0.0324. The central comparison through
20 MHz is much more reliable for the single square than high-drive repeated
reference predictions. Abrupt phase inversions and Markovian noise are model
assumptions; hardware bandwidth and colored noise have not been simulated.</p>
<p>The independent native square check differs from the exact exponential by
0.0021 at its worst tested point. Exact exponentials, not native RK4, supply
every square result. An initial uniform-envelope four-level root check was
under-resolved; the converged transformed-grid check above supersedes it.
See <a href="validation.json">initial validation</a> and
<a href="additional_validation.json">converged level checks</a> for all values.</p>
</details>
<p><a href="metrics.csv">All metrics CSV</a> · <a href="responses.npz">Response arrays</a> ·
<a href="configuration.json">Simulation configuration and original source hashes</a> ·
<a href="postprocessing_provenance.json">Postprocessing source and array hashes</a></p>
<p>Software checks for this change: 30 focused tests passed using the simulation
virtual environment; Ruff passed on all three added source/test files. Full-suite
collection was blocked by missing optional torch. With that test module excluded,
204 tests passed and one unrelated Figure 3 statistics test failed. Repository-wide
lint reported 692 issues outside the added files. See
<a href="software_validation.json">software validation details</a>.</p>
<p>Reproduce from the repository root:</p>
<pre>PYTHONPATH=. .venv/bin/python scripts/compare_square_root_echo.py</pre>
<p>Use <code>--reuse</code> to regenerate plots and this report from saved main responses.</p>
</html>"""
    (output / "report.html").write_text(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/square_root_echo_comparison_20us")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    amps = np.unique(np.round(np.r_[np.geomspace(.05, 60., 121), SLICES,
                                    [1., 3., 10., 20., 40., 60.]], 12))
    cfg = SimulationConfig(
        levels=3, amplitude_mhz=tuple(amps),
        detuning_mhz=tuple(np.linspace(-.5, .5, 401)), pulse_type="square",
        duration_us=20., cutoff=.005, order=.5, t1_us=43.8, t2_us=7.6,
        anharmonicity_mhz=-216., num_steps_per_half=20000,
    )
    if args.reuse:
        saved_config = json.loads((output / "configuration.json").read_text())[
            "configuration"
        ]
        if json.dumps(saved_config, sort_keys=True) != json.dumps(
            cfg.__dict__, sort_keys=True
        ):
            raise ValueError("Saved configuration differs; run without --reuse")
        with np.load(output / "responses.npz") as saved:
            data = dict(saved)
    else:
        pulse = EchoTheoryPulse(duration_us=cfg.duration_us, cutoff=cfg.cutoff)
        paths = [Path(__file__), ROOT / "echospec/simulation/rotary_echo.py",
                 ROOT / "echospec/simulation/echo_theory.py",
                 ROOT / "echospec/simulation/qutrit.py",
                 ROOT / "echospec/simulation/qutrit_rk4.cpp",
                 ROOT / "echospec/simulation/piecewise.py",
                 ROOT / "echospec/analysis/dip_metrics.py"]
        provenance = dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            git_head=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                              cwd=ROOT, text=True).strip(),
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(
                p.read_bytes()).hexdigest() for p in paths},
            configuration=cfg.__dict__,
            root_tau_us=pulse.tau_us,
            root_to_square_integrated_drive_squared=(
                2*pulse.tau_us*np.arctan(cfg.duration_us/(2*pulse.tau_us))
                / cfg.duration_us
            ),
            repeated_cycle_rule="max(1, floor(peak_rabi_MHz * duration_us + 0.5))",
            pulse_conditions="Abrupt sign inversions; no DRAG, edge window or Stark correction",
            model="Rotating-wave Duffing Hamiltonian; constant Markovian T1/T2",
            sensitivity="Known local operating point and amplitude; perfect Pe readout; no overhead",
            frequency_window_mhz=[-.5, .5],
            scope="Numerical benchmark only; no acquisition or manuscript modification",
        )
        (output / "configuration.json").write_text(json.dumps(provenance, indent=2)+"\n")
        data = build_data(cfg, output)
    rows = metrics(data, output)
    plot(data, rows, output, cfg)
    extra_controls(cfg, data, output, force=not args.reuse)
    write_report(cfg, data, rows, output)
    source_paths = [Path(__file__), ROOT / "echospec/simulation/rotary_echo.py",
                    ROOT / "tests/test_rotary_echo.py"]
    (output / "postprocessing_provenance.json").write_text(json.dumps(dict(
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in source_paths},
        response_sha256=hashlib.sha256((output / "responses.npz").read_bytes()).hexdigest(),
        command="PYTHONPATH=. .venv/bin/python scripts/compare_square_root_echo.py"
    ), indent=2)+"\n")
    print(f"Saved comparison in {output}", flush=True)


if __name__ == "__main__":
    main()
