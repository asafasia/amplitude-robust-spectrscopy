"""Add matched 2D pulse comparisons to the polynomial echo research note.

Run with the repository Python environment. Reuses the polynomial and C2 maps and
computes the hard-cutoff beta=0 map, including a timestep refinement check.
"""

from __future__ import annotations

import csv
import hashlib
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
from echospec.analysis.dip_metrics import dip_metrics  # noqa: E402
from scripts import make_echo_edge_ramp_comparison as base  # noqa: E402
from scripts.echo_sweep_html import T2_FWHM_KHZ, write_interactive_sweeps  # noqa: E402
from scripts.plot_polynomial_echo_tradeoff import COMMON  # noqa: E402

OUT = ROOT / "outputs/polynomial_echo_note_2d"
FIG = ROOT / "figures/paper/polynomial_echo_note"
POLYNOMIAL = ROOT / "outputs/optimized_echo_tail/polynomial5_full.npz"
C2 = ROOT / "outputs/optimized_echo_tail/c2_full.npz"
HARD_DRAG = ROOT / "outputs/polynomial_power_30us_scan/c0.001_hard.npz"


def half_depth_points(x, y):
    """Locate actual half-depth crossings, retaining the existing width convention."""
    metric = dip_metrics(x, y)
    if not metric["accepted"]:
        return np.nan, np.nan, metric
    central = np.flatnonzero(abs(x) <= .25)
    edges = abs(x) >= .4
    depth = np.polyval(np.polyfit(x[edges], y[edges], 1), x) - y
    peak = central[np.argmax(depth[central])]
    half = depth[peak] / 2
    li = np.flatnonzero(depth[:peak] <= half)[-1]
    ri = np.flatnonzero(depth[peak + 1:] <= half)[0] + peak + 1
    left = np.interp(half, depth[li:li + 2], x[li:li + 2])
    right = np.interp(half, depth[ri - 1:ri + 1][::-1], x[ri - 1:ri + 1][::-1])
    np.testing.assert_allclose((right - left) * 1e6, metric["fwhm_hz"],
                               rtol=1e-12, atol=1e-7)
    return float(left * 1000), float(right * 1000), metric


def load_checked(path, beta):
    with np.load(path) as data:
        config = json.loads(str(data["config"]))
        for key, value in dict(COMMON, drag_beta=beta).items():
            assert config[key] == value, (path, key)
        assert config.get("cutoff", config.get("cutoff_core")) == .001
        dt = (config["dt_ns"] if "dt_ns" in config else
              1000 * config["duration_us"] / (2 * config["num_steps_per_half"]))
        assert dt == .5
        np.testing.assert_array_equal(data["rabi_mhz"], base.AMPS)
        np.testing.assert_array_equal(data["detuning_mhz"], base.DETUNING)
        p = data["populations"].copy()
    assert p.shape == (3, 100, 201)
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    np.testing.assert_allclose(p.sum(axis=0), 1., atol=1e-9)
    np.testing.assert_allclose(p[:, 0], np.tile([[1.], [0.], [0.]], (1, 201)))
    return p, config


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    base.configure_experiment(30., OUT)
    hard_key = base.run_case(.001, "hard", "none", 0., beta=0.)
    hard_path = OUT / f"{hard_key}.npz"
    polynomial, pcfg = load_checked(POLYNOMIAL, -.5)
    c2, ccfg = load_checked(C2, -.5)
    hard, hcfg = load_checked(hard_path, 0.)
    hard_drag, dcfg = load_checked(HARD_DRAG, -.5)
    assert pcfg["family"] == "polynomial5"
    assert ccfg["family"] == "c2"
    assert hcfg["edge_window_kind"] == "none"
    assert dcfg["edge_window_kind"] == "none"
    indices = [10, 50, 99]
    refine_key = base.run_case(
        .001, "hard", "none", 0., beta=0., amplitudes=base.AMPS[indices],
        steps=60000, suffix="_refine")
    with np.load(OUT / f"{refine_key}.npz") as data:
        difference = float(np.max(np.abs(data["populations"] - hard[:, indices])))
    assert difference < 1e-4, difference
    prior = ROOT / "outputs/echo_edge_ramp_30us_comparison/c0.001_hard_beta0_beta_control.npz"
    with np.load(prior) as data:
        old_indices = [int(np.flatnonzero(base.AMPS == a)[0])
                       for a in data["rabi_mhz"]]
        prior_difference = float(np.max(abs(data["populations"] - hard[:, old_indices])))
    assert prior_difference < 1e-10, prior_difference
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True,
                             layout="constrained")
    rows = []
    cases = ["hard_beta0", "hard_beta-0.5", "polynomial5", "c2"]
    for ax, p, case, title in zip(axes.flat, [hard, hard_drag, polynomial, c2], cases, [
            "Hard cutoff (no envelope correction)\nDRAG β = 0",
            "Hard cutoff (no envelope correction)\nDRAG β = −0.5",
            "Polynomial p = 5 + cutoff subtraction\nDRAG β = −0.5",
            "C² subtraction\nDRAG β = −0.5"], strict=True):
        mesh = ax.pcolormesh(base.DETUNING * 1000, base.AMPS, p[1],
                             shading="nearest", cmap="viridis", vmin=0, vmax=1,
                             rasterized=True)
        ax.set(title=title, xlabel="Drive detuning (kHz)", xlim=(-500, 500),
               ylim=(0, 100))
        ax.set_xticks([-500, -250, 0, 250, 500])
        crossings = []
        for amp, spectrum in zip(base.AMPS, p[1], strict=True):
            left, right, metric = half_depth_points(base.DETUNING, spectrum)
            crossings.append([left, right])
            rows.append(dict(case=case, amplitude_mhz=float(amp),
                             left_half_depth_khz=left, right_half_depth_khz=right,
                             **metric))
        crossings = np.asarray(crossings)
        for column in range(2):
            ax.scatter(crossings[:, column], base.AMPS, s=9, facecolors="white",
                       edgecolors="#333333", linewidths=.25, zorder=3,
                       label="Half-depth crossings" if column == 0 else None)
        for direction in [-1, 1]:
            ax.axvline(direction * T2_FWHM_KHZ / 2, color="#ff6b9d", ls="--",
                       lw=1.2, zorder=4,
                       label=(f"T₂ reference FWHM = {T2_FWHM_KHZ:.2f} kHz"
                              if direction == -1 else None))
    for ax in axes[:, 0]:
        ax.set_ylabel("Nominal Rabi amplitude A (MHz)")
    for ax in axes[0]:
        ax.set_xlabel("")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center",
               frameon=False, markerscale=1.5, ncol=2)
    with (FIG / "sweeps_2d_fwhm_points.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig.colorbar(mesh, ax=axes, label="Final excited-state population Pₑ",
                 ticks=[0, .2, .4, .6, .8, 1], shrink=.94)
    fig.suptitle("2D echo spectroscopy: envelope correction and DRAG comparison\n"
                 "30 µs · cutoff 0.001 · 16 ns transition · 100 amplitude points",
                 fontsize=14)
    fig.savefig(FIG / "sweeps_2d.png", dpi=180)
    fig.savefig(FIG / "sweeps_2d.pdf")
    plt.close(fig)
    write_interactive_sweeps(FIG / "sweeps_2d.html", base.AMPS, base.DETUNING,
                             [hard, hard_drag, polynomial, c2], rows, cases)
    sources = [POLYNOMIAL, C2, HARD_DRAG, hard_path, OUT / f"{refine_key}.npz", prior,
               Path(__file__).resolve(),
               ROOT / "scripts/echo_sweep_html.py",
               ROOT / "echospec/analysis/dip_metrics.py",
               ROOT / "scripts/make_echo_edge_ramp_comparison.py",
               ROOT / "echospec/simulation/qutrit.py",
               ROOT / "echospec/simulation/qutrit_native.py",
               ROOT / "echospec/simulation/qutrit_rk4.cpp"]
    provenance = dict(
        generated_date="2026-10-13", polynomial_config=pcfg, hard_config=hcfg,
        c2_config=ccfg, hard_drag_config=dcfg, panel_order=cases,
        amplitude_points=100, detuning_points=201,
        population_order=["ground", "excited", "second_excited"],
        polynomial_source=str(POLYNOMIAL.relative_to(ROOT)),
        c2_source=str(C2.relative_to(ROOT)),
        hard_drag_source=str(HARD_DRAG.relative_to(ROOT)),
        hard_source=str(hard_path.relative_to(ROOT)),
        color_scale=[0, 1], row_normalization=False,
        t2_reference=dict(t2_us=7.31, fwhm_khz=T2_FWHM_KHZ,
                          line_detunings_khz=[-T2_FWHM_KHZ / 2, T2_FWHM_KHZ / 2]),
        interactive_html="sweeps_2d.html",
        interactive_color_scale=dict(vmin=0, vmax_default=1,
                                    vmax_range=[.05, 1], step=.01),
        refinement=dict(amplitudes_mhz=base.AMPS[indices].tolist(),
                        original_dt_ns=.5, refined_dt_ns=.25,
                        maximum_population_change=difference),
        prior_control_maximum_population_change=prior_difference,
        half_depth_points=dict(
            method="Nearest half-depth crossings after linear background subtraction",
            background_region_mhz="0.4 <= |detuning| <= 0.5",
            valid_rows={case: sum(r["accepted"] for r in rows if r["case"] == case)
                        for case in cases},
            width_consistency="All crossing separations checked against dip_metrics",
            zero_drive="No dip; no half-depth markers"),
        interpretation=("Hard cutoff at both beta=0 and beta=-0.5; polynomial and "
                        "C2 at beta=-0.5 isolate envelope correction effects."),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(
            p.read_bytes()).hexdigest() for p in sources},
        reproduce="python scripts/plot_polynomial_echo_2d.py")
    (FIG / "sweeps_2d_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance["refinement"], indent=2), flush=True)
    print(FIG / "sweeps_2d.png", flush=True)


if __name__ == "__main__":
    main()
