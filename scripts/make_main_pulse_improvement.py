"""PRL Figure 3: matched echo, DRAG, and polynomial-envelope simulations.

Reuses the validated maps documented in paper/notes/POLYNOMIAL_ECHO_PULSE.md.
Run with PYTHONPATH=. MPLBACKEND=Agg python scripts/make_main_pulse_improvement.py.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from echospec.analysis.dip_metrics import dip_metrics
from echospec.figures import FigureVariant, apply_figure_style, save_figure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/paper"
STEM = "03_pulse_improvement"
CASES = (
    ("echo", "Echo", "outputs/polynomial_echo_note_2d/c0.001_hard_beta0.npz", 0.0),
    ("drag", "+ DRAG", "outputs/polynomial_power_30us_scan/c0.001_hard.npz", -0.5),
    ("polynomial", "+ polynomial", "outputs/optimized_echo_tail/polynomial5_full.npz", -0.5),
)
COLORS = ("#555555", "#0072b2", "#d55e00")
LINESTYLES = ("--", "-.", "-")
COMMON = {
    "duration_us": 30.0,
    "anharmonicity_mhz": -216.0,
    "echo": True,
    "echo_transition_shape": "sine",
    "echo_transition_us": 0.016,
    "order": 0.5,
    "stark_kappa_mhz_inv": 0.0,
    "t1_us": 51.24,
    "t_phi_us": 7.871480508563624,
}


def center_stability(amps, center_hz):
    """Summarize finite centers at 10--100 MHz, with RMS referenced to zero."""
    amps, center_hz = np.asarray(amps), np.asarray(center_hz)
    mask = (amps >= 10) & (amps <= 100) & np.isfinite(center_hz)
    if not mask.any():
        raise ValueError("No finite resonance centers in the 10--100 MHz range")
    selected = center_hz[mask]
    return {
        "sample_count": int(mask.sum()),
        "rms_from_zero_hz": float(np.sqrt(np.mean(selected ** 2))),
        "mean_hz": float(np.mean(selected)),
        "standard_deviation_hz": float(np.std(selected)),
    }


def main():
    amps, det = np.linspace(0, 100, 100), np.linspace(-0.5, 0.5, 201)
    prior_path = OUT / "polynomial_echo_note/sweeps_2d_provenance.json"
    prior = json.loads(prior_path.read_text())
    populations, configs, summaries, hashes, rows = {}, {}, {}, {}, []
    for key, _, source, beta in CASES:
        path = ROOT / source
        hashes[source] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashes[source] == prior["source_sha256"][source], source
        with np.load(path) as data:
            cfg = json.loads(str(data["config"]))
            for name, value in dict(COMMON, drag_beta=beta).items():
                assert cfg[name] == value, (key, name)
            assert cfg.get("cutoff", cfg.get("cutoff_core")) == 0.001
            dt = cfg.get("dt_ns")
            if dt is None:
                dt = 1000 * cfg["duration_us"] / (2 * cfg["num_steps_per_half"])
            assert dt == 0.5
            if key == "polynomial":
                assert cfg["family"] == "polynomial5" and cfg["parameters"] is None
            else:
                assert cfg["edge_window_kind"] == "none"
            np.testing.assert_array_equal(data["rabi_mhz"], amps)
            np.testing.assert_array_equal(data["detuning_mhz"], det)
            p = data["populations"].copy()
        assert p.shape == (3, 100, 201) and np.isfinite(p).all()
        assert p.min() >= 0 and p.max() <= 1
        np.testing.assert_allclose(p.sum(axis=0), 1, atol=1e-9)
        populations[key], configs[key] = p, cfg
        local = [
            dict(case=key, amplitude_mhz=float(a), **dip_metrics(det, y))
            for a, y in zip(amps, p[1], strict=True)
        ]
        assert not local[0]["accepted"]
        assert all(r["accepted"] for r in local[10:])
        assert all(0 <= r["contrast"] <= 1 for r in local if r["accepted"])
        summaries[key] = {
            "at_100_mhz": local[-1],
            "minimum_contrast_10_to_100_mhz": min(r["contrast"] for r in local[10:]),
            "maximum_fwhm_khz_10_to_100_mhz": max(
                r["fwhm_hz"] / 1000 for r in local[10:]
            ),
        }
        center_hz = np.array([r["center_hz"] for r in local])
        summaries[key]["center_stability_10_to_100_mhz"] = center_stability(
            amps, center_hz
        )
        rows.extend(local)

    apply_figure_style(FigureVariant.PAPER)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6,
        "axes.grid": False, "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.bbox": None,
    })
    width, height = 3.38, 4.77
    fig = plt.figure(figsize=(width, height))
    t2_us = 1 / (1 / (2 * COMMON["t1_us"]) + 1 / COMMON["t_phi_us"])
    coherence_fwhm_khz = 1000 / (np.pi * t2_us)
    coherence_hwhm_mhz = coherence_fwhm_khz / 2000

    def axes(x, y, w, h):
        ax = fig.add_axes([x / width, y / height, w / width, h / height])
        ax.tick_params(direction="in", top=True, right=True, length=2, pad=2)
        return ax

    for index, (key, title, _, beta) in enumerate(CASES):
        ax = axes(0.4 + index * 0.96, 3.18, 0.91, 1.12)
        mesh = ax.pcolormesh(
            det, amps, populations[key][1], shading="nearest", cmap="magma",
            vmin=0, vmax=0.8, rasterized=True,
        )
        for sign in (-1, 1):
            ax.axvline(
                sign * coherence_hwhm_mhz,
                color="white", lw=0.45, ls=(0, (3, 15)),
            )
        ax.set(xlim=(-0.5, 0.5), ylim=(0, 100), xticks=[-0.4, 0, 0.4],
               yticks=[0, 50, 100])
        ax.tick_params(labelleft=index == 0)
        if index == 0:
            ax.set_ylabel(r"$\Omega_0/2\pi$ (MHz)", labelpad=2)
        ax.set_title(title, color=COLORS[index], pad=4)
        ax.text(0.06, 0.94, f"({chr(97 + index)})", transform=ax.transAxes,
                va="top", color="white", fontsize=7,
                path_effects=[pe.withStroke(linewidth=0.65, foreground="0.2")])
        ax.text(0.94, 0.08, rf"$\beta={beta:g}$", transform=ax.transAxes,
                ha="right", color="0.15", fontsize=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))
    fig.text(1.82 / width, 2.93 / height, r"$\Delta/2\pi$ (MHz)", ha="center")
    cb_ax = fig.add_axes([1.225 / width, 4.57 / height, 1.18 / width, 0.055 / height])
    cb = fig.colorbar(mesh, cax=cb_ax, orientation="horizontal", ticks=[0, 0.4, 0.8])
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    fig.text(2.485 / width, 4.60 / height, r"$P_e$", va="center")

    center_ax = axes(0.4, 1.82, 2.82, 0.95)
    fwhm_ax = axes(0.4, 0.42, 1.08, 1.03)
    contrast_ax = axes(2.14, 0.42, 1.08, 1.03)
    center_ax.axhline(0, color="0.75", lw=0.5, zorder=0)
    for index, (key, _, _, _) in enumerate(CASES):
        local = [r for r in rows if r["case"] == key]
        style = dict(color=COLORS[index], ls=LINESTYLES[index], lw=1.0)
        contrast_ax.plot(amps, [r["contrast"] for r in local], **style)
        fwhm_ax.plot(amps, [r["fwhm_hz"] / 1000 for r in local], **style)
        center_ax.plot(amps, [r["center_hz"] / 1000 for r in local], **style)
    center_ax.set(ylabel=r"$\Delta_{\mathrm{res}}/2\pi$ (kHz)",
                  ylim=(-22, 22), yticks=[-20, 0, 20])
    contrast_ax.set(ylabel="Contrast", ylim=(0, 1), yticks=[0, 0.5, 1])
    fwhm_ax.set(ylabel="FWHM (kHz)", ylim=(0, 550), yticks=[0, 250, 500])
    normalized_axis = fwhm_ax.secondary_yaxis(
        "right", functions=(lambda value: value / coherence_fwhm_khz,
                            lambda value: value * coherence_fwhm_khz)
    )
    normalized_axis.set_ylabel(r"FWHM/$\Gamma_{T_2}$", fontsize=7, labelpad=2)
    normalized_axis.set_yticks([0, 4, 8, 12])
    normalized_axis.tick_params(direction="in", length=2, pad=2, labelsize=6)
    fwhm_ax.tick_params(right=False)
    fwhm_ax.axhline(coherence_fwhm_khz, color="0.5", ls=":", lw=0.65)
    fwhm_ax.text(95, 63, r"$1/(\pi T_2)$", ha="right", color="0.35", fontsize=6)
    for ax, label in ((center_ax, "(d)"), (fwhm_ax, "(e)"), (contrast_ax, "(f)")):
        ax.set(xlim=(0, 100), xticks=[0, 50, 100], xlabel=r"$\Omega_0/2\pi$ (MHz)")
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
        ax.text(0.94 if ax is contrast_ax else 0.04, 0.93, label, va="top",
                ha="right" if ax is contrast_ax else "left", transform=ax.transAxes)
    handles = [
        Line2D([], [], color=color, ls=style, lw=1)
        for color, style in zip(COLORS, LINESTYLES, strict=True)
    ]
    fig.legend(handles, ["(a) Echo", "(b) + DRAG", "(c) + polynomial"],
               loc="lower center", bbox_to_anchor=(0.53, 0), ncol=3, frameon=False,
               fontsize=6, handlelength=1.5, columnspacing=0.8, handletextpad=0.4)
    paths = save_figure(fig, STEM, variant=FigureVariant.PAPER,
                        formats=("pdf", "png", "svg"), dpi=350)
    plt.close(fig)
    with (OUT / f"{STEM}_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(OUT / f"{STEM}.npz", rabi_mhz=amps, detuning_mhz=det,
                        **populations,
                        **{f"{key}_center_hz": np.array([
                            r["center_hz"] for r in rows if r["case"] == key
                        ]) for key, _, _, _ in CASES},
                        center_rms_mask=(amps >= 10) & (amps <= 100))
    for source in ("scripts/make_main_pulse_improvement.py",
                   "echospec/analysis/dip_metrics.py",
                   str(prior_path.relative_to(ROOT))):
        hashes[source] = hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
    provenance = {
        "generator": "scripts/make_main_pulse_improvement.py",
        "included_in_manuscript": True, "source_configs": configs,
        "source_sha256": hashes, "figure_size_inches": [width, height],
        "color_limits": [0, 0.8], "row_normalization": False,
        "population_order": ["ground", "excited", "second_excited"],
        "summaries": summaries,
        "resonance_center": {
            "panel": "d",
            "definition": "Three-point quadratic interpolation of the raw Pe "
            "minimum within +/-0.25 MHz, as returned by dip_metrics",
            "display": "Raw interpolated centers; no smoothing, detrending, "
            "or subtraction of a case-dependent mean",
            "rms_definition": "sqrt(mean(center_hz**2)) relative to zero detuning",
            "rms_amplitude_range_mhz": [10, 100],
            "detuning_grid_spacing_khz": float(1000 * (det[1] - det[0])),
            "limitation": "Sub-grid interpolation is not an independent "
            "validation of sub-kHz numerical accuracy",
        },
        "coherence_reference": {
            "panels": ["a", "b", "c"],
            "t2_us": t2_us,
            "detuning_mhz": [-coherence_hwhm_mhz, coherence_hwhm_mhz],
            "fwhm_khz": coherence_fwhm_khz,
            "definition": "Vertical lines at +/-1/(2*pi*T2); "
            "separation is the coherence-limited FWHM 1/(pi*T2)",
            "color": "white",
            "linewidth_pt": 0.45,
            "linestyle": "dashed",
            "dash_pattern": [3, 15],
        },
        "secondary_fwhm_axis": {"t2_us": t2_us,
                                "reference_fwhm_khz": coherence_fwhm_khz,
                                "definition": "FWHM / [1/(pi*T2)]"},
        "contrast": "Dip depth in excited-state probability relative to the fitted "
                    "linear background at the dip; dimensionless, plotted on [0, 1]",
        "fwhm": "Nearest half-depth crossings relative to a linear background "
                "at 0.4 <= |detuning| <= 0.5 MHz; zero-drive metrics omitted",
        "comparison": "All cases have a 16 ns sine midpoint transition. "
                      "(a) to (b) isolates DRAG; (b) to (c) adds cutoff subtraction "
                      "and the p=5 polynomial window. No area or peak rematching.",
        "validation": "Source NPZ hashes match the existing note's provenance; "
                      "parameters, grids, normalization, and metric validity checked.",
    }
    (OUT / f"{STEM}_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summaries, indent=2))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
