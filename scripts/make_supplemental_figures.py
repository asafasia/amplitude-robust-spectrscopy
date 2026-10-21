"""Render the supplemental figures in the extended cutoff map's common style.

Uses saved arrays and fitted metrics, without new simulation or fitting.
Reproduce with `make paper-figures-supplemental PYTHON=.venv/bin/python`.
Supplement-only assets leave the main-paper and historical report figures intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.figures.supplemental import (  # noqa: E402
    FIGURE_WIDTH,
    STYLE_PATH,
    apply_supplemental_style,
)

OUT = ROOT / "figures/paper/supplemental"
SOURCES: dict[str, str] = {}
OUTPUTS: list[str] = []
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7D5BA6", "#8C6D31", "#303030"]


def source_path(relative: str) -> Path:
    path = ROOT / relative
    SOURCES[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def load(relative: str) -> dict[str, np.ndarray]:
    with np.load(source_path(relative), allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def save(fig: plt.Figure, stem: str) -> None:
    """Save fixed-width canvases; tight cropping would change printed type sizes."""
    for extension in ("pdf", "png", "svg"):
        path = OUT / f"{stem}.{extension}"
        fig.savefig(path, dpi=300)
        OUTPUTS.append(str(path.relative_to(ROOT)))
    plt.close(fig)


def normalize_existing(fig: plt.Figure) -> None:
    """Bring reused schematic/plot artists into the same typography and axes."""
    apply_supplemental_style()
    for text in fig.findobj(Text):
        text.set_fontfamily("DejaVu Sans")
        text.set_math_fontfamily("dejavusans")
        text.set_fontsize(8)
        text.set_fontweight("normal")
    for ax in fig.axes:
        for label in (ax.xaxis.label, ax.yaxis.label, ax.title):
            label.set_fontsize(9)
        ax.tick_params(which="major", direction="out", length=3.5, width=0.6)
        ax.minorticks_off()
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)


def schematic() -> None:
    from scripts.make_cryogenic_measurement_diagram import (
        BLUE,
        GRAY,
        GREEN,
        INK,
        build_figure,
        circulator,
        hemt,
        low_pass_filter,
    )

    source_path("scripts/make_cryogenic_measurement_diagram.py")
    fig = build_figure()
    normalize_existing(fig)
    fig.set_size_inches(FIGURE_WIDTH, 6.2)
    fig.axes[0].set_position([0.21, 0.02, 0.78, 0.96])
    fig.axes[0].set_xlim(-0.4, 10.2)
    for text in fig.axes[0].texts:
        if text.get_text() == "300 MHz":
            text.set_fontsize(7.5)
    ax = fig.add_axes([0.015, 0.30, 0.175, 0.55])
    ax.set(xlim=(0, 2.1), ylim=(0, 5.7), aspect="equal")
    ax.axis("off")
    ax.text(0.05, 5.65, "Component\nkey", fontsize=9, va="top")
    # Match the symbols and colors actually used in the wiring diagram.
    x = 0.3
    rows = [4.6, 3.8, 3.0, 2.2, 1.4, 0.6]
    ax.add_patch(
        Rectangle(
            (x - 0.105, rows[0] - 0.20),
            0.21,
            0.40,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=1,
        )
    )
    low_pass_filter(ax, x, rows[1], label=None)
    circulator(ax, x, rows[2])
    hemt(ax, x, rows[3])
    ax.add_patch(
        FancyArrowPatch(
            (x - 0.24, rows[4]),
            (x + 0.24, rows[4]),
            arrowstyle="-|>",
            mutation_scale=8,
            color=INK,
            linewidth=1.0,
        )
    )
    ax.plot([x - 0.24, x + 0.24], [rows[5], rows[5]], color=GREEN, lw=1.1)
    labels = [
        "Attenuator\n(dB)",
        "Low-pass\nfilter",
        "Circulator",
        "HEMT\namplifier",
        "RF signal\npath",
        "DC bias",
    ]
    for y, label in zip(rows, labels, strict=True):
        ax.text(x + 0.36, y, label, fontsize=8, va="center", color=INK)
    # Fit a square-corner frame to the key with equal padding on every side.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = Bbox.union(
        [artist.get_window_extent(renderer)
         for artist in [*ax.texts, *ax.patches, *ax.lines]]
    ).padded(5 * fig.dpi / 72).transformed(fig.transFigure.inverted())
    fig.add_artist(
        Rectangle(
            (bounds.x0, bounds.y0),
            bounds.width,
            bounds.height,
            transform=fig.transFigure,
            facecolor="none",
            edgecolor=GRAY,
            linewidth=0.7,
            zorder=0,
        )
    )
    save(fig, "s01_measurement_chain")


def duration_metrics() -> None:
    from scripts.make_duration_resolution_comparison import _build_figure

    source_path("scripts/make_duration_resolution_comparison.py")
    data = load("figures/paper/03_echo_duration_resolution_comparison.npz")
    collections = []
    for prefix in ("clean", "measured"):
        collections.append(
            {
                float(length): {
                    name: data[f"{prefix}_{name}_{int(length)}us"]
                    for name in ("fwhm_t2_units", "contrast", "product")
                }
                for length in data["durations_us"]
            }
        )
    measurements = {
        float(length): SimpleNamespace(
            rabi_mhz=data[f"measured_rabi_mhz_{int(length)}us"]
        )
        for length in data["durations_us"]
    }
    fig = _build_figure(
        *collections,
        measurements,
        data["constant_fwhm_t2_units"],
        data["constant_contrast"],
        data["constant_product"],
    )
    normalize_existing(fig)
    fig.set_size_inches(FIGURE_WIDTH, 5.5)
    for index, ax in enumerate(fig.axes):
        for text in list(ax.texts):
            if text.get_text() in ("(a)", "(b)", "(c)"):
                text.remove()
        ax.set_title(f"({chr(97 + index)})", loc="left")
        legend = ax.get_legend()
        if legend is not None:
            legend.set_loc("upper left")
            legend.set_frame_on(False)
    save(fig, "s03_duration_resolution")


def spectrum_slices() -> None:
    from scripts.make_supplemental_spectrum_slices import build_figure

    source_path("scripts/make_supplemental_spectrum_slices.py")
    source_path("figures/paper/supplemental/s02_spectrum_slices_provenance.json")
    data = load("figures/paper/supplemental/s02_spectrum_slices.npz")
    save(build_figure(data), "s02_spectrum_slices")


def dense_linewidths() -> None:
    exp = load("paper/data/experimental/12_dense_pulse_length.npz")
    sim = load("paper/data/numerical/12_dense_pulse_length.npz")
    meta = json.loads(
        source_path("paper/data/experimental/12_dense_pulse_length.json").read_text()
    )["provenance"]
    reference = meta["coherence_reference_khz"]
    np.testing.assert_array_equal(exp["rabi_frequency_mhz"], sim["rabi_frequency_mhz"])
    indices = np.flatnonzero(exp["pulse_length_us"] <= 25)
    fmt = FuncFormatter(lambda value, _: f"{value:g}")
    apply_supplemental_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 4.5), layout="constrained")
    for i, color in zip(indices, COLORS, strict=True):
        x = exp["rabi_frequency_mhz"][i]
        ax.plot(
            x,
            np.where(sim["accepted"][i], sim["fwhm_khz"][i], np.nan),
            color=color,
            ls="--",
        )
        ax.scatter(
            x,
            np.where(exp["accepted"][i], exp["fwhm_khz"][i], np.nan),
            color=color,
            s=7,
            alpha=0.7,
            edgecolors="none",
        )
    handles = [
        Line2D([], [], color=color, label=rf"${exp['pulse_length_us'][i]:g}\,\mu$s")
        for i, color in zip(indices, COLORS, strict=True)
    ]
    legend = ax.legend(
        handles=handles,
        title="Pulse / template length",
        loc="upper center",
        ncol=7,
        columnspacing=1,
        handlelength=1.5,
    )
    ax.add_artist(legend)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="none", color=".4", label="Experiment"),
            Line2D([], [], ls="--", color=".4", label="Simulation"),
        ],
        loc="lower right",
    )
    ax.set(
        xscale="log",
        yscale="log",
        ylim=(10, 2000),
        xlim=(exp["rabi_frequency_mhz"].min(), exp["rabi_frequency_mhz"].max()),
        xlabel=r"Nominal peak Rabi frequency $f_R$ (MHz)",
        ylabel="FWHM (kHz)",
        xticks=[1, 2, 5, 10, 20, 40],
        yticks=[10, 20, 50, 100, 200, 500, 1000, 2000],
    )
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.minorticks_off()
    right = ax.secondary_yaxis(
        "right", functions=(lambda w: w / reference, lambda r: r * reference)
    )
    right.set_ylabel(r"FWHM / $\Gamma_{T_2^*}$")
    right.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20])
    right.yaxis.set_major_formatter(fmt)
    right.yaxis.set_minor_formatter(NullFormatter())
    ax.axhline(reference, color=".4", ls=":")
    ax.annotate(
        rf"$\Gamma_{{T_2^*}}={reference:.1f}$ kHz",
        xy=(0.98, reference),
        xycoords=ax.get_yaxis_transform(),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        fontsize=8,
    )
    save(fig, "s04_duration_linewidths")

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 4.6), layout="constrained")
    total = 0
    for i, color in zip(indices, COLORS, strict=True):
        pair = exp["accepted"][i] & sim["accepted"][i]
        total += int(pair.sum())
        ax.scatter(
            sim["fwhm_khz"][i, pair],
            exp["fwhm_khz"][i, pair],
            s=8,
            alpha=0.65,
            color=color,
            edgecolors="none",
            label=rf"${exp['pulse_length_us'][i]:g}\,\mu$s ($n={pair.sum()}$)",
        )
    assert total == 1278
    ax.plot([15, 1600], [15, 1600], ls="--", color=".4")
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(15, 1600),
        ylim=(15, 1600),
        xlabel="Simulated FWHM (kHz)",
        ylabel="Experimental FWHM (kHz)",
    )
    ax.set_aspect("equal", adjustable="box")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks([20, 50, 100, 200, 500, 1000])
        axis.set_major_formatter(fmt)
    ax.minorticks_off()
    ax.legend(
        loc="upper left",
        title=f"Paired linewidths (N={total})",
        bbox_to_anchor=(1.02, 1),
    )
    save(fig, "s05_paired_linewidths")


def phase_data(duration: int) -> dict[str, np.ndarray]:
    return load(
        f"data/generated/accumulated_phase_duration_sweep/{duration}us/results.npz"
    )


def phase_waveform() -> None:
    data = phase_data(20)
    # Same analytic waveform and trapezoidal phase integral as the report generator.
    source_path("scripts/make_accumulated_phase_duration_report.py")
    length, cutoff, order = (
        float(data[key]) for key in ("duration_us", "cutoff", "order")
    )
    kappa = float(data["selected_kappa_mhz_inv"])
    sigma = length / 2 / np.sqrt(cutoff ** (-1 / order) - 1)
    time = np.linspace(-length / 2, length / 2, 20_001)
    base = (1 + (time / sigma) ** 2) ** -order
    real = 40 * np.where(time < 0, 1, -1) * base
    correction = kappa * (40 * base) ** 2
    phase = np.r_[
        0, np.cumsum(np.pi * (correction[1:] + correction[:-1]) * np.diff(time))
    ]
    corrected = real * np.exp(-1j * phase)
    zoom = np.abs(time) <= min(50 * sigma, length / 2)
    # Document exactly the waveform arrays shown in this supplemental export.
    np.savez_compressed(
        OUT / "s06_phase_waveform.npz",
        time_us=time,
        original_i=real,
        corrected_i=corrected.real,
        corrected_q=corrected.imag,
        correction_mhz=correction,
        phase_rad=phase,
    )
    apply_supplemental_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH, 2.7), layout="constrained")
    for ax, i, q, title in [
        (axes[0], real, np.zeros_like(real), "(a)  Original pulse"),
        (axes[1], corrected.real, corrected.imag, "(b)  Accumulated phase"),
    ]:
        ax.plot(time[zoom], i[zoom], label="I")
        ax.plot(time[zoom], q[zoom], ls="--", label="Q")
        ax.axvline(0, color=".5", lw=0.6, ls=":")
        ax.set(xlabel=r"Time $t$ ($\mu$s)", title=title, ylim=(-45, 45))
        ax.legend(loc="lower right")
    axes[0].set_ylabel(r"$I,Q$ (MHz)")
    axes[1].plot(
        time[zoom], np.abs(corrected[zoom]), color=".45", ls=":", label=r"$|I+iQ|$"
    )
    axes[1].legend(loc="lower right")
    axes[2].plot(time, correction, color="C3", label=r"$\delta f_{\rm corr}$")
    axes[2].set(
        xlabel=r"Time $t$ ($\mu$s)",
        ylabel=r"$\delta f_{\rm corr}$ (MHz)",
        title="(c)  Correction and phase",
    )
    phase_ax = axes[2].twinx()
    phase_ax.plot(time, phase / (2 * np.pi), color="C2", label=r"$\phi/(2\pi)$")
    phase_ax.set_ylabel(r"$\phi/(2\pi)$ (cycles)")
    lines = axes[2].lines + phase_ax.lines
    axes[2].legend(lines, [line.get_label() for line in lines], loc="upper left")
    save(fig, "s06_phase_waveform")


def phase_optimization() -> None:
    data = phase_data(20)
    apply_supplemental_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH, 6.3), layout="constrained")
    groups = fig.add_gridspec(2, 1, height_ratios=[1, 1.7])
    top = groups[0].subgridspec(1, 2)
    left, right = [fig.add_subplot(top[0, col]) for col in range(2)]
    kappas = data["optimization_kappas_mhz_inv"]
    selected = float(data["selected_kappa_mhz_inv"])
    left.plot(kappas, 1000 * data["optimization_rms_mhz"], "o-")
    left.axvline(selected, color="C3", ls="--")
    left.axvline(float(data["kappa_theory_mhz_inv"]), color=".35", ls=":")
    left.set(
        xlabel=r"$\kappa$ (MHz$^{-1}$)",
        ylabel="Center RMS from zero (kHz)",
        title="(a)  Quadratic correction scan",
    )
    for kappa, label in [(0, r"$\kappa=0$"), (selected, "Optimized")]:
        index = np.flatnonzero(np.isclose(kappas, kappa, atol=1e-10, rtol=0)).item()
        right.plot(
            data["opt_rabi_mhz"],
            1000 * data["optimization_centers_mhz"][index],
            "o-",
            label=label,
        )
    right.axhline(0, color=".35", ls=":")
    right.set(
        xlabel=r"$\Omega_0/(2\pi)$ (MHz)",
        ylabel="Fitted center (kHz)",
        title="(b)  Central minimum",
    )
    right.legend()
    bottom = groups[1].subgridspec(2, 3)
    axes = []
    for col, length in enumerate((5, 10, 20)):
        d = phase_data(length)
        for row, key in enumerate(("plain_pe", "corrected_pe")):
            ax = fig.add_subplot(bottom[row, col])
            axes.append(ax)
            mesh = ax.pcolormesh(
                d["detuning_mhz"],
                d["rabi_mhz"],
                d[key],
                shading="auto",
                cmap="magma",
                vmin=0,
                vmax=0.5,
                rasterized=True,
            )
            ax.axvline(0, color="white", lw=0.7, ls="--")
            for bound in (-0.5, 0.5):
                ax.axvline(
                    bound * float(d["t2_limit_fwhm_mhz"]),
                    color="white",
                    lw=0.65,
                    ls=":",
                )
            ax.set(xlim=(-1, 1), ylim=(0, 80), xticks=[-1, 0, 1], yticks=[0, 40, 80])
            ax.set_title(
                f"({chr(99 + 3 * row + col)})  " + rf"$L={length}\,\mu\mathrm{{s}}$",
                loc="left",
            )
            if row == 1:
                ax.set_xlabel(r"$\Delta/(2\pi)$ (MHz)")
            else:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel(
                    ("Without correction\n" if row == 0 else "Accumulated phase\n")
                    + r"$\Omega_0/(2\pi)$ (MHz)"
                )
            else:
                ax.tick_params(labelleft=False)
    bar = fig.colorbar(mesh, ax=axes, location="right", fraction=0.025, pad=0.015)
    bar.set_label(r"$P_e$")
    save(fig, "s07_phase_optimization")


def midpoint_smoothing_leakage() -> None:
    """Plot saved final leakage, without rerunning dynamics or fitting."""
    directory = "outputs/sine_echo_transition_comparison"
    metadata_path = source_path(f"{directory}/provenance.json")
    metadata = json.loads(metadata_path.read_text())
    widths = np.array([0, 4, 8, 16])
    cases = [load(f"{directory}/T{width}ns.npz") for width in widths]
    detuning = cases[0]["narrow_detuning_mhz"]
    rabi = cases[0]["narrow_rabi_mhz"]
    for case in cases:
        np.testing.assert_array_equal(case["narrow_detuning_mhz"], detuning)
        np.testing.assert_array_equal(case["narrow_rabi_mhz"], rabi)
    pf = np.stack([case["narrow_populations"][2] for case in cases])
    maximum = pf.max(axis=2)
    np.testing.assert_allclose(
        maximum.max(axis=1), [m["max_final_pf"] for m in metadata["metrics"]],
        rtol=1e-14,
    )
    stem = "midpoint_smoothing_leakage"
    arrays_path = OUT / f"{stem}.npz"
    np.savez_compressed(
        arrays_path, transition_ns=widths, detuning_mhz=detuning, rabi_mhz=rabi,
        final_pf=pf, max_final_pf_over_detuning=maximum,
    )
    OUTPUTS.append(str(arrays_path.relative_to(ROOT)))
    apply_supplemental_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.95))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.79, wspace=0.32)
    for j, (width, color, style) in enumerate(zip(
        widths, COLORS[:4], ["-", "--", "-.", ":"], strict=True
    )):
        label = "0 ns (abrupt)" if width == 0 else f"{width} ns"
        axes[0].plot(detuning, pf[j, -1], color=color, ls=style, label=label)
        # The zero-drive values are exactly zero and cannot appear on a log axis.
        positive = rabi > 0
        axes[1].plot(rabi[positive], maximum[j, positive], color=color, ls=style,
                     marker="o", ms=2.5, label=label)
    axes[0].set(
        title=f"(a)  Final leakage at {rabi[-1]:.2f} MHz drive",
        xlabel=r"Detuning $\Delta/(2\pi)$ (MHz)", ylabel=r"Final $P_f$",
        xlim=(detuning[0], detuning[-1]), ylim=(1e-4, 5e-2), yscale="log",
    )
    axes[1].set(
        title="(b)  Maximum over detuning",
        xlabel=r"Peak Rabi frequency $\Omega_0/(2\pi)$ (MHz)",
        ylabel=r"$\max_{\Delta} P_f$", xlim=(0, 44), ylim=(3e-7, 5e-2),
        yscale="log",
    )
    for ax in axes:
        ax.minorticks_off()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               title="Midpoint transition duration", bbox_to_anchor=(0.5, 1.01))
    save(fig, stem)
    provenance = {
        "parameters": metadata["parameters"],
        "transition_ns": widths.tolist(),
        "dt_ns": metadata["dt_ns"],
        "convergence_max_abs_population_error": metadata[
            "convergence_max_abs_population_error"
        ],
        "observable": "Final Pf; not maximum transient leakage",
        "panels": {
            "a": "All 201 detunings at the highest saved drive amplitude",
            "b": "Maximum over 201 detunings at each saved drive; zero-drive Pf=0 omitted on log axis",
        },
        "max_final_pf": maximum.max(axis=1).tolist(),
        "processing": "No fitting, smoothing, rescaling, or new dynamics; lines connect saved samples",
        "source_sha256": {
            k: v for k, v in SOURCES.items() if k.startswith(directory + "/")
        },
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        "reproduce": ".venv/bin/python scripts/make_supplemental_figures.py --only midpoint_smoothing_leakage",
    }
    sidecar = OUT / f"{stem}_provenance.json"
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    OUTPUTS.append(str(sidecar.relative_to(ROOT)))


def drag_beta_optimization() -> None:
    import csv

    from echospec.simulation.qutrit import sine_echo_sign

    scan = load("outputs/drag_beta_center_scan/scan.npz")
    scan_meta = json.loads(source_path("outputs/drag_beta_center_scan/provenance.json").read_text())
    experiment = load("paper/data/experimental/04_main_ac_stark_correction_maps.npz")
    source_path("paper/data/experimental/04_main_ac_stark_correction_maps.json")
    source_path("echospec/simulation/qutrit.py")
    source_path("scripts/scan_drag_beta_center.py")
    csv_path = source_path("outputs/sine_echo_negative_beta_comparison/direct_resonance_metrics.csv")
    source_path("outputs/sine_echo_negative_beta_comparison/provenance.json")
    with csv_path.open() as handle:
        records = list(csv.DictReader(handle))
    time = np.linspace(-0.1, 0.1, 4001)
    sigma = 10 / np.sqrt(0.005**-2 - 1)
    g = (1 + (time / sigma)**2)**-0.5
    dg = -time / sigma**2 * (1 + (time / sigma)**2)**-1.5
    sign, dsign = np.array([sine_echo_sign(t, 0.016) for t in time]).T
    i = g * sign
    q = 0.5 * (dg * sign + g * dsign) / (2 * np.pi * 216)
    displayed = dict(
        time_us=time, i_over_omega0=i, q_over_omega0=q,
        scan_beta=scan["beta"], scan_rabi_mhz=scan["rabi_mhz"],
        scan_centers_hz=scan["centers_hz"], scan_rms_center_hz=scan["rms_center_hz"],
    )
    apply_supplemental_style()
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 5.25), layout="constrained")
    wave, objective, simulated, measured = axes.flat
    wave.plot(time * 1000, i, color=COLORS[0], label=r"$I/\Omega_0$")
    q_axis = wave.twinx()
    q_axis.plot(time * 1000, q, color=COLORS[1], label=r"$Q/\Omega_0$")
    wave.set(xlim=(-80, 80), ylim=(-1.1, 1.1), title=r"(a)  Adding Q at $\beta=-0.5$",
             xlabel="Time from midpoint (ns)", ylabel=r"$I/\Omega_0$")
    q_axis.set(ylim=(-0.08, 0.08), ylabel=r"$Q/\Omega_0$")
    q_axis.tick_params(axis="y", colors=COLORS[1])
    q_axis.yaxis.label.set_color(COLORS[1])
    wave.yaxis.label.set_color(COLORS[0])
    wave.tick_params(axis="y", colors=COLORS[0])
    wave.axhline(0, color="0.6", lw=0.5)
    best_beta = scan_meta["best_sampled_beta"]
    objective.plot(scan["beta"], scan["rms_center_hz"] / 1000,
                   color=COLORS[0], marker="o", ms=2.7)
    objective.axvline(-0.5, color="0.5", lw=0.7, ls=":")
    objective.scatter([best_beta], [scan_meta["best_rms_center_hz"] / 1000],
                      marker="*", s=65, color=COLORS[1], zorder=5)
    objective.set(title="(b)  Simulated beta scan", xlabel=r"DRAG coefficient $\beta$",
                  ylabel="RMS center about zero (kHz)", yscale="log", xlim=(-1.03, 0.03))
    objective.text(0.5, 0.95, rf"Best sampled $\beta={best_beta:g}$",
                   transform=objective.transAxes, fontsize=8, ha="center", va="top")
    for beta, color, style in zip((0, -0.5, -1), COLORS[:3], ("-", "--", "-."), strict=True):
        selected = [row for row in records if float(row["beta"]) == beta
                    and row["accepted"] == "True" and float(row["rabi_mhz"]) >= 10]
        amp = np.array([float(row["rabi_mhz"]) for row in selected])
        center = np.array([float(row["center_hz"]) for row in selected])
        simulated.plot(amp, center / 1000, color=color, ls=style, label=rf"$\beta={beta:g}$")
        displayed[f"simulation_beta{beta:g}_rabi_mhz"] = amp
        displayed[f"simulation_beta{beta:g}_centers_hz"] = center
    simulated.set(title="(c)  Simulated center versus drive",
                  xlabel=r"Peak Rabi frequency $\Omega_0/(2\pi)$ (MHz)",
                  ylabel="Dip center (kHz)", xlim=(10, 43))
    simulated.axhline(0, color="0.6", lw=0.5)
    simulated.legend(fontsize=7.5, loc="upper left")
    for index, prefix in enumerate(("plain", "drag")):
        mask = experiment[f"{prefix}_display_center_mask"]
        amp = experiment["rabi_mhz"][mask]
        center = experiment[f"{prefix}_display_centers_mhz"][mask] * 1000
        beta = experiment["drag_beta"][index]
        transition = experiment["applied_echo_transition_ns"][index]
        measured.plot(amp, center, ls="none", marker="o", ms=2.2,
                      color=COLORS[index], label=rf"$\beta={beta:g}$, $T={transition:g}$ ns")
        displayed[f"experiment_{prefix}_rabi_mhz"] = amp
        displayed[f"experiment_{prefix}_center_variation_khz"] = center
    measured.set(title="(d)  Experiment: selected settings",
                 xlabel=r"Calibrated Rabi frequency $\Omega_{\rm cal}/(2\pi)$ (MHz)",
                 ylabel="Center minus run mean (kHz)", xlim=(10, 60))
    measured.axhline(0, color="0.6", lw=0.5)
    measured.legend(fontsize=7.5, loc="lower left")
    for ax in (*axes.flat, q_axis):
        ax.minorticks_off()
    stem = "drag_beta_optimization"
    save(fig, stem)
    arrays = OUT / f"{stem}.npz"
    np.savez_compressed(arrays, **displayed)
    OUTPUTS.append(str(arrays.relative_to(ROOT)))
    provenance = dict(
        simulated_scan=scan_meta,
        experiment=dict(duration_us=50, cutoff=0.00075, beta=[0, -0.22],
                        transition_ns=[0, 16], shots=2000,
                        processing="Saved accepted Gaussian centers, run mean subtracted over 10-60 MHz; no extra smoothing",
                        limitation="Two selected settings with different midpoint smoothing; not an experimental beta scan"),
        sources={key: value for key, value in SOURCES.items() if any(
            part in key for part in ("drag_beta", "negative_beta", "ac_stark_correction", "qutrit.py")
        )},
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        arrays_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest(),
        reproduce=".venv/bin/python scripts/make_supplemental_figures.py --only drag_beta_optimization",
    )
    sidecar = OUT / f"{stem}_provenance.json"
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    OUTPUTS.append(str(sidecar.relative_to(ROOT)))


def edge_smoothing_results() -> None:
    import csv

    from echospec.analysis.spectroscopy_quality import spectroscopy_quality

    data = load("figures/paper/03_pulse_improvement.npz")
    metadata = json.loads(source_path("figures/paper/03_pulse_improvement_provenance.json").read_text())
    source_path("echospec/analysis/spectroscopy_quality.py")
    source_path("echospec/analysis/dip_metrics.py")
    # The manuscript pulse includes endpoint subtraction and a p=5 window.
    # Older plain-window edge-ramp reports describe a different pulse.
    assert metadata["source_configs"]["polynomial"]["family"] == "polynomial5"
    for key in ("drag", "polynomial"):
        cfg = metadata["source_configs"][key]
        assert cfg["drag_beta"] == -0.5 and cfg["duration_us"] == 30
        assert cfg["echo_transition_us"] == 0.016
        assert cfg.get("cutoff", cfg.get("cutoff_core")) == 0.001
    amp, detuning = data["rabi_mhz"], data["detuning_mhz"]
    high = amp >= 50
    metrics, rows, summaries = {}, [], {}
    displayed = dict(rabi_mhz=amp, detuning_mhz=detuning)
    for key in ("drag", "polynomial"):
        local = [dict(case=key, amplitude_mhz=float(a),
                      **spectroscopy_quality(detuning, population, shots=1000))
                 for a, population in zip(amp, data[key][1], strict=True)]
        rows.extend(local)
        metrics[key] = {name: np.array([r[name] for r in local]) for name in
                        ("accepted", "center_hz", "fwhm_hz", "contrast",
                         "contrast_snr", "shoulder_contrast")}
        assert metrics[key]["accepted"][high].all()
        assert np.isfinite(metrics[key]["contrast_snr"][high]).all()
        for name, values in metrics[key].items():
            displayed[f"{key}_{name}"] = values
        displayed[f"{key}_pe_100mhz"] = data[key][1, -1]
        summaries[key] = dict(
            at_100_mhz=local[-1],
            max_fwhm_khz_50_to_100=float(metrics[key]["fwhm_hz"][high].max()/1000),
            min_contrast_50_to_100=float(metrics[key]["contrast"][high].min()),
            min_ideal_snr_50_to_100=float(metrics[key]["contrast_snr"][high].min()),
            passing_200khz_snr10=int(np.sum(high & (metrics[key]["fwhm_hz"] <= 2e5)
                                           & (metrics[key]["contrast_snr"] >= 10))),
        )
    apply_supplemental_style()
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 5.15))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.105, top=0.89,
                        wspace=0.28, hspace=0.50)
    spectrum, width, contrast, snr = axes.flat
    labels = ("Hard cutoff + DRAG", "Polynomial ($p=5$) + DRAG")
    for key, color, style, label in zip(
        ("drag", "polynomial"), ("#586579", "#176eaa"), ("-", "--"), labels,
        strict=True,
    ):
        spectrum.plot(detuning, data[key][1, -1], color=color, ls=style, label=label)
        for ax, metric, scale in ((width, "fwhm_hz", 1e-3),
                                  (contrast, "contrast", 1), (snr, "contrast_snr", 1)):
            values = np.where(metrics[key]["accepted"], metrics[key][metric]*scale, np.nan)
            ax.plot(amp, values, color=color, ls=style, label=label)
    spectrum.set(title="(a)  Spectrum at 100 MHz drive",
                 xlabel=r"Detuning $\Delta/(2\pi)$ (MHz)", ylabel=r"Final $P_e$",
                 xlim=(-0.5, 0.5), ylim=(0.24, 0.62))
    width.set(title="(b)  Resistance to power broadening", ylabel="FWHM (kHz)",
              ylim=(0, 550))
    width.axhline(200, color="0.5", ls=":", lw=0.7)
    width.text(3, 210, "200 kHz guide", color="0.4", fontsize=7)
    contrast.set(title="(c)  Dip visibility", ylabel="Contrast C", ylim=(0, 0.55))
    snr.set(title="(d)  Ideal shot-noise SNR", ylabel="SNR (1000 total shots)",
            ylim=(0, 19))
    snr.axhline(10, color="0.5", ls=":", lw=0.7)
    for ax in (width, contrast, snr):
        ax.set(xlim=(0, 100), xlabel=r"Peak Rabi frequency $\Omega_0/(2\pi)$ (MHz)")
    handles, labels = spectrum.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    stem = "edge_smoothing_results"
    save(fig, stem)
    arrays = OUT / f"{stem}.npz"
    np.savez_compressed(arrays, **displayed)
    metrics_path = OUT / f"{stem}_metrics.csv"
    with metrics_path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUTS.extend([str(arrays.relative_to(ROOT)), str(metrics_path.relative_to(ROOT))])
    provenance = dict(
        source_configs={k: metadata["source_configs"][k] for k in ("drag", "polynomial")},
        summaries=summaries,
        high_drive_mhz=[float(amp[high][0]), float(amp[-1])], high_drive_points=int(high.sum()),
        contrast="Background-referenced dip depth from dip_metrics, as in the main-paper figure",
        snr="Ideal Bernoulli readout, 500 shots at interpolated dip center and 250 at each +/-0.45 MHz shoulder; no assignment error, drift, technical noise or center-search cost",
        snr_signal="Mean of two shoulder populations minus center population; distinct from fitted-background contrast C",
        processing="Saved populations unchanged; no extra smoothing, fitting, normalization or new dynamics; lines connect samples",
        source_sha256={k: v for k, v in SOURCES.items() if any(s in k for s in
                      ("03_pulse_improvement", "spectroscopy_quality.py", "dip_metrics.py"))},
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        arrays_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest(),
        reproduce=".venv/bin/python scripts/make_supplemental_figures.py --only edge_smoothing_results",
    )
    sidecar = OUT / f"{stem}_provenance.json"
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    OUTPUTS.append(str(sidecar.relative_to(ROOT)))


def pulse_modifications() -> None:
    from echospec.simulation.polynomial_echo import PolynomialEchoPulse

    source_path("echospec/simulation/polynomial_echo.py")
    model = PolynomialEchoPulse()
    full = np.unique(np.r_[np.linspace(-16, 16, 16001),
                           np.linspace(-0.08, 0.08, 1601)])
    left = np.linspace(-15.5, -10, 2201)
    center = np.linspace(-0.04, 0.04, 3201)
    data = {}
    for key, time in (("full", full), ("left", left), ("center", center)):
        hard = np.where(np.abs(time) < model.duration_us / 2,
                        (1 + (time / model.sigma_us)**2)**-0.5, 0.0)
        data[f"{key}_time_us"] = time
        data[f"{key}_envelopes"] = np.array([hard, model.envelope(time)[0]])
    g, dg = model.envelope(center)
    inside = np.abs(center) < model.transition_us / 2
    sign = np.where(inside, -np.sin(np.pi * center / model.transition_us),
                    np.where(center < 0, 1.0, -1.0))
    dsign = np.where(inside, -np.pi / model.transition_us
                     * np.cos(np.pi * center / model.transition_us), 0.0)
    coefficient = -model.drag_beta / (2 * np.pi * abs(model.anharmonicity_mhz))
    data["center_i_over_omega0"] = data["center_envelopes"] * sign
    data["center_q_envelope_over_omega0"] = coefficient * dg * sign
    data["center_q_midpoint_over_omega0"] = coefficient * g * dsign
    data["center_q_over_omega0"] = -model.drive(center).imag
    np.testing.assert_allclose(
        data["center_q_over_omega0"],
        data["center_q_envelope_over_omega0"]
        + data["center_q_midpoint_over_omega0"], atol=1e-16,
    )
    path = OUT / "s08_pulse_modifications.npz"
    np.savez_compressed(path, **data)
    OUTPUTS.append(str(path.relative_to(ROOT)))
    apply_supplemental_style()
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 4.9), layout="constrained")
    overview, edge, pulse, quadrature = axes.flat
    colors = ["#586579", "#176eaa"]
    labels = ["Hard cutoff", "Polynomial ($p=5$)"]

    def curves(ax, x, values):
        for y, color, label, style in zip(
            values, colors, labels, ["-", "--"], strict=True
        ):
            ax.plot(x, y, color=color, label=label, ls=style)

    for ax, key, title, limit in zip(
        (overview, edge), ("full", "left"),
        ("(a)  Full envelope", "(b)  Turn-on edge"), (1e-8, 1e-9), strict=True,
    ):
        time = data[f"{key}_time_us"]
        curves(ax, time, data[f"{key}_envelopes"])
        ax.set_yscale("symlog", linthresh=limit)
        ax.set(xlim=(time[0], time[-1]), title=title,
               xlabel="Time from midpoint (µs)", ylabel=r"Envelope $g(t)$")
        if key == "full":
            ax.set(ylim=(-1e-9, 1.5), yticks=[0, 1e-6, 1e-4, 1e-2, 1])
        else:
            ax.set(ylim=(-2e-10, 0.002), yticks=[0, 1e-8, 1e-6, 1e-4, 1e-3])
            ax.axvline(-15, color="#9c3549", lw=0.6, ls=":")
        ax.minorticks_off()
    overview.legend(loc="lower center", fontsize=7.5)
    time = 1000 * center
    curves(pulse, time, data["center_i_over_omega0"])
    abrupt = np.where(time < 0, 1, -1) * data["center_envelopes"][0]
    pulse.plot(time, abrupt, color="black", ls=":", label="Abrupt inversion")
    pulse.set(title="(c)  In-phase waveform", xlabel="Time from midpoint (ns)",
              ylabel=r"$I(t)/\Omega_0$", xlim=(-40, 40), ylim=(-1.12, 1.12))
    pulse.legend(handles=[pulse.lines[-1]], loc="upper right", fontsize=7.5)
    quadrature.plot(time, data["center_q_over_omega0"], color=colors[1],
                    label="Total Q", lw=1.4)
    quadrature.plot(time, data["center_q_envelope_over_omega0"], color="#d47823",
                    ls="--", label=r"Envelope term: $g_p' s_T$")
    quadrature.plot(time, data["center_q_midpoint_over_omega0"], color="#70439b",
                    ls=":", label=r"Midpoint term: $g_p s_T'$")
    quadrature.set(title="(d)  DRAG quadrature (polynomial)",
                   xlabel="Time from midpoint (ns)", ylabel=r"$Q(t)/\Omega_0$",
                   xlim=(-40, 40), ylim=(-0.08, 0.06))
    quadrature.legend(loc="upper center", fontsize=7)
    for ax in (pulse, quadrature):
        ax.axhline(0, color="gray", lw=0.6)
        for boundary in (-8, 8):
            ax.axvline(boundary, color="#70439b", lw=0.6, ls=":")
    save(fig, "s08_pulse_modifications")
    metadata = {
        "parameters": vars(model),
        "convention": "I-iQ; Q/Omega0=-beta*(g_prime*s+g*s_prime)/(2*pi*abs(alpha_MHz))",
        "families": labels,
        "normalization": "Unsigned g(0)=1; no area or signed-peak rescaling",
        "quadrature_panel": "Polynomial pulse; total and its two derivative contributions",
        "reproduce": ".venv/bin/python scripts/make_supplemental_figures.py --only pulse_modifications",
        "source_sha256": {
            key: SOURCES[key] for key in ("echospec/simulation/polynomial_echo.py",)
        },
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arrays_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    sidecar = OUT / "s08_pulse_modifications_provenance.json"
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
    OUTPUTS.append(str(sidecar.relative_to(ROOT)))

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    renderers = (
        schematic,
        spectrum_slices,
        duration_metrics,
        dense_linewidths,
        phase_waveform,
        phase_optimization,
        midpoint_smoothing_leakage,
        drag_beta_optimization,
        pulse_modifications,
        edge_smoothing_results,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=[f.__name__ for f in renderers])
    args = parser.parse_args()
    for render in renderers:
        if args.only and render.__name__ not in args.only:
            continue
        print(f"Rendering {render.__name__}", flush=True)
        render()
    source_path(str(STYLE_PATH.relative_to(ROOT)))
    source_path("echospec/figures/supplemental.py")
    source_path("scripts/make_supplemental_figures.py")
    # No source numerical arrays, masks, or fitted metrics may change during rendering.
    for relative, digest in SOURCES.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    provenance_path = OUT / "provenance.json"
    previous = json.loads(provenance_path.read_text()) if args.only else {}
    prior_sources = previous.get("source_sha256", {})
    for stale in (
        "figures/paper/03_lorentzian_echo_slices_data.csv",
        "figures/paper/03_lorentzian_echo_slices_aligned_simulation.csv",
        "scripts/make_figure3_left_exp_sim.py",
    ):
        prior_sources.pop(stale, None)
    prior_outputs = [p for p in previous.get("outputs", [])
                     if "s02_duration_resolution" not in p
                     and "s03_spectrum_slices" not in p]
    provenance = {
        "reproduction": "make paper-figures-supplemental PYTHON=.venv/bin/python",
        "style_reference": "Extended cutoff-duration map",
        "style": str(STYLE_PATH.relative_to(ROOT)),
        "canvas_width_inches": FIGURE_WIDTH,
        "source_sha256": {**prior_sources, **SOURCES},
        "outputs": sorted(set(prior_outputs + OUTPUTS)),
        "data_policy": "Saved numerical arrays unchanged during rendering; S2 uses documented mean subtraction and stacking, with no gain or frequency fit.",
        "pulse_modification_waveform": "See s08_pulse_modifications_provenance.json for the analytic I and Q waveforms and parameters.",
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
