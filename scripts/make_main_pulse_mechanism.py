"""Render compact Fig. 1 and a separate, conceptual level-diagram alternative.

The compact figure preserves the parameters and numerical model from
notebooks/paper/30_main_pulse_mechanism.ipynb. The schematic is not included
in the manuscript and its energy spacings are illustrative, not calibrated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, sesolve

from echospec.figures import FigureVariant, apply_figure_style, save_figure
from echospec.simulation.hamiltonian import Hamiltonian
from echospec.simulation.operators import N_dim, sz
from echospec.simulation.pulses import (
    PulseArgs,
    PulseType,
    echo_lorentzian_envelope,
)
from echospec.utils.parameters import Parameters
from echospec.utils.units import Units as u

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures/paper"
PULSE_LENGTH_US = 10.0
CUTOFF = 5e-3
ORDER = 0.5
RABI_MHZ = 15.0
DETUNINGS_MHZ = (0.0, 0.10, 2.00)
COLORS = ("#555555", "#d55e00", "#0072b2")
LINESTYLES = ("-", "--", "-.")
COMPACT_STYLE = {
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": None,
}


def calculate():
    """Evaluate the existing unitary two-level model without changing parameters."""
    pulse_length = PULSE_LENGTH_US * u.us
    args = PulseArgs(pulse_length=pulse_length, cutoff=CUTOFF, order=ORDER)
    t = np.linspace(-pulse_length / 2, pulse_length / 2, 2**14, endpoint=False)
    omega = RABI_MHZ * echo_lorentzian_envelope(t, args)
    dt = t[1] - t[0]
    padded = np.pad(omega, (8 * t.size, 8 * t.size))
    frequency = np.fft.fftshift(np.fft.fftfreq(padded.size, dt)) / 1e6
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(padded)))
    spectrum /= spectrum.max()
    time = np.linspace(-pulse_length / 2, pulse_length / 2, 1600)
    populations = []
    for detuning in DETUNINGS_MHZ:
        params = Parameters(
            eco_pulse=True,
            pulse_type=PulseType.LORENTZIAN,
            pulse_length=pulse_length,
            cutoff=CUTOFF,
            order=ORDER,
            rabi_frequency=RABI_MHZ * 2 * np.pi * u.MHz,
            detuning=detuning * 2 * np.pi * u.MHz,
        )
        result = sesolve(
            Hamiltonian(params=params).get_hamiltonian(),
            basis(N_dim, 0),
            time,
            e_ops=[sz],
        )
        populations.append((1 - np.asarray(result.expect[0])) / 2)
    return {
        "waveform_time_us": t / u.us,
        "waveform_mhz": omega,
        "frequency_mhz": frequency,
        "spectrum_normalized": spectrum,
        "evolution_time_us": time / u.us,
        "detunings_mhz": np.asarray(DETUNINGS_MHZ),
        "populations": np.asarray(populations),
    }


def save(fig, stem):
    paths = save_figure(
        fig,
        stem,
        variant=FigureVariant.PAPER,
        formats=("pdf", "png", "svg"),
        dpi=350,
    )
    for path in paths:
        print(path)
    plt.close(fig)


def compact_figure(data):
    fig = plt.figure(figsize=(3.38, 3.05), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.08))
    time_ax = fig.add_subplot(grid[0, 0])
    fft_ax = fig.add_subplot(grid[0, 1])
    population_ax = fig.add_subplot(grid[1, :])
    time_ax.plot(data["waveform_time_us"], data["waveform_mhz"], color=COLORS[0])
    time_ax.axhline(0, color="0.65", lw=0.5)
    time_ax.set(
        xlabel=r"$t$ ($\mu$s)",
        ylabel=r"$\Omega/2\pi$ (MHz)",
        xticks=[-5, 0, 5],
        yticks=[-15, 0, 15],
    )
    mask = np.abs(data["frequency_mhz"]) <= 1
    fft_ax.plot(
        data["frequency_mhz"][mask],
        data["spectrum_normalized"][mask],
        color="#009e73",
    )
    fft_ax.set(
        xlabel=r"$f$ (MHz)",
        ylabel=r"$|\widetilde{\Omega}(f)|$ (norm.)",
        ylim=(-0.02, 1.05),
        xticks=[-1, 0, 1],
        yticks=[0, 0.5, 1],
    )
    for color, linestyle, detuning, population in zip(
        COLORS,
        LINESTYLES,
        DETUNINGS_MHZ,
        data["populations"],
        strict=True,
    ):
        population_ax.plot(
            data["evolution_time_us"],
            population,
            color=color,
            ls=linestyle,
            label=rf"${detuning:.2f}$",
        )
    population_ax.set(
        xlabel=r"$t$ ($\mu$s)",
        ylabel=r"$P_e$",
        ylim=(-0.03, 1.05),
        xticks=[-5, 0, 5],
        yticks=[0, 0.5, 1],
    )
    population_ax.legend(
        title=r"$\Delta/2\pi$ (MHz)",
        loc="center left",
        fontsize=6,
        title_fontsize=6,
        frameon=False,
        handlelength=1.4,
    )
    for index, ax in enumerate((time_ax, fft_ax, population_ax)):
        ax.axvline(0, color="0.65", ls=":", lw=0.6, zorder=0)
        ax.tick_params(direction="in", top=True, right=True, length=2, pad=2)
        ax.tick_params(which="minor", direction="in", top=True, right=True,
                       length=1.2, width=0.4)
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
        ax.text(0.04, 0.94, f"({chr(97 + index)})", transform=ax.transAxes,
                va="top", fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))
    save(fig, "01_pulse_mechanism")


def schematic_variant():
    """A qualitative level ladder and exact sign reversal of the drive envelope."""
    red, gray = "#ed2424", "#999999"
    with plt.rc_context(
        {"font.family": "sans-serif", "mathtext.fontset": "dejavusans"}
    ):
        fig = plt.figure(figsize=(6.9, 2.85))
        ladder = fig.add_axes((0.065, 0.22, 0.31, 0.65))
        pulse = fig.add_axes((0.55, 0.28, 0.42, 0.59))
        ladder.set(xlim=(0, 1), ylim=(-0.2, 2.12))
        ladder.axis("off")
        ladder.annotate(
            "",
            xy=(0.03, 2.05),
            xytext=(0.03, -0.08),
            arrowprops={"arrowstyle": "-|>", "lw": 1.3, "color": "black"},
        )
        ladder.text(-0.025, 2.08, r"$E$", fontsize=12)
        for energy, state, color in (
            (0, r"$|g\rangle$", "black"),
            (1, r"$|e\rangle$", "black"),
            (1.78, r"$|f\rangle$", gray),
        ):
            ladder.plot([0.18, 0.80], [energy, energy], color=color, lw=2)
            ladder.text(0.86, energy, state, va="center", color=color, fontsize=12)
        ladder.plot([0.18, 0.80], [2, 2], color=gray, lw=1, ls="--")
        ladder.annotate(
            "",
            xy=(0.68, 2),
            xytext=(0.68, 1.78),
            arrowprops={"arrowstyle": "<->", "color": gray, "lw": 1},
        )
        ladder.text(0.84, 2.02, r"$\hbar|\alpha|$", color=gray, fontsize=9)
        ladder.annotate(
            "",
            xy=(0.48, 0.97),
            xytext=(0.48, 0.03),
            arrowprops={"arrowstyle": "<->", "color": red, "lw": 1.8},
        )
        ladder.text(0.54, 0.5, r"$\Omega(t)$", color=red, va="center", fontsize=12)
        ladder.text(0.16, 0.5, r"$\hbar\omega_{01}$", va="center", fontsize=10)
        ladder.text(0.18, -0.22, "Illustrative level spacing", fontsize=8, color="0.4")

        # Show the ideal pulse on its natural time scale, not an artificial
        # broadening of the very narrow peak in the main figure.
        x = np.linspace(-6, 0, 900)
        envelope = 1 / np.sqrt(1 + x**2)
        pulse.axvspan(-6, 0, color="0.91", zorder=0)
        pulse.axhline(0, color="0.7", lw=0.8)
        pulse.axvline(0, color="0.45", ls=(0, (5, 4)), lw=1)
        pulse.plot(x, envelope, color=red, lw=2)
        pulse.plot(-x[::-1], -envelope[::-1], color="black", lw=2)
        pulse.plot([0, 0], [1, -1], color=red, lw=1.2)
        pulse.set(
            xlim=(-6, 6),
            ylim=(-1.2, 1.2),
            xlabel=r"$t/\tau$",
            ylabel=r"$\Omega(t)/\Omega_0$",
            xticks=[-6, -3, 0, 3, 6],
            yticks=[-1, 0, 1],
        )
        pulse.text(-3.2, 0.93, r"$\phi=0$", ha="center", fontsize=11)
        pulse.text(3.2, 0.93, r"$\phi=\pi$", ha="center", fontsize=11)
        pulse.text(0, 1.27, r"$\pi$ phase inversion", ha="center", fontsize=10)
        pulse.tick_params(direction="in", top=True, right=True, labelsize=9)
        for spine in pulse.spines.values():
            spine.set_linewidth(1.1)
        fig.text(0.02, 0.96, "(a) Driven qubit", fontsize=12, va="top")
        fig.text(0.46, 0.96, "(b) Echo reversal", fontsize=12, va="top")
        fig.text(
            0.76,
            0.045,
            r"$\Delta=0:\quad U_{\pi}=U_0^\dagger,\quad U_{\pi}U_0=\mathbb{I}$",
            ha="center",
            fontsize=11,
        )
        fig.text(0.065, 0.05, "Ideal two-level, unitary cancellation", fontsize=8)
        save(fig, "01_pulse_mechanism_schematic_variant")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-data", action="store_true",
                        help="Restyle saved arrays without rerunning the simulation")
    parser.add_argument("--main-only", action="store_true",
                        help="Leave the separate schematic variant unchanged")
    options = parser.parse_args()
    apply_figure_style(FigureVariant.PAPER)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 0.9,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    if options.reuse_data:
        with np.load(OUTPUT / "01_pulse_mechanism.npz", allow_pickle=False) as saved:
            data = {key: saved[key].copy() for key in saved.files}
        np.testing.assert_array_equal(data["detunings_mhz"], DETUNINGS_MHZ)
    else:
        data = calculate()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not options.reuse_data:
        np.savez_compressed(OUTPUT / "01_pulse_mechanism.npz", **data)
    with plt.rc_context(COMPACT_STYLE):
        compact_figure(data)
    if not options.main_only:
        schematic_variant()
    provenance = {
        "generator": "scripts/make_main_pulse_mechanism.py",
        "original_workflow": "notebooks/paper/30_main_pulse_mechanism.ipynb",
        "model": "Unitary two-level; no relaxation or dephasing",
        "pulse_length_us": PULSE_LENGTH_US,
        "cutoff": CUTOFF,
        "order": ORDER,
        "rabi_mhz": RABI_MHZ,
        "detunings_mhz": list(DETUNINGS_MHZ),
        "final_populations": data["populations"][:, -1].tolist(),
        "main_figure_size_inches": [3.38, 3.05],
        "main_panel_titles": False,
        "main_figure_style": COMPACT_STYLE,
        "population_colors": list(COLORS),
        "population_linestyles": list(LINESTYLES),
        "spectrum_color": "#009e73",
        "schematic_variant": {
            "included_in_paper": False,
            "energy_levels": "Illustrative weakly anharmonic ladder, not to scale",
            "pulse": "Ideal root-Lorentzian, displayed over -6 <= t/tau <= 6",
            "cancellation": "Ideal symmetric unitary two-level pulse at resonance",
        },
    }
    (OUTPUT / "01_pulse_mechanism_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print("Final populations:", provenance["final_populations"])


if __name__ == "__main__":
    main()
