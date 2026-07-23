"""Generate the accumulated-phase duration sweep and tabbed HTML report."""

from __future__ import annotations

# Backend and local-source setup must precede pyplot and echospec imports.
# ruff: noqa: E402, I001

import argparse
import base64
import csv
import html
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.analysis.fwhm import fwhm_half_depth
from echospec.figures import FigureVariant, apply_figure_style
from echospec.simulation.qutrit import simulate_qutrit_map


DURATIONS_US = (3.0, 5.0, 7.0, 10.0, 20.0, 30.0)
CUTOFF = 0.001
ORDER = 0.5
# Repeated q1 OPX1000 two-photon scans place f02/2 at 4.159106667 GHz while
# f01=4.267106667 GHz, hence alpha/(2*pi)=2*(f02/2-f01)=-216 MHz.
ANHARMONICITY_MHZ = -216.0
T1_US = 51.24
T2_US = 7.31
T_PHI_US = 1.0 / (1.0 / T2_US - 1.0 / (2.0 * T1_US))
DRAG_BETA = 0.0
ECHO = True
RABI_MHZ = np.linspace(0.0, 80.0, 41)
DETUNING_MHZ = np.linspace(-1.0, 1.0, 201)
OPT_RABI_MHZ = np.linspace(20.0, 80.0, 13)
OPT_DETUNING_MHZ = np.linspace(-1.0, 1.0, 101)
STEPS_PER_US = 1000
FIT_WINDOW_MHZ = 0.35
SWEEP_RABI_MHZ = np.asarray([10.0, 20.0, 40.0, 60.0, 80.0])
PULSE_PLOT_RABI_MHZ = 40.0
T2_LIMIT_FWHM_MHZ = 1.0 / (np.pi * T2_US)
T2_LIMIT_HALF_WIDTH_MHZ = 0.5 * T2_LIMIT_FWHM_MHZ
KAPPA_THEORY_MHZ_INV = -1.0 / (2.0 * ANHARMONICITY_MHZ)

DATA_DIR = ROOT / "data" / "generated" / "accumulated_phase_duration_sweep"
FIGURE_DIR = ROOT / "figures" / "paper" / "accumulated_phase_duration_sweep"
REPORT_PATH = ROOT / "outputs" / "accumulated_phase_duration_report.html"
SUMMARY_CSV_PATH = DATA_DIR / "duration_summary.csv"
AMPLITUDE_CSV_PATH = DATA_DIR / "amplitude_metrics.csv"


def duration_label(duration_us: float) -> str:
    return f"{duration_us:g}us".replace(".", "p")


def fitted_central_minima(
    detuning_mhz: np.ndarray,
    excited_population: np.ndarray,
) -> np.ndarray:
    mask = np.abs(detuning_mhz) <= FIT_WINDOW_MHZ
    x_values = detuning_mhz[mask]
    centers: list[float] = []
    for row in excited_population[:, mask]:
        index = int(np.argmin(row))
        center = float(x_values[index])
        if 0 < index < x_values.size - 1:
            coefficients = np.polyfit(
                x_values[index - 1 : index + 2],
                row[index - 1 : index + 2],
                2,
            )
            if coefficients[0] > 0:
                candidate = float(-coefficients[1] / (2.0 * coefficients[0]))
                if x_values[index - 1] <= candidate <= x_values[index + 1]:
                    center = candidate
        centers.append(center)
    return np.asarray(centers)


def linewidths(excited_population: np.ndarray) -> np.ndarray:
    central_mask = np.abs(DETUNING_MHZ) <= FIT_WINDOW_MHZ
    central_detuning = DETUNING_MHZ[central_mask]
    return np.asarray(
        [
            fwhm_half_depth(
                central_detuning,
                row[central_mask],
                smooth_sigma_pts=0,
            )[0]
            for row in excited_population
        ]
    )


def simulation_kwargs(duration_us: float) -> dict[str, object]:
    return {
        "duration_us": duration_us,
        "detuning_mhz": DETUNING_MHZ,
        "rabi_mhz": RABI_MHZ,
        "t1_us": T1_US,
        "t_phi_us": T_PHI_US,
        "anharmonicity_mhz": ANHARMONICITY_MHZ,
        "num_steps_per_half": int(
            np.ceil((duration_us / 2.0) * STEPS_PER_US)
        ),
        "cutoff": CUTOFF,
        "echo": ECHO,
        "order": ORDER,
        "stark_correction_mode": "accumulated_phase",
    }


def optimize_kappa(
    duration_us: float,
) -> tuple[float, dict[float, dict[str, object]]]:
    optimization: dict[float, dict[str, object]] = {}
    common = simulation_kwargs(duration_us)
    common.update(
        detuning_mhz=OPT_DETUNING_MHZ,
        rabi_mhz=OPT_RABI_MHZ,
        drag_beta=DRAG_BETA,
    )

    def evaluate(kappa: float) -> dict[str, object]:
        key = float(np.round(kappa, 8))
        if key not in optimization:
            result = simulate_qutrit_map(
                **common,
                stark_kappa_mhz_inv=key,
            )
            centers = fitted_central_minima(
                OPT_DETUNING_MHZ,
                result.excited,
            )
            optimization[key] = {
                "centers": centers,
                "rms_zero": float(np.sqrt(np.mean(centers**2))),
                "rms_centered": float(np.std(centers)),
                "max_abs": float(np.max(np.abs(centers))),
                "max_leakage": float(np.max(result.second_excited)),
            }
        return optimization[key]

    coarse_kappas = np.linspace(-0.005, 0.005, 11)
    for kappa in coarse_kappas:
        evaluate(float(kappa))
    coarse_best = min(
        coarse_kappas,
        key=lambda value: float(evaluate(float(value))["rms_zero"]),
    )
    fine_kappas = np.linspace(coarse_best - 0.001, coarse_best + 0.001, 9)
    for kappa in fine_kappas:
        evaluate(float(kappa))
    selected = min(
        optimization,
        key=lambda value: float(optimization[value]["rms_zero"]),
    )
    return selected, optimization


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(
            path,
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def add_t2_bounds(ax: plt.Axes, *, color: str) -> None:
    ax.axvline(0, color=color, lw=0.65, ls="--", alpha=0.8)
    for bound in (-T2_LIMIT_HALF_WIDTH_MHZ, T2_LIMIT_HALF_WIDTH_MHZ):
        ax.axvline(bound, color=color, lw=0.8, ls=":", alpha=0.95)


def plot_pulse_waveform(
    directory: Path,
    duration_us: float,
    selected_kappa: float,
) -> None:
    """Compare the played complex envelope before and after phase correction."""
    half_duration = duration_us / 2.0
    sigma_us = half_duration / np.sqrt(CUTOFF ** (-1.0 / ORDER) - 1.0)
    time_us = np.linspace(-half_duration, half_duration, 20_001)
    base = (1.0 + (time_us / sigma_us) ** 2) ** (-ORDER)
    echo_sign = np.where(time_us < 0.0, 1.0, -1.0 if ECHO else 1.0)
    old_envelope = PULSE_PLOT_RABI_MHZ * echo_sign * base
    correction_mhz = selected_kappa * (PULSE_PLOT_RABI_MHZ * base) ** 2
    increments = (
        np.pi
        * (correction_mhz[1:] + correction_mhz[:-1])
        * np.diff(time_us)
    )
    phase_rad = np.concatenate(([0.0], np.cumsum(increments)))
    corrected_envelope = old_envelope * np.exp(-1j * phase_rad)

    zoom = np.abs(time_us) <= min(50.0 * sigma_us, half_duration)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9.2, 2.65),
        constrained_layout=True,
    )
    axes[0].plot(time_us[zoom], old_envelope[zoom], label="I")
    axes[0].plot(time_us[zoom], np.zeros(np.count_nonzero(zoom)), "--", label="Q")
    axes[0].set(title="Original pulse", ylabel=r"$I,Q$ (MHz)")
    axes[1].plot(time_us[zoom], corrected_envelope.real[zoom], label="I")
    axes[1].plot(time_us[zoom], corrected_envelope.imag[zoom], "--", label="Q")
    axes[1].plot(
        time_us[zoom],
        np.abs(corrected_envelope[zoom]),
        color="0.45",
        lw=0.7,
        ls=":",
        label=r"$|I+iQ|$",
    )
    axes[1].set(title="Accumulated-phase pulse")
    for axis in axes[:2]:
        axis.axvline(0.0, color="0.5", lw=0.6, ls=":")
        axis.set_xlabel(r"$t$ ($\mu$s), central $|t|\leq50\sigma$")
        axis.legend(fontsize=6.5)

    axes[2].plot(time_us, correction_mhz, color="C3", label=r"$\delta f_{\rm corr}$")
    axes[2].set(
        title="Frequency correction and phase",
        xlabel=r"$t$ ($\mu$s), full pulse",
        ylabel=r"$\delta f_{\rm corr}$ (MHz)",
    )
    phase_axis = axes[2].twinx()
    phase_axis.plot(
        time_us,
        phase_rad / (2.0 * np.pi),
        color="C2",
        label=r"$\phi/2\pi$",
    )
    phase_axis.set_ylabel(r"Accumulated phase $\phi/2\pi$ (cycles)")
    lines = axes[2].lines + phase_axis.lines
    axes[2].legend(lines, [line.get_label() for line in lines], fontsize=6.5)
    fig.suptitle(
        rf"$L={duration_us:g}\,\mu$s, $\Omega_0/2\pi={PULSE_PLOT_RABI_MHZ:g}$ MHz, "
        rf"$\kappa={selected_kappa:+.6f}$ MHz$^{{-1}}$",
        fontsize=8,
    )
    save_figure(fig, directory, "00_pulse_waveform_comparison")


def plot_optimization(
    directory: Path,
    selected_kappa: float,
    optimization: dict[float, dict[str, object]],
) -> None:
    ordered = np.asarray(sorted(optimization))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.8),
        constrained_layout=True,
    )
    axes[0].plot(
        ordered,
        [1e3 * float(optimization[kappa]["rms_zero"]) for kappa in ordered],
        "o-",
    )
    axes[0].axvline(selected_kappa, color="C3", ls="--", lw=0.9)
    axes[0].axvline(KAPPA_THEORY_MHZ_INV, color="0.35", ls=":", lw=0.8)
    axes[0].set(
        xlabel=r"$\kappa$ (MHz$^{-1}$)",
        ylabel="Center RMS from zero (kHz)",
    )
    for kappa, label in (
        (0.0, r"$\kappa=0$"),
        (selected_kappa, "optimized"),
    ):
        axes[1].plot(
            OPT_RABI_MHZ,
            1e3 * np.asarray(optimization[kappa]["centers"]),
            "o-",
            label=label,
        )
    axes[1].axhline(0, color="0.35", lw=0.7, ls=":")
    axes[1].set(
        xlabel=r"$\Omega_0/2\pi$ (MHz)",
        ylabel="Fitted center (kHz)",
    )
    axes[1].legend(fontsize=7)
    save_figure(fig, directory, "01_kappa_optimization")


def plot_pe_maps(directory: Path, plain: object, corrected: object) -> None:
    comparison = (
        ("Without correction", plain),
        ("With accumulated-phase correction", corrected),
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for ax, (title, result) in zip(axes, comparison, strict=True):
        image = ax.pcolormesh(
            DETUNING_MHZ,
            RABI_MHZ,
            result.excited,
            shading="auto",
            cmap="viridis",
            vmin=0,
            vmax=1,
            rasterized=True,
        )
        add_t2_bounds(ax, color="white")
        ax.set_title(title)
        ax.set_xlabel(r"$\Delta/2\pi$ (MHz)")
    axes[0].set_ylabel(r"$\Omega_0/2\pi$ (MHz)")
    fig.colorbar(
        image,
        ax=axes,
        pad=0.02,
        label=r"Excited-state population $P_e$",
    )
    save_figure(fig, directory, "02_pe_maps")


def plot_1d_sweeps(directory: Path, plain: object, corrected: object) -> None:
    comparison = (
        ("Without correction", plain),
        ("With accumulated-phase correction", corrected),
    )
    indices = [
        int(np.argmin(np.abs(RABI_MHZ - value))) for value in SWEEP_RABI_MHZ
    ]
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.2, 4.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()
    for ax, index in zip(flat_axes, indices, strict=False):
        for title, result in comparison:
            ax.plot(
                DETUNING_MHZ,
                result.excited[index],
                lw=1.15,
                label=title,
            )
        add_t2_bounds(ax, color="0.35")
        ax.set_title(rf"$\Omega_0/2\pi={RABI_MHZ[index]:g}$ MHz")
        ax.set_xlabel(r"$\Delta/2\pi$ (MHz)")
        ax.set_ylabel(r"$P_e$")
    flat_axes[-1].axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    flat_axes[-1].legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        fontsize=8,
    )
    save_figure(fig, directory, "03_pe_1d_detuning_sweeps")


def plot_fit_metrics(
    directory: Path,
    plain_centers: np.ndarray,
    corrected_centers: np.ndarray,
    plain_widths: np.ndarray,
    corrected_widths: np.ndarray,
) -> None:
    fit_rabi = RABI_MHZ[1:]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.9),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(
        fit_rabi,
        1e3 * plain_centers,
        "o-",
        ms=3,
        label="Without correction",
    )
    axes[0].plot(
        fit_rabi,
        1e3 * corrected_centers,
        "o-",
        ms=3,
        label="With accumulated-phase correction",
    )
    axes[1].plot(
        fit_rabi,
        plain_widths / T2_LIMIT_FWHM_MHZ,
        "o-",
        ms=3,
        label="Without correction",
    )
    axes[1].plot(
        fit_rabi,
        corrected_widths / T2_LIMIT_FWHM_MHZ,
        "o-",
        ms=3,
        label="With accumulated-phase correction",
    )
    axes[0].axhline(0, color="0.35", lw=0.7, ls=":")
    axes[1].axhline(1, color="0.35", lw=0.7, ls=":")
    axes[0].set_ylabel(r"Fitted $f_{10}$ offset (kHz)")
    axes[1].set_ylabel(r"$P_e$ dip FWHM / $[1/(\pi T_2)]$")
    for ax in axes:
        ax.set_xlabel(r"$\Omega_0/2\pi$ (MHz)")
    axes[0].legend(fontsize=7)
    save_figure(fig, directory, "04_f10_fwhm")


def plot_populations(directory: Path, plain: object, corrected: object) -> None:
    results = (
        ("Echo-root-Lorentzian", plain),
        ("Accumulated-phase echo-root-Lorentzian", corrected),
    )
    population_names = (
        ("ground", r"$P_g$"),
        ("excited", r"$P_e$"),
        ("second_excited", r"$P_f$"),
    )
    pf_vmax = max(result.second_excited.max() for _, result in results)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(8.2, 5.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    column_images: list[object | None] = [None, None, None]
    for row, (protocol, result) in enumerate(results):
        for col, (attribute, label) in enumerate(population_names):
            ax = axes[row, col]
            image = ax.pcolormesh(
                DETUNING_MHZ,
                RABI_MHZ,
                getattr(result, attribute),
                shading="auto",
                cmap="viridis",
                vmin=0,
                vmax=pf_vmax if attribute == "second_excited" else 1,
                rasterized=True,
            )
            column_images[col] = image
            add_t2_bounds(ax, color="white")
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(protocol + "\n" + r"$\Omega_0/2\pi$ (MHz)")
            if row == 1:
                ax.set_xlabel(r"$\Delta/2\pi$ (MHz)")
    for col, image in enumerate(column_images):
        fig.colorbar(
            image,
            ax=axes[:, col],
            pad=0.012,
            fraction=0.045,
            label="Final population" if col < 2 else r"Leakage $P_f$",
        )
    save_figure(fig, directory, "05_all_populations")


def plot_leakage(
    directory: Path,
    plain: object,
    corrected: object,
    selected_kappa: float,
) -> None:
    delta_pf = corrected.second_excited - plain.second_excited
    limit = float(np.max(np.abs(delta_pf)))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.8),
        constrained_layout=True,
    )
    difference = axes[0].pcolormesh(
        DETUNING_MHZ,
        RABI_MHZ,
        delta_pf,
        shading="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        rasterized=True,
    )
    add_t2_bounds(axes[0], color="0.15")
    axes[0].set_xlabel(r"$\Delta/2\pi$ (MHz)")
    axes[0].set_ylabel(r"$\Omega_0/2\pi$ (MHz)")
    axes[0].set_title(r"$P_f^{\rm corrected}-P_f^{\rm plain}$")
    fig.colorbar(difference, ax=axes[0], pad=0.02, label=r"$\Delta P_f$")
    axes[1].plot(
        RABI_MHZ,
        plain.second_excited.max(axis=1),
        label="Plain pulse",
    )
    axes[1].plot(
        RABI_MHZ,
        corrected.second_excited.max(axis=1),
        label=rf"Accumulated phase, $\beta=0$, $\kappa={selected_kappa:+.4f}$",
    )
    axes[1].set_xlabel(r"$\Omega_0/2\pi$ (MHz)")
    axes[1].set_ylabel(r"$\max_{\Delta} P_f$")
    axes[1].set_title("Worst leakage in detuning window")
    axes[1].legend(fontsize=7)
    save_figure(fig, directory, "06_leakage")


def summary_from_npz(path: Path) -> dict[str, float | int | str]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "duration_us": float(data["duration_us"]),
            "kappa_mhz_inv": float(data["selected_kappa_mhz_inv"]),
            "optimization_center_rms_khz": float(
                data["optimization_center_rms_mhz"] * 1e3
            ),
            "optimization_max_center_khz": float(
                data["optimization_max_center_mhz"] * 1e3
            ),
            "plain_max_center_khz": float(
                np.max(np.abs(data["plain_centers_mhz"])) * 1e3
            ),
            "corrected_max_center_khz": float(
                np.max(np.abs(data["corrected_centers_mhz"])) * 1e3
            ),
            "plain_fwhm_t2_min": float(
                np.nanmin(data["plain_fwhm_mhz"] / T2_LIMIT_FWHM_MHZ)
            ),
            "plain_fwhm_t2_max": float(
                np.nanmax(data["plain_fwhm_mhz"] / T2_LIMIT_FWHM_MHZ)
            ),
            "corrected_fwhm_t2_min": float(
                np.nanmin(data["corrected_fwhm_mhz"] / T2_LIMIT_FWHM_MHZ)
            ),
            "corrected_fwhm_t2_max": float(
                np.nanmax(data["corrected_fwhm_mhz"] / T2_LIMIT_FWHM_MHZ)
            ),
            "plain_unresolved": int(
                np.count_nonzero(~np.isfinite(data["plain_fwhm_mhz"]))
            ),
            "corrected_unresolved": int(
                np.count_nonzero(~np.isfinite(data["corrected_fwhm_mhz"]))
            ),
            "plain_max_leakage": float(np.max(data["plain_pf"])),
            "corrected_max_leakage": float(np.max(data["corrected_pf"])),
            "elapsed_s": float(data["elapsed_s"]),
            "data_path": str(path),
        }


def run_duration(duration_us: float, *, force: bool) -> dict[str, object]:
    apply_figure_style(FigureVariant.PAPER)
    label = duration_label(duration_us)
    duration_data_dir = DATA_DIR / label
    duration_figure_dir = FIGURE_DIR / label
    duration_data_dir.mkdir(parents=True, exist_ok=True)
    duration_figure_dir.mkdir(parents=True, exist_ok=True)
    data_path = duration_data_dir / "results.npz"
    if data_path.exists() and not force:
        print(f"L={duration_us:g} us: using {data_path}", flush=True)
        return summary_from_npz(data_path)

    started = time.perf_counter()
    print(f"L={duration_us:g} us: optimizing kappa", flush=True)
    selected_kappa, optimization = optimize_kappa(duration_us)
    best = optimization[selected_kappa]
    plot_optimization(duration_figure_dir, selected_kappa, optimization)

    common = simulation_kwargs(duration_us)
    print(f"L={duration_us:g} us: simulating plain map", flush=True)
    plain = simulate_qutrit_map(
        **common,
        drag_beta=0.0,
        stark_kappa_mhz_inv=0.0,
    )
    print(f"L={duration_us:g} us: simulating corrected map", flush=True)
    corrected = simulate_qutrit_map(
        **common,
        drag_beta=DRAG_BETA,
        stark_kappa_mhz_inv=selected_kappa,
    )

    for name, result in (("plain", plain), ("corrected", corrected)):
        total = result.ground + result.excited + result.second_excited
        if not np.allclose(total, 1.0, atol=2e-6):
            raise RuntimeError(f"L={duration_us:g} us {name}: normalization failed")
        for population in (
            result.ground,
            result.excited,
            result.second_excited,
        ):
            if population.min() < -2e-6 or population.max() > 1.0 + 2e-6:
                raise RuntimeError(
                    f"L={duration_us:g} us {name}: population out of range"
                )

    fit_mask = RABI_MHZ > 0
    plain_centers = fitted_central_minima(
        DETUNING_MHZ,
        plain.excited[fit_mask],
    )
    corrected_centers = fitted_central_minima(
        DETUNING_MHZ,
        corrected.excited[fit_mask],
    )
    plain_widths = linewidths(plain.excited[fit_mask])
    corrected_widths = linewidths(corrected.excited[fit_mask])

    plot_pe_maps(duration_figure_dir, plain, corrected)
    plot_1d_sweeps(duration_figure_dir, plain, corrected)
    plot_fit_metrics(
        duration_figure_dir,
        plain_centers,
        corrected_centers,
        plain_widths,
        corrected_widths,
    )
    plot_populations(duration_figure_dir, plain, corrected)
    plot_leakage(duration_figure_dir, plain, corrected, selected_kappa)

    ordered_kappas = np.asarray(sorted(optimization))
    elapsed_s = time.perf_counter() - started
    np.savez_compressed(
        data_path,
        duration_us=duration_us,
        cutoff=CUTOFF,
        order=ORDER,
        anharmonicity_mhz=ANHARMONICITY_MHZ,
        t1_us=T1_US,
        t2_us=T2_US,
        t_phi_us=T_PHI_US,
        drag_beta=DRAG_BETA,
        echo=ECHO,
        steps_per_us=STEPS_PER_US,
        t2_limit_fwhm_mhz=T2_LIMIT_FWHM_MHZ,
        kappa_theory_mhz_inv=KAPPA_THEORY_MHZ_INV,
        selected_kappa_mhz_inv=selected_kappa,
        optimization_center_rms_mhz=float(best["rms_zero"]),
        optimization_centered_rms_mhz=float(best["rms_centered"]),
        optimization_max_center_mhz=float(best["max_abs"]),
        optimization_max_leakage=float(best["max_leakage"]),
        optimization_kappas_mhz_inv=ordered_kappas,
        optimization_rms_mhz=np.asarray(
            [optimization[kappa]["rms_zero"] for kappa in ordered_kappas]
        ),
        optimization_centers_mhz=np.stack(
            [optimization[kappa]["centers"] for kappa in ordered_kappas]
        ),
        opt_rabi_mhz=OPT_RABI_MHZ,
        detuning_mhz=DETUNING_MHZ,
        rabi_mhz=RABI_MHZ,
        fit_rabi_mhz=RABI_MHZ[fit_mask],
        plain_pg=plain.ground,
        plain_pe=plain.excited,
        plain_pf=plain.second_excited,
        corrected_pg=corrected.ground,
        corrected_pe=corrected.excited,
        corrected_pf=corrected.second_excited,
        plain_centers_mhz=plain_centers,
        corrected_centers_mhz=corrected_centers,
        plain_fwhm_mhz=plain_widths,
        corrected_fwhm_mhz=corrected_widths,
        plain_max_pf_by_rabi=plain.second_excited.max(axis=1),
        corrected_max_pf_by_rabi=corrected.second_excited.max(axis=1),
        elapsed_s=elapsed_s,
    )
    summary = summary_from_npz(data_path)
    print(
        f"L={duration_us:g} us: kappa={selected_kappa:+.6f}, "
        f"max center {summary['plain_max_center_khz']:.2f} -> "
        f"{summary['corrected_max_center_khz']:.2f} kHz, "
        f"elapsed={elapsed_s:.1f} s",
        flush=True,
    )
    return summary


def write_csv_outputs(summaries: list[dict[str, object]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "duration_us",
        "kappa_mhz_inv",
        "optimization_center_rms_khz",
        "optimization_max_center_khz",
        "plain_max_center_khz",
        "corrected_max_center_khz",
        "plain_fwhm_t2_min",
        "plain_fwhm_t2_max",
        "corrected_fwhm_t2_min",
        "corrected_fwhm_t2_max",
        "plain_unresolved",
        "corrected_unresolved",
        "plain_max_leakage",
        "corrected_max_leakage",
        "elapsed_s",
        "data_path",
    ]
    with SUMMARY_CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(
            {field: summary[field] for field in summary_fields}
            for summary in summaries
        )

    amplitude_fields = [
        "duration_us",
        "rabi_mhz",
        "plain_center_khz",
        "corrected_center_khz",
        "plain_fwhm_mhz",
        "corrected_fwhm_mhz",
        "plain_fwhm_t2",
        "corrected_fwhm_t2",
        "plain_max_leakage",
        "corrected_max_leakage",
    ]
    with AMPLITUDE_CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=amplitude_fields)
        writer.writeheader()
        for summary in summaries:
            with np.load(str(summary["data_path"]), allow_pickle=False) as data:
                for index, rabi_mhz in enumerate(data["fit_rabi_mhz"]):
                    full_index = index + 1
                    writer.writerow(
                        {
                            "duration_us": summary["duration_us"],
                            "rabi_mhz": float(rabi_mhz),
                            "plain_center_khz": float(
                                1e3 * data["plain_centers_mhz"][index]
                            ),
                            "corrected_center_khz": float(
                                1e3 * data["corrected_centers_mhz"][index]
                            ),
                            "plain_fwhm_mhz": float(
                                data["plain_fwhm_mhz"][index]
                            ),
                            "corrected_fwhm_mhz": float(
                                data["corrected_fwhm_mhz"][index]
                            ),
                            "plain_fwhm_t2": float(
                                data["plain_fwhm_mhz"][index]
                                / T2_LIMIT_FWHM_MHZ
                            ),
                            "corrected_fwhm_t2": float(
                                data["corrected_fwhm_mhz"][index]
                                / T2_LIMIT_FWHM_MHZ
                            ),
                            "plain_max_leakage": float(
                                data["plain_max_pf_by_rabi"][full_index]
                            ),
                            "corrected_max_leakage": float(
                                data["corrected_max_pf_by_rabi"][full_index]
                            ),
                        }
                    )


def plot_duration_summary(summaries: list[dict[str, object]]) -> None:
    durations = np.asarray([summary["duration_us"] for summary in summaries])
    kappas = np.asarray([summary["kappa_mhz_inv"] for summary in summaries])
    plain_centers = np.asarray(
        [summary["plain_max_center_khz"] for summary in summaries]
    )
    corrected_centers = np.asarray(
        [summary["corrected_max_center_khz"] for summary in summaries]
    )
    plain_fwhm = np.asarray(
        [summary["plain_fwhm_t2_max"] for summary in summaries]
    )
    corrected_fwhm = np.asarray(
        [summary["corrected_fwhm_t2_max"] for summary in summaries]
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(8.4, 2.7),
        constrained_layout=True,
    )
    axes[0].plot(durations, kappas, "o-")
    axes[0].axhline(
        KAPPA_THEORY_MHZ_INV,
        color="0.35",
        ls=":",
        lw=0.8,
        label="weak-drive theory",
    )
    axes[0].set(
        xlabel=r"Pulse length $L$ ($\mu$s)",
        ylabel=r"$\kappa$ (MHz$^{-1}$)",
    )
    axes[0].legend(fontsize=6.5)
    axes[1].plot(durations, plain_centers, "o-", label="without correction")
    axes[1].plot(durations, corrected_centers, "o-", label="corrected")
    axes[1].set(
        xlabel=r"Pulse length $L$ ($\mu$s)",
        ylabel=r"Max $|f_{10}|$ offset (kHz)",
    )
    axes[1].legend(fontsize=6.5)
    axes[2].plot(durations, plain_fwhm, "o-", label="without correction")
    axes[2].plot(durations, corrected_fwhm, "o-", label="corrected")
    axes[2].axhline(1, color="0.35", ls=":", lw=0.8)
    axes[2].set(
        xlabel=r"Pulse length $L$ ($\mu$s)",
        ylabel=r"Max FWHM / $[1/(\pi T_2)]$",
    )
    axes[2].legend(fontsize=6.5)
    save_figure(fig, FIGURE_DIR, "duration_summary")


def data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def format_number(value: object, digits: int = 2) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def summary_table(summaries: list[dict[str, object]]) -> str:
    rows = []
    for summary in summaries:
        rows.append(
            "<tr>"
            f"<td>{summary['duration_us']:g}</td>"
            f"<td>{float(summary['kappa_mhz_inv']):+.6f}</td>"
            f"<td>{format_number(summary['plain_max_center_khz'])}</td>"
            f"<td>{format_number(summary['corrected_max_center_khz'])}</td>"
            f"<td>{format_number(summary['plain_fwhm_t2_min'])}–"
            f"{format_number(summary['plain_fwhm_t2_max'])}</td>"
            f"<td>{format_number(summary['corrected_fwhm_t2_min'])}–"
            f"{format_number(summary['corrected_fwhm_t2_max'])}</td>"
            f"<td>{format_number(summary['plain_max_leakage'], 4)}</td>"
            f"<td>{format_number(summary['corrected_max_leakage'], 4)}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Length (µs)</th><th>κ (MHz⁻¹)</th>"
        "<th>Plain max |center| (kHz)</th>"
        "<th>Corrected max |center| (kHz)</th>"
        "<th>Plain FWHM/T₂</th><th>Corrected FWHM/T₂</th>"
        "<th>Plain max P<sub>f</sub></th>"
        "<th>Corrected max P<sub>f</sub></th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def amplitude_table(data: np.lib.npyio.NpzFile) -> str:
    rows = []
    for index, rabi_mhz in enumerate(data["fit_rabi_mhz"]):
        full_index = index + 1
        rows.append(
            "<tr>"
            f"<td>{float(rabi_mhz):.1f}</td>"
            f"<td>{1e3 * float(data['plain_centers_mhz'][index]):.3f}</td>"
            f"<td>{1e3 * float(data['corrected_centers_mhz'][index]):.3f}</td>"
            f"<td>{format_number(data['plain_fwhm_mhz'][index] / T2_LIMIT_FWHM_MHZ, 3)}</td>"
            f"<td>{format_number(data['corrected_fwhm_mhz'][index] / T2_LIMIT_FWHM_MHZ, 3)}</td>"
            f"<td>{float(data['plain_max_pf_by_rabi'][full_index]):.5f}</td>"
            f"<td>{float(data['corrected_max_pf_by_rabi'][full_index]):.5f}</td>"
            "</tr>"
        )
    return (
        "<details><summary>Amplitude-resolved numerical table</summary>"
        '<div class="table-wrap amplitude-table"><table><thead><tr>'
        "<th>Ω₀/2π (MHz)</th><th>Plain center (kHz)</th>"
        "<th>Corrected center (kHz)</th><th>Plain FWHM/T₂</th>"
        "<th>Corrected FWHM/T₂</th><th>Plain max P<sub>f</sub></th>"
        "<th>Corrected max P<sub>f</sub></th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></details>"
    )


def figure_block(path: Path, caption: str) -> str:
    return (
        "<figure>"
        f'<img src="{data_uri(path)}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def duration_panel(summary: dict[str, object]) -> str:
    duration_us = float(summary["duration_us"])
    label = duration_label(duration_us)
    figure_dir = FIGURE_DIR / label
    with np.load(str(summary["data_path"]), allow_pickle=False) as data:
        metric_line = (
            '<div class="metrics">'
            f'<span><b>κ</b> {float(summary["kappa_mhz_inv"]):+.6f} MHz⁻¹</span>'
            f'<span><b>Max |center|</b> {format_number(summary["plain_max_center_khz"])} → '
            f'{format_number(summary["corrected_max_center_khz"])} kHz</span>'
            f'<span><b>Corrected FWHM/T₂</b> '
            f'{format_number(summary["corrected_fwhm_t2_min"])}–'
            f'{format_number(summary["corrected_fwhm_t2_max"])}</span>'
            f'<span><b>Max corrected P<sub>f</sub></b> '
            f'{format_number(summary["corrected_max_leakage"], 4)}</span>'
            "</div>"
        )
        figures = "".join(
            (
                figure_block(
                    figure_dir / "00_pulse_waveform_comparison.png",
                    f"Original and corrected {duration_us:g} µs pulse at "
                    f"{PULSE_PLOT_RABI_MHZ:g} MHz",
                ),
                figure_block(
                    figure_dir / "01_kappa_optimization.png",
                    "Quadratic phase-coefficient optimization",
                ),
                figure_block(
                    figure_dir / "02_pe_maps.png",
                    "Excited-state maps with T2-limit bounds",
                ),
                figure_block(
                    figure_dir / "03_pe_1d_detuning_sweeps.png",
                    "One-dimensional detuning sweeps",
                ),
                figure_block(
                    figure_dir / "04_f10_fwhm.png",
                    "Fitted f10 offsets and FWHM in T2 units",
                ),
                figure_block(
                    figure_dir / "05_all_populations.png",
                    "Complete Pg, Pe, and Pf maps",
                ),
                figure_block(
                    figure_dir / "06_leakage.png",
                    "Leakage difference and worst-case leakage",
                ),
            )
        )
        table = amplitude_table(data)
    return (
        f'<section class="tab-panel" role="tabpanel" id="panel-{label}" '
        f'aria-labelledby="tab-{label}" hidden>'
        + metric_line
        + f'<div class="figure-grid">{figures}</div>'
        + table
        + "</section>"
    )


def theory_panel() -> str:
    kappa_theory = KAPPA_THEORY_MHZ_INV
    return f"""<section class="tab-panel theory" role="tabpanel" id="panel-theory"
aria-labelledby="tab-theory">
<h2>Why the accumulated phase cancels the AC-Stark shift</h2>
<p>A strong off-resonant coupling to the transmon's second transition moves the
effective <i>0→1</i> resonance. To leading order this displacement is quadratic
in the in-phase drive amplitude. We therefore add an equal-and-opposite
quadratic frequency correction.</p>
<div class="equation" role="math" aria-label="delta f correction equals kappa times f I squared">
δf<sub>corr</sub>(t) = κ f<sub>I</sub><sup>2</sup>(t),
&nbsp;&nbsp; f<sub>I</sub>(t) ≡ Ω<sub>I</sub>(t)/(2π)
</div>
<p>Here frequencies are in MHz, time is in µs, and κ has units MHz⁻¹. The
correction added to the swept drive detuning is</p>
<div class="equation" role="math" aria-label="effective detuning equals swept detuning plus correction">
Δ<sub>eff</sub>(t) = Δ<sub>sweep</sub> + Δ<sub>corr</sub>(t),
&nbsp;&nbsp; Δ<sub>corr</sub>(t) = 2πκ f<sub>I</sub><sup>2</sup>(t).
</div>
<p>For a weakly anharmonic transmon, second-order perturbation theory gives the
starting estimate</p>
<div class="equation" role="math" aria-label="kappa approximately minus one over twice the anharmonicity">
κ<sub>theory</sub> ≈ −1/[2(α/2π)] = {kappa_theory:.4f} MHz<sup>−1</sup>
&nbsp;&nbsp; for α/2π = {ANHARMONICITY_MHZ:g} MHz.
</div>
<p>The numerical sweep then refines κ because the finite pulse, strong drive,
decoherence, and third-level dynamics are not fully captured by the weak-drive
estimate.</p>

<h2>Hardware form: integrate the correction into phase</h2>
<p>Instead of programming a separate time-dependent detuning, accumulate its
phase and rotate the complex envelope:</p>
<div class="equation" role="math" aria-label="phase is the integral of the quadratic frequency correction">
φ(t) = ∫<sub>t₀</sub><sup>t</sup> Δ<sub>corr</sub>(t′)dt′
= 2πκ ∫<sub>t₀</sub><sup>t</sup> f<sub>I</sub><sup>2</sup>(t′)dt′,
</div>
<div class="equation" role="math" aria-label="corrected complex envelope equals original envelope times exponential minus i phase">
[I(t)+iQ(t)]<sub>corr</sub> = [I(t)+iQ(t)] exp[−iφ(t)].
</div>
{figure_block(
    FIGURE_DIR / "20us" / "00_pulse_waveform_comparison.png",
    "Original and accumulated-phase-corrected 20 µs pulse at 40 MHz",
)}
<p>Because dφ/dt = Δ<sub>corr</sub>(t), this phase modulation produces exactly
the desired instantaneous frequency addition. It is <strong>not</strong>
exp[−iΔ<sub>corr</sub>(t)t]: differentiating that expression would also create
the unwanted term t·dΔ<sub>corr</sub>/dt.</p>

<h2>Connection to DRAG intuition</h2>
<p>DRAG uses a quadrature proportional to the envelope derivative to suppress
leakage and compensate drive-induced phase errors. The same physical intuition
applies here: the pulse intensity produces an AC-Stark frequency error, and a
phase ramp with matching instantaneous frequency cancels it. In this report
β = {DRAG_BETA:g}, so no derivative quadrature is used—the only modification is
the accumulated phase proportional to ∫Ω<sub>I</sub><sup>2</sup>(t)dt.
This is the AC-Stark correction derived in
<a href="https://arxiv.org/abs/0901.0534"
target="_blank" rel="noopener noreferrer"><i>Motzoi et al.</i>,
Phys. Rev. Lett. <b>103</b>, 110501 (2009)</a>
(DOI: 10.1103/PhysRevLett.103.110501): see Eqs. (8)–(10), and the
following phase-ramping prescription when real-time detuning is unavailable.</p>
<aside class="note"><b>Sign convention.</b> Detuning is drive frequency minus
the bare 0→1 frequency. Positive κ therefore adds positive drive detuning. The
optimized sign is the one that opposes the fitted AC-Stark center displacement.</aside>
</section>"""


def write_report(summaries: list[dict[str, object]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tabs = [
        '<button type="button" role="tab" id="tab-theory" '
        'aria-controls="panel-theory" aria-selected="true">Theory</button>',
        '<button type="button" role="tab" id="tab-summary" '
        'aria-controls="panel-summary" aria-selected="false">Summary</button>',
    ]
    panels = [
        theory_panel(),
        '<section class="tab-panel" role="tabpanel" id="panel-summary" '
        'aria-labelledby="tab-summary" hidden>'
        + figure_block(
            FIGURE_DIR / "duration_summary.png",
            "Cross-duration summary",
        )
        + summary_table(summaries)
        + "</section>"
    ]
    for summary in summaries:
        duration_us = float(summary["duration_us"])
        label = duration_label(duration_us)
        tabs.append(
            f'<button type="button" role="tab" id="tab-{label}" '
            f'aria-controls="panel-{label}" aria-selected="false">'
            f"{duration_us:g} µs</button>"
        )
        panels.append(duration_panel(summary))

    metadata = {
        "durations_us": [summary["duration_us"] for summary in summaries],
        "cutoff": CUTOFF,
        "beta": DRAG_BETA,
        "t1_us": T1_US,
        "t2_us": T2_US,
        "t_phi_us": T_PHI_US,
        "anharmonicity_mhz": ANHARMONICITY_MHZ,
        "rabi_grid": [float(RABI_MHZ.min()), float(RABI_MHZ.max()), RABI_MHZ.size],
        "detuning_grid": [
            float(DETUNING_MHZ.min()),
            float(DETUNING_MHZ.max()),
            DETUNING_MHZ.size,
        ],
        "steps_per_us": STEPS_PER_US,
        "t2_limit_fwhm_khz": 1e3 * T2_LIMIT_FWHM_MHZ,
    }
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Accumulated-phase spectroscopy duration sweep</title>
<style>
:root {{ color-scheme: light dark; --bg: #fff; --fg: #17212b; --muted: #586575; --line: #d7dde5; --accent: #1769aa; --panel: #f7f9fb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #11161c; --fg: #e9eef4; --muted: #aab5c2; --line: #35404d; --accent: #63b3ed; --panel: #171e26; }} }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 24px; background: var(--bg); color: var(--fg); font: 15px/1.45 system-ui, sans-serif; }}
main {{ max-width: 1500px; margin: 0 auto; }}
a {{ color: var(--accent); }}
h1 {{ margin: 0 0 6px; font-size: clamp(22px, 3vw, 34px); font-weight: 600; }}
h2 {{ margin: 26px 0 8px; font-size: clamp(18px, 2vw, 23px); font-weight: 600; }}
.theory {{ max-width: 980px; }}
.theory p {{ max-width: 82ch; }}
.equation {{ margin: 14px 0; padding: 14px 18px; overflow-x: auto; background: var(--panel); border-left: 4px solid var(--accent); font: 18px/1.6 Georgia, "Times New Roman", serif; white-space: nowrap; }}
.note {{ display: block; margin-top: 22px; padding: 12px 14px; border: 1px solid var(--line); border-radius: 8px; color: var(--muted); }}
.subtitle {{ margin: 0 0 18px; color: var(--muted); }}
.config {{ display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 0 0 18px; padding: 12px 0; border-block: 1px solid var(--line); color: var(--muted); }}
.tabs {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 18px; }}
.tabs button {{ appearance: none; border: 1px solid var(--line); border-radius: 999px; padding: 8px 14px; background: transparent; color: var(--fg); cursor: pointer; font: inherit; }}
.tabs button[aria-selected="true"] {{ background: var(--accent); border-color: var(--accent); color: white; }}
.tabs button:focus-visible {{ outline: 3px solid var(--accent); outline-offset: 2px; }}
.metrics {{ display: flex; flex-wrap: wrap; gap: 8px 22px; margin-bottom: 18px; padding: 12px 14px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
.figure-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px; }}
figure {{ margin: 0; min-width: 0; }}
figure img {{ display: block; width: 100%; height: auto; background: white; border: 1px solid var(--line); }}
figcaption {{ margin-top: 6px; color: var(--muted); }}
.table-wrap {{ width: 100%; overflow-x: auto; margin-top: 22px; }}
table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--muted); font-weight: 600; }}
details {{ margin-top: 24px; }}
summary {{ cursor: pointer; font-weight: 600; }}
.amplitude-table {{ max-height: 620px; overflow: auto; }}
.amplitude-table thead {{ position: sticky; top: 0; background: var(--bg); }}
footer {{ margin-top: 30px; padding-top: 14px; border-top: 1px solid var(--line); color: var(--muted); }}
@media (max-width: 860px) {{ body {{ padding: 14px; }} .figure-grid {{ grid-template-columns: 1fr; }} }}
@media print {{ body {{ padding: 0; }} .tabs {{ display: none; }} .tab-panel[hidden] {{ display: block; page-break-before: always; }} details > * {{ display: block; }} .figure-grid {{ grid-template-columns: 1fr 1fr; gap: 12px; }} }}
</style>
</head>
<body>
<main>
<h1>Accumulated-phase spectroscopy duration sweep</h1>
<p class="subtitle">Complete qutrit simulations with independent κ optimization for every pulse length.</p>
<div class="config">
<span><b>L</b> = {", ".join(f"{value:g}" for value in DURATIONS_US)} µs</span>
<span><b>cutoff</b> = {CUTOFF:g}</span><span><b>β</b> = {DRAG_BETA:g}</span>
<span><b>T₁</b> = {T1_US:.2f} µs</span><span><b>T₂</b> = {T2_US:.2f} µs</span>
<span><b>α/2π</b> = {ANHARMONICITY_MHZ:g} MHz</span>
<span><b>T₂-limit FWHM</b> = {1e3 * T2_LIMIT_FWHM_MHZ:.2f} kHz</span>
</div>
<nav class="tabs" role="tablist" aria-label="Report sections">{"".join(tabs)}</nav>
{"".join(panels)}
<footer>Raw maps and extracted arrays are stored in compressed NPZ files; summary and amplitude-resolved values are also available as CSV.</footer>
<script type="application/json" id="report-metadata">{html.escape(json.dumps(metadata, separators=(",", ":")))}</script>
</main>
<script>
const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
const panels = Array.from(document.querySelectorAll('[role="tabpanel"]'));
function selectTab(tab) {{
  tabs.forEach((item) => item.setAttribute('aria-selected', String(item === tab)));
  panels.forEach((panel) => {{ panel.hidden = panel.id !== tab.getAttribute('aria-controls'); }});
  history.replaceState(null, '', '#' + tab.id.replace('tab-', ''));
}}
tabs.forEach((tab) => {{
  tab.addEventListener('click', () => selectTab(tab));
  tab.addEventListener('keydown', (event) => {{
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const current = tabs.indexOf(tab);
    let next = current;
    if (event.key === 'ArrowLeft') next = (current - 1 + tabs.length) % tabs.length;
    if (event.key === 'ArrowRight') next = (current + 1) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    tabs[next].focus();
    selectTab(tabs[next]);
  }});
}});
const requested = location.hash.slice(1);
const initial = document.getElementById('tab-' + requested);
if (initial) selectTab(initial);
</script>
</body>
</html>
"""
    REPORT_PATH.write_text(document)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute duration data even when cached NPZ files exist.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of durations to simulate concurrently.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_figure_style(FigureVariant.PAPER)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            future_to_duration = {
                executor.submit(run_duration, duration, force=args.force): duration
                for duration in DURATIONS_US
            }
            for future in as_completed(future_to_duration):
                summaries.append(future.result())
    else:
        for duration in DURATIONS_US:
            summaries.append(run_duration(duration, force=args.force))

    summaries.sort(key=lambda summary: float(summary["duration_us"]))
    for summary in summaries:
        duration_us = float(summary["duration_us"])
        plot_pulse_waveform(
            FIGURE_DIR / duration_label(duration_us),
            duration_us,
            float(summary["kappa_mhz_inv"]),
        )
    write_csv_outputs(summaries)
    plot_duration_summary(summaries)
    write_report(summaries)
    print(SUMMARY_CSV_PATH)
    print(AMPLITUDE_CSV_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
