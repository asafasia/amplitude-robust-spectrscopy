"""Generate the main-text noisy pulse-duration resolution comparison."""

from __future__ import annotations

# Backend and local-source setup must precede pyplot and echospec imports.
# ruff: noqa: E402, I001

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.figures import FigureVariant, apply_figure_style, save_figure
from echospec.experiments.torrey_resonance import (
    OptionsTorreyResonance,
    TorreyFwhmVsRabiExperiment,
)
from echospec.simulation.pulses import PulseArgs, PulseType, choose_pulse
from echospec.utils.parameters import Parameters
from echospec.utils.units import Units as u


DURATIONS_US = (10.0, 20.0, 30.0, 40.0)
CUTOFF = 0.002
DETUNING_MHZ = np.linspace(-1.0, 1.0, 501)
RABI_MHZ = np.geomspace(1e-3, 15.5, 100)
EDGE_POINTS = 20
SMOOTH_SIGMA_POINTS = 0.5
MAX_FWHM_MHZ = 0.12
MAX_ABS_CENTER_MHZ = 0.05
FIT_HALF_WIDTH_MHZ = 0.25
FIT_BASELINE_INNER_MHZ = 0.15
Q_THRESHOLD = 0.10
NOISE_STD = 0.002
MIN_SIGNAL_TO_NOISE = 3.0
NOISE_SEED = 4103
CONSTANT_RABI_MHZ = np.geomspace(RABI_MHZ.min(), RABI_MHZ.max(), 20_001)


def _latex_macro_float(name: str) -> float:
    text = (ROOT / "paper" / "coherence_parameters.tex").read_text()
    match = re.search(rf"\\newcommand\{{\\{name}\}}\{{([0-9.]+)", text)
    if match is None:
        raise ValueError(f"Missing numeric macro {name}")
    return float(match.group(1))


T1_US = _latex_macro_float("MeasuredTOne")
T_PHI_US = _latex_macro_float("EffectiveTPhi")
T2_LIMIT_FWHM_MHZ = _latex_macro_float("EffectiveCoherenceFwhm") / 1e3


def _simulate_echo(duration_us: float) -> np.ndarray:
    """Return final excited-state probability on the metric-extraction grid."""
    duration_s = duration_us * 1e-6
    t1_s = T1_US * 1e-6
    t_phi_s = T_PHI_US * 1e-6
    t2_s = 1.0 / (1.0 / t_phi_s + 1.0 / (2.0 * t1_s))
    detuning_rad_s = 2.0 * np.pi * DETUNING_MHZ[None, :] * 1e6
    rabi_rad_s = 2.0 * np.pi * RABI_MHZ[:, None] * 1e6

    x = np.zeros((RABI_MHZ.size, DETUNING_MHZ.size), dtype=float)
    y = np.zeros_like(x)
    z = np.ones_like(x)

    # Keep the physical time step at or below 10 ns and retain the 2000-step
    # resolution used for the 20-us main-text simulation.
    num_steps = max(2000, int(np.ceil(duration_us * 100.0)))
    dt = duration_s / num_steps
    time_midpoints = (
        -duration_s / 2.0
        + (np.arange(num_steps, dtype=float) + 0.5) * dt
    )
    pulse = choose_pulse(PulseType.LORENTZIAN, True)
    pulse_args = PulseArgs(
        pulse_length=duration_s,
        cutoff=CUTOFF,
        order=0.5,
        zeroed_pulse=False,
    )
    envelope = pulse(time_midpoints, pulse_args)

    def derivatives(
        x_value: np.ndarray,
        y_value: np.ndarray,
        z_value: np.ndarray,
        omega_rad_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            -detuning_rad_s * y_value - x_value / t2_s,
            detuning_rad_s * x_value - omega_rad_s * z_value - y_value / t2_s,
            omega_rad_s * y_value + (1.0 - z_value) / t1_s,
        )

    for envelope_value in envelope:
        omega = rabi_rad_s * envelope_value
        k1 = derivatives(x, y, z, omega)
        k2 = derivatives(
            x + 0.5 * dt * k1[0],
            y + 0.5 * dt * k1[1],
            z + 0.5 * dt * k1[2],
            omega,
        )
        k3 = derivatives(
            x + 0.5 * dt * k2[0],
            y + 0.5 * dt * k2[1],
            z + 0.5 * dt * k2[2],
            omega,
        )
        k4 = derivatives(
            x + dt * k3[0],
            y + dt * k3[1],
            z + dt * k3[2],
            omega,
        )
        x += dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        y += dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        z += dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0

    population = ((1.0 - z) / 2.0).T
    if not np.all(np.isfinite(population)):
        raise RuntimeError(f"Nonfinite population for L={duration_us:g} us")
    if population.min() < -1e-5 or population.max() > 1.0 + 1e-5:
        raise RuntimeError(
            f"Unphysical population for L={duration_us:g} us: "
            f"{population.min():.6g} to {population.max():.6g}"
        )
    return np.clip(population, 0.0, 1.0)


def _dip_gaussian(
    detuning_mhz: np.ndarray,
    offset: float,
    depth: float,
    center_mhz: float,
    sigma_mhz: float,
) -> np.ndarray:
    return offset - depth * np.exp(
        -0.5 * ((detuning_mhz - center_mhz) / sigma_mhz) ** 2
    )


def _extract_metrics(
    traces: np.ndarray,
    *,
    noise_std: float | None,
) -> dict[str, np.ndarray]:
    """Extract centered features, optionally rejecting fits below a noise floor."""
    x_values = DETUNING_MHZ[EDGE_POINTS:-EDGE_POINTS]
    fit_mask = np.abs(x_values) <= FIT_HALF_WIDTH_MHZ
    fit_x_values = x_values[fit_mask]
    step = float(np.median(np.diff(fit_x_values)))
    span = float(np.ptp(fit_x_values))
    fwhm_t2_units = np.full(RABI_MHZ.size, np.nan)
    resolution = np.full(RABI_MHZ.size, np.nan)
    contrast = np.full(RABI_MHZ.size, np.nan)

    for index in range(RABI_MHZ.size):
        trace = np.asarray(traces[:, index], dtype=float)
        interior = gaussian_filter1d(
            trace[EDGE_POINTS:-EDGE_POINTS],
            SMOOTH_SIGMA_POINTS,
        )
        baseline_mask = (
            (np.abs(x_values) >= FIT_BASELINE_INNER_MHZ)
            & (np.abs(x_values) <= FIT_HALF_WIDTH_MHZ)
        )
        baseline = float(np.median(interior[baseline_mask]))
        center_index = int(np.argmin(np.abs(x_values)))
        # At long duration the centered feature can change from a depletion to
        # a peak as relaxation reshapes the background.  Orient every slice so
        # that the centered feature is fitted as a dip, while retaining the
        # absolute physical contrast.
        orientation = 1.0 if interior[center_index] <= baseline else -1.0
        trace = orientation * trace
        interior = orientation * interior
        baseline = orientation * baseline
        fit_interior = interior[fit_mask]
        minimum = float(np.min(fit_interior))
        raw_depth = baseline - minimum
        if not np.isfinite(raw_depth) or raw_depth <= 1e-12:
            continue

        normalized_trace = (trace - minimum) / raw_depth
        normalized_interior = gaussian_filter1d(
            normalized_trace[EDGE_POINTS:-EDGE_POINTS],
            SMOOTH_SIGMA_POINTS,
        )
        normalized_fit = normalized_interior[fit_mask]
        minimum_index = int(np.argmin(normalized_fit))
        center0 = float(fit_x_values[minimum_index])
        offset0 = 1.0
        depth0 = max(offset0 - float(normalized_fit[minimum_index]), 0.1)
        try:
            values, _ = curve_fit(
                _dip_gaussian,
                fit_x_values,
                normalized_fit,
                p0=[offset0, depth0, center0, 0.04],
                bounds=(
                    [-0.2, 0.0, fit_x_values.min(), abs(step) / 2.0],
                    [1.2, 1.2, fit_x_values.max(), span],
                ),
                maxfev=20_000,
            )
        except (RuntimeError, ValueError):
            continue

        _, normalized_depth, center_mhz, sigma_mhz = values
        fwhm_mhz = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(sigma_mhz)
        if (
            not np.isfinite(fwhm_mhz)
            or fwhm_mhz <= 0.0
            or fwhm_mhz > MAX_FWHM_MHZ
            or abs(center_mhz) > MAX_ABS_CENTER_MHZ
        ):
            continue
        physical_contrast = normalized_depth * raw_depth
        if (
            noise_std is not None
            and physical_contrast / noise_std < MIN_SIGNAL_TO_NOISE
        ):
            continue
        fwhm_t2_units[index] = fwhm_mhz / T2_LIMIT_FWHM_MHZ
        resolution[index] = T2_LIMIT_FWHM_MHZ / fwhm_mhz
        contrast[index] = physical_contrast

    return {
        "fwhm_t2_units": fwhm_t2_units,
        "resolution": resolution,
        "contrast": contrast,
        "product": contrast * resolution,
    }


def _build_figure(
    clean_metrics_by_duration: dict[float, dict[str, np.ndarray]],
    noisy_metrics_by_duration: dict[float, dict[str, np.ndarray]],
    constant_resolution: np.ndarray,
    constant_product: np.ndarray,
) -> plt.Figure:
    apply_figure_style(FigureVariant.PAPER)
    plt.rcParams.update(
        {
            "figure.figsize": (6.8, 6.0),
            "axes.labelsize": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        constrained_layout=True,
    )
    colors = ("#0072b2", "#6a1b9a", "#d55e00", "#009e73")
    markers = ("o", "s", "^", "D")
    constant_color = "#c62828"
    constant_fwhm = np.divide(
        1.0,
        constant_resolution,
        out=np.full_like(constant_resolution, np.nan),
        where=constant_resolution > 0.0,
    )
    axes[0].plot(
        CONSTANT_RABI_MHZ,
        constant_fwhm,
        color=constant_color,
        lw=1.25,
        label="Constant (Torrey)",
        zorder=1,
    )
    axes[1].plot(
        CONSTANT_RABI_MHZ,
        constant_resolution,
        color=constant_color,
        lw=1.25,
        zorder=1,
    )
    axes[2].plot(
        CONSTANT_RABI_MHZ,
        constant_product,
        color=constant_color,
        lw=1.25,
        zorder=1,
    )
    duration_handles: list[Line2D] = []
    for duration_us, color, marker in zip(
        DURATIONS_US,
        colors,
        markers,
    ):
        clean_metrics = clean_metrics_by_duration[duration_us]
        noisy_metrics = noisy_metrics_by_duration[duration_us]
        label = rf"$L={duration_us:.0f}~\mu\mathrm{{s}}$"
        duration_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=1.0,
                marker=marker,
                ms=3.0,
                mec="white",
                mew=0.25,
                label=label,
            )
        )
        for ax, metric_name in zip(
            axes,
            ("fwhm_t2_units", "resolution", "product"),
        ):
            ax.plot(
                RABI_MHZ,
                clean_metrics[metric_name],
                color=color,
                lw=1.0,
                zorder=2,
            )
            finite = np.isfinite(noisy_metrics[metric_name])
            ax.plot(
                RABI_MHZ[finite],
                noisy_metrics[metric_name][finite],
                ls="none",
                marker=marker,
                ms=2.8,
                color=color,
                mec="white",
                mew=0.25,
                zorder=3,
            )

    axes[0].axhline(1.0, color="0.35", ls="--", lw=0.65)
    axes[1].axhline(1.0, color="0.35", ls="--", lw=0.65)
    axes[2].axhline(Q_THRESHOLD, color="0.35", ls="--", lw=0.65)
    axes[0].set_ylabel(r"$\Gamma_f/\Gamma_{f,T_2}$")
    axes[1].set_ylabel(r"$R=\Gamma_{f,T_2}/\Gamma_f$")
    axes[2].set_ylabel(r"$Q=A\,\Gamma_{f,T_2}/\Gamma_f$")
    axes[2].set_xlabel(r"$\Omega_0/2\pi$ (MHz)")
    for panel, ax in zip(("(g)", "(h)", "(i)"), axes):
        ax.set_xscale("log")
        ax.set_xlim(RABI_MHZ.min(), RABI_MHZ.max())
        ax.text(
            0.015,
            0.96,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    fwhm_max = max(
        float(np.nanmax(metrics["fwhm_t2_units"]))
        for collection in (
            clean_metrics_by_duration,
            noisy_metrics_by_duration,
        )
        for metrics in collection.values()
    )
    axes[0].set_ylim(0.0, max(1.2, 1.06 * fwhm_max))
    resolution_max = max(
        float(np.nanmax(metrics["resolution"]))
        for collection in (
            clean_metrics_by_duration,
            noisy_metrics_by_duration,
        )
        for metrics in collection.values()
    )
    axes[1].set_ylim(0.0, max(1.05, 1.06 * resolution_max))
    product_max = max(
        float(np.nanmax(metrics["product"]))
        for collection in (
            clean_metrics_by_duration,
            noisy_metrics_by_duration,
        )
        for metrics in collection.values()
    )
    axes[2].set_ylim(0.0, max(0.24, 1.08 * product_max))
    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=constant_color,
                lw=1.25,
                label="Constant (Torrey)",
            ),
            *duration_handles,
        ],
        loc="lower left",
        frameon=True,
        framealpha=0.88,
        edgecolor="none",
        ncol=3,
        fontsize=6.7,
        handlelength=2.3,
        borderpad=0.3,
        labelspacing=0.25,
    )
    return fig


def main() -> None:
    output_dir = ROOT / "figures" / "paper"
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_metrics_by_duration: dict[float, dict[str, np.ndarray]] = {}
    noisy_metrics_by_duration: dict[float, dict[str, np.ndarray]] = {}
    clean_traces_by_duration: dict[float, np.ndarray] = {}
    noisy_traces_by_duration: dict[float, np.ndarray] = {}
    rng = np.random.default_rng(NOISE_SEED)
    for duration_us in DURATIONS_US:
        print(f"Simulating L={duration_us:.0f} us, c={CUTOFF:g}...")
        clean_traces = _simulate_echo(duration_us)
        noisy_traces = clean_traces + rng.normal(
            0.0,
            NOISE_STD,
            clean_traces.shape,
        )
        clean_metrics = _extract_metrics(clean_traces, noise_std=None)
        noisy_metrics = _extract_metrics(noisy_traces, noise_std=NOISE_STD)
        finite_count = int(np.isfinite(noisy_metrics["resolution"]).sum())
        if finite_count < 10:
            raise RuntimeError(
                f"Only {finite_count}/{RABI_MHZ.size} finite fits for "
                f"L={duration_us:g} us"
            )
        clean_metrics_by_duration[duration_us] = clean_metrics
        noisy_metrics_by_duration[duration_us] = noisy_metrics
        clean_traces_by_duration[duration_us] = clean_traces
        noisy_traces_by_duration[duration_us] = noisy_traces
        print(
            f"  finite fits: {finite_count}/{RABI_MHZ.size}; "
            f"max noisy Q={np.nanmax(noisy_metrics['product']):.3f}"
        )

    constant_params = Parameters(
        T1=T1_US * u.us,
        T_dephasing=T_PHI_US * u.us,
    )
    constant_result = TorreyFwhmVsRabiExperiment(
        CONSTANT_RABI_MHZ * u.pi2 * u.MHz,
        constant_params,
        OptionsTorreyResonance(
            plot=False,
            save=False,
            plot_product=True,
            product_threshold=Q_THRESHOLD,
        ),
    ).run()
    constant_resolution = constant_result.inverse_fwhm_t2_units
    constant_product = constant_result.inverse_fwhm_snr_product
    fig = _build_figure(
        clean_metrics_by_duration,
        noisy_metrics_by_duration,
        constant_resolution,
        constant_product,
    )
    stem = "10_simulated_duration_resolution_comparison"
    saved = save_figure(
        fig,
        stem,
        variant=FigureVariant.PAPER,
        formats=("pdf", "png", "svg"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)

    data_path = output_dir / f"{stem}.npz"
    np.savez_compressed(
        data_path,
        durations_us=np.asarray(DURATIONS_US),
        cutoff=CUTOFF,
        detuning_convention="drive_minus_qubit",
        detuning_mhz=DETUNING_MHZ,
        rabi_mhz=RABI_MHZ,
        t1_us=T1_US,
        t_phi_us=T_PHI_US,
        t2_limit_fwhm_mhz=T2_LIMIT_FWHM_MHZ,
        noise_std=NOISE_STD,
        min_signal_to_noise=MIN_SIGNAL_TO_NOISE,
        noise_seed=NOISE_SEED,
        constant_rabi_mhz=CONSTANT_RABI_MHZ,
        constant_resolution=constant_resolution,
        constant_product=constant_product,
        **{
            f"clean_echo_state_{int(duration_us)}us": (
                clean_traces_by_duration[duration_us]
            )
            for duration_us in DURATIONS_US
        },
        **{
            f"noisy_echo_state_{int(duration_us)}us": (
                noisy_traces_by_duration[duration_us]
            )
            for duration_us in DURATIONS_US
        },
        **{
            f"{metric_name}_{int(duration_us)}us": metric_values
            for duration_us, metrics in noisy_metrics_by_duration.items()
            for metric_name, metric_values in metrics.items()
        },
        **{
            f"clean_{metric_name}_{int(duration_us)}us": metric_values
            for duration_us, metrics in clean_metrics_by_duration.items()
            for metric_name, metric_values in metrics.items()
        },
    )
    for path in (*saved, data_path):
        print(path)


if __name__ == "__main__":
    main()
