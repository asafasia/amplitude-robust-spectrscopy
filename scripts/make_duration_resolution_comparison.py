"""Generate the supplemental pulse-duration resolution comparison."""

from __future__ import annotations

# Backend and local-source setup must precede pyplot and echospec imports.
# ruff: noqa: E402, I001

import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.figures import FigureVariant, apply_figure_style, save_figure
from echospec.paper_data import save_paper_dataset


DURATIONS_US = (5.0, 10.0, 15.0, 20.0, 30.0)
CUTOFF = 0.005
RUN_IDS = {
    5.0: "17-00-01-732960",
    10.0: "16-47-44-334581",
    15.0: "17-05-56-223645",
    20.0: "14-02-28-518579",
    30.0: "16-55-50-249375",
}
DETUNING_MHZ = np.linspace(-0.5, 0.5, 201)
RABI_MHZ = np.geomspace(1e-3, 50.0, 100)
EDGE_POINTS = 20
SMOOTH_SIGMA_POINTS = 0.5
MAX_FWHM_MHZ = 0.30
MAX_ABS_CENTER_MHZ = 0.05
FIT_HALF_WIDTH_MHZ = 0.25
FIT_BASELINE_INNER_MHZ = 0.15
Q_THRESHOLD = 0.10
MIN_SIGNAL_TO_NOISE = 3.0
U_STEPS_PER_HALF = 1600
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


@dataclass(frozen=True)
class Measurement:
    run_id: str
    duration_us: float
    detuning_mhz: np.ndarray
    rabi_mhz: np.ndarray
    excited: np.ndarray
    requested_shots: int


def _opx1000_data_root() -> Path:
    """Resolve the read-only OPX1000 data repository."""
    configured = os.environ.get("OPX1000_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    sibling = ROOT.parent / "data_opx1000"
    if sibling.exists():
        return sibling

    # Compatibility with the Windows lab checkout, where ``data`` still
    # lives inside the OPX1000 control-code repository.
    lab_checkout = ROOT.parent / "opx1000-codes" / "data"
    if lab_checkout.exists():
        return lab_checkout
    return sibling


def _load_measurement(duration_us: float) -> Measurement:
    run_id = RUN_IDS[duration_us]
    data_root = _opx1000_data_root()
    campaign_dir = (
        data_root / "calibrations" / "2026-08-10" / "echo_lorentzian"
    )
    if not (campaign_dir / run_id).exists() and (data_root / run_id).exists():
        campaign_dir = data_root
    if not (campaign_dir / run_id).exists():
        archived = ROOT / "data/experimental/2026-08-10/echo_lorentzian_duration"
        if (archived / run_id).exists():
            campaign_dir = archived
    run_dir = campaign_dir / run_id
    parameters = json.loads((run_dir / "parameters.json").read_text())
    pulses = json.loads((run_dir / "profile" / "pulses.json").read_text())
    with np.load(run_dir / "sweep.npz", allow_pickle=False) as sweep:
        qubits = np.asarray(sweep["qubit"])
        detuning_hz = np.asarray(sweep["detuning"], dtype=float)
        amp_prefactor = np.asarray(sweep["amp_prefactor"], dtype=float)
    with np.load(run_dir / "results.npz", allow_pickle=False) as results:
        state = np.asarray(results["state"], dtype=float)

    if qubits.size != 1 or str(qubits[0]) != "q1":
        raise ValueError(f"{run_id}: expected the q1 measurement")
    if state.shape != (1, detuning_hz.size, amp_prefactor.size):
        raise ValueError(f"{run_id}: unexpected state shape {state.shape}")
    measured_duration_us = float(parameters["lorentzian_length_in_ns"]) / 1000.0
    if not np.isclose(measured_duration_us, duration_us):
        raise ValueError(
            f"{run_id}: expected {duration_us:g} us, got {measured_duration_us:g} us"
        )
    if parameters["pulse_shape"] != "root_lorentzian" or not parameters["echo"]:
        raise ValueError(f"{run_id}: expected an echo-root-Lorentzian pulse")
    if not np.isclose(float(parameters["cutoff"]), CUTOFF):
        raise ValueError(f"{run_id}: expected cutoff c={CUTOFF:g}")

    peak_amplitude_v = float(parameters["lorentzian_peak_amplitude"])
    pi_pulse = pulses["pulses"]["q1"]["x180_const"]
    pi_amplitude_v = float(pi_pulse["amplitude"])
    pi_length_ns = float(pi_pulse["length_ns"])
    pi_rabi_hz = 1.0 / (2.0 * pi_length_ns * 1e-9)
    rabi_mhz = (
        amp_prefactor * peak_amplitude_v / pi_amplitude_v * pi_rabi_hz / 1e6
    )
    return Measurement(
        run_id=run_id,
        duration_us=duration_us,
        detuning_mhz=detuning_hz / 1e6,
        rabi_mhz=rabi_mhz,
        excited=state[0].T,
        requested_shots=int(parameters["num_shots"]),
    )


def _simulate_echo(duration_us: float) -> np.ndarray:
    """Return the two-level Bloch result using t=sigma*sinh(u)."""
    order = 0.5
    sigma_us = (duration_us / 2.0) / np.sqrt(CUTOFF ** (-1.0 / order) - 1.0)
    u_edge = float(np.arcsinh((duration_us / 2.0) / sigma_us))
    detuning, rabi = np.meshgrid(
        2.0 * np.pi * DETUNING_MHZ,
        2.0 * np.pi * RABI_MHZ,
    )
    bloch = np.zeros((3, *detuning.shape), dtype=float)
    bloch[2] = 1.0
    inv_t1 = 1.0 / T1_US
    inv_t2 = 1.0 / (2.0 * T1_US) + 1.0 / T_PHI_US

    def integrate_half(
        state: np.ndarray,
        u_start: float,
        u_stop: float,
        drive_sign: float,
    ) -> np.ndarray:
        du = (u_stop - u_start) / U_STEPS_PER_HALF

        def derivative(values: np.ndarray, u_value: float) -> np.ndarray:
            x, y, z = values
            envelope = 1.0 / np.cosh(u_value)
            dt_du = sigma_us * np.cosh(u_value)
            drive = drive_sign * rabi * envelope
            return dt_du * np.stack(
                (
                    -detuning * y - inv_t2 * x,
                    detuning * x - drive * z - inv_t2 * y,
                    drive * y + inv_t1 * (1.0 - z),
                )
            )

        u_value = u_start
        for _ in range(U_STEPS_PER_HALF):
            k1 = derivative(state, u_value)
            k2 = derivative(state + 0.5 * du * k1, u_value + 0.5 * du)
            k3 = derivative(state + 0.5 * du * k2, u_value + 0.5 * du)
            k4 = derivative(state + du * k3, u_value + du)
            state = state + (du / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            u_value += du
        return state

    bloch = integrate_half(bloch, -u_edge, 0.0, 1.0)
    bloch = integrate_half(bloch, 0.0, u_edge, -1.0)
    population = ((1.0 - bloch[2]) / 2.0).T
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
    detuning_mhz: np.ndarray = DETUNING_MHZ,
    rabi_mhz: np.ndarray = RABI_MHZ,
    noise_std: float | None,
) -> dict[str, np.ndarray]:
    """Extract centered features, optionally rejecting fits below a noise floor."""
    if traces.shape != (detuning_mhz.size, rabi_mhz.size):
        raise ValueError(
            "Trace shape must be (detuning, Rabi); got "
            f"{traces.shape} for {(detuning_mhz.size, rabi_mhz.size)}"
        )
    edge_points = min(EDGE_POINTS, max(2, detuning_mhz.size // 10))
    x_values = detuning_mhz[edge_points:-edge_points]
    fit_mask = np.abs(x_values) <= FIT_HALF_WIDTH_MHZ
    fit_x_values = x_values[fit_mask]
    step = float(np.median(np.diff(fit_x_values)))
    span = float(np.ptp(fit_x_values))
    fwhm_t2_units = np.full(rabi_mhz.size, np.nan)
    resolution = np.full(rabi_mhz.size, np.nan)
    contrast = np.full(rabi_mhz.size, np.nan)

    for index in range(rabi_mhz.size):
        trace = np.asarray(traces[:, index], dtype=float)
        interior = gaussian_filter1d(
            trace[edge_points:-edge_points],
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
            normalized_trace[edge_points:-edge_points],
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
        fitted_trace = _dip_gaussian(fit_x_values, *values)
        residual_sum_squares = float(np.sum((normalized_fit - fitted_trace) ** 2))
        total_sum_squares = float(
            np.sum((normalized_fit - np.mean(normalized_fit)) ** 2)
        )
        r_squared = (
            1.0 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0.0
            else np.nan
        )
        fwhm_mhz = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(sigma_mhz)
        if (
            not np.isfinite(fwhm_mhz)
            or fwhm_mhz < 2.0 * abs(step)
            or fwhm_mhz > MAX_FWHM_MHZ
            or abs(center_mhz) > MAX_ABS_CENTER_MHZ
            or not np.isfinite(r_squared)
            or r_squared < 0.5
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
        "fwhm_khz": fwhm_t2_units * T2_LIMIT_FWHM_MHZ * 1e3,
        "fwhm_t2_units": fwhm_t2_units,
        "resolution": resolution,
        "contrast": contrast,
        "product": contrast * resolution,
    }


def _build_figure(
    clean_metrics_by_duration: dict[float, dict[str, np.ndarray]],
    measured_metrics_by_duration: dict[float, dict[str, np.ndarray]],
    measurements: dict[float, Measurement],
    constant_fwhm_t2_units: np.ndarray,
    constant_contrast: np.ndarray,
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
    colors = ("#0072b2", "#6a1b9a", "#cc79a7", "#d55e00", "#009e73")
    markers = ("o", "s", "P", "^", "D")
    constant_color = "#c62828"
    axes[0].plot(
        CONSTANT_RABI_MHZ,
        constant_fwhm_t2_units,
        color=constant_color,
        lw=1.25,
        label="Constant (Bloch)",
        zorder=1,
    )
    axes[1].plot(
        CONSTANT_RABI_MHZ,
        constant_contrast,
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
        strict=True,
    ):
        clean_metrics = clean_metrics_by_duration[duration_us]
        measured_metrics = measured_metrics_by_duration[duration_us]
        measurement = measurements[duration_us]
        label = rf"$L={duration_us:.0f}~\mu\mathrm{{s}}$"
        duration_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=1.0,
                marker=marker,
                ms=3.0,
                mec="none",
                mew=0.0,
                label=label,
            )
        )
        for ax, metric_name in zip(
            axes,
            ("fwhm_t2_units", "contrast", "product"),
            strict=True,
        ):
            ax.plot(
                RABI_MHZ,
                clean_metrics[metric_name],
                color=color,
                lw=1.0,
                zorder=2,
            )
            finite = np.isfinite(measured_metrics[metric_name])
            ax.plot(
                measurement.rabi_mhz[finite],
                measured_metrics[metric_name][finite],
                ls="none",
                marker=marker,
                ms=2.6,
                color=color,
                mec="none",
                mew=0.0,
                zorder=3,
            )

    axes[0].axhline(
        1.0,
        color="0.35",
        ls="--",
        lw=0.65,
    )
    axes[2].axhline(Q_THRESHOLD, color="0.35", ls="--", lw=0.65)
    axes[0].set_ylabel(r"FWHM $\Gamma_f/\Gamma_{f,T_2}$")
    axes[1].set_ylabel(r"Fitted contrast $A$")
    axes[2].set_ylabel(r"$Q=A\,\Gamma_{f,T_2}/\Gamma_f$")
    axes[2].set_xlabel(r"$\Omega_0/2\pi$ (MHz)")
    rabi_ticks = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    rabi_tick_labels = ("0.001", "0.01", "0.1", "1", "10", "100")
    for panel, ax in zip(("(a)", "(b)", "(c)"), axes, strict=True):
        ax.set_xscale("log")
        ax.set_xlim(RABI_MHZ.min(), 25.0)
        ax.xaxis.set_major_locator(FixedLocator(rabi_ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(rabi_tick_labels))
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
            measured_metrics_by_duration,
        )
        for metrics in collection.values()
    )
    axes[0].set_ylim(0.0, max(2.4, 1.06 * fwhm_max))
    contrast_max = max(
        float(np.nanmax(metrics["contrast"]))
        for collection in (
            clean_metrics_by_duration,
            measured_metrics_by_duration,
        )
        for metrics in collection.values()
    )
    axes[1].set_ylim(0.0, max(0.1, 1.06 * contrast_max))
    product_max = max(
        float(np.nanmax(metrics["product"]))
        for collection in (
            clean_metrics_by_duration,
            measured_metrics_by_duration,
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
                label="Constant (Bloch)",
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
    axes[1].legend(
        handles=[
            Line2D([0], [0], color="0.2", lw=1.0, label="Two-level simulation"),
            Line2D(
                [0],
                [0],
                color="0.2",
                ls="none",
                marker="o",
                ms=3.0,
                markeredgecolor="none",
                markeredgewidth=0.0,
                label="OPX1000",
            ),
        ],
        loc="upper right",
        frameon=True,
        framealpha=0.88,
        edgecolor="none",
        fontsize=6.7,
        borderpad=0.3,
        labelspacing=0.25,
    )
    return fig


def main() -> None:
    output_dir = ROOT / "figures" / "paper"
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_metrics_by_duration: dict[float, dict[str, np.ndarray]] = {}
    measured_metrics_by_duration: dict[float, dict[str, np.ndarray]] = {}
    measurements: dict[float, Measurement] = {}
    clean_traces_by_duration: dict[float, np.ndarray] = {}
    for duration_us in DURATIONS_US:
        print(f"Simulating L={duration_us:.0f} us, c={CUTOFF:g}...")
        clean_traces = _simulate_echo(duration_us)
        clean_metrics = _extract_metrics(clean_traces, noise_std=None)
        measurement = _load_measurement(duration_us)
        measured_metrics = _extract_metrics(
            measurement.excited.T,
            detuning_mhz=measurement.detuning_mhz,
            rabi_mhz=measurement.rabi_mhz,
            noise_std=None,
        )
        finite_count = int(np.isfinite(measured_metrics["resolution"]).sum())
        if finite_count < 10:
            raise RuntimeError(
                f"Only {finite_count}/{measurement.rabi_mhz.size} finite fits for "
                f"L={duration_us:g} us"
            )
        clean_metrics_by_duration[duration_us] = clean_metrics
        measured_metrics_by_duration[duration_us] = measured_metrics
        measurements[duration_us] = measurement
        clean_traces_by_duration[duration_us] = clean_traces
        print(
            f"  OPX run {measurement.run_id}: {finite_count}/"
            f"{measurement.rabi_mhz.size} finite fits; "
            f"max measured Q={np.nanmax(measured_metrics['product']):.3f}"
        )

    t1_s = T1_US * 1e-6
    t_phi_s = T_PHI_US * 1e-6
    t2_s = 1.0 / (1.0 / t_phi_s + 1.0 / (2.0 * t1_s))
    constant_omega = 2.0 * np.pi * CONSTANT_RABI_MHZ * 1e6
    saturation = constant_omega**2 * t1_s * t2_s
    constant_fwhm_t2_units = np.sqrt(1.0 + saturation)
    constant_resolution = 1.0 / constant_fwhm_t2_units
    constant_contrast = saturation / (2.0 * (1.0 + saturation))
    constant_product = constant_contrast * constant_resolution
    fig = _build_figure(
        clean_metrics_by_duration,
        measured_metrics_by_duration,
        measurements,
        constant_fwhm_t2_units,
        constant_contrast,
        constant_product,
    )
    stem = "03_echo_duration_resolution_comparison"
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
    paper_arrays = {
        "durations_us": np.asarray(DURATIONS_US),
        "cutoff": np.asarray(CUTOFF),
        "detuning_convention": np.asarray("drive_minus_qubit"),
        "detuning_mhz": DETUNING_MHZ,
        "rabi_mhz": RABI_MHZ,
        "t1_us": np.asarray(T1_US),
        "t_phi_us": np.asarray(T_PHI_US),
        "t2_limit_fwhm_mhz": np.asarray(T2_LIMIT_FWHM_MHZ),
        "opx1000_data_dir": np.asarray(str(_opx1000_data_root())),
        "constant_rabi_mhz": CONSTANT_RABI_MHZ,
        "constant_fwhm_t2_units": constant_fwhm_t2_units,
        "constant_contrast": constant_contrast,
        "constant_resolution": constant_resolution,
        "constant_product": constant_product,
        **{
            f"clean_echo_state_{int(duration_us)}us": (
                clean_traces_by_duration[duration_us]
            )
            for duration_us in DURATIONS_US
        },
        **{
            f"measured_echo_state_{int(duration_us)}us": (
                measurements[duration_us].excited
            )
            for duration_us in DURATIONS_US
        },
        **{
            f"measured_{metric_name}_{int(duration_us)}us": metric_values
            for duration_us, metrics in measured_metrics_by_duration.items()
            for metric_name, metric_values in metrics.items()
        },
        **{
            f"measured_rabi_mhz_{int(duration_us)}us": (
                measurements[duration_us].rabi_mhz
            )
            for duration_us in DURATIONS_US
        },
        **{
            f"measured_detuning_mhz_{int(duration_us)}us": (
                measurements[duration_us].detuning_mhz
            )
            for duration_us in DURATIONS_US
        },
        **{
            f"measured_run_id_{int(duration_us)}us": (
                measurements[duration_us].run_id
            )
            for duration_us in DURATIONS_US
        },
        **{
            f"clean_{metric_name}_{int(duration_us)}us": metric_values
            for duration_us, metrics in clean_metrics_by_duration.items()
            for metric_name, metric_values in metrics.items()
        },
    }
    np.savez_compressed(
        data_path,
        **paper_arrays,
    )
    paper_data_paths = save_paper_dataset(
        stem,
        category="experimental",
        arrays=paper_arrays,
        provenance={
            "figure_asset": f"figures/paper/{stem}.pdf",
            "manuscript_scope": "supplemental",
            "generator": "scripts/make_duration_resolution_comparison.py",
            "reproduction_command": (
                "PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python "
                "scripts/make_duration_resolution_comparison.py"
            ),
            "model": "measured q1 OPX1000 data plus dissipative Bloch simulation",
            "detuning_convention": "drive_minus_qubit",
            "population_definition": "P_e",
            "pulse_shape": "echo-root-Lorentzian",
            "fit_model": "Gaussian depletion feature",
            "array_dimensions": {
                "*_echo_state_*us": ["detuning_mhz", "rabi_mhz"],
                "*_fwhm_*us": ["rabi_mhz"],
                "*_contrast_*us": ["rabi_mhz"],
                "*_resolution_*us": ["rabi_mhz"],
                "*_product_*us": ["rabi_mhz"],
                "constant_*": ["constant_rabi_mhz"],
            },
        },
    )
    summary_path = output_dir / "03_echo_duration_resolution_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "duration_us",
                "run_id",
                "best_q_rabi_mhz",
                "fwhm_khz",
                "resolution_t2_units",
                "contrast",
                "q",
            )
        )
        for duration_us in DURATIONS_US:
            metrics = measured_metrics_by_duration[duration_us]
            best_index = int(np.nanargmax(metrics["product"]))
            writer.writerow(
                (
                    f"{duration_us:g}",
                    measurements[duration_us].run_id,
                    f"{measurements[duration_us].rabi_mhz[best_index]:.6g}",
                    f"{metrics['fwhm_khz'][best_index]:.6g}",
                    f"{metrics['resolution'][best_index]:.6g}",
                    f"{metrics['contrast'][best_index]:.6g}",
                    f"{metrics['product'][best_index]:.6g}",
                )
            )
    for path in (*saved, data_path, summary_path, *paper_data_paths):
        print(path)


if __name__ == "__main__":
    main()
