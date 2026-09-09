"""Build a self-contained interactive report for duration/cutoff leakage sweeps.

The report compares the current abrupt AC-Stark-corrected echo pulse against a
smooth midpoint, smooth midpoint plus DRAG at the original kappa, and smooth
midpoint plus DRAG after retuning kappa.  All dynamics use the repository's
three-level Lindblad/RK4 model.
"""

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
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echospec.figures import FigureVariant, apply_figure_style
from echospec.simulation.qutrit import _smooth_echo_peak, simulate_qutrit_map


DURATIONS_US = (3.0, 5.0, 7.0, 10.0, 20.0, 30.0)
CUTOFFS = (0.001, 0.005, 0.02)
ORDER = 0.5
ANHARMONICITY_MHZ = -216.0
T1_US = 51.24
T2_US = 7.31
T_PHI_US = 1.0 / (1.0 / T2_US - 1.0 / (2.0 * T1_US))
STEPS_PER_US = 1000
ECHO_TRANSITION_NS = 2.0
DRAG_BETA = 1.0
KAPPA_REFERENCE_MHZ_INV = 0.00225
FFT_RABI_MHZ = 80.0
FFT_TIME_STEP_NS = 0.25
FFT_MAX_FREQUENCY_MHZ = 350.0
FFT_FLOOR_DB = -100.0
SCAN_PF_FLOOR = 1e-6
FOCUSED_DURATION_US = 20.0
FOCUSED_CUTOFF = 0.0025

RABI_MHZ = np.linspace(10.0, 80.0, 15)
DETUNING_MHZ = np.linspace(-0.35, 0.35, 71)
FIT_WINDOW_MHZ = 0.12
OPT_RABI_MHZ = RABI_MHZ.copy()
OPT_DETUNING_MHZ = np.linspace(-FIT_WINDOW_MHZ, FIT_WINDOW_MHZ, 25)

OUTPUT_DIR = ROOT / "outputs" / "smooth_echo_drag_sweep"
DATA_DIR = OUTPUT_DIR / "data"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "outputs" / "smooth_echo_drag_sweep_report.html"
SUMMARY_CSV_PATH = OUTPUT_DIR / "summary.csv"
CACHE_VERSION = 4
FOCUSED_CACHE_VERSION = 1


@dataclass(frozen=True)
class Protocol:
    key: str
    label: str
    smooth: bool
    drag_beta: float
    kappa_source: str
    color: str


PROTOCOLS = (
    Protocol(
        "abrupt_kappa_only",
        "Abrupt midpoint, kappa only",
        False,
        0.0,
        "reference",
        "#6c757d",
    ),
    Protocol(
        "smooth_no_drag",
        "Smooth midpoint, no DRAG",
        True,
        0.0,
        "reference",
        "#0072b2",
    ),
    Protocol(
        "smooth_drag_same_kappa",
        "Smooth + DRAG, same kappa",
        True,
        DRAG_BETA,
        "reference",
        "#009e73",
    ),
    Protocol(
        "smooth_drag_retuned",
        "Smooth + DRAG, retuned kappa",
        True,
        DRAG_BETA,
        "retuned",
        "#d55e00",
    ),
)


def value_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def configuration_path(duration_us: float, cutoff: float) -> Path:
    return DATA_DIR / (
        f"duration_{value_label(duration_us)}us_cutoff_{value_label(cutoff)}.npz"
    )


def focused_configuration_path() -> Path:
    return DATA_DIR / (
        f"focused_duration_{value_label(FOCUSED_DURATION_US)}us_"
        f"cutoff_{value_label(FOCUSED_CUTOFF)}.npz"
    )


def focused_configuration_signature() -> str:
    payload = {
        "cache_version": FOCUSED_CACHE_VERSION,
        "duration_us": FOCUSED_DURATION_US,
        "cutoff": FOCUSED_CUTOFF,
        "order": ORDER,
        "anharmonicity_mhz": ANHARMONICITY_MHZ,
        "t1_us": T1_US,
        "t2_us": T2_US,
        "steps_per_us": STEPS_PER_US,
        "echo_transition_ns": ECHO_TRANSITION_NS,
        "drag_beta": DRAG_BETA,
        "rabi_mhz": RABI_MHZ.tolist(),
        "detuning_mhz": DETUNING_MHZ.tolist(),
        "comparison": "smooth_none_vs_smooth_drag_retuned_kappa",
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def configuration_signature(duration_us: float, cutoff: float) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "duration_us": duration_us,
        "cutoff": cutoff,
        "order": ORDER,
        "anharmonicity_mhz": ANHARMONICITY_MHZ,
        "t1_us": T1_US,
        "t2_us": T2_US,
        "steps_per_us": STEPS_PER_US,
        "echo_transition_ns": ECHO_TRANSITION_NS,
        "drag_beta": DRAG_BETA,
        "kappa_reference_mhz_inv": KAPPA_REFERENCE_MHZ_INV,
        "rabi_mhz": RABI_MHZ.tolist(),
        "detuning_mhz": DETUNING_MHZ.tolist(),
        "opt_rabi_mhz": OPT_RABI_MHZ.tolist(),
        "opt_detuning_mhz": OPT_DETUNING_MHZ.tolist(),
        "fit_window_mhz": FIT_WINDOW_MHZ,
        "center_fit": "nearest_local_minimum",
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def fitted_central_minima(
    detuning_mhz: np.ndarray,
    excited: np.ndarray,
) -> np.ndarray:
    """Fit the local population minimum nearest zero detuning."""
    mask = np.abs(detuning_mhz) <= FIT_WINDOW_MHZ
    x_values = detuning_mhz[mask]
    centers = []
    for full_row in excited:
        row = full_row[mask]
        local_minima = np.flatnonzero(
            (row[1:-1] <= row[:-2]) & (row[1:-1] <= row[2:])
        ) + 1
        if local_minima.size:
            index = int(local_minima[np.argmin(np.abs(x_values[local_minima]))])
        else:
            index = int(np.argmin(row))
        center = float(x_values[index])
        if 0 < index < x_values.size - 1:
            coefficients = np.polyfit(
                x_values[index - 1 : index + 2],
                row[index - 1 : index + 2],
                2,
            )
            if coefficients[0] > 0.0:
                candidate = float(-coefficients[1] / (2.0 * coefficients[0]))
                if (
                    x_values[index - 1] <= candidate <= x_values[index + 1]
                ):
                    center = candidate
        centers.append(center)
    return np.asarray(centers)


def num_steps_per_half(duration_us: float, cutoff: float) -> int:
    """Resolve both the transmon rotation and the narrow pulse waist."""
    half_duration = duration_us / 2.0
    sigma_us = half_duration / np.sqrt(cutoff ** (-1.0 / ORDER) - 1.0)
    target_step_us = min(1.0 / STEPS_PER_US, sigma_us / 4.0)
    return int(np.ceil(half_duration / target_step_us))


def simulation_kwargs(
    duration_us: float,
    cutoff: float,
    *,
    optimization: bool,
) -> dict[str, object]:
    return {
        "duration_us": duration_us,
        "detuning_mhz": OPT_DETUNING_MHZ if optimization else DETUNING_MHZ,
        "rabi_mhz": OPT_RABI_MHZ if optimization else RABI_MHZ,
        "t1_us": T1_US,
        "t_phi_us": T_PHI_US,
        "anharmonicity_mhz": ANHARMONICITY_MHZ,
        "num_steps_per_half": num_steps_per_half(duration_us, cutoff),
        "cutoff": cutoff,
        "echo": True,
        "order": ORDER,
        "stark_correction_mode": "accumulated_phase",
    }


def optimize_drag_kappa(
    duration_us: float,
    cutoff: float,
) -> tuple[float, dict[float, tuple[float, np.ndarray]]]:
    """Retune kappa with a coarse scan followed by a local fine scan."""
    common = simulation_kwargs(duration_us, cutoff, optimization=True)
    evaluations: dict[float, tuple[float, np.ndarray]] = {}

    def evaluate(kappa: float) -> tuple[float, np.ndarray]:
        key = float(np.round(kappa, 8))
        if key not in evaluations:
            result = simulate_qutrit_map(
                **common,
                echo_transition_us=ECHO_TRANSITION_NS / 1000.0,
                drag_beta=DRAG_BETA,
                stark_kappa_mhz_inv=key,
            )
            centers = fitted_central_minima(
                OPT_DETUNING_MHZ,
                result.excited,
            )
            evaluations[key] = (
                float(np.sqrt(np.mean(centers**2))),
                centers,
            )
        return evaluations[key]

    coarse = np.asarray([-0.006, -0.004, -0.002, 0.0, 0.002])
    for kappa in coarse:
        evaluate(float(kappa))
    coarse_best = min(coarse, key=lambda key: evaluate(float(key))[0])
    offsets = np.asarray([-0.001, -0.0005, 0.0005, 0.001])
    if coarse_best == coarse[0]:
        offsets = np.asarray([-0.002, -0.001, -0.0005, 0.0005, 0.001])
    elif coarse_best == coarse[-1]:
        offsets = np.asarray([-0.001, -0.0005, 0.0005, 0.001, 0.002])
    for kappa in coarse_best + offsets:
        evaluate(float(np.clip(kappa, -0.012, 0.012)))
    selected = min(evaluations, key=lambda key: evaluations[key][0])
    return selected, evaluations


def protocol_kappa(protocol: Protocol, retuned_kappa: float) -> float:
    if protocol.kappa_source == "retuned":
        return retuned_kappa
    return KAPPA_REFERENCE_MHZ_INV


def summary_from_npz(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        protocols = {}
        for protocol in PROTOCOLS:
            centers = data[f"{protocol.key}_centers_mhz"]
            protocols[protocol.key] = {
                "label": protocol.label,
                "kappa_mhz_inv": float(data[f"{protocol.key}_kappa"]),
                "maximum_final_pf": float(data[f"{protocol.key}_pf"].max()),
                "center_rms_khz": float(1e3 * np.sqrt(np.mean(centers**2))),
                "maximum_abs_center_khz": float(
                    1e3 * np.max(np.abs(centers))
                ),
                "minimum_pe_contrast": float(
                    data[f"{protocol.key}_pe_contrast"].min()
                ),
            }
        return {
            "duration_us": float(data["duration_us"]),
            "cutoff": float(data["cutoff"]),
            "kappa_drag_retuned_mhz_inv": float(
                data["kappa_drag_retuned_mhz_inv"]
            ),
            "optimization_center_rms_khz": float(
                1e3 * data["optimization_selected_center_rms_mhz"]
            ),
            "elapsed_s": float(data["elapsed_s"]),
            "protocols": protocols,
            "data_path": str(path),
        }


def cache_is_current(path: Path, duration_us: float, cutoff: float) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return str(data["configuration_signature"]) == (
                configuration_signature(duration_us, cutoff)
            )
    except (KeyError, OSError, ValueError):
        return False


def run_configuration(
    duration_us: float,
    cutoff: float,
    force: bool,
) -> dict[str, object]:
    path = configuration_path(duration_us, cutoff)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and cache_is_current(path, duration_us, cutoff):
        print(
            f"L={duration_us:g} us, c={cutoff:g}: using cache",
            flush=True,
        )
        return summary_from_npz(path)

    started = time.perf_counter()
    retuned_kappa, optimization = optimize_drag_kappa(duration_us, cutoff)
    arrays: dict[str, object] = {
        "configuration_signature": configuration_signature(duration_us, cutoff),
        "duration_us": duration_us,
        "cutoff": cutoff,
        "rabi_mhz": RABI_MHZ,
        "detuning_mhz": DETUNING_MHZ,
        "kappa_drag_retuned_mhz_inv": retuned_kappa,
    }
    common = simulation_kwargs(duration_us, cutoff, optimization=False)
    for protocol in PROTOCOLS:
        kappa = protocol_kappa(protocol, retuned_kappa)
        result = simulate_qutrit_map(
            **common,
            echo_transition_us=(
                ECHO_TRANSITION_NS / 1000.0 if protocol.smooth else 0.0
            ),
            drag_beta=protocol.drag_beta,
            stark_kappa_mhz_inv=kappa,
        )
        total = result.ground + result.excited + result.second_excited
        if not np.allclose(total, 1.0, atol=2e-6):
            raise RuntimeError(
                f"L={duration_us:g}, c={cutoff:g}, {protocol.key}: trace failure"
            )
        centers = fitted_central_minima(DETUNING_MHZ, result.excited)
        arrays.update(
            {
                f"{protocol.key}_kappa": kappa,
                f"{protocol.key}_pg": result.ground,
                f"{protocol.key}_pe": result.excited,
                f"{protocol.key}_pf": result.second_excited,
                f"{protocol.key}_centers_mhz": centers,
                f"{protocol.key}_max_pf_by_rabi": (
                    result.second_excited.max(axis=1)
                ),
                f"{protocol.key}_pe_contrast": (
                    result.excited.max(axis=1) - result.excited.min(axis=1)
                ),
            }
        )
    ordered_kappas = np.asarray(sorted(optimization))
    arrays["optimization_kappas_mhz_inv"] = ordered_kappas
    arrays["optimization_center_rms_mhz"] = np.asarray(
        [optimization[kappa][0] for kappa in ordered_kappas]
    )
    arrays["optimization_selected_center_rms_mhz"] = optimization[
        retuned_kappa
    ][0]
    arrays["elapsed_s"] = time.perf_counter() - started
    np.savez_compressed(path, **arrays)
    summary = summary_from_npz(path)
    print(
        f"L={duration_us:g} us, c={cutoff:g}: "
        f"kappa_drag={retuned_kappa:+.6f}, "
        f"Pf={summary['protocols']['smooth_drag_retuned']['maximum_final_pf']:.3g}, "
        f"elapsed={summary['elapsed_s']:.1f} s",
        flush=True,
    )
    return summary


def focused_summary_from_npz(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        protocols = {}
        for key, label in (
            ("without", "Smooth echo, no DRAG and no kappa"),
            ("with", "Smooth echo with DRAG and retuned kappa"),
        ):
            centers = data[f"{key}_centers_mhz"]
            protocols[key] = {
                "label": label,
                "drag_beta": float(data[f"{key}_drag_beta"]),
                "kappa_mhz_inv": float(data[f"{key}_kappa"]),
                "maximum_final_pf": float(data[f"{key}_pf"].max()),
                "center_rms_khz": float(1e3 * np.sqrt(np.mean(centers**2))),
                "maximum_abs_center_khz": float(
                    1e3 * np.max(np.abs(centers))
                ),
                "minimum_pe_contrast": float(
                    (
                        data[f"{key}_pe"].max(axis=1)
                        - data[f"{key}_pe"].min(axis=1)
                    ).min()
                ),
            }
        return {
            "duration_us": float(data["duration_us"]),
            "cutoff": float(data["cutoff"]),
            "retuned_kappa_mhz_inv": float(data["retuned_kappa_mhz_inv"]),
            "protocols": protocols,
            "leakage_reduction": (
                protocols["without"]["maximum_final_pf"]
                / protocols["with"]["maximum_final_pf"]
            ),
            "elapsed_s": float(data["elapsed_s"]),
            "data_path": str(path),
        }


def focused_cache_is_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return str(data["configuration_signature"]) == (
                focused_configuration_signature()
            )
    except (KeyError, OSError, ValueError):
        return False


def run_focused_comparison(force: bool) -> dict[str, object]:
    """Run the requested 20 us, c=0.0025 with/without comparison."""
    path = focused_configuration_path()
    if not force and focused_cache_is_current(path):
        print(
            f"Focused L={FOCUSED_DURATION_US:g} us, c={FOCUSED_CUTOFF:g}: "
            "using cache",
            flush=True,
        )
        return focused_summary_from_npz(path)

    started = time.perf_counter()
    retuned_kappa, optimization = optimize_drag_kappa(
        FOCUSED_DURATION_US,
        FOCUSED_CUTOFF,
    )
    arrays: dict[str, object] = {
        "configuration_signature": focused_configuration_signature(),
        "duration_us": FOCUSED_DURATION_US,
        "cutoff": FOCUSED_CUTOFF,
        "rabi_mhz": RABI_MHZ,
        "detuning_mhz": DETUNING_MHZ,
        "retuned_kappa_mhz_inv": retuned_kappa,
    }
    common = simulation_kwargs(
        FOCUSED_DURATION_US,
        FOCUSED_CUTOFF,
        optimization=False,
    )
    for key, drag_beta, kappa in (
        ("without", 0.0, 0.0),
        ("with", DRAG_BETA, retuned_kappa),
    ):
        result = simulate_qutrit_map(
            **common,
            echo_transition_us=ECHO_TRANSITION_NS / 1000.0,
            drag_beta=drag_beta,
            stark_kappa_mhz_inv=kappa,
        )
        total = result.ground + result.excited + result.second_excited
        if not np.allclose(total, 1.0, atol=2e-6):
            raise RuntimeError(f"Focused comparison {key}: trace failure")
        centers = fitted_central_minima(DETUNING_MHZ, result.excited)
        arrays.update(
            {
                f"{key}_drag_beta": drag_beta,
                f"{key}_kappa": kappa,
                f"{key}_pg": result.ground,
                f"{key}_pe": result.excited,
                f"{key}_pf": result.second_excited,
                f"{key}_centers_mhz": centers,
                f"{key}_max_pf_by_rabi": result.second_excited.max(axis=1),
            }
        )
    ordered_kappas = np.asarray(sorted(optimization))
    arrays["optimization_kappas_mhz_inv"] = ordered_kappas
    arrays["optimization_center_rms_mhz"] = np.asarray(
        [optimization[kappa][0] for kappa in ordered_kappas]
    )
    arrays["elapsed_s"] = time.perf_counter() - started
    np.savez_compressed(path, **arrays)
    summary = focused_summary_from_npz(path)
    print(
        f"Focused L={FOCUSED_DURATION_US:g} us, c={FOCUSED_CUTOFF:g}: "
        f"kappa={retuned_kappa:+.6f}, "
        f"max Pf {summary['protocols']['without']['maximum_final_pf']:.3g} -> "
        f"{summary['protocols']['with']['maximum_final_pf']:.3g}",
        flush=True,
    )
    return summary


def save_figure(figure: plt.Figure, stem: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{stem}.png"
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)
    return png_path


def metric_matrix(
    summaries: list[dict[str, object]],
    protocol_key: str,
    metric: str,
) -> np.ndarray:
    lookup = {
        (float(item["duration_us"]), float(item["cutoff"])): item
        for item in summaries
    }
    return np.asarray(
        [
            [
                lookup[(duration, cutoff)]["protocols"][protocol_key][metric]
                for cutoff in CUTOFFS
            ]
            for duration in DURATIONS_US
        ]
    )


def plot_overview(
    summaries: list[dict[str, object]],
    metric: str,
    title: str,
    colorbar_label: str,
    stem: str,
    *,
    logarithmic: bool,
) -> Path:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(8.2, 6.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    matrices = [
        metric_matrix(summaries, protocol.key, metric) for protocol in PROTOCOLS
    ]
    plotted = [np.log10(matrix) if logarithmic else matrix for matrix in matrices]
    vmin = min(float(matrix.min()) for matrix in plotted)
    vmax = max(float(matrix.max()) for matrix in plotted)
    images = []
    for axis, protocol, values, raw in zip(
        axes.flat,
        PROTOCOLS,
        plotted,
        matrices,
        strict=True,
    ):
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="magma" if logarithmic else "viridis",
        )
        images.append(image)
        axis.set_title(protocol.label)
        axis.set_xticks(range(len(CUTOFFS)), [f"{value:g}" for value in CUTOFFS])
        axis.set_yticks(
            range(len(DURATIONS_US)),
            [f"{value:g}" for value in DURATIONS_US],
        )
        for row in range(raw.shape[0]):
            for column in range(raw.shape[1]):
                value = raw[row, column]
                label = (
                    f"{100 * value:.2g}%"
                    if metric == "maximum_final_pf"
                    else f"{value:.2g}"
                )
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="white" if values[row, column] < (vmin + vmax) / 2 else "black",
                )
    for axis in axes[-1]:
        axis.set_xlabel("Cutoff c")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"Pulse length $L$ ($\mu$s)")
    figure.suptitle(title)
    figure.colorbar(images[-1], ax=axes, label=colorbar_label, pad=0.02)
    return save_figure(figure, stem)


def plot_kappa_summary(summaries: list[dict[str, object]]) -> Path:
    figure, axis = plt.subplots(figsize=(5.4, 3.2), constrained_layout=True)
    for cutoff in CUTOFFS:
        selected = sorted(
            (
                item
                for item in summaries
                if float(item["cutoff"]) == cutoff
            ),
            key=lambda item: float(item["duration_us"]),
        )
        axis.plot(
            [item["duration_us"] for item in selected],
            [item["kappa_drag_retuned_mhz_inv"] for item in selected],
            "o-",
            label=f"c={cutoff:g}",
        )
    axis.axhline(
        KAPPA_REFERENCE_MHZ_INV,
        color="0.35",
        ls=":",
        label="reference kappa",
    )
    axis.set(
        xlabel=r"Pulse length $L$ ($\mu$s)",
        ylabel=r"Retuned $\kappa$ (MHz$^{-1}$)",
        title="DRAG requires a joint kappa calibration",
    )
    axis.legend(frameon=False, ncol=2)
    return save_figure(figure, "02_kappa_retuning")


def plot_cutoff_curves(
    summaries: list[dict[str, object]],
    cutoff: float,
    metric_key: str,
    ylabel: str,
    stem_suffix: str,
    *,
    logarithmic: bool,
) -> Path:
    selected = sorted(
        (item for item in summaries if float(item["cutoff"]) == cutoff),
        key=lambda item: float(item["duration_us"]),
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(9.2, 5.2),
        sharex=True,
        constrained_layout=True,
    )
    for axis, item in zip(axes.flat, selected, strict=True):
        with np.load(item["data_path"], allow_pickle=False) as data:
            for protocol in PROTOCOLS:
                y_values = data[f"{protocol.key}_{metric_key}"]
                plot = axis.semilogy if logarithmic else axis.plot
                plot(
                    RABI_MHZ,
                    y_values,
                    "o-",
                    ms=2.5,
                    color=protocol.color,
                    label=protocol.label,
                )
        axis.set_title(f"L={item['duration_us']:g} us")
        axis.set_xlabel(r"$\Omega_0/2\pi$ (MHz)")
        axis.set_ylabel(ylabel)
        if metric_key == "centers_mhz":
            axis.axhline(0.0, color="0.5", lw=0.6, ls=":")
    axes[0, 0].legend(fontsize=6.0, frameon=False)
    figure.suptitle(f"Cutoff c={cutoff:g}")
    return save_figure(
        figure,
        f"cutoff_{value_label(cutoff)}_{stem_suffix}",
    )


def played_waveform(
    duration_us: float,
    cutoff: float,
    protocol: Protocol,
    kappa_mhz_inv: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the complete complex envelope sent by the waveform generator."""
    half_duration_us = duration_us / 2.0
    sigma_us = half_duration_us / np.sqrt(cutoff ** (-1.0 / ORDER) - 1.0)
    intervals = int(np.ceil(duration_us / (FFT_TIME_STEP_NS / 1000.0)))
    time_us = np.linspace(-half_duration_us, half_duration_us, intervals + 1)

    scaled_time = time_us / sigma_us
    base = (1.0 + scaled_time**2) ** (-ORDER)
    base_derivative = (
        -2.0
        * ORDER
        * time_us
        / sigma_us**2
        * (1.0 + scaled_time**2) ** (-ORDER - 1.0)
    )
    if protocol.smooth:
        transition_us = ECHO_TRANSITION_NS / 1000.0
        midpoint_sign = -np.tanh(time_us / transition_us)
        midpoint_derivative = -(1.0 - midpoint_sign**2) / transition_us
        peak = _smooth_echo_peak(
            half_duration_us=half_duration_us,
            sigma_us=sigma_us,
            order=ORDER,
            transition_us=transition_us,
        )
        in_phase = midpoint_sign * base / peak
        in_phase_derivative = (
            midpoint_derivative * base + midpoint_sign * base_derivative
        ) / peak
    else:
        midpoint_sign = np.where(time_us < 0.0, 1.0, -1.0)
        in_phase = midpoint_sign * base
        # The ideal jump's distribution-valued derivative is intentionally
        # excluded, matching the qutrit simulator's abrupt-echo convention.
        in_phase_derivative = midpoint_sign * base_derivative

    alpha_angular_per_us = 2.0 * np.pi * ANHARMONICITY_MHZ
    quadrature = (
        -protocol.drag_beta * in_phase_derivative / alpha_angular_per_us
    )
    correction_mhz = kappa_mhz_inv * (FFT_RABI_MHZ * in_phase) ** 2
    phase = np.zeros_like(time_us)
    phase[1:] = np.cumsum(
        np.pi
        * (correction_mhz[:-1] + correction_mhz[1:])
        * np.diff(time_us)
    )
    played = FFT_RABI_MHZ * (in_phase + 1j * quadrature) * np.exp(-1j * phase)
    return time_us, played


def waveform_fft(
    time_us: np.ndarray,
    waveform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a two-sided amplitude FFT; cyclic frequency is in MHz."""
    time_step_us = float(time_us[1] - time_us[0])
    fft_points = 1 << int(np.ceil(np.log2(waveform.size)))
    frequency_mhz = np.fft.fftshift(np.fft.fftfreq(fft_points, time_step_us))
    spectrum = time_step_us * np.abs(
        np.fft.fftshift(np.fft.fft(waveform, n=fft_points))
    )
    return frequency_mhz, spectrum


def plot_cutoff_fft(
    summaries: list[dict[str, object]],
    cutoff: float,
) -> Path:
    """Plot every protocol's complete played-envelope spectrum."""
    selected = sorted(
        (item for item in summaries if float(item["cutoff"]) == cutoff),
        key=lambda item: float(item["duration_us"]),
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(9.2, 5.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, item in zip(axes.flat, selected, strict=True):
        spectra = []
        for protocol in PROTOCOLS:
            kappa_mhz_inv = float(
                item["protocols"][protocol.key]["kappa_mhz_inv"]
            )
            time_us, waveform = played_waveform(
                float(item["duration_us"]),
                cutoff,
                protocol,
                kappa_mhz_inv,
            )
            frequency_mhz, spectrum = waveform_fft(time_us, waveform)
            spectra.append((protocol, frequency_mhz, spectrum))
        reference = max(float(spectrum.max()) for _, _, spectrum in spectra)
        for protocol, frequency_mhz, spectrum in spectra:
            visible = np.abs(frequency_mhz) <= FFT_MAX_FREQUENCY_MHZ
            spectrum_db = 20.0 * np.log10(
                np.maximum(spectrum / reference, 10.0 ** (FFT_FLOOR_DB / 20.0))
            )
            axis.plot(
                frequency_mhz[visible],
                spectrum_db[visible],
                color=protocol.color,
                lw=0.9,
                label=protocol.label,
            )
        for frequency_mhz in (
            -abs(ANHARMONICITY_MHZ),
            abs(ANHARMONICITY_MHZ),
        ):
            axis.axvline(frequency_mhz, color="0.35", lw=0.65, ls=":")
        axis.set_title(f"L={item['duration_us']:g} us")
        axis.set_xlabel("Offset frequency (MHz)")
        axis.set_ylabel("FFT amplitude (dB)")
        axis.set_xlim(-FFT_MAX_FREQUENCY_MHZ, FFT_MAX_FREQUENCY_MHZ)
        axis.set_ylim(FFT_FLOOR_DB, 3.0)
    axes[0, 0].legend(fontsize=6.0, frameon=False, loc="lower left")
    figure.suptitle(
        "Played-envelope FFT, "
        f"c={cutoff:g}, $\\Omega_0/2\\pi={FFT_RABI_MHZ:g}$ MHz"
    )
    return save_figure(figure, f"cutoff_{value_label(cutoff)}_fft")


def plot_focused_maps(summary: dict[str, object]) -> Path:
    """Plot the focused with/without detuning-amplitude population maps."""
    protocols = (
        ("without", "Without DRAG or kappa"),
        ("with", "With DRAG and retuned kappa"),
    )
    with np.load(summary["data_path"], allow_pickle=False) as data:
        maximum_pf = max(float(data[f"{key}_pf"].max()) for key, _ in protocols)
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(8.2, 5.7),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        excited_image = None
        leakage_image = None
        for column, (key, label) in enumerate(protocols):
            excited_image = axes[0, column].pcolormesh(
                DETUNING_MHZ,
                RABI_MHZ,
                data[f"{key}_pe"],
                shading="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                rasterized=True,
            )
            leakage_image = axes[1, column].pcolormesh(
                DETUNING_MHZ,
                RABI_MHZ,
                np.maximum(data[f"{key}_pf"], SCAN_PF_FLOOR),
                shading="auto",
                cmap="magma",
                norm=LogNorm(vmin=SCAN_PF_FLOOR, vmax=maximum_pf),
                rasterized=True,
            )
            centers_mhz = data[f"{key}_centers_mhz"]
            for axis in axes[:, column]:
                axis.plot(centers_mhz, RABI_MHZ, color="white", lw=1.1)
                axis.plot(
                    centers_mhz,
                    RABI_MHZ,
                    color="black",
                    lw=0.35,
                    ls=":",
                )
                axis.set_xlabel(r"Detuning $\Delta/2\pi$ (MHz)")
            axes[0, column].set_title(label)
        axes[0, 0].set_ylabel(r"Amplitude $\Omega_0/2\pi$ (MHz)")
        axes[1, 0].set_ylabel(r"Amplitude $\Omega_0/2\pi$ (MHz)")
        figure.colorbar(
            excited_image,
            ax=axes[0, :],
            pad=0.02,
            label=r"$P_e$",
        )
        figure.colorbar(
            leakage_image,
            ax=axes[1, :],
            pad=0.02,
            label=r"$P_f$",
        )
        figure.suptitle(
            f"Focused qutrit sweep: L={FOCUSED_DURATION_US:g} us, "
            f"c={FOCUSED_CUTOFF:g}"
        )
    return save_figure(figure, "focused_20us_cutoff_0p0025_maps")


def plot_focused_metrics(summary: dict[str, object]) -> Path:
    """Plot amplitude-resolved centers and leakage for the focused sweep."""
    protocols = (
        ("without", "Without DRAG or kappa", "#0072b2"),
        ("with", "With DRAG and retuned kappa", "#d55e00"),
    )
    with np.load(summary["data_path"], allow_pickle=False) as data:
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(8.0, 3.1),
            constrained_layout=True,
        )
        for key, label, color in protocols:
            axes[0].plot(
                RABI_MHZ,
                1e3 * data[f"{key}_centers_mhz"],
                "o-",
                ms=3.0,
                color=color,
                label=label,
            )
            axes[1].semilogy(
                RABI_MHZ,
                data[f"{key}_max_pf_by_rabi"],
                "o-",
                ms=3.0,
                color=color,
                label=label,
            )
        axes[0].axhline(0.0, color="0.5", lw=0.7, ls=":")
        axes[0].set(
            xlabel=r"Amplitude $\Omega_0/2\pi$ (MHz)",
            ylabel="Fitted center (kHz)",
        )
        axes[1].set(
            xlabel=r"Amplitude $\Omega_0/2\pi$ (MHz)",
            ylabel=r"$\max_{\Delta} P_f$",
        )
        axes[0].legend(frameon=False, fontsize=7)
    return save_figure(figure, "focused_20us_cutoff_0p0025_metrics")


def data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def figure_block(path: Path, caption: str) -> str:
    return (
        "<figure>"
        f'<img src="{data_uri(path)}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def summary_rows(summaries: list[dict[str, object]], cutoff: float) -> str:
    rows = []
    selected = sorted(
        (item for item in summaries if float(item["cutoff"]) == cutoff),
        key=lambda item: float(item["duration_us"]),
    )
    for item in selected:
        for protocol in PROTOCOLS:
            values = item["protocols"][protocol.key]
            rows.append(
                "<tr>"
                f"<td>{item['duration_us']:g}</td>"
                f"<td>{html.escape(protocol.label)}</td>"
                f"<td>{values['kappa_mhz_inv']:+.6f}</td>"
                f"<td>{100 * values['maximum_final_pf']:.4f}%</td>"
                f"<td>{values['center_rms_khz']:.3f}</td>"
                f"<td>{values['maximum_abs_center_khz']:.3f}</td>"
                f"<td>{values['minimum_pe_contrast']:.4f}</td>"
                "</tr>"
            )
    return "".join(rows)


def write_summary_csv(summaries: list[dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "duration_us",
                "cutoff",
                "protocol",
                "kappa_mhz_inv",
                "maximum_final_pf",
                "center_rms_khz",
                "maximum_abs_center_khz",
                "minimum_pe_contrast",
            )
        )
        for item in summaries:
            for protocol in PROTOCOLS:
                values = item["protocols"][protocol.key]
                writer.writerow(
                    (
                        item["duration_us"],
                        item["cutoff"],
                        protocol.key,
                        values["kappa_mhz_inv"],
                        values["maximum_final_pf"],
                        values["center_rms_khz"],
                        values["maximum_abs_center_khz"],
                        values["minimum_pe_contrast"],
                    )
                )


def encode_scan(values: np.ndarray) -> str:
    """Encode normalized display values as little-endian unsigned 16-bit data."""
    quantized = np.rint(65535.0 * np.clip(values, 0.0, 1.0)).astype("<u2")
    return base64.b64encode(quantized.tobytes()).decode("ascii")


def build_scan_payload(summaries: list[dict[str, object]]) -> dict[str, object]:
    """Pack all individual detuning/amplitude maps for the HTML explorer."""
    maximum_pf = max(
        float(item["protocols"][protocol.key]["maximum_final_pf"])
        for item in summaries
        for protocol in PROTOCOLS
    )
    log_pf_floor = float(np.log10(SCAN_PF_FLOOR))
    log_pf_ceiling = float(np.log10(maximum_pf))
    maps = {}
    for item in summaries:
        cutoff_label = value_label(float(item["cutoff"]))
        duration_label = value_label(float(item["duration_us"]))
        with np.load(item["data_path"], allow_pickle=False) as data:
            for protocol in PROTOCOLS:
                excited = np.asarray(data[f"{protocol.key}_pe"])
                leakage = np.asarray(data[f"{protocol.key}_pf"])
                leakage_log = (
                    np.log10(np.maximum(leakage, SCAN_PF_FLOOR))
                    - log_pf_floor
                ) / (log_pf_ceiling - log_pf_floor)
                key = f"{cutoff_label}|{duration_label}|{protocol.key}"
                values = item["protocols"][protocol.key]
                maps[key] = {
                    "pe": encode_scan(excited),
                    "pf": encode_scan(leakage_log),
                    "centers_mhz": np.round(
                        data[f"{protocol.key}_centers_mhz"], 8
                    ).tolist(),
                    "kappa_mhz_inv": float(values["kappa_mhz_inv"]),
                    "maximum_final_pf": float(values["maximum_final_pf"]),
                    "center_rms_khz": float(values["center_rms_khz"]),
                }
    return {
        "rows": int(RABI_MHZ.size),
        "columns": int(DETUNING_MHZ.size),
        "rabi_mhz": RABI_MHZ.tolist(),
        "detuning_mhz": DETUNING_MHZ.tolist(),
        "pf_log_floor": log_pf_floor,
        "pf_log_ceiling": log_pf_ceiling,
        "maps": maps,
    }


def scan_button_group(
    label: str,
    values: list[tuple[str, str]],
    selected: str,
) -> str:
    buttons = "".join(
        f'<button type="button" data-scan-group="{html.escape(label.lower())}" '
        f'data-scan-value="{html.escape(value)}" '
        f'aria-pressed="{str(value == selected).lower()}">{html.escape(text)}</button>'
        for value, text in values
    )
    return (
        f'<div class="scan-control" role="group" aria-label="{html.escape(label)}">'
        f"<b>{html.escape(label)}</b>{buttons}</div>"
    )


def scan_explorer_panel() -> str:
    initial_duration = value_label(10.0)
    initial_protocol = "smooth_drag_retuned"
    controls = "".join(
        (
            scan_button_group(
                "Cutoff",
                [(value_label(value), f"c={value:g}") for value in CUTOFFS],
                value_label(CUTOFFS[0]),
            ),
            scan_button_group(
                "Duration",
                [
                    (value_label(value), f"{value:g} µs")
                    for value in DURATIONS_US
                ],
                initial_duration,
            ),
            scan_button_group(
                "Protocol",
                [(protocol.key, protocol.label) for protocol in PROTOCOLS],
                initial_protocol,
            ),
            scan_button_group(
                "Population",
                [("pe", "P_e"), ("pf", "P_f (log)")],
                "pe",
            ),
        )
    )
    return (
        '<section role="tabpanel" id="panel-scans" '
        'aria-labelledby="tab-scans" hidden>'
        '<h2>Individual detuning-amplitude scans</h2>'
        f'<div class="scan-switches">{controls}</div>'
        '<div id="scan-selection" class="scan-summary" aria-live="polite"></div>'
        '<div class="scan-view"><canvas id="scan-canvas" width="960" height="540" '
        'role="img" aria-label="Selected detuning-amplitude population map">'
        'Interactive detuning-amplitude population map.</canvas></div>'
        '<p class="scan-note"><span id="scan-hover" aria-live="polite">Move over the map for a value.</span> '
        'The white curve is the fitted local minimum; P_f uses one shared logarithmic scale.</p>'
        "</section>"
    )


def focused_comparison_panel(
    summary: dict[str, object],
    maps_path: Path,
    metrics_path: Path,
) -> str:
    without = summary["protocols"]["without"]
    with_correction = summary["protocols"]["with"]
    rows = "".join(
        (
            "<tr><td>Without DRAG or kappa</td>"
            f"<td>{without['drag_beta']:.1f}</td>"
            f"<td>{without['kappa_mhz_inv']:+.6f}</td>"
            f"<td>{100 * without['maximum_final_pf']:.4f}%</td>"
            f"<td>{without['center_rms_khz']:.3f}</td>"
            f"<td>{without['maximum_abs_center_khz']:.3f}</td>"
            f"<td>{without['minimum_pe_contrast']:.4f}</td></tr>",
            "<tr><td>With DRAG and retuned kappa</td>"
            f"<td>{with_correction['drag_beta']:.1f}</td>"
            f"<td>{with_correction['kappa_mhz_inv']:+.6f}</td>"
            f"<td>{100 * with_correction['maximum_final_pf']:.4f}%</td>"
            f"<td>{with_correction['center_rms_khz']:.3f}</td>"
            f"<td>{with_correction['maximum_abs_center_khz']:.3f}</td>"
            f"<td>{with_correction['minimum_pe_contrast']:.4f}</td></tr>",
        )
    )
    return (
        '<section role="tabpanel" id="panel-focused" '
        'aria-labelledby="tab-focused" hidden>'
        f'<h2>Focused comparison: L={summary["duration_us"]:g} us, '
        f'c={summary["cutoff"]:g}</h2>'
        '<p>The baseline is the same smooth echo with both DRAG and kappa set '
        'to zero. The corrected case uses beta=1 and kappa optimized on this '
        'exact duration/cutoff configuration.</p>'
        '<div class="cards">'
        f'<div><b>{with_correction["kappa_mhz_inv"]:+.6f}</b><span>retuned kappa (MHz^-1)</span></div>'
        f'<div><b>{summary["leakage_reduction"]:.1f}x</b><span>worst-case leakage reduction</span></div>'
        f'<div><b>{without["center_rms_khz"]:.2f} &rarr; {with_correction["center_rms_khz"]:.2f}</b>'
        '<span>center RMS (kHz)</span></div></div>'
        '<div class="grid">'
        + figure_block(
            maps_path,
            "Individual detuning-amplitude P_e and P_f maps",
        )
        + figure_block(
            metrics_path,
            "Amplitude-resolved fitted centers and worst leakage",
        )
        + '</div><div class="table-wrap"><table><thead><tr>'
        '<th>Protocol</th><th>beta</th><th>kappa (MHz^-1)</th>'
        '<th>max P_f</th><th>center RMS (kHz)</th>'
        '<th>max |center| (kHz)</th><th>min contrast</th>'
        f"</tr></thead><tbody>{rows}</tbody></table></div></section>"
    )


def write_report(
    summaries: list[dict[str, object]],
    overview_leakage: Path,
    overview_centers: Path,
    kappa_summary: Path,
    cutoff_figures: dict[float, tuple[Path, Path, Path]],
    focused_summary: dict[str, object],
    focused_maps: Path,
    focused_metrics: Path,
) -> None:
    scan_payload = build_scan_payload(summaries)
    scan_json = json.dumps(scan_payload, separators=(",", ":"))
    tabs = [
        '<button role="tab" id="tab-overview" aria-controls="panel-overview" '
        'aria-selected="true">Overview</button>',
        '<button role="tab" id="tab-method" aria-controls="panel-method" '
        'aria-selected="false">Method</button>',
        '<button role="tab" id="tab-scans" aria-controls="panel-scans" '
        'aria-selected="false">Individual maps</button>',
        '<button role="tab" id="tab-focused" aria-controls="panel-focused" '
        'aria-selected="false">20 us / c=0.0025</button>',
    ]
    panels = [
        '<section role="tabpanel" id="panel-overview" aria-labelledby="tab-overview">'
        f'<div class="cards"><div><b>{len(DURATIONS_US) * len(CUTOFFS)}+1</b>'
        '<span>grid settings plus focused comparison</span></div>'
        '<div><b>4</b><span>protocols per shape</span></div>'
        '<div><b>&le;1 ns</b><span>RK4 time step</span></div></div>'
        '<div class="grid">'
        + figure_block(overview_leakage, "Maximum final leakage across amplitude and detuning")
        + figure_block(overview_centers, "RMS fitted center displacement across amplitudes")
        + figure_block(kappa_summary, "Retuned DRAG kappa across duration and cutoff")
        + "</div></section>",
        '<section role="tabpanel" id="panel-method" aria-labelledby="tab-method" hidden>'
        '<h2>Pulse construction</h2>'
        '<p>The current echo uses an instantaneous sign reversal. The smooth '
        'versions replace it by a peak-normalized <code>-tanh(t/tau)</code> '
        f'zero crossing with <b>tau={ECHO_TRANSITION_NS:g} ns</b>. DRAG uses '
        '<code>Omega_Q=-beta d(Omega_I)/dt/alpha</code> with beta=1 and the '
        'derivative of the complete signed waveform.</p>'
        '<h2>AC-Stark phase and kappa</h2>'
        '<div class="equation">delta f_corr(t) = kappa [Omega_I(t)/(2 pi)]^2; '
        'phi(t) = 2 pi integral delta f_corr(t) dt</div>'
        '<p>The first three protocols retain the reference '
        f'<b>kappa={KAPPA_REFERENCE_MHZ_INV:+.5f} MHz^-1</b>. The fourth '
        'protocol retunes kappa independently for every duration/cutoff pair '
        'by minimizing the RMS fitted center over the complete 10-80 MHz grid.</p>'
        f'<p>The center tracker follows the local minimum nearest zero inside '
        f'+/-{1e3 * FIT_WINDOW_MHZ:g} kHz. Large residuals at weak contrast '
        'mean that one amplitude-independent kappa no longer describes the '
        'line reliably; they should not be interpreted as precise frequency shifts.</p>'
        '<h2>FFT convention</h2>'
        f'<p>The spectra use the complete complex played envelope at '
        f'<b>Omega0/(2 pi)={FFT_RABI_MHZ:g} MHz</b>, including the accumulated '
        'kappa phase and, where applicable, the DRAG quadrature. Each pulse-length '
        'panel uses one shared amplitude normalization for all four protocols. '
        f'Dotted lines mark +/-|alpha|/(2 pi)={abs(ANHARMONICITY_MHZ):g} MHz. '
        'No analysis window is applied, so the finite pulse truncation remains '
        f'visible; zero padding only samples the FFT more densely. The '
        f'{FFT_FLOOR_DB:g} dB floor is for display.</p>'
        '<h2>Individual-map display</h2>'
        '<p>The map explorer contains all 72 duration/cutoff/protocol scans. '
        'P_e uses the fixed 0-1 population scale; P_f uses one shared logarithmic '
        f'scale with a {SCAN_PF_FLOOR:g} display floor. The embedded maps are '
        '16-bit display encodings, while the NPZ files retain full precision.</p>'
        '<h2>Interpretation boundary</h2>'
        '<p>This is a three-level rotating-wave simulation. The reported '
        'leakage is final P_f, not maximum transient leakage. No measured '
        'microwave transfer function or predistortion is included, and the '
        'fitted coefficients are not hardware calibrations. The FFT is a '
        'waveform diagnostic, not the nonlinear qutrit response itself.</p>'
        '<p>Literature basis: <a href="https://doi.org/10.1103/PhysRevLett.103.110501">'
        'Motzoi et al. (2009)</a>, <a href="https://doi.org/10.1103/PhysRevA.83.012308">'
        'Gambetta et al. (2011)</a>, and '
        '<a href="https://doi.org/10.1103/PRXQuantum.5.030353">Hyyppa et al. (2024)</a>.</p>'
        "</section>",
        scan_explorer_panel(),
        focused_comparison_panel(
            focused_summary,
            focused_maps,
            focused_metrics,
        ),
    ]
    for cutoff in CUTOFFS:
        label = value_label(cutoff)
        tabs.append(
            f'<button role="tab" id="tab-cutoff-{label}" '
            f'aria-controls="panel-cutoff-{label}" aria-selected="false">'
            f"c={cutoff:g}</button>"
        )
        leakage_path, centers_path, fft_path = cutoff_figures[cutoff]
        panels.append(
            f'<section role="tabpanel" id="panel-cutoff-{label}" '
            f'aria-labelledby="tab-cutoff-{label}" hidden><div class="grid">'
            + figure_block(leakage_path, f"Worst final leakage for cutoff {cutoff:g}")
            + figure_block(centers_path, f"Fitted centers for cutoff {cutoff:g}")
            + figure_block(
                fft_path,
                f"Complete played-envelope FFT for cutoff {cutoff:g}",
            )
            + '</div><div class="table-wrap"><table><thead><tr>'
            '<th>L (us)</th><th>Protocol</th><th>kappa (MHz^-1)</th>'
            '<th>max P_f</th><th>center RMS (kHz)</th>'
            '<th>max |center| (kHz)</th><th>min contrast</th>'
            "</tr></thead><tbody>"
            + summary_rows(summaries, cutoff)
            + "</tbody></table></div></section>"
        )

    metadata = {
        "durations_us": DURATIONS_US,
        "cutoffs": CUTOFFS,
        "protocols": [protocol.key for protocol in PROTOCOLS],
        "rabi_mhz": RABI_MHZ.tolist(),
        "detuning_mhz": DETUNING_MHZ.tolist(),
        "steps_per_us": STEPS_PER_US,
        "t1_us": T1_US,
        "t2_us": T2_US,
        "anharmonicity_mhz": ANHARMONICITY_MHZ,
        "fft_rabi_mhz": FFT_RABI_MHZ,
        "fft_time_step_ns": FFT_TIME_STEP_NS,
        "fft_max_frequency_mhz": FFT_MAX_FREQUENCY_MHZ,
        "individual_scan_count": len(scan_payload["maps"]),
        "scan_encoding": "uint16 normalized display values",
        "scan_pf_floor": SCAN_PF_FLOOR,
        "focused_comparison": {
            "duration_us": focused_summary["duration_us"],
            "cutoff": focused_summary["cutoff"],
            "retuned_kappa_mhz_inv": focused_summary[
                "retuned_kappa_mhz_inv"
            ],
            "baseline": "smooth echo with beta=0 and kappa=0",
            "corrected": "smooth echo with beta=1 and retuned kappa",
        },
    }
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smooth echo and DRAG leakage sweep</title>
<style>
:root{{--bg:#fbfcfe;--fg:#18212b;--muted:#5d6877;--line:#d8dee7;--panel:#fff;--accent:#0b6fb8}}
*{{box-sizing:border-box}} body{{margin:0;padding:24px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto}} h1{{margin:0 0 4px;font-size:clamp(24px,3vw,38px)}} h2{{margin-top:28px}}
.subtitle{{color:var(--muted);margin:0 0 18px}} .config,.tabs,.cards{{display:flex;flex-wrap:wrap;gap:9px 18px}}
.config{{padding:12px 0;margin-bottom:16px;border-block:1px solid var(--line);color:var(--muted)}}
.tabs{{margin-bottom:20px}} button{{border:1px solid var(--line);border-radius:999px;padding:8px 15px;background:var(--panel);color:var(--fg);font:inherit;cursor:pointer}}
button[aria-selected="true"]{{background:var(--accent);border-color:var(--accent);color:white}}
.scan-switches{{display:grid;gap:9px;margin:0 0 14px;max-width:1200px}}
.scan-control{{display:flex;flex-wrap:wrap;align-items:center;gap:7px}}
.scan-control>b{{min-width:82px}} .scan-control button{{padding:6px 11px}}
.scan-control button[aria-pressed="true"]{{background:var(--accent);border-color:var(--accent);color:white}}
.scan-summary{{max-width:960px;margin:10px 0;color:var(--muted);font-variant-numeric:tabular-nums}}
.scan-view{{max-width:960px}} #scan-canvas{{display:block;width:100%;height:auto;background:white;border:1px solid var(--line)}}
.scan-note{{max-width:960px;margin:7px 0;color:var(--muted)}}
.cards{{margin-bottom:20px}} .cards div{{min-width:150px;padding:14px 18px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}}
.cards b{{display:block;font-size:24px}} .cards span,figcaption{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}} figure{{margin:0;min-width:0}}
img{{display:block;width:100%;height:auto;background:white;border:1px solid var(--line)}} figcaption{{margin-top:6px}}
.equation{{padding:14px 18px;border-left:4px solid var(--accent);background:var(--panel);font:17px Georgia,serif}}
.table-wrap{{overflow:auto;margin-top:24px;max-height:680px}} table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th:nth-child(2),td:nth-child(2){{text-align:left}}
thead{{position:sticky;top:0;background:var(--bg)}} footer{{margin-top:28px;color:var(--muted)}} a{{color:var(--accent)}}
@media(max-width:850px){{body{{padding:14px}}.grid{{grid-template-columns:1fr}}}}
@media print{{.tabs{{display:none}}[role="tabpanel"][hidden]{{display:block;page-break-before:always}}}}
</style></head><body><main>
<h1>Smooth echo and DRAG leakage sweep</h1>
<p class="subtitle">Three-level AC-Stark-corrected spectroscopy across pulse length and Lorentzian cutoff.</p>
<div class="config"><span><b>L</b> = {", ".join(f"{v:g}" for v in DURATIONS_US)} us</span>
<span><b>cutoff c</b> = {", ".join(f"{v:g}" for v in CUTOFFS)}</span>
<span><b>Rabi</b> = {RABI_MHZ.min():g}-{RABI_MHZ.max():g} MHz</span>
<span><b>alpha/2pi</b> = {ANHARMONICITY_MHZ:g} MHz</span><span><b>T1/T2</b> = {T1_US:.2f}/{T2_US:.2f} us</span></div>
<nav class="tabs" role="tablist">{"".join(tabs)}</nav>{"".join(panels)}
<footer>Self-contained report. Numerical maps are also saved as NPZ and the summary as CSV.</footer>
<script type="application/json" id="report-metadata">{html.escape(json.dumps(metadata, separators=(",", ":")))}</script>
<script type="application/json" id="scan-data">{scan_json}</script>
</main><script>
const tabs=[...document.querySelectorAll('[role="tab"]')],panels=[...document.querySelectorAll('[role="tabpanel"]')];
const scanData=JSON.parse(document.getElementById('scan-data').textContent);
const scanState={{cutoff:'0p001',duration:'10',protocol:'smooth_drag_retuned',population:'pe'}};
let activeScan=null;
function select(tab){{
  tabs.forEach(x=>x.setAttribute('aria-selected',String(x===tab)));
  panels.forEach(x=>x.hidden=x.id!==tab.getAttribute('aria-controls'));
  history.replaceState(null,'','#'+tab.id.replace('tab-',''));
  if(tab.id==='tab-scans') drawScan();
}}
tabs.forEach((tab,index)=>{{
  tab.addEventListener('click',()=>select(tab));
  tab.addEventListener('keydown',event=>{{
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
    event.preventDefault();
    let next=index;
    if(event.key==='ArrowLeft') next=(index-1+tabs.length)%tabs.length;
    if(event.key==='ArrowRight') next=(index+1)%tabs.length;
    if(event.key==='Home') next=0;
    if(event.key==='End') next=tabs.length-1;
    tabs[next].focus(); select(tabs[next]);
  }});
}});

function decodeMap(encoded){{
  const raw=atob(encoded), values=new Float32Array(raw.length/2);
  for(let i=0;i<values.length;i++){{
    values[i]=(raw.charCodeAt(2*i)|(raw.charCodeAt(2*i+1)<<8))/65535;
  }}
  return values;
}}
const viridis=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
function colorAt(value){{
  const position=Math.max(0,Math.min(1,value))*(viridis.length-1);
  const index=Math.min(viridis.length-2,Math.floor(position));
  const fraction=position-index, a=viridis[index], b=viridis[index+1];
  return a.map((component,i)=>Math.round(component+fraction*(b[i]-component)));
}}
function populationValue(normalized,population){{
  if(population==='pe') return normalized;
  return Math.pow(10,scanData.pf_log_floor+normalized*(scanData.pf_log_ceiling-scanData.pf_log_floor));
}}
function formatPopulation(value,population){{
  if(population==='pe') return value.toFixed(4);
  return value<0.001?value.toExponential(2):value.toFixed(4);
}}
function drawScan(){{
  const key=[scanState.cutoff,scanState.duration,scanState.protocol].join('|');
  const entry=scanData.maps[key];
  if(!entry) return;
  const values=decodeMap(entry[scanState.population]);
  const canvas=document.getElementById('scan-canvas'),ctx=canvas.getContext('2d');
  const width=canvas.width,height=canvas.height;
  const margin={{left:82,right:132,top:52,bottom:72}};
  const plot={{x:margin.left,y:margin.top,w:width-margin.left-margin.right,h:height-margin.top-margin.bottom}};
  const rows=scanData.rows,columns=scanData.columns;
  const detuning=scanData.detuning_mhz,rabi=scanData.rabi_mhz;
  const xMin=detuning[0],xMax=detuning[detuning.length-1];
  const yMin=rabi[0],yMax=rabi[rabi.length-1];
  const protocolButton=document.querySelector('[data-scan-group="protocol"][aria-pressed="true"]');
  const protocolLabel=protocolButton.textContent;
  const populationLabel=scanState.population==='pe'?'P_e':'P_f';
  ctx.clearRect(0,0,width,height); ctx.fillStyle='#fff'; ctx.fillRect(0,0,width,height);
  const offscreen=document.createElement('canvas'); offscreen.width=columns; offscreen.height=rows;
  const offscreenContext=offscreen.getContext('2d');
  const image=offscreenContext.createImageData(columns,rows);
  for(let row=0;row<rows;row++){{
    for(let column=0;column<columns;column++){{
      const source=row*columns+column,destination=((rows-1-row)*columns+column)*4;
      const color=colorAt(values[source]);
      image.data[destination]=color[0]; image.data[destination+1]=color[1];
      image.data[destination+2]=color[2]; image.data[destination+3]=255;
    }}
  }}
  offscreenContext.putImageData(image,0,0);
  ctx.imageSmoothingEnabled=false; ctx.drawImage(offscreen,plot.x,plot.y,plot.w,plot.h);
  const mapX=value=>plot.x+(value-xMin)/(xMax-xMin)*plot.w;
  const mapY=value=>plot.y+plot.h-(value-yMin)/(yMax-yMin)*plot.h;
  ctx.strokeStyle='rgba(0,0,0,.8)'; ctx.lineWidth=4; ctx.beginPath();
  entry.centers_mhz.forEach((value,index)=>{{const x=mapX(value),y=mapY(rabi[index]);if(index===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}});ctx.stroke();
  ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
  ctx.strokeStyle='#26313d';ctx.lineWidth=1;ctx.strokeRect(plot.x,plot.y,plot.w,plot.h);
  ctx.fillStyle='#18212b';ctx.font='14px system-ui,sans-serif';ctx.textAlign='center';
  for(const tick of [-0.3,-0.2,-0.1,0,0.1,0.2,0.3]){{
    const x=mapX(tick);ctx.beginPath();ctx.moveTo(x,plot.y+plot.h);ctx.lineTo(x,plot.y+plot.h+6);ctx.stroke();
    ctx.fillText(tick===0?'0':tick.toFixed(1),x,plot.y+plot.h+24);
  }}
  ctx.textAlign='right';
  for(const tick of [10,20,30,40,50,60,70,80]){{
    const y=mapY(tick);ctx.beginPath();ctx.moveTo(plot.x-6,y);ctx.lineTo(plot.x,y);ctx.stroke();ctx.fillText(String(tick),plot.x-11,y+5);
  }}
  ctx.textAlign='center';ctx.font='16px system-ui,sans-serif';
  ctx.fillText('Detuning Δ/2π (MHz)',plot.x+plot.w/2,height-18);
  ctx.save();ctx.translate(24,plot.y+plot.h/2);ctx.rotate(-Math.PI/2);ctx.fillText('Amplitude Ω₀/2π (MHz)',0,0);ctx.restore();
  ctx.font='16px system-ui,sans-serif';
  ctx.fillText(protocolLabel+' · c='+scanState.cutoff.replace('p','.')+' · L='+scanState.duration.replace('p','.')+' µs',plot.x+plot.w/2,26);
  const barX=plot.x+plot.w+36,barWidth=24;
  for(let pixel=0;pixel<plot.h;pixel++){{
    const normalized=1-pixel/(plot.h-1),color=colorAt(normalized);
    ctx.fillStyle='rgb('+color.join(',')+')';ctx.fillRect(barX,plot.y+pixel,barWidth,1);
  }}
  ctx.strokeStyle='#26313d';ctx.strokeRect(barX,plot.y,barWidth,plot.h);
  ctx.fillStyle='#18212b';ctx.font='13px system-ui,sans-serif';ctx.textAlign='left';
  for(const normalized of [0,0.25,0.5,0.75,1]){{
    const y=plot.y+(1-normalized)*plot.h;
    ctx.beginPath();ctx.moveTo(barX+barWidth,y);ctx.lineTo(barX+barWidth+5,y);ctx.stroke();
    ctx.fillText(formatPopulation(populationValue(normalized,scanState.population),scanState.population),barX+barWidth+9,y+4);
  }}
  ctx.save();ctx.translate(width-17,plot.y+plot.h/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.font='14px system-ui,sans-serif';ctx.fillText(populationLabel+(scanState.population==='pf'?' (log scale)':''),0,0);ctx.restore();
  const summary=protocolLabel+' · κ='+Number(entry.kappa_mhz_inv).toFixed(6)+' MHz⁻¹ · max P_f='+Number(entry.maximum_final_pf).toExponential(3)+' · center RMS='+Number(entry.center_rms_khz).toFixed(2)+' kHz';
  document.getElementById('scan-selection').textContent=summary;
  canvas.setAttribute('aria-label',populationLabel+' detuning-amplitude map for '+protocolLabel+', cutoff '+scanState.cutoff.replace('p','.')+', duration '+scanState.duration.replace('p','.')+' microseconds');
  activeScan={{values,entry,population:scanState.population,plot,detuning,rabi}};
  document.getElementById('scan-hover').textContent='Move over the map for a value.';
}}
document.querySelectorAll('[data-scan-group]').forEach(button=>{{
  button.addEventListener('click',()=>{{
    const group=button.dataset.scanGroup;
    scanState[group]=button.dataset.scanValue;
    document.querySelectorAll('[data-scan-group="'+group+'"]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
    drawScan();
  }});
}});
document.getElementById('scan-canvas').addEventListener('pointermove',event=>{{
  if(!activeScan) return;
  const canvas=event.currentTarget,rect=canvas.getBoundingClientRect();
  const x=(event.clientX-rect.left)*canvas.width/rect.width,y=(event.clientY-rect.top)*canvas.height/rect.height;
  const plot=activeScan.plot;
  if(x<plot.x||x>plot.x+plot.w||y<plot.y||y>plot.y+plot.h) return;
  const column=Math.max(0,Math.min(scanData.columns-1,Math.floor((x-plot.x)/plot.w*scanData.columns)));
  const row=Math.max(0,Math.min(scanData.rows-1,scanData.rows-1-Math.floor((y-plot.y)/plot.h*scanData.rows)));
  const normalized=activeScan.values[row*scanData.columns+column];
  const value=populationValue(normalized,activeScan.population);
  document.getElementById('scan-hover').textContent='Δ/2π='+activeScan.detuning[column].toFixed(3)+' MHz · Ω₀/2π='+activeScan.rabi[row].toFixed(1)+' MHz · '+(activeScan.population==='pe'?'P_e':'P_f')+'='+formatPopulation(value,activeScan.population)+'.';
}});
document.getElementById('scan-canvas').addEventListener('pointerleave',()=>{{document.getElementById('scan-hover').textContent='Move over the map for a value.';}});
drawScan();
const requested=document.getElementById('tab-'+location.hash.slice(1));if(requested)select(requested);
</script></body></html>"""
    REPORT_PATH.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="ignore cached NPZ files")
    parser.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(duration, cutoff) for cutoff in CUTOFFS for duration in DURATIONS_US]
    summaries = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(run_configuration, duration, cutoff, args.force): (
                duration,
                cutoff,
            )
            for duration, cutoff in jobs
        }
        for future in as_completed(futures):
            summaries.append(future.result())
    summaries.sort(key=lambda item: (float(item["cutoff"]), float(item["duration_us"])))
    focused_summary = run_focused_comparison(args.force)

    apply_figure_style(FigureVariant.PAPER)
    overview_leakage = plot_overview(
        summaries,
        "maximum_final_pf",
        "Worst final leakage",
        r"$\log_{10}(\max P_f)$",
        "00_overview_leakage",
        logarithmic=True,
    )
    overview_centers = plot_overview(
        summaries,
        "center_rms_khz",
        "Fitted center stability",
        "Center RMS (kHz)",
        "01_overview_centers",
        logarithmic=False,
    )
    kappa_summary = plot_kappa_summary(summaries)
    focused_maps = plot_focused_maps(focused_summary)
    focused_metrics = plot_focused_metrics(focused_summary)
    cutoff_figures = {}
    for cutoff in CUTOFFS:
        cutoff_figures[cutoff] = (
            plot_cutoff_curves(
                summaries,
                cutoff,
                "max_pf_by_rabi",
                r"$\max_{\Delta} P_f$",
                "leakage",
                logarithmic=True,
            ),
            plot_cutoff_curves(
                summaries,
                cutoff,
                "centers_mhz",
                "Fitted center (MHz)",
                "centers",
                logarithmic=False,
            ),
            plot_cutoff_fft(summaries, cutoff),
        )
    write_summary_csv(summaries)
    write_report(
        summaries,
        overview_leakage,
        overview_centers,
        kappa_summary,
        cutoff_figures,
        focused_summary,
        focused_maps,
        focused_metrics,
    )
    provenance = {
        "description": "Three-level smooth-echo and DRAG duration/cutoff sweep",
        "report": str(REPORT_PATH),
        "durations_us": DURATIONS_US,
        "cutoffs": CUTOFFS,
        "configuration_count": len(summaries),
        "protocol_count": len(PROTOCOLS),
        "focused_comparison": {
            "duration_us": focused_summary["duration_us"],
            "cutoff": focused_summary["cutoff"],
            "retuned_kappa_mhz_inv": focused_summary[
                "retuned_kappa_mhz_inv"
            ],
            "leakage_reduction": focused_summary["leakage_reduction"],
            "data_path": focused_summary["data_path"],
        },
        "limitations": [
            "Final P_f is reported; maximum transient leakage is not retained.",
            "The model has three levels and uses the rotating-wave approximation.",
            "No measured transfer function or waveform predistortion is applied.",
            "Simulation coefficients are not hardware calibrations.",
            "FFT curves diagnose the played waveform, not the nonlinear qutrit response.",
        ],
        "fft": {
            "rabi_mhz": FFT_RABI_MHZ,
            "time_step_ns": FFT_TIME_STEP_NS,
            "frequency_limit_mhz": FFT_MAX_FREQUENCY_MHZ,
            "display_floor_db": FFT_FLOOR_DB,
            "window": None,
            "normalization": "shared maximum across protocols in each panel",
        },
        "individual_scan_explorer": {
            "map_count": len(summaries) * len(PROTOCOLS),
            "populations": ["P_e", "P_f"],
            "encoding": "uint16 display values; full precision remains in NPZ",
            "pf_scale": "shared logarithmic scale",
            "pf_floor": SCAN_PF_FLOOR,
        },
        "reproduction_command": (
            "PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python "
            "scripts/make_smooth_echo_drag_sweep_report.py"
        ),
    }
    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved HTML report to {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
