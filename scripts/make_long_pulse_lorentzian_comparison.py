"""Plot matched experimental Lorentzian and echo-Lorentzian length scans."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from echospec.figures import FigureVariant, apply_figure_style, save_figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("OPX1000_DATA_DIR", PROJECT_ROOT.parent / "data_opx1000")
)


@dataclass(frozen=True)
class ScanSelection:
    duration_us: int
    no_echo_run: str
    echo_run: str


# These six runs are one controlled q9 sequence acquired on 2026-06-23. All
# use the same detuning/amplitude grid, 100 shots, cutoff=5e-4, tau=8 ns, and
# peak waveform amplitude=0.5. For the 160-us pulse, QUA stretches a 60-us
# stored template to the physical duration recorded in the parameters.
SELECTIONS = (
    ScanSelection(10, "08-21-04-530458", "13-11-08-347658"),
    ScanSelection(60, "08-44-25-321864", "13-35-23-672149"),
    ScanSelection(160, "10-00-01-972580", "14-50-56-162603"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the six-panel experimental pulse-duration comparison for "
            "the Supplemental Material."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root of the data_opx1000 repository.",
    )
    return parser.parse_args()


def run_directory(data_root: Path, run: str) -> Path:
    return (
        data_root
        / "calibrations"
        / "2026-06-23"
        / "echo_lorentzian"
        / run
    )


def load_scan(
    data_root: Path,
    run: str,
    *,
    expected_duration_us: int,
    expected_echo: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directory = run_directory(data_root, run)
    parameters_path = directory / "parameters.json"
    sweep_path = directory / "sweep.npz"
    results_path = directory / "results.npz"
    missing = [path for path in (parameters_path, sweep_path, results_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files for {run}: {missing}")

    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    expected = {
        "lorentzian_length_in_ns": expected_duration_us * 1000,
        "echo": expected_echo,
        "lorentzian_tau_in_ns": 8.0,
        "lorentzian_peak_amplitude": 0.5,
        "cutoff": 0.0005,
        "num_shots": 100,
    }
    mismatches = {
        key: (parameters.get(key), value)
        for key, value in expected.items()
        if parameters.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Unexpected parameters in {run}: {mismatches}")

    with np.load(sweep_path) as sweep:
        qubits = tuple(sweep["qubit"].tolist())
        detuning_mhz = np.asarray(sweep["detuning"], dtype=float) / 1e6
        amplitude_prefactor = np.asarray(sweep["amp_prefactor"], dtype=float)
    if qubits != ("q9",):
        raise ValueError(f"Expected q9 data in {run}, found {qubits}")

    with np.load(results_path) as results:
        state = np.asarray(results["state"], dtype=float)
    expected_shape = (1, detuning_mhz.size, amplitude_prefactor.size)
    if state.shape != expected_shape:
        raise ValueError(
            f"Unexpected state shape in {run}: {state.shape}, expected {expected_shape}"
        )
    if not np.isfinite(state).all():
        raise ValueError(f"Non-finite state values in {run}")

    return detuning_mhz, amplitude_prefactor, state[0].T


def build_figure(data_root: Path) -> plt.Figure:
    apply_figure_style(FigureVariant.PAPER)
    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 4.15),
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    loaded: dict[tuple[int, bool], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for selection in SELECTIONS:
        loaded[(selection.duration_us, False)] = load_scan(
            data_root,
            selection.no_echo_run,
            expected_duration_us=selection.duration_us,
            expected_echo=False,
        )
        loaded[(selection.duration_us, True)] = load_scan(
            data_root,
            selection.echo_run,
            expected_duration_us=selection.duration_us,
            expected_echo=True,
        )

    reference_detuning, reference_amplitude, _ = loaded[(SELECTIONS[0].duration_us, False)]
    for key, (detuning, amplitude, _) in loaded.items():
        if not np.array_equal(detuning, reference_detuning):
            raise ValueError(f"Detuning grid differs for selection {key}")
        if not np.array_equal(amplitude, reference_amplitude):
            raise ValueError(f"Amplitude grid differs for selection {key}")

    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, constrained_layout=True)
    panel_labels = iter("abcdef")
    image = None
    for row, echo in enumerate((False, True)):
        for column, selection in enumerate(SELECTIONS):
            ax = axes[row, column]
            detuning, amplitude, state = loaded[(selection.duration_us, echo)]
            image = ax.pcolormesh(
                detuning,
                amplitude,
                state,
                shading="auto",
                cmap="viridis",
                vmin=0.10,
                vmax=0.85,
                rasterized=True,
            )
            ax.text(
                0.03,
                0.94,
                f"({next(panel_labels)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="white",
                fontsize=7,
            )
            if row == 0:
                ax.set_title(rf"$L={selection.duration_us}\,\mu\mathrm{{s}}$")
            if row == 1:
                ax.set_xlabel(r"Drive detuning $(f_d-f_{01})$ (MHz)")
            if column == 0:
                protocol = "Echo-root-Lorentzian" if echo else "Root-Lorentzian"
                ax.set_ylabel(f"{protocol}\nAmplitude prefactor")
    if image is None:
        raise RuntimeError("No panels were generated")
    colorbar = fig.colorbar(image, ax=axes, pad=0.015, fraction=0.035)
    colorbar.set_label(r"$P_e$")
    colorbar.ax.tick_params(labelsize=6)
    return fig


def main() -> None:
    args = parse_args()
    fig = build_figure(args.data_root.expanduser().resolve())
    paths = save_figure(
        fig,
        "06_long_pulse_lorentzian_comparison",
        variant=FigureVariant.PAPER,
        formats=("pdf", "png", "svg"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06,
    )
    plt.close(fig)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
