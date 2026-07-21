from __future__ import annotations

# Backend and local-source setup must precede pyplot and echospec imports.
# ruff: noqa: E402, I001

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.figures import FigureVariant, apply_figure_style, save_figure


DATA_ROOT = Path(
    os.environ.get("OPX1000_DATA_DIR", ROOT.parent / "data_opx1000")
).expanduser()
OUTPUT_STEM = "07_high_amplitude_cutoff_comparison"
RABI_MIN_MHZ = 50.0
RABI_MAX_MHZ = 80.0
BLOCK_SIZE = 3


@dataclass(frozen=True)
class ScanSelection:
    run_id: str
    expected_cutoff: float

    @property
    def directory(self) -> Path:
        return (
            DATA_ROOT
            / "calibrations"
            / "2026-06-29"
            / "echo_lorentzian"
            / self.run_id
        )


SELECTIONS = (
    ScanSelection("18-23-52-853018", 0.001),
    ScanSelection("18-22-48-501753", 0.005),
    ScanSelection("18-21-36-642521", 0.010),
)


def load_q1_rabi_calibration(directory: Path) -> tuple[float, float]:
    """Return the q1 pi-pulse amplitude and Rabi rate saved with a scan."""
    profile = directory / "profile"
    qubits = json.loads((profile / "qubits.json").read_text())["qubits"]
    pulses = json.loads((profile / "pulses.json").read_text())["pulses"]
    x180_name = qubits["q1"]["operations"]["x180"]
    pi_pulse = pulses["q1"][x180_name]
    pi_amplitude = float(pi_pulse["amplitude"])
    pi_rabi_mhz = 1000.0 / (2.0 * float(pi_pulse["length_ns"]))
    return pi_amplitude, pi_rabi_mhz


def load_scan(
    selection: ScanSelection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    directory = selection.directory
    required = (
        directory / "parameters.json",
        directory / "metadata.json",
        directory / "sweep.npz",
        directory / "results.npz",
        directory / "profile" / "qubits.json",
        directory / "profile" / "pulses.json",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")

    parameters = json.loads((directory / "parameters.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    if not np.isclose(float(parameters["cutoff"]), selection.expected_cutoff):
        raise ValueError(
            f"{selection.run_id}: expected cutoff {selection.expected_cutoff}, "
            f"found {parameters['cutoff']}"
        )
    if parameters["echo"] is not True:
        raise ValueError(f"{selection.run_id}: scan is not an echo sequence")
    if int(parameters["lorentzian_length_in_ns"]) != 10_000:
        raise ValueError(f"{selection.run_id}: pulse duration is not 10 us")
    if int(parameters["num_shots"]) != 40:
        raise ValueError(f"{selection.run_id}: expected 40 repetitions per point")

    with np.load(directory / "sweep.npz", allow_pickle=False) as sweep:
        qubits = np.asarray(sweep["qubit"])
        detuning_mhz = np.asarray(sweep["detuning"], dtype=float) / 1e6
        amplitude = np.asarray(sweep["amp_prefactor"], dtype=float)
    with np.load(directory / "results.npz", allow_pickle=False) as results:
        state = np.asarray(results["state"], dtype=float)

    if qubits.tolist() != ["q1"]:
        raise ValueError(f"{selection.run_id}: expected only q1, found {qubits}")
    expected_shape = (1, detuning_mhz.size, amplitude.size)
    if state.shape != expected_shape:
        raise ValueError(
            f"{selection.run_id}: expected state shape {expected_shape}, "
            f"found {state.shape}"
        )
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{selection.run_id}: state map contains nonfinite values")

    pi_amplitude, pi_rabi_mhz = load_q1_rabi_calibration(directory)
    peak_amplitude = float(parameters["lorentzian_peak_amplitude"])
    rabi_mhz = amplitude * peak_amplitude / pi_amplitude * pi_rabi_mhz
    high_amplitude = (rabi_mhz >= RABI_MIN_MHZ) & (rabi_mhz <= RABI_MAX_MHZ)
    if np.count_nonzero(high_amplitude) < 2:
        raise ValueError(f"{selection.run_id}: no resolved 50-80 MHz band")

    provenance = {
        "run_id": selection.run_id,
        "cutoff": float(parameters["cutoff"]),
        "pulse_length_us": float(parameters["lorentzian_length_in_ns"]) / 1000.0,
        "num_shots": int(parameters["num_shots"]),
        "timestamp": metadata["timestamp"],
        "pi_amplitude": pi_amplitude,
        "pi_rabi_mhz": pi_rabi_mhz,
        "peak_amplitude": peak_amplitude,
        "raw_rabi_min_mhz": float(rabi_mhz[high_amplitude].min()),
        "raw_rabi_max_mhz": float(rabi_mhz[high_amplitude].max()),
    }
    return (
        detuning_mhz,
        rabi_mhz[high_amplitude],
        state[0][:, high_amplitude].T,
        provenance,
    )


def block_average_map(
    detuning_mhz: np.ndarray,
    rabi_mhz: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average non-overlapping BLOCK_SIZE x BLOCK_SIZE measured cells."""
    rows = state.shape[0] // BLOCK_SIZE * BLOCK_SIZE
    columns = state.shape[1] // BLOCK_SIZE * BLOCK_SIZE
    if rows == 0 or columns == 0:
        raise ValueError("Map is too small for the requested block average")
    state = state[:rows, :columns]
    averaged = state.reshape(
        rows // BLOCK_SIZE,
        BLOCK_SIZE,
        columns // BLOCK_SIZE,
        BLOCK_SIZE,
    ).mean(axis=(1, 3))
    rabi_binned = rabi_mhz[:rows].reshape(-1, BLOCK_SIZE).mean(axis=1)
    detuning_binned = (
        detuning_mhz[:columns].reshape(-1, BLOCK_SIZE).mean(axis=1)
    )
    return detuning_binned, rabi_binned, averaged


def main() -> None:
    apply_figure_style(FigureVariant.PAPER)
    raw_scans = [load_scan(selection) for selection in SELECTIONS]
    scans = [
        (*block_average_map(detuning, rabi, state), provenance)
        for detuning, rabi, state, provenance in raw_scans
    ]

    calibrations = {
        (provenance["pi_amplitude"], provenance["pi_rabi_mhz"])
        for _, _, _, provenance in scans
    }
    if len(calibrations) != 1:
        raise ValueError(
            f"Selected scans have different q1 calibrations: {calibrations}"
        )
    pi_amplitude, pi_rabi_mhz = calibrations.pop()

    reference_detuning, reference_rabi, _, _ = scans[0]
    for detuning_mhz, rabi_mhz, _, provenance in scans[1:]:
        if not np.array_equal(detuning_mhz, reference_detuning):
            raise ValueError(
                f"{provenance['run_id']}: detuning grid differs from first scan"
            )
        if not np.array_equal(rabi_mhz, reference_rabi):
            raise ValueError(
                f"{provenance['run_id']}: amplitude grid differs from first scan"
            )

    combined = np.concatenate([state.ravel() for _, _, state, _ in scans])
    color_min, color_max = np.quantile(combined, [0.01, 0.99])

    fig, axes = plt.subplots(
        1,
        len(scans),
        figsize=(7.0, 2.35),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    mesh = None
    for panel, (ax, (detuning_mhz, rabi_mhz, state, provenance)) in enumerate(
        zip(axes, scans, strict=True)
    ):
        mesh = ax.pcolormesh(
            detuning_mhz,
            rabi_mhz,
            state,
            shading="auto",
            cmap="viridis",
            vmin=color_min,
            vmax=color_max,
            rasterized=True,
        )
        ax.axvline(0.0, color="white", lw=0.55, ls="--", alpha=0.8)
        ax.set_title(rf"$c={provenance['cutoff']:.3f}$")
        ax.set_xlabel(r"Drive detuning $(f_d-f_{01})$ (MHz)")
        ax.text(
            0.03,
            0.97,
            f"({chr(ord('a') + panel)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="white",
            fontweight="bold",
        )
    axes[0].set_ylabel(r"$\Omega_0/2\pi$ (MHz)")
    if mesh is None:
        raise RuntimeError("No scans were plotted")
    colorbar = fig.colorbar(mesh, ax=axes, pad=0.02, aspect=24)
    colorbar.set_label(r"$P_e$")

    saved = save_figure(
        fig,
        OUTPUT_STEM,
        variant=FigureVariant.PAPER,
        formats=("pdf", "png", "svg"),
        dpi=300,
    )
    plt.close(fig)

    print(f"q1 pi-pulse calibration amplitude: {pi_amplitude:.12g}")
    print(f"q1 pi-pulse Rabi rate: {pi_rabi_mhz:.6g} MHz")
    print(f"Non-overlapping block average: {BLOCK_SIZE} x {BLOCK_SIZE}")
    print(
        "Displayed Rabi range: "
        f"{reference_rabi.min():.3f}-{reference_rabi.max():.3f} MHz"
    )
    for _, _, state, provenance in scans:
        print(
            f"{provenance['run_id']}: cutoff={provenance['cutoff']:.3g}, "
            f"raw Rabi range={provenance['raw_rabi_min_mhz']:.3f}-"
            f"{provenance['raw_rabi_max_mhz']:.3f} MHz, "
            f"displayed state range={state.min():.3f}-{state.max():.3f}"
        )
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
