"""Build main-text Fig. 2 from the 2026-08-10 q1 measurements."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from echospec.figures import FigureVariant, apply_figure_style, save_figure


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/experimental/2026-08-10/echo_lorentzian"
OUTPUT_STEM = "02_central_spectroscopy"
VMIN = 0.0
VMAX = 0.6

RUNS = {
    ("broad", "constant_echo"): "14-46-06-038411",
    ("broad", "root"): "14-16-19-887278",
    ("broad", "echo"): "14-22-03-566753",
    ("narrow", "constant_echo"): "14-52-31-810398",
    ("narrow", "root"): "14-09-56-777281",
    ("narrow", "echo"): "14-02-28-518579",
}


@dataclass(frozen=True)
class Measurement:
    run_id: str
    detuning_mhz: np.ndarray
    rabi_mhz: np.ndarray
    excited: np.ndarray
    requested_shots: int
    effective_shots: int
    duration_us: float
    cutoff: float
    echo: bool
    peak_amplitude_v: float
    qubit: str
    f01_ghz: float


def load_measurement(run_id: str) -> Measurement:
    run_dir = DATA_ROOT / run_id
    parameters = json.loads((run_dir / "parameters.json").read_text())
    metadata = json.loads((run_dir / "metadata.json").read_text())
    qubits = json.loads((run_dir / "profile/qubits.json").read_text())
    pulses = json.loads((run_dir / "profile/pulses.json").read_text())

    with np.load(run_dir / "sweep.npz", allow_pickle=False) as sweep:
        qubit_values = np.asarray(sweep["qubit"])
        detuning_hz = np.asarray(sweep["detuning"], dtype=float)
        amp_prefactor = np.asarray(sweep["amp_prefactor"], dtype=float)
    with np.load(run_dir / "results.npz", allow_pickle=False) as results:
        state = np.asarray(results["state"], dtype=float)

    if qubit_values.size != 1:
        raise ValueError(f"{run_id}: expected one qubit, got {qubit_values}")
    qubit = str(qubit_values[0])
    if state.shape != (1, detuning_hz.size, amp_prefactor.size):
        raise ValueError(f"{run_id}: unexpected state shape {state.shape}")
    if not parameters["use_state_discrimination"]:
        raise ValueError(f"{run_id}: state-discriminated data are required")
    if parameters["pulse_shape"] != "root_lorentzian":
        raise ValueError(f"{run_id}: expected a root-Lorentzian pulse")
    if metadata["results"]["state"]["shape"] != list(state.shape):
        raise ValueError(f"{run_id}: metadata and result shapes disagree")

    peak_amplitude_v = float(parameters["lorentzian_peak_amplitude"])
    pi_pulse = pulses["pulses"][qubit]["x180_const"]
    pi_amplitude_v = float(pi_pulse["amplitude"])
    pi_length_ns = float(pi_pulse["length_ns"])
    pi_rabi_hz = 1.0 / (2.0 * pi_length_ns * 1e-9)
    rabi_mhz = (
        amp_prefactor * peak_amplitude_v / pi_amplitude_v * pi_rabi_hz / 1e6
    )

    qubit_profile = qubits["qubits"][qubit]
    denominators = {
        Fraction(float(value)).limit_denominator(10_000).denominator
        for value in np.unique(state)
    }
    effective_shots = 1
    for denominator in denominators:
        effective_shots = lcm(effective_shots, denominator)
    if not np.allclose(state * effective_shots, np.rint(state * effective_shots)):
        raise ValueError(f"{run_id}: cannot infer the completed shot count")

    return Measurement(
        run_id=run_id,
        detuning_mhz=detuning_hz / 1e6,
        rabi_mhz=rabi_mhz,
        excited=state[0].T,
        requested_shots=int(parameters["num_shots"]),
        effective_shots=effective_shots,
        duration_us=float(parameters["lorentzian_length_in_ns"]) / 1000.0,
        cutoff=float(parameters["cutoff"]),
        echo=bool(parameters["echo"]),
        peak_amplitude_v=peak_amplitude_v,
        qubit=qubit,
        f01_ghz=float(qubit_profile["frequencies_hz"]["qubit_f01"]) / 1e9,
    )


def validate(measurements: dict[tuple[str, str], Measurement]) -> None:
    reference = next(iter(measurements.values()))
    for key, measurement in measurements.items():
        if measurement.qubit != reference.qubit:
            raise ValueError(f"{key}: qubit mismatch")
        if not np.isclose(measurement.duration_us, reference.duration_us):
            raise ValueError(f"{key}: duration mismatch")
        if not np.array_equal(measurement.rabi_mhz, reference.rabi_mhz):
            raise ValueError(f"{key}: calibrated Rabi grid mismatch")
        if measurement.excited.shape != (
            measurement.rabi_mhz.size,
            measurement.detuning_mhz.size,
        ):
            raise ValueError(f"{key}: transposed population shape mismatch")
        if not np.isfinite(measurement.excited).all():
            raise ValueError(f"{key}: nonfinite measured populations")

    expected_grids = {
        "broad": (-50.0, 50.0, 0.5),
        "narrow": (-0.5, 0.5, 0.005),
    }
    for domain, (start, stop, step) in expected_grids.items():
        for protocol in ("constant_echo", "root", "echo"):
            measurement = measurements[(domain, protocol)]
            if not np.isclose(measurement.detuning_mhz[0], start):
                raise ValueError(f"{domain}/{protocol}: wrong detuning start")
            if not np.isclose(measurement.detuning_mhz[-1], stop):
                raise ValueError(f"{domain}/{protocol}: wrong detuning stop")
            if not np.allclose(np.diff(measurement.detuning_mhz), step):
                raise ValueError(f"{domain}/{protocol}: wrong detuning step")

    expected_parameters = {
        ("broad", "constant_echo"): (0.99, True, 200),
        ("broad", "root"): (0.005, False, 200),
        ("broad", "echo"): (0.005, True, 200),
        ("narrow", "constant_echo"): (0.99, True, 200),
        ("narrow", "root"): (0.005, False, 1000),
        ("narrow", "echo"): (0.005, True, 1000),
    }
    for key, (cutoff, echo, requested_shots) in expected_parameters.items():
        measurement = measurements[key]
        if not np.isclose(measurement.cutoff, cutoff):
            raise ValueError(f"{key}: wrong cutoff")
        if measurement.echo is not echo:
            raise ValueError(f"{key}: wrong echo flag")
        if measurement.requested_shots != requested_shots:
            raise ValueError(f"{key}: wrong requested shot count")

    expected_effective_shots = {
        ("broad", "constant_echo"): 200,
        ("broad", "root"): 200,
        ("broad", "echo"): 200,
        ("narrow", "constant_echo"): 200,
        ("narrow", "root"): 216,
        ("narrow", "echo"): 1000,
    }
    for key, expected_shots in expected_effective_shots.items():
        if measurements[key].effective_shots != expected_shots:
            raise ValueError(
                f"{key}: expected {expected_shots} completed averages, got "
                f"{measurements[key].effective_shots}"
            )


def main() -> None:
    measurements = {key: load_measurement(run_id) for key, run_id in RUNS.items()}
    validate(measurements)
    reference = measurements[("broad", "constant_echo")]

    apply_figure_style(FigureVariant.PAPER)
    plt.rcParams.update(
        {
            "figure.figsize": (7.1, 4.55),
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(2, 3, sharey=True, constrained_layout=True)
    protocols = (
        ("constant_echo", "Constant (echo)"),
        ("root", "Root-Lorentzian"),
        ("echo", "Echo-root-Lorentzian"),
    )
    domains = (("broad", "Broad domain"), ("narrow", "Narrow domain"))
    panel_labels = iter("abcdef")
    image = None

    for row, (domain, domain_label) in enumerate(domains):
        for column, (protocol, protocol_label) in enumerate(protocols):
            measurement = measurements[(domain, protocol)]
            ax = axes[row, column]
            image = ax.pcolormesh(
                measurement.detuning_mhz,
                measurement.rabi_mhz,
                measurement.excited,
                cmap="magma",
                vmin=VMIN,
                vmax=VMAX,
                shading="auto",
                rasterized=True,
            )
            ax.axvline(0.0, color="white", lw=0.45, ls="--", alpha=0.65)
            ax.set_xlim(measurement.detuning_mhz[0], measurement.detuning_mhz[-1])
            ax.set_ylim(reference.rabi_mhz[0], reference.rabi_mhz[-1])
            ax.set_box_aspect(1.0)
            panel_text = ax.text(
                0.035,
                0.95,
                f"({next(panel_labels)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="white",
                fontweight="bold",
            )
            panel_text.set_path_effects(
                [path_effects.withStroke(linewidth=1.25, foreground="black")]
            )
            if row == 0:
                ax.set_title(protocol_label)
            if row == 1:
                ax.set_xlabel(r"Drive detuning $(f_d-f_{01})$ (MHz)")
            if column == 0:
                ax.set_ylabel(r"Peak Rabi frequency $\Omega_0/2\pi$ (MHz)")
                ax.text(
                    -0.26,
                    0.5,
                    domain_label,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=6.2,
                )

    if image is None:
        raise RuntimeError("No panels were generated")
    colorbar = fig.colorbar(image, ax=axes, pad=0.018, fraction=0.055)
    colorbar.set_label(r"Measured $P_e$")
    colorbar.ax.tick_params(labelsize=6.0)

    saved = save_figure(
        fig,
        OUTPUT_STEM,
        variant=FigureVariant.PAPER,
        formats=("pdf", "png", "svg"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    for path in saved:
        print(path)
    print(
        f"q1 f01={reference.f01_ghz:.9f} GHz; "
        f"Rabi grid={reference.rabi_mhz[0]:.6g}--{reference.rabi_mhz[-1]:.6g} MHz; "
        f"L={reference.duration_us:g} us; "
        "cutoffs=0.99 (constant echo), 0.005 (root/echo)"
    )


if __name__ == "__main__":
    main()
