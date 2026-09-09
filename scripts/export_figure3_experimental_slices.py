"""Refresh Figure 3 experimental slices from the August 2026 sweep set."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    ROOT
    / "data/experimental/2026-08-25/six_detuning_amplitude_sweeps"
)
TRACE_PATH = ROOT / "figures/paper/03_lorentzian_echo_slices_data.csv"
TARGET_RABI_MHZ = (2.5, 10.0, 25.0)
EXPERIMENT_RUNS = (
    (
        "current_noecho_experiment",
        "05_narrow_1mhz_cutoff_0p005_no_echo",
        False,
    ),
    (
        "current_echo_experiment",
        "06_narrow_1mhz_cutoff_0p005_echo",
        True,
    ),
)
FIELDNAMES = (
    "series",
    "target_rabi_mhz",
    "actual_rabi_mhz",
    "detuning_mhz",
    "excited_probability",
)


def load_experimental_rows(
    series: str, run_name: str, expected_echo: bool
) -> tuple[list[dict[str, str]], list[tuple[float, float]]]:
    run_dir = DATA_ROOT / run_name
    parameters = json.loads((run_dir / "parameters.json").read_text())
    qubits = json.loads((run_dir / "profile/qubits.json").read_text())
    pulses = json.loads((run_dir / "profile/pulses.json").read_text())

    with np.load(run_dir / "sweep.npz", allow_pickle=False) as sweep:
        qubit_values = np.asarray(sweep["qubit"])
        detuning_mhz = np.asarray(sweep["detuning"], dtype=float) / 1e6
        amp_prefactor = np.asarray(sweep["amp_prefactor"], dtype=float)
    with np.load(run_dir / "results.npz", allow_pickle=False) as results:
        state = np.asarray(results["state"], dtype=float)

    if qubit_values.size != 1:
        raise ValueError(f"{run_name}: expected one qubit")
    if state.shape != (1, detuning_mhz.size, amp_prefactor.size):
        raise ValueError(f"{run_name}: unexpected state shape {state.shape}")
    if not parameters["use_state_discrimination"]:
        raise ValueError(f"{run_name}: state discrimination is required")
    if not np.isclose(parameters["cutoff"], 0.005):
        raise ValueError(f"{run_name}: expected cutoff 0.005")
    if bool(parameters["echo"]) is not expected_echo:
        raise ValueError(f"{run_name}: wrong echo flag")
    if parameters["num_shots"] != 2000:
        raise ValueError(f"{run_name}: expected 2000 shots")
    if not np.allclose(
        (detuning_mhz[0], detuning_mhz[-1]), (-0.5, 0.5)
    ):
        raise ValueError(f"{run_name}: expected a -0.5 to 0.5 MHz sweep")

    qubit = str(qubit_values[0])
    qubit_profile = qubits["qubits"][qubit]
    x180_operation = qubit_profile["operations"]["x180"]
    pi_pulse = pulses["pulses"][qubit][x180_operation]
    peak_amplitude_v = float(parameters["lorentzian_peak_amplitude"])
    pi_amplitude_v = float(pi_pulse["amplitude"])
    pi_length_ns = float(pi_pulse["length_ns"])
    rabi_mhz = (
        amp_prefactor
        * peak_amplitude_v
        / pi_amplitude_v
        / (2.0 * pi_length_ns * 1e-9)
        / 1e6
    )

    rows: list[dict[str, str]] = []
    selected: list[tuple[float, float]] = []
    for target_rabi_mhz in TARGET_RABI_MHZ:
        amplitude_index = int(np.argmin(np.abs(rabi_mhz - target_rabi_mhz)))
        actual_rabi_mhz = float(rabi_mhz[amplitude_index])
        selected.append((target_rabi_mhz, actual_rabi_mhz))
        for detuning, probability in zip(
            detuning_mhz, state[0, :, amplitude_index], strict=True
        ):
            rows.append(
                {
                    "series": series,
                    "target_rabi_mhz": f"{target_rabi_mhz:.17g}",
                    "actual_rabi_mhz": f"{actual_rabi_mhz:.17g}",
                    "detuning_mhz": f"{float(detuning):.17g}",
                    "excited_probability": f"{float(probability):.17g}",
                }
            )
    return rows, selected


def load_simulation_rows() -> list[dict[str, str]]:
    with TRACE_PATH.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["series"].endswith("_simulation")
        ]
    if not rows:
        raise ValueError(f"{TRACE_PATH}: no simulation rows found")
    return rows


def main() -> None:
    rows: list[dict[str, str]] = []
    selections: dict[str, list[tuple[float, float]]] = {}
    for series, run_name, expected_echo in EXPERIMENT_RUNS:
        experimental_rows, selected = load_experimental_rows(
            series, run_name, expected_echo
        )
        rows.extend(experimental_rows)
        selections[series] = selected
    rows.extend(load_simulation_rows())

    with TRACE_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(TRACE_PATH)
    for series, selected in selections.items():
        values = ", ".join(
            f"{target:g}->{actual:.6g} MHz" for target, actual in selected
        )
        print(f"{series}: {values}")


if __name__ == "__main__":
    main()
