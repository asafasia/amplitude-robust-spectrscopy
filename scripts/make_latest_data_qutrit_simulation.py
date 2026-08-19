"""Compare the latest narrow q1 scans with a matched three-level simulation."""

from __future__ import annotations

# Backend and local-source setup must precede pyplot and echospec imports.
# ruff: noqa: E402, I001

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ars-matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from echospec.figures import FigureVariant, apply_figure_style, save_figure
from echospec.simulation.qutrit import simulate_qutrit_map
from make_latest_data_pulse_parameter_simulation import load_measurement


DATA_ROOT = ROOT / "data/experimental/2026-08-10/echo_lorentzian"
RUNS = {
    "root": "14-09-56-777281",
    "echo": "14-02-28-518579",
}
REFERENCE_RUN = "14-52-31-810398"
ORDER = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1-us", type=float, default=30.0)
    parser.add_argument("--t2-us", type=float, default=6.0)
    parser.add_argument("--duration-us", type=float, default=20.0)
    parser.add_argument("--cutoff", type=float, default=0.005)
    parser.add_argument("--detuning-span-mhz", type=float, default=1.0)
    parser.add_argument("--detuning-points", type=int, default=101)
    parser.add_argument("--max-rabi-mhz", type=float, default=25.0)
    parser.add_argument("--rabi-points", type=int, default=51)
    parser.add_argument("--steps-per-half", type=int, default=8000)
    parser.add_argument(
        "--anharmonicity-mhz",
        type=float,
        default=None,
        help="Defaults to the negative magnitude saved for q1.",
    )
    return parser.parse_args()


def saved_q1_anharmonicity_mhz() -> float:
    profile_path = DATA_ROOT / REFERENCE_RUN / "profile/qubits.json"
    profile = json.loads(profile_path.read_text())
    magnitude_hz = float(
        profile["qubits"]["q1"]["transmon"]["anharmonicity_hz"]
    )
    return -abs(magnitude_hz) / 1e6


def t_phi_from_t1_t2(t1_us: float, t2_us: float) -> float:
    inv_t_phi = 1.0 / t2_us - 1.0 / (2.0 * t1_us)
    if inv_t_phi <= 0.0:
        raise ValueError("T2 must be less than 2*T1 for positive pure dephasing")
    return 1.0 / inv_t_phi


def main() -> None:
    args = parse_args()
    if not (0.0 < args.cutoff < 1.0):
        raise ValueError("cutoff must lie between zero and one")
    if min(args.t1_us, args.t2_us, args.duration_us, args.max_rabi_mhz) <= 0.0:
        raise ValueError("times and maximum Rabi frequency must be positive")

    anharmonicity_mhz = (
        saved_q1_anharmonicity_mhz()
        if args.anharmonicity_mhz is None
        else args.anharmonicity_mhz
    )
    t_phi_us = t_phi_from_t1_t2(args.t1_us, args.t2_us)
    detuning_mhz = np.linspace(
        -args.detuning_span_mhz / 2.0,
        args.detuning_span_mhz / 2.0,
        args.detuning_points,
    )
    rabi_mhz = np.linspace(0.0, args.max_rabi_mhz, args.rabi_points)
    measurements = {name: load_measurement(run_id) for name, run_id in RUNS.items()}

    simulations = {}
    for name in ("root", "echo"):
        print(f"Simulating {name} qutrit map...", flush=True)
        simulations[name] = simulate_qutrit_map(
            duration_us=args.duration_us,
            detuning_mhz=detuning_mhz,
            rabi_mhz=rabi_mhz,
            t1_us=args.t1_us,
            t_phi_us=t_phi_us,
            anharmonicity_mhz=anharmonicity_mhz,
            num_steps_per_half=args.steps_per_half,
            cutoff=args.cutoff,
            echo=name == "echo",
            order=ORDER,
        )

    measured_crops = {}
    excitation_arrays = []
    leakage_arrays = []
    for name in ("root", "echo"):
        measured_detuning, measured_rabi, measured_excited, parameters = measurements[name]
        if not np.isclose(float(parameters["lorentzian_length_in_ns"]) / 1000.0, args.duration_us):
            raise ValueError(f"{name} measurement duration does not match")
        if not np.isclose(float(parameters["cutoff"]), args.cutoff):
            raise ValueError(f"{name} measurement cutoff does not match")
        keep = measured_rabi <= args.max_rabi_mhz + 1e-12
        measured_crops[name] = (
            measured_detuning,
            measured_rabi[keep],
            measured_excited[keep],
        )
        excitation_arrays.extend(
            (
                measured_excited[keep],
                simulations[name].excited + simulations[name].second_excited,
            )
        )
        leakage_arrays.append(simulations[name].second_excited)

    excitation_vmin = min(float(values.min()) for values in excitation_arrays)
    excitation_vmax = max(float(values.max()) for values in excitation_arrays)
    leakage_vmin = min(float(values.min()) for values in leakage_arrays)
    leakage_vmax = max(float(values.max()) for values in leakage_arrays)

    apply_figure_style(FigureVariant.PAPER)
    plt.rcParams.update({"figure.figsize": (7.0, 7.0), "svg.fonttype": "none"})
    figure, axes = plt.subplots(3, 2, sharex=True, sharey=True, constrained_layout=True)
    excitation_image = None
    leakage_image = None
    for column, name in enumerate(("root", "echo")):
        measured_detuning, measured_rabi, measured_excited = measured_crops[name]
        excitation_image = axes[0, column].pcolormesh(
            measured_detuning,
            measured_rabi,
            measured_excited,
            shading="auto",
            cmap="viridis",
            vmin=excitation_vmin,
            vmax=excitation_vmax,
            rasterized=True,
        )
        result = simulations[name]
        total_excitation = result.excited + result.second_excited
        axes[1, column].pcolormesh(
            detuning_mhz,
            rabi_mhz,
            total_excitation,
            shading="auto",
            cmap="viridis",
            vmin=excitation_vmin,
            vmax=excitation_vmax,
            rasterized=True,
        )
        leakage_image = axes[2, column].pcolormesh(
            detuning_mhz,
            rabi_mhz,
            result.second_excited,
            shading="auto",
            cmap="magma",
            vmin=leakage_vmin,
            vmax=leakage_vmax,
            rasterized=True,
        )
        axes[0, column].set_title(
            "Root-Lorentzian" if name == "root" else "Echo root-Lorentzian"
        )
        axes[2, column].set_xlabel(r"$\Delta/2\pi$ (MHz)")
        for row in range(3):
            axes[row, column].axvline(
                0.0, color="white", lw=0.5, ls="--", alpha=0.8
            )

    axes[0, 0].set_ylabel("Measured\n" + r"$\Omega_0/2\pi$ (MHz)")
    axes[1, 0].set_ylabel(r"Qutrit $P_1+P_2$" + "\n" + r"$\Omega_0/2\pi$ (MHz)")
    axes[2, 0].set_ylabel(r"Leakage $P_2$" + "\n" + r"$\Omega_0/2\pi$ (MHz)")
    figure.suptitle(
        rf"$L={args.duration_us:g}\,\mu s$, $c={args.cutoff:g}$, "
        rf"$T_1={args.t1_us:g}\,\mu s$, $T_2={args.t2_us:g}\,\mu s$, "
        rf"$\alpha/2\pi={anharmonicity_mhz:.3f}\,$MHz",
        fontsize=8.5,
    )
    if excitation_image is None or leakage_image is None:
        raise RuntimeError("No maps were plotted")
    excitation_colorbar = figure.colorbar(
        excitation_image, ax=axes[:2], pad=0.02, fraction=0.035
    )
    excitation_colorbar.set_label(r"Excitation probability")
    leakage_colorbar = figure.colorbar(
        leakage_image, ax=axes[2], pad=0.02, fraction=0.035
    )
    leakage_colorbar.set_label(r"$P_2$")

    stem = "21_latest_data_qutrit_simulation"
    saved = save_figure(
        figure,
        stem,
        variant=FigureVariant.PAPER,
        formats=("png", "pdf", "svg"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)
    data_path = ROOT / "figures" / "paper" / f"{stem}.npz"
    np.savez_compressed(
        data_path,
        detuning_convention="drive_minus_qubit",
        detuning_mhz=detuning_mhz,
        rabi_mhz=rabi_mhz,
        root_p0=simulations["root"].ground,
        root_p1=simulations["root"].excited,
        root_p2=simulations["root"].second_excited,
        echo_p0=simulations["echo"].ground,
        echo_p1=simulations["echo"].excited,
        echo_p2=simulations["echo"].second_excited,
        t1_us=args.t1_us,
        t2_us=args.t2_us,
        t_phi_us=t_phi_us,
        duration_us=args.duration_us,
        cutoff=args.cutoff,
        order=ORDER,
        anharmonicity_mhz=anharmonicity_mhz,
        steps_per_half=args.steps_per_half,
        root_run_id=RUNS["root"],
        echo_run_id=RUNS["echo"],
    )
    for path in (*saved, data_path):
        print(path)


if __name__ == "__main__":
    main()
