"""Compare the latest narrow q1 scans with a matched two-level simulation."""

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

from echospec.figures import FigureVariant, apply_figure_style, save_figure


DATA_ROOT = ROOT / "data/experimental/2026-08-10/echo_lorentzian"
RUNS = {
    "root": "14-09-56-777281",
    "echo": "14-02-28-518579",
}
ORDER = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1-us", type=float, default=30.0)
    parser.add_argument("--t2-us", type=float, default=6.0)
    parser.add_argument("--duration-us", type=float, default=20.0)
    parser.add_argument("--cutoff", type=float, default=0.005)
    parser.add_argument(
        "--detuning-span-mhz",
        type=float,
        default=1.0,
        help="Total detuning span centered at zero.",
    )
    parser.add_argument("--max-rabi-mhz", type=float, default=25.0)
    parser.add_argument("--rabi-points", type=int, default=126)
    parser.add_argument("--steps-per-half", type=int, default=1600)
    return parser.parse_args()


def load_measurement(run_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    run_dir = DATA_ROOT / run_id
    parameters = json.loads((run_dir / "parameters.json").read_text())
    pulses = json.loads((run_dir / "profile/pulses.json").read_text())
    with np.load(run_dir / "sweep.npz", allow_pickle=False) as sweep:
        detuning_mhz = np.asarray(sweep["detuning"], dtype=float) / 1e6
        amplitude = np.asarray(sweep["amp_prefactor"], dtype=float)
        qubit = str(np.asarray(sweep["qubit"])[0])
    with np.load(run_dir / "results.npz", allow_pickle=False) as results:
        excited = np.asarray(results["state"], dtype=float)[0].T

    pi_pulse = pulses["pulses"][qubit]["x180_const"]
    pi_rabi_mhz = 1000.0 / (2.0 * float(pi_pulse["length_ns"]))
    rabi_mhz = (
        amplitude
        * float(parameters["lorentzian_peak_amplitude"])
        / float(pi_pulse["amplitude"])
        * pi_rabi_mhz
    )
    return detuning_mhz, rabi_mhz, excited, parameters


def rhs(
    bloch: np.ndarray,
    detuning: np.ndarray,
    drive: np.ndarray,
    inv_t1: float,
    inv_t2: float,
) -> np.ndarray:
    x_value, y_value, z_value = bloch
    return np.stack(
        (
            -detuning * y_value - inv_t2 * x_value,
            detuning * x_value - drive * z_value - inv_t2 * y_value,
            drive * y_value + inv_t1 * (1.0 - z_value),
        )
    )


def integrate_half(
    bloch: np.ndarray,
    *,
    u_start: float,
    u_stop: float,
    sigma_us: float,
    detuning: np.ndarray,
    rabi: np.ndarray,
    drive_sign: float,
    t1_us: float,
    t2_us: float,
    steps: int,
) -> np.ndarray:
    du = (u_stop - u_start) / steps
    inv_t1 = 1.0 / t1_us
    inv_t2 = 1.0 / t2_us

    def derivative(state: np.ndarray, u_value: float) -> np.ndarray:
        envelope = 1.0 / np.cosh(u_value)
        dt_du = sigma_us * np.cosh(u_value)
        return dt_du * rhs(
            state,
            detuning,
            drive_sign * rabi * envelope,
            inv_t1,
            inv_t2,
        )

    u_value = u_start
    for _ in range(steps):
        k1 = derivative(bloch, u_value)
        k2 = derivative(bloch + 0.5 * du * k1, u_value + 0.5 * du)
        k3 = derivative(bloch + 0.5 * du * k2, u_value + 0.5 * du)
        k4 = derivative(bloch + du * k3, u_value + du)
        bloch += (du / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        u_value += du
    return bloch


def simulate(
    detuning_mhz: np.ndarray,
    rabi_mhz: np.ndarray,
    *,
    echo: bool,
    duration_us: float,
    cutoff: float,
    t1_us: float,
    t2_us: float,
    steps_per_half: int,
) -> np.ndarray:
    sigma_us = (duration_us / 2.0) / np.sqrt(cutoff ** (-1.0 / ORDER) - 1.0)
    u_edge = float(np.arcsinh((duration_us / 2.0) / sigma_us))
    detuning, rabi = np.meshgrid(
        2.0 * np.pi * detuning_mhz,
        2.0 * np.pi * rabi_mhz,
    )
    bloch = np.zeros((3, *detuning.shape), dtype=float)
    bloch[2] = 1.0
    common = {
        "sigma_us": sigma_us,
        "detuning": detuning,
        "rabi": rabi,
        "t1_us": t1_us,
        "t2_us": t2_us,
        "steps": steps_per_half,
    }
    bloch = integrate_half(
        bloch,
        u_start=-u_edge,
        u_stop=0.0,
        drive_sign=1.0,
        **common,
    )
    bloch = integrate_half(
        bloch,
        u_start=0.0,
        u_stop=u_edge,
        drive_sign=-1.0 if echo else 1.0,
        **common,
    )
    population = (1.0 - bloch[2]) / 2.0
    if not np.all(np.isfinite(population)):
        raise RuntimeError("Simulation produced nonfinite populations")
    if population.min() < -1e-6 or population.max() > 1.0 + 1e-6:
        raise RuntimeError(
            "Simulation left the physical probability interval: "
            f"{population.min():.6g} to {population.max():.6g}"
        )
    return np.clip(population, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    if not (0.0 < args.cutoff < 1.0):
        raise ValueError("cutoff must lie between zero and one")
    if min(args.t1_us, args.t2_us, args.duration_us, args.max_rabi_mhz) <= 0.0:
        raise ValueError("times and maximum Rabi frequency must be positive")

    measurements = {name: load_measurement(run_id) for name, run_id in RUNS.items()}
    detuning_mhz = np.linspace(
        -args.detuning_span_mhz / 2.0,
        args.detuning_span_mhz / 2.0,
        201,
    )
    rabi_mhz = np.linspace(0.0, args.max_rabi_mhz, args.rabi_points)
    simulations = {
        name: simulate(
            detuning_mhz,
            rabi_mhz,
            echo=name == "echo",
            duration_us=args.duration_us,
            cutoff=args.cutoff,
            t1_us=args.t1_us,
            t2_us=args.t2_us,
            steps_per_half=args.steps_per_half,
        )
        for name in ("root", "echo")
    }

    apply_figure_style(FigureVariant.PAPER)
    plt.rcParams.update({"figure.figsize": (7.0, 5.2), "svg.fonttype": "none"})
    figure, axes = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True)
    image = None
    for column, name in enumerate(("root", "echo")):
        measured_detuning, measured_rabi, measured_excited, parameters = measurements[name]
        if not np.isclose(float(parameters["lorentzian_length_in_ns"]) / 1000.0, args.duration_us):
            raise ValueError(f"{name} measurement duration does not match requested duration")
        if not np.isclose(float(parameters["cutoff"]), args.cutoff):
            raise ValueError(f"{name} measurement cutoff does not match requested cutoff")
        keep = measured_rabi <= args.max_rabi_mhz + 1e-12
        image = axes[0, column].pcolormesh(
            measured_detuning,
            measured_rabi[keep],
            measured_excited[keep],
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=0.6,
            rasterized=True,
        )
        axes[1, column].pcolormesh(
            detuning_mhz,
            rabi_mhz,
            simulations[name],
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=0.6,
            rasterized=True,
        )
        axes[0, column].set_title("Root-Lorentzian" if name == "root" else "Echo root-Lorentzian")
        axes[1, column].set_xlabel(r"$\Delta/2\pi$ (MHz)")
        for row in range(2):
            axes[row, column].axvline(0.0, color="white", lw=0.5, ls="--", alpha=0.8)

    axes[0, 0].set_ylabel("Measured\n" + r"$\Omega_0/2\pi$ (MHz)")
    axes[1, 0].set_ylabel("Simulated\n" + r"$\Omega_0/2\pi$ (MHz)")
    figure.suptitle(
        rf"$L={args.duration_us:g}\,\mu s$, $c={args.cutoff:g}$, "
        rf"$T_1={args.t1_us:g}\,\mu s$, $T_2={args.t2_us:g}\,\mu s$",
        fontsize=9,
    )
    if image is None:
        raise RuntimeError("No maps were plotted")
    colorbar = figure.colorbar(image, ax=axes, pad=0.02, fraction=0.035)
    colorbar.set_label(r"$P_e$")

    stem = "20_latest_data_pulse_parameter_simulation"
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
        root_population=simulations["root"],
        echo_population=simulations["echo"],
        t1_us=args.t1_us,
        t2_us=args.t2_us,
        duration_us=args.duration_us,
        cutoff=args.cutoff,
        order=ORDER,
        steps_per_half=args.steps_per_half,
        root_run_id=RUNS["root"],
        echo_run_id=RUNS["echo"],
    )
    for path in (*saved, data_path):
        print(path)


if __name__ == "__main__":
    main()
