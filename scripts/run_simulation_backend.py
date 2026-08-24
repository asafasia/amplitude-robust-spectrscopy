"""Run the same finite-level configuration with a selected solver backend."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--solver",
        choices=("qutip", "torch-cpu", "mps"),
        default="torch-cpu",
    )
    result.add_argument("--levels", type=int, default=4)
    result.add_argument("--duration-us", type=float, default=0.1)
    result.add_argument("--steps-per-half", type=int, default=400)
    result.add_argument("--output", type=Path)
    return result


def main() -> None:
    from echospec.simulation.backends import run_with_solver
    from echospec.simulation.config import SimulationConfig

    args = parser().parse_args()
    config = SimulationConfig(
        levels=args.levels,
        amplitude_mhz=(0.0, 12.5, 25.0),
        detuning_mhz=(-0.5, 0.0, 0.5),
        duration_us=args.duration_us,
        num_steps_per_half=args.steps_per_half,
        cutoff=0.005,
        echo=True,
    )
    start = time.perf_counter()
    simulation = run_with_solver(config, solver=args.solver)
    elapsed = time.perf_counter() - start
    summary = {
        "solver": args.solver,
        "device": simulation.tensor_device,
        "elapsed_seconds": elapsed,
        "population_shape": list(simulation.populations.shape),
        "maximum_trace_error": simulation.trace_error,
        "maximum_hermiticity_error": simulation.hermiticity_error,
        "minimum_density_matrix_eigenvalue": simulation.minimum_eigenvalue,
        "maximum_leakage": float(simulation.leakage.max()),
    }
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output,
            solver=args.solver,
            amplitude_mhz=config.amplitude_mhz,
            detuning_mhz=config.detuning_mhz,
            density_matrices=simulation.density_matrices,
            populations=simulation.populations,
            leakage=simulation.leakage,
        )
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
