"""Validate and benchmark QuTiP, PyTorch CPU, and Apple MPS backends."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from echospec.simulation.config import SimulationConfig, representative_config
from echospec.simulation.qutip.runner import run_simulation as run_qutip_reference

from .runner import run_simulation
from .solver import mps_is_available


def _synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _timed(function, *, device: str) -> tuple[object, float]:
    _synchronize(device)
    start = time.perf_counter()
    result = function()
    _synchronize(device)
    return result, time.perf_counter() - start


def _differences(left, right) -> dict[str, float]:
    return {
        "maximum_population_difference": float(
            np.max(np.abs(left.populations - right.populations))
        ),
        "maximum_density_matrix_difference": float(
            np.max(np.abs(left.density_matrices - right.density_matrices))
        ),
    }


def benchmark(config: SimulationConfig, *, include_full_qutip: bool) -> dict:
    validation = representative_config(config)
    reference, qutip_validation_time = _timed(
        lambda: run_qutip_reference(validation), device="cpu"
    )
    torch_validation = run_simulation(validation, device="cpu")
    validation_comparison = _differences(torch_validation, reference)

    # Warm-up is deliberately outside the steady-state MPS timer.
    warmup_time = None
    if mps_is_available():
        warmup = replace(
            validation,
            amplitude_mhz=validation.amplitude_mhz[:1],
            detuning_mhz=validation.detuning_mhz[:1],
            num_steps_per_half=min(4, validation.num_steps_per_half),
        )
        _, warmup_time = _timed(
            lambda: run_simulation(warmup, device="mps"), device="mps"
        )

    torch_cpu, torch_cpu_time = _timed(
        lambda: run_simulation(config, device="cpu"), device="cpu"
    )
    mps_result = None
    mps_time = None
    if mps_is_available():
        mps_result, mps_time = _timed(
            lambda: run_simulation(config, device="mps"), device="mps"
        )

    qutip_result = reference
    qutip_time = qutip_validation_time
    qutip_scope = "representative validation grid"
    if include_full_qutip:
        qutip_result, qutip_time = _timed(
            lambda: run_qutip_reference(config), device="cpu"
        )
        qutip_scope = "full grid"

    validation_points = len(validation.amplitude_mhz) * len(validation.detuning_mhz)
    full_points = len(config.amplitude_mhz) * len(config.detuning_mhz)
    projected_qutip_time = None
    if not include_full_qutip:
        projected_qutip_time = qutip_time * full_points / validation_points

    doubled = replace(
        validation, num_steps_per_half=2 * validation.num_steps_per_half
    )
    converged = run_simulation(doubled, device="cpu")
    convergence = _differences(torch_validation, converged)
    report = {
        "config": asdict(config),
        "qutip_scope": qutip_scope,
        "timings_seconds": {
            "qutip_cpu": qutip_time,
            "pytorch_cpu": torch_cpu_time,
            "mps_warmup": warmup_time,
            "mps_gpu": mps_time,
        },
        "speedups": {
            "mps_over_qutip_measured_full_grid": None
            if mps_time is None or not include_full_qutip
            else qutip_time / mps_time,
            "mps_over_qutip_projected_from_representative_points": None
            if mps_time is None or projected_qutip_time is None
            else projected_qutip_time / mps_time,
            "mps_over_pytorch_cpu": None
            if mps_time is None
            else torch_cpu_time / mps_time,
        },
        "projected_qutip_full_grid_seconds": projected_qutip_time,
        "validation_pytorch_cpu_vs_qutip": validation_comparison,
        "selected_timestep_vs_half_timestep": convergence,
        "physical_validity": {
            "pytorch_cpu": {
                "maximum_raw_trace_drift": torch_cpu.raw_trace_error,
                "maximum_trace_error": torch_cpu.trace_error,
                "maximum_hermiticity_error": torch_cpu.hermiticity_error,
                "minimum_density_matrix_eigenvalue": torch_cpu.minimum_eigenvalue,
                "maximum_leakage": float(torch_cpu.leakage.max()),
            },
            "mps": None
            if mps_result is None
            else {
                "maximum_raw_trace_drift": mps_result.raw_trace_error,
                "maximum_trace_error": mps_result.trace_error,
                "maximum_hermiticity_error": mps_result.hermiticity_error,
                "minimum_density_matrix_eigenvalue": mps_result.minimum_eigenvalue,
                "maximum_leakage": float(mps_result.leakage.max()),
                **_differences(mps_result, torch_cpu),
            },
        },
        "devices": {
            "pytorch_cpu": torch_cpu.tensor_device,
            "mps": None if mps_result is None else mps_result.tensor_device,
            "mps_available": mps_is_available(),
        },
        "complex_representation": torch_cpu.complex_representation,
    }
    tolerances = {
        "maximum_population_difference": 2e-5,
        "maximum_density_matrix_difference": 1e-2,
        "maximum_trace_error": 2e-5,
        "maximum_hermiticity_error": 2e-5,
        "minimum_density_matrix_eigenvalue": -2e-5,
        "pytorch_cpu_vs_mps_density_matrix_difference": 5e-5,
    }
    report["tolerances"] = tolerances
    mps_difference = None if mps_result is None else _differences(mps_result, torch_cpu)
    report["validation_passed"] = bool(
        validation_comparison["maximum_population_difference"]
        <= tolerances["maximum_population_difference"]
        and validation_comparison["maximum_density_matrix_difference"]
        <= tolerances["maximum_density_matrix_difference"]
        and torch_cpu.trace_error <= tolerances["maximum_trace_error"]
        and torch_cpu.hermiticity_error
        <= tolerances["maximum_hermiticity_error"]
        and torch_cpu.minimum_eigenvalue
        >= tolerances["minimum_density_matrix_eigenvalue"]
        and convergence["maximum_population_difference"]
        <= tolerances["maximum_population_difference"]
        and convergence["maximum_density_matrix_difference"]
        <= tolerances["maximum_density_matrix_difference"]
        and (
            mps_difference is None
            or mps_difference["maximum_density_matrix_difference"]
            <= tolerances["pytorch_cpu_vs_mps_density_matrix_difference"]
        )
    )
    if include_full_qutip:
        report["full_grid_pytorch_cpu_vs_qutip"] = _differences(
            torch_cpu, qutip_result
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=("small", "full"), default="small")
    parser.add_argument("--full-qutip", action="store_true")
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--steps-per-half", type=int)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/mps_benchmark")
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.grid == "small":
        config = SimulationConfig(
            levels=args.levels,
            amplitude_mhz=(0.0, 12.5, 25.0),
            detuning_mhz=(-0.5, 0.0, 0.5),
            duration_us=0.1,
            num_steps_per_half=args.steps_per_half or 400,
        )
    else:
        config = SimulationConfig(
            levels=args.levels,
            num_steps_per_half=args.steps_per_half or 30_000,
        )
    report = benchmark(config, include_full_qutip=args.full_qutip)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"benchmark_{args.grid}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
