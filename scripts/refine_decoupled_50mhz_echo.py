"""Run a bounded, single-worker strict refinement of the decoupled echo.

Example: .venv/bin/python scripts/refine_decoupled_50mhz_echo.py
The default run starts from core_phase/best.json and saves strict_local/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.simulation.decoupled_echo import DecoupledEchoPulse  # noqa: E402
from echospec.simulation.polynomial_echo import (  # noqa: E402
    PolynomialEchoPulse,
    simulate_polynomial_echo,
)
from scripts.optimize_50mhz_echo import metrics, peak_scale  # noqa: E402
from scripts.optimize_decoupled_50mhz_echo import (  # noqa: E402
    AMPS,
    BOUNDS,
    DET,
    NAMES,
    OUT,
    clean,
    constraints,
    encode,
    make_pulse,
)


def write_json(path: Path, value: dict) -> None:
    """Keep each evaluation and campaign summary independently inspectable."""
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="strict_local")
    parser.add_argument("--start", type=Path, default=OUT / "core_phase/best.json")
    parser.add_argument("--max-evaluations", type=int, default=250)
    args = parser.parse_args()
    if not 1 <= args.max_evaluations <= 250:
        parser.error("--max-evaluations must be between 1 and 250")
    if Path(args.campaign).name != args.campaign:
        parser.error("--campaign must be a single directory name")

    start_path = args.start.resolve()
    start_bytes = start_path.read_bytes()
    start_record = json.loads(start_bytes)
    start_pulse = DecoupledEchoPulse(**start_record["pulse"])
    bounds = [(8.0, BOUNDS[0][1]), *BOUNDS[1:], (-3.0, 3.0), (-2.0, -0.7)]
    lower, upper = np.asarray(bounds).T
    initial_parameters = np.clip(encode(start_pulse, two_phase=True), lower, upper)
    initial_normalized = (initial_parameters - lower) / (upper - lower)

    original = PolynomialEchoPulse()
    baseline_populations, baseline_diagnostics = simulate_polynomial_echo(
        original, AMPS / peak_scale(original), DET, dt_ns=0.5
    )
    baseline = metrics(baseline_populations, AMPS, DET)
    source_paths = [
        Path(__file__),
        ROOT / "scripts/optimize_decoupled_50mhz_echo.py",
        ROOT / "scripts/optimize_50mhz_echo.py",
        ROOT / "echospec/simulation/decoupled_echo.py",
        ROOT / "echospec/simulation/polynomial_echo.py",
        ROOT / "echospec/simulation/qutrit_rk4.cpp",
        ROOT / "echospec/analysis/dip_metrics.py",
    ]
    configuration = dict(
        method="Powell",
        normalized_bounds=[(0.0, 1.0)] * len(bounds),
        bounds=bounds,
        parameter_names=[*NAMES, "phase2_rad", "log10_phase2_scale_us"],
        initial_parameters=initial_parameters.tolist(),
        start_path=str(start_path),
        start_sha256=hashlib.sha256(start_bytes).hexdigest(),
        max_candidate_evaluations=args.max_evaluations,
        baseline_evaluations=1,
        workers=1,
        optimizer_options=dict(maxfev=args.max_evaluations, xtol=0.002, ftol=0.0001),
        amplitudes_mhz=AMPS.tolist(),
        detuning_mhz=DET.tolist(),
        dt_ns=0.5,
        t1_us=51.24,
        t2_us=7.31,
        baseline=baseline,
        baseline_diagnostics=baseline_diagnostics,
        constraints=(
            "C>0.4 across 5--60 MHz; no width/contrast regression vs original "
            "at all samples; unchanged strict objective and feasibility checks"
        ),
        contrast_floor_scope="5--60 MHz",
        no_regression_contrast_tolerance=0.001,
        no_regression_width_tolerance_khz=0.25,
        loss="0.7*width50_khz + 0.3*max_width_khz + strict violation",
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__),
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
    )
    configuration = clean(configuration)
    config_hash = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()
    directory = OUT / args.campaign
    evaluation_directory = directory / "evaluations"
    evaluation_directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        parser.error("campaign configuration differs; use a new --campaign name")
    write_json(config_path, configuration)
    records: dict[str, dict] = {}

    def objective(normalized: np.ndarray) -> float:
        parameters = lower + np.asarray(normalized) * (upper - lower)
        key = hashlib.sha256(parameters.astype(float).tobytes()).hexdigest()
        if key in records:
            return records[key]["loss"]
        path = evaluation_directory / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("configuration_sha256") == config_hash:
                records[key] = record
                return record["loss"]
        pulse = make_pulse(parameters)
        norm = peak_scale(pulse)
        populations, diagnostics = simulate_polynomial_echo(
            pulse, AMPS / norm, DET, dt_ns=0.5
        )
        rows = metrics(populations, AMPS, DET)
        quality = constraints(rows, baseline, contrast_at50_only=False)
        loss = (
            0.7 * quality["width50_khz"]
            + 0.3 * quality["max_width_khz"]
            + quality["violation"]
        )
        record = dict(
            parameters=parameters.tolist(),
            normalized_parameters=np.asarray(normalized).tolist(),
            pulse=asdict(pulse),
            peak_scale=norm,
            rows=rows,
            quality=quality,
            loss=loss,
            diagnostics=diagnostics,
            configuration_sha256=config_hash,
        )
        write_json(path, record)
        records[key] = record
        if len(records) % 25 == 0:
            best = min(records.values(), key=lambda item: item["loss"])
            print(json.dumps(dict(evaluations=len(records), loss=best["loss"],
                                  quality=best["quality"])), flush=True)
        return loss

    result = minimize(
        objective,
        initial_normalized,
        method="Powell",
        bounds=[(0.0, 1.0)] * len(bounds),
        options=configuration["optimizer_options"],
    )
    ranked = sorted(records.values(), key=lambda item: item["loss"])
    feasible = sorted(
        (record for record in ranked if record["quality"]["feasible"]),
        key=lambda record: (
            0.7 * record["quality"]["width50_khz"]
            + 0.3 * record["quality"]["max_width_khz"]
        ),
    )
    summary = dict(
        candidates=feasible[:8] if feasible else ranked[:8],
        feasible_count=len(feasible),
        evaluations=len(records),
        optimizer_function_calls=int(result.nfev),
        optimizer_iterations=int(result.nit),
        optimizer_loss=float(result.fun),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        interpretation="Finite bounded family search; no infeasibility theorem.",
        configuration_sha256=config_hash,
    )
    write_json(directory / "best.json", ranked[0])
    write_json(directory / "candidates.json", summary)
    print("DONE", len(records), "feasible", len(feasible),
          "best", ranked[0]["quality"], flush=True)


if __name__ == "__main__":
    main()
