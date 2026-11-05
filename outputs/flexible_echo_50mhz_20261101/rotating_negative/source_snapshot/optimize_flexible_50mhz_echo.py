"""Ensemble-optimize smooth echo shapes and continuous central I/Q rotations.

The complete complex waveform is normalized before every amplitude sweep.
The original polynomial is the no-regression reference at every drive level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.simulation.flexible_echo import FlexibleEchoPulse  # noqa: E402
from echospec.simulation.polynomial_echo import (  # noqa: E402
    PolynomialEchoPulse,
    simulate_polynomial_echo,
)
from scripts.optimize_50mhz_echo import metrics, peak_scale  # noqa: E402
from scripts.optimize_decoupled_50mhz_echo import DET, clean  # noqa: E402

OUT = ROOT / "outputs/flexible_echo_50mhz_20261101"
PREVIOUS = ROOT / "outputs/decoupled_echo_50mhz_20261101/selected_candidates.json"
AMPS = np.array([5., 6., 7.5, 10., 12., 15., 20., 25., 30., 40., 50., 60.])
BOUNDS = [(8., 45.), (-2.6989, -.7), (-1., .7), (-1.7, .7), (-.3, 1.5),
          (0., 20.), (.03, .98), (-3., 3.), (-1.7, .9), (-2.4, -.3)]
BOUNDS += [(-1.5, 1.5)] * 4 + [(-3., 3.)] * 5 + [(-1., 1.)] + [(-3., 3.)] * 3
NAMES = ["duration_us", "log10_core_us", "log10_tail_gain", "log10_tail_join_us",
         "log10_tail_scale_us", "tail_power", "edge_fraction", "phase_rad",
         "log10_phase_scale_us", "log10_transition_us"]
NAMES += [f"amp_coefficient_{i}" for i in range(4)]
NAMES += [f"phase_coefficient_{i}" for i in range(5)]
NAMES += ["rotation_mix"] + [f"odd_phase_coefficient_{i}" for i in range(3)]


def make_pulse(p):
    return FlexibleEchoPulse(
        duration_us=p[0], core_us=10**p[1], tail_gain=10**p[2],
        tail_join_us=10**p[3], tail_scale_us=10**p[4], tail_power=p[5],
        edge_width_us=p[6] * p[0] / 2, phase_rad=p[7], phase_scale_us=10**p[8],
        transition_us=10**p[9], amp_coefficients=tuple(p[10:14]),
        phase_coefficients=tuple(p[14:19]), rotation_mix=p[19],
        odd_phase_coefficients=tuple(p[20:23]),
    )


def encode(fields):
    p = fields
    return np.array([
        p["duration_us"], np.log10(p["core_us"]), np.log10(p["tail_gain"]),
        np.log10(p["tail_join_us"]), np.log10(p["tail_scale_us"]), p["tail_power"],
        2*p["edge_width_us"]/p["duration_us"], p["phase_rad"],
        np.log10(p["phase_scale_us"]), np.log10(p["transition_us"]),
        *p.get("amp_coefficients", [0.]*4), *p.get("phase_coefficients", [0.]*5),
        p.get("rotation_mix", 0.), *p.get("odd_phase_coefficients", [0.]*3),
    ], dtype=float)


def assess(rows, baseline):
    """User acceptance and auxiliary spectral checks, plus smooth inequalities."""
    n = len(rows)
    if not all(r["accepted"] for r in rows):
        return dict(feasible=False, width50_khz=1e4, loss=1e6), np.full(5*n+1, -10.)
    widths = np.array([r["fwhm_hz"]/1000 for r in rows])
    contrast = np.array([r["contrast"] for r in rows])
    bw = np.array([r["fwhm_hz"]/1000 for r in baseline])
    bc = np.array([r["contrast"] for r in baseline])
    satellites = np.array([r["satellite_prominence"] for r in rows])
    centers = np.array([abs(r["center_hz"])/1000 for r in rows])
    leakage = np.array([r["final_leakage_max"] for r in rows])
    index50 = next(i for i, r in enumerate(rows) if r["rabi_mhz"] == 50.)
    inequalities = np.r_[10*(contrast-bc-.0005), (bw-widths-.2)/100,
                         10*(contrast[index50]-.402), 10*(.01-satellites),
                         (2-centers)/10, 100*(.005-leakage)]
    user_pass = (contrast[index50] > .4 and np.all(contrast >= bc-.0001)
                 and np.all(widths <= bw+.05))
    guardrails = np.all(satellites <= .01) and np.all(centers <= 2) and np.all(leakage <= .005)
    objective = (.85*widths[index50] + .15*widths.mean())/100
    penalty = 100*np.maximum(-inequalities, 0).sum()
    return dict(feasible=bool(user_pass and guardrails), user_constraints_pass=bool(user_pass),
                guardrails_pass=bool(guardrails), width50_khz=float(widths[index50]),
                contrast50=float(contrast[index50]), min_contrast=float(contrast.min()),
                max_width_khz=float(widths.max()),
                max_width_regression_khz=float((widths-bw).max()),
                max_contrast_regression=float((bc-contrast).max()),
                max_center_khz=float(centers.max()), max_satellite=float(satellites.max()),
                max_leakage=float(leakage.max()), objective=float(objective),
                loss=float(objective+penalty)), inequalities


def save(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


class EvaluationBudget(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--mode", choices=("base", "shape", "rotation"), default="shape")
    parser.add_argument("--method", choices=("slsqp", "nm", "de"), default="slsqp")
    parser.add_argument("--start", type=Path)
    parser.add_argument("--start-name", default="preserve_original")
    parser.add_argument("--rotation-seed", type=float)
    parser.add_argument("--max-evaluations", type=int, default=600)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--gradient-step", type=float, default=1e-4)
    parser.add_argument("--extra-amplitudes", type=float, nargs="*", default=[])
    parser.add_argument("--rotation-range", type=float, nargs=2)
    args = parser.parse_args()
    amplitudes = np.unique(np.r_[AMPS, args.extra_amplitudes])
    start_file = args.start or PREVIOUS
    data = json.loads(start_file.read_text())
    source = (next(r for r in data["candidates"] if r.get("name") == args.start_name)
              if not args.start else data.get("candidates", [data])[0])
    fields = dict(source["pulse"])
    if "amp_coefficients" not in fields:
        fields["transition_us"] *= 1.875
    full_start = encode(fields)
    if args.rotation_seed is not None:
        full_start[19] = args.rotation_seed
    dimensions = {"base": 10, "shape": 19, "rotation": 23}[args.mode]
    bounds = list(BOUNDS)
    if args.rotation_range:
        bounds[19] = tuple(args.rotation_range)
        if not -1 <= bounds[19][0] < bounds[19][1] <= 1:
            parser.error("--rotation-range requires -1 <= lower < upper <= 1")
    lower, upper = np.array(bounds).T
    full_start = np.clip(full_start, lower, upper)
    initial = ((full_start-lower)/(upper-lower))[:dimensions]
    directory = OUT / args.campaign
    (directory / "evaluations").mkdir(parents=True, exist_ok=True)
    baseline_pulse = PolynomialEchoPulse()
    pop, _ = simulate_polynomial_echo(baseline_pulse, amplitudes/peak_scale(baseline_pulse), DET, dt_ns=.5)
    baseline = metrics(pop, amplitudes, DET)
    sources = [Path(__file__), ROOT/"echospec/simulation/flexible_echo.py",
               ROOT/"echospec/simulation/decoupled_echo.py",
               ROOT/"echospec/simulation/polynomial_echo.py",
               ROOT/"echospec/simulation/qutrit_rk4.cpp",
               ROOT/"echospec/analysis/dip_metrics.py", ROOT/"scripts/optimize_50mhz_echo.py"]
    config = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  start_file=str(start_file), start_sha256=hashlib.sha256(start_file.read_bytes()).hexdigest(),
                  parameter_names=NAMES, bounds=bounds, initial_parameters=full_start.tolist(),
                  active_dimensions=dimensions, amplitudes_mhz=amplitudes.tolist(), detuning_mhz=DET.tolist(),
                  dt_ns=.5, t1_us=51.24, t2_us=7.31, baseline=baseline,
                  constraints="C50>0.4; no linewidth/contrast regression vs original over5–60MHz",
                  spectral_guardrails="center<=2kHz, satellite<=.01, finalP2<=.005",
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sources})
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = directory / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        parser.error("Existing campaign configuration differs; use a new name")
    save(config_path, config)
    snapshot = directory / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for source_path in sources:
        (snapshot / source_path.name).write_bytes(source_path.read_bytes())
    records, gradients = {}, {}
    lock = threading.Lock()
    submitted = 0

    def evaluate(z):
        nonlocal submitted
        z = np.clip(np.asarray(z, float), 0, 1)
        key = hashlib.sha256(z.tobytes()).hexdigest()
        with lock:
            if key in records:
                return records[key]
            if submitted >= args.max_evaluations:
                raise EvaluationBudget
            submitted += 1
        parameters = full_start.copy()
        parameters[:dimensions] = lower[:dimensions] + z*(upper[:dimensions]-lower[:dimensions])
        pulse = make_pulse(parameters)
        norm = peak_scale(pulse)
        pop, diag = simulate_polynomial_echo(pulse, amplitudes/norm, DET, dt_ns=.5)
        rows = metrics(pop, amplitudes, DET)
        quality, inequalities = assess(rows, baseline)
        record = dict(parameters=parameters.tolist(), normalized_parameters=z.tolist(),
                      pulse=asdict(pulse), peak_scale=norm, rows=rows, quality=quality,
                      loss=quality["loss"], inequalities=inequalities.tolist(), diagnostics=diag,
                      configuration_sha256=config_hash)
        save(directory / "evaluations" / f"{key}.json", record)
        with lock:
            records[key] = record
            if len(records) % 50 == 0:
                best = min(records.values(), key=lambda r: r["loss"])
                print(json.dumps(dict(evaluations=len(records), best=best["quality"])), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        def jacobians(z):
            key = hashlib.sha256(np.asarray(z, float).tobytes()).hexdigest()
            if key in gradients:
                return gradients[key]
            base = evaluate(z)
            points, steps = [], []
            for j in range(dimensions):
                step = args.gradient_step if z[j] + args.gradient_step <= 1 else -args.gradient_step
                point = np.array(z, copy=True)
                point[j] += step
                points.append(point)
                steps.append(step)
            perturbed = list(pool.map(evaluate, points))
            fgrad = np.array([(r["quality"].get("objective", 100)-base["quality"].get("objective", 100))/h
                              for r, h in zip(perturbed, steps, strict=True)])
            cgrad = np.column_stack([(np.array(r["inequalities"])-base["inequalities"])/h
                                     for r, h in zip(perturbed, steps, strict=True)])
            gradients[key] = fgrad, cgrad
            return gradients[key]

        status = "evaluation budget reached"
        try:
            if args.method == "slsqp":
                result = minimize(
                    lambda z: evaluate(z)["quality"].get("objective", 100), initial,
                    jac=lambda z: jacobians(z)[0], method="SLSQP", bounds=[(0, 1)]*dimensions,
                    constraints={"type": "ineq", "fun": lambda z: np.array(evaluate(z)["inequalities"]),
                                 "jac": lambda z: jacobians(z)[1]},
                    options={"maxiter": args.max_iterations, "ftol": 2e-5, "disp": True},
                )
            elif args.method == "nm":
                simplex = np.tile(initial, (dimensions+1, 1))
                for j in range(dimensions):
                    simplex[j+1, j] += .012 if initial[j] < .98 else -.012
                result = minimize(lambda z: evaluate(z)["loss"], initial, method="Nelder-Mead",
                                  bounds=[(0, 1)]*dimensions,
                                  options={"maxfev": args.max_evaluations, "initial_simplex": simplex,
                                           "xatol": 1e-4, "fatol": 1e-4})
            else:
                rng = np.random.default_rng(args.seed)
                population = np.clip(initial + rng.normal(0, .035, (48, dimensions)), 0, 1)
                population[0] = initial
                result = differential_evolution(lambda z: evaluate(z)["loss"], [(0, 1)]*dimensions,
                                                 init=population, workers=pool.map, updating="deferred",
                                                 maxiter=args.max_iterations, polish=False, seed=args.seed,
                                                 mutation=(.3, .8), recombination=.8)
            status = str(result.message)
        except EvaluationBudget:
            pass
    ranked = sorted(records.values(), key=lambda r: r["loss"])
    feasible = sorted((r for r in ranked if r["quality"]["feasible"]),
                      key=lambda r: r["quality"]["width50_khz"])
    save(directory/"best.json", ranked[0])
    save(directory/"candidates.json", dict(candidates=feasible[:10] if feasible else ranked[:10],
                                          evaluations=len(records), feasible_count=len(feasible),
                                          optimizer_status=status))
    print("DONE", len(records), "feasible", len(feasible), "best", ranked[0]["quality"], flush=True)


if __name__ == "__main__":
    main()
