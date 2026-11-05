"""Optimize one fixed echo shape over actual peaks 5--60 MHz.

Core, normalized tail weight, tail decay and endpoint closure vary independently.
The previous waveform generators and output directory are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.simulation.decoupled_echo import DecoupledEchoPulse  # noqa: E402
from echospec.simulation.polynomial_echo import (  # noqa: E402
    PolynomialEchoPulse,
    simulate_polynomial_echo,
)
from scripts.optimize_50mhz_echo import metrics, peak_scale  # noqa: E402

OUT = ROOT / "outputs/decoupled_echo_50mhz_20261101"
AMPS = np.array([5., 7.5, 10., 15., 20., 30., 40., 50., 60.])
DET = np.unique(np.round(np.r_[np.arange(-.5, -.15, .025),
                               np.arange(-.15, .151, .005), np.arange(.175, .501, .025)], 8))
NAMES = ["duration_us", "log10_core_us", "log10_normalized_tail_us", "log10_tail_join_us",
         "log10_tail_scale_us", "tail_power", "edge_fraction", "phase_rad", "log10_phase_scale_us"]
BOUNDS = [(16., 44.), (-2.7, -1.), (-2.7, -1.), (-1.7, .5),
          (-.3, 1.5), (0., 20.), (.04, .9), (-3., 3.), (-.7, .8)]


def make_pulse(parameters):
    length, core, weight, join, scale, power, edge, phase, phase_scale = parameters[:9]
    pulse = DecoupledEchoPulse(duration_us=length, core_us=10**core,
                               tail_join_us=10**join, tail_scale_us=10**scale,
                               tail_power=power, edge_width_us=edge * length / 2,
                               phase_rad=phase, phase_scale_us=10**phase_scale,
                               phase2_rad=parameters[9] if len(parameters) > 9 else 0.,
                               phase2_scale_us=10**parameters[10] if len(parameters) > 9 else 4.)
    # Approximate normalized asymptotic tail coefficient, removing most of the
    # core/gain correlation. Final normalization uses the actual global peak.
    gain = 10**weight * peak_scale(pulse) / pulse.core_us
    return replace(pulse, tail_gain=gain)


def encode(pulse, two_phase=False):
    unit = replace(pulse, tail_gain=1.)
    values = [pulse.duration_us, np.log10(pulse.core_us),
              np.log10(pulse.tail_gain * pulse.core_us / peak_scale(unit)),
              np.log10(pulse.tail_join_us), np.log10(pulse.tail_scale_us),
              pulse.tail_power, 2 * pulse.edge_width_us / pulse.duration_us,
              pulse.phase_rad, np.log10(pulse.phase_scale_us)]
    if two_phase:
        values += [pulse.phase2_rad, np.log10(pulse.phase2_scale_us)]
    return np.array(values)


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def constraints(rows, baseline):
    if not all(r["accepted"] for r in rows):
        return dict(feasible=False, violation=1e4, width50_khz=1e4,
                    max_width_khz=1e4, min_contrast=0.)
    width = np.array([r["fwhm_hz"] / 1000 for r in rows])
    contrast = np.array([r["contrast"] for r in rows])
    bw = np.array([r["fwhm_hz"] / 1000 for r in baseline])
    bc = np.array([r["contrast"] for r in baseline])
    target_c = np.maximum(.405, bc + .001)
    deficit = np.maximum(target_c - contrast, 0)
    regression = np.maximum(width - bw, 0)
    satellite = max(r["satellite_prominence"] for r in rows)
    center = max(abs(r["center_hz"]) / 1000 for r in rows)
    leakage = max(r["final_leakage_max"] for r in rows)
    violation = (1e4 * deficit.max() + 2e3 * deficit.mean()
                 + 6 * regression.max() + 3 * regression.mean()
                 + 1e4 * max(0., satellite - .01)
                 + 30 * max(0., center - 2.) + 2e3 * max(0., leakage - .005))
    return dict(feasible=bool(np.all(contrast > .4) and np.all(contrast >= bc - .001)
                              and np.all(width <= bw + .25) and satellite <= .01
                              and center <= 2. and leakage <= .005),
                violation=float(violation), width50_khz=float(width[7]),
                max_width_khz=float(width.max()), min_contrast=float(contrast.min()),
                max_contrast_regression=float((bc - contrast).max()),
                max_width_regression_khz=float((width - bw).max()),
                max_satellite=float(satellite), max_center_khz=float(center),
                max_leakage=float(leakage))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="initial")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--population", type=int, default=54)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--two-phase", action="store_true")
    parser.add_argument("--warm-start", type=Path)
    args = parser.parse_args()
    directory = OUT / args.campaign
    (directory / "evaluations").mkdir(parents=True, exist_ok=True)
    baseline_pop, diagnostics = simulate_polynomial_echo(
        PolynomialEchoPulse(), AMPS / peak_scale(PolynomialEchoPulse()), DET, dt_ns=.5)
    baseline = metrics(baseline_pop, AMPS, DET)
    bounds = BOUNDS + ([(-3, 3), (.0, 1.1)] if args.two_phase else [])
    source_paths = [Path(__file__), ROOT / "echospec/simulation/decoupled_echo.py",
                    ROOT / "echospec/simulation/polynomial_echo.py",
                    ROOT / "echospec/simulation/qutrit_rk4.cpp",
                    ROOT / "scripts/optimize_50mhz_echo.py"]
    config = dict(amplitudes_mhz=AMPS.tolist(), detuning_mhz=DET.tolist(), dt_ns=.5,
                  t1_us=51.24, t2_us=7.31, seed=args.seed, population=args.population,
                  iterations=args.iterations, bounds=bounds, parameter_names=NAMES +
                  (["phase2_rad", "log10_phase2_scale_us"] if args.two_phase else []),
                  baseline=baseline, constraints="C>0.4 across5-60MHz; no width/contrast regression vs original at all samples",
                  no_regression_contrast_tolerance=.001, no_regression_width_tolerance_khz=.25,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in source_paths})
    (directory / "configuration.json").write_text(json.dumps(config, indent=2))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def objective(parameters):
        key = hashlib.sha256(np.asarray(parameters, dtype=float).tobytes()).hexdigest()
        path = directory / "evaluations" / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("configuration_sha256") == config_hash:
                return record["loss"]
        pulse = make_pulse(parameters)
        norm = peak_scale(pulse)
        pop, diag = simulate_polynomial_echo(pulse, AMPS / norm, DET, dt_ns=.5)
        rows = metrics(pop, AMPS, DET)
        quality = constraints(rows, baseline)
        loss = .7 * quality["width50_khz"] + .3 * quality["max_width_khz"] + quality["violation"]
        record = dict(parameters=list(parameters), pulse=asdict(pulse), peak_scale=norm,
                      rows=rows, quality=quality, loss=loss, diagnostics=diag,
                      configuration_sha256=config_hash)
        path.write_text(json.dumps(clean(record), indent=2, allow_nan=False))
        return loss

    seeds = []
    seed_file = OUT / "seed_fits.json"
    if seed_file.exists():
        data = json.loads(seed_file.read_text())
        entries = data.get("fits", data.get("cases", []))
        if isinstance(entries, dict):
            entries = list(entries.values())
        for row in entries:
            seeds.append(encode(DecoupledEchoPulse(**row["pulse"]), args.two_phase))
    if args.warm_start:
        saved = json.loads(args.warm_start.read_text())
        records = saved if isinstance(saved, list) else saved.get("candidates", [saved])
        for r in records:
            seeds.append(encode(DecoupledEchoPulse(**r["pulse"]), args.two_phase))
    for core in (.008, .015, .03, .06):
        for phase in (0., -1.):
            seed = [30., np.log10(core), -1.8, -1.1, .6, 1.5, .4, phase, -.1]
            if args.two_phase:
                seed += [0., .6]
            seeds.append(np.array(seed))
    lower, upper = np.array(bounds).T
    seeds = np.clip(np.array(seeds), lower, upper)
    rng = np.random.default_rng(args.seed)
    initial = [*seeds]
    while len(initial) < args.population:
        base = seeds[len(initial) % len(seeds)]
        jitter = rng.normal(0, .10 if len(initial) < args.population * .7 else .25, len(bounds))
        initial.append(np.clip(base + jitter * (upper - lower), lower, upper))
    initial = np.array(initial[:args.population])
    generation = 0

    def callback(x, convergence):
        nonlocal generation
        generation += 1
        value = objective(x)
        print(json.dumps(dict(generation=generation, loss=value, parameters=x.tolist())), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        result = differential_evolution(objective, bounds, init=initial,
                                        maxiter=args.iterations, seed=args.seed,
                                        workers=pool.map, updating="deferred", polish=False,
                                        callback=callback, tol=.001, mutation=(.4, 1.),
                                        recombination=.8)
    records = [json.loads(p.read_text()) for p in (directory / "evaluations").glob("*.json")]
    records.sort(key=lambda r: r["loss"])
    feasible = sorted([r for r in records if r["quality"]["feasible"]],
                      key=lambda r: .7 * r["quality"]["width50_khz"] + .3 * r["quality"]["max_width_khz"])
    (directory / "best.json").write_text(json.dumps(records[0], indent=2))
    (directory / "candidates.json").write_text(json.dumps(dict(
        candidates=feasible[:8] if feasible else records[:8], feasible_count=len(feasible),
        evaluations=len(records), optimizer_loss=float(result.fun)), indent=2))
    print("DONE", len(records), "feasible", len(feasible), "best", records[0]["quality"], flush=True)


if __name__ == "__main__":
    main()
