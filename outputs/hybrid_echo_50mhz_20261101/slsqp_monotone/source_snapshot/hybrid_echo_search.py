"""Evolutionary exploration followed by constrained smooth-notch refinement.

All final acceptance still uses the existing extracted FWHM and contrast.
The local half-height surrogate avoids differentiating the selected crossings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.analysis.notch_constraints import (  # noqa: E402
    centered_notch_features,
    half_depth_residuals,
)
from echospec.simulation.polynomial_echo import simulate_polynomial_echo  # noqa: E402
from scripts.optimize_50mhz_echo import metrics, peak_scale  # noqa: E402
from scripts.optimize_flexible_50mhz_echo import (  # noqa: E402
    BOUNDS,
    NAMES,
    encode,
    make_pulse,
)
from scripts.report_decoupled_50mhz_echo import (  # noqa: E402
    clean,
    local_shoulder_metrics,
)

SOURCE = ROOT / "outputs/flexible_echo_50mhz_20261101"
OUT = ROOT / "outputs/hybrid_echo_50mhz_20261101"
GLOBAL_AMPS = np.array([5., 7.5, 10., 15., 17., 20., 25., 35., 50., 60.])
LOCAL_AMPS = np.array([5., 6., 7., 7.5, 10., 12., 15., 16.5, 17., 17.5,
                       20., 21.5, 25., 30., 40., 50., 55., 60.])
GLOBAL_DET = np.unique(np.round(np.r_[np.linspace(-.2, .2, 81), np.linspace(-.5, .5, 41)], 12))
LOCAL_DET = np.linspace(-.5, .5, 201)
LIMITS = list(BOUNDS) + [(-3., 3.), (-1.7, 1.), (-.65, -.35)]
LIMITS[0] = (9., 35.)
LIMITS[6] = (.005, .95)
LIMITS[7] = (-4., 4.)
LIMITS[10:14] = [(-2., 2.)] * 4
LIMITS[14:19] = [(-4., 4.)] * 5
LOW, HIGH = np.array(LIMITS).T
SPAN = HIGH - LOW
PARAMETER_NAMES = NAMES + ["phase2_rad", "log10_phase2_scale_us", "drag_beta"]
MODEL_PATHS = [ROOT / "echospec/simulation" / n for n in (
    "flexible_echo.py", "decoupled_echo.py", "polynomial_echo.py", "qutrit_rk4.cpp")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def to_normalized(fields):
    p = np.r_[encode(fields), fields.get("phase2_rad", 0),
              np.log10(fields.get("phase2_scale_us", 4)), fields.get("drag_beta", -.5)]
    return np.clip((p - LOW) / SPAN, 0, 1)


def from_normalized(z):
    p = LOW + np.asarray(z) * SPAN
    return replace(make_pulse(p[:23]), phase2_rad=p[23], phase2_scale_us=10**p[24],
                   drag_beta=p[25])


class EvaluationBudget(Exception):
    pass


class Engine:
    def __init__(self, directory, amplitudes, detuning, dt_ns, budget,
                 target_contrast=None, monotone_local=False):
        self.directory = directory
        (directory / "evaluations").mkdir(exist_ok=True, parents=True)
        self.amplitudes, self.detuning, self.dt_ns = amplitudes, detuning, dt_ns
        self.budget, self.count, self.lock = budget, 0, threading.Lock()
        self.records, self.arrays = {}, {}
        self.target_contrast, self.monotone_local = target_contrast, monotone_local
        self.i50 = int(np.flatnonzero(amplitudes == 50)[0])
        with np.load(SOURCE / "validated_spectra.npz") as old:
            ia = np.searchsorted(old["rabi_mhz"], amplitudes)
            ix = np.rint((detuning + .5) / .0025).astype(int)
            np.testing.assert_allclose(old["detuning_mhz"][ix], detuning, atol=1e-12)
            baseline_pop = old["baseline"][:, ia][:, :, ix]
        self.baseline = metrics(baseline_pop, amplitudes, detuning)
        self.bw = np.array([r["fwhm_hz"] / 1000 for r in self.baseline])
        self.bc = np.array([r["contrast"] for r in self.baseline])
        config = dict(amplitudes_mhz=amplitudes.tolist(), detuning_mhz=detuning.tolist(),
                      dt_ns=dt_ns, t1_us=51.24, t2_us=7.31, baseline=self.baseline,
                      bounds=LIMITS, parameter_names=PARAMETER_NAMES,
                      target_contrast=target_contrast, monotone_local=monotone_local,
                      model_sources={str(p.relative_to(ROOT)): sha(p) for p in MODEL_PATHS},
                      baseline_source_sha256=sha(SOURCE / "validated_spectra.npz"))
        self.config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if (directory / "configuration.json").exists():
            assert json.loads((directory / "configuration.json").read_text()) == clean(config)
        save(directory / "configuration.json", config)

    def evaluate(self, z):
        z = np.clip(np.asarray(z[:26], float), 0, 1)
        key = hashlib.sha256(z.tobytes()).hexdigest()
        with self.lock:
            if key in self.records:
                return self.records[key], self.arrays.get(key)
            path = self.directory / "evaluations" / f"{key}.npz"
            if path.exists():
                with np.load(path) as old:
                    record = json.loads(str(old["record"]))
                    assert record["configuration_sha256"] == self.config_hash
                    pe = old["pe"].copy()
                self.records[key], self.arrays[key] = record, pe
                return record, pe
            if self.count >= self.budget:
                raise EvaluationBudget
            self.count += 1
        pulse = from_normalized(z)
        try:
            norm = peak_scale(pulse)
            pop, diag = simulate_polynomial_echo(pulse, self.amplitudes / norm,
                                                 self.detuning, dt_ns=self.dt_ns)
            rows = metrics(pop, self.amplitudes, self.detuning)
            local = local_shoulder_metrics(self.detuning, pop[1, self.i50])
            record = dict(valid=all(r["accepted"] for r in rows), pulse=asdict(pulse),
                          parameters=(LOW + z * SPAN).tolist(), normalized_parameters=z.tolist(),
                          peak_scale=norm, rows=rows, local50=local, diagnostics=diag,
                          configuration_sha256=self.config_hash, key=key)
            pe = pop[1]
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            record = dict(valid=False, error=str(exc), pulse=asdict(pulse),
                          normalized_parameters=z.tolist(), configuration_sha256=self.config_hash, key=key)
            pe = np.empty(0)
        np.savez_compressed(path, record=json.dumps(clean(record)), pe=pe)
        with self.lock:
            self.records[key], self.arrays[key] = record, pe
        return record, pe

    def quality(self, record, track):
        if not record["valid"]:
            return dict(feasible=False, loss=1e6, width50_khz=1e4), np.full(5 * len(self.amplitudes) + 2, -100.)
        rows = record["rows"]
        w = np.array([r["fwhm_hz"] / 1000 for r in rows])
        c = np.array([r["contrast"] for r in rows])
        sat = np.array([r["satellite_prominence"] for r in rows])
        center = np.array([abs(r["center_hz"]) / 1000 for r in rows])
        leak = np.array([r["final_leakage_max"] for r in rows])
        target = self.target_contrast or (.403 if track == "primary" else .42)
        local = record["local50"].get("contrast")
        local = float(local) if local is not None and np.isfinite(local) else 0.
        local_g = (local - .4) / .02 if track == "local" else 1.
        g = np.r_[(c - self.bc - .001) / .02, (self.bw - w - .3) / 10,
                  (c[self.i50] - target) / .02, (.01 - sat) / .01,
                  (2 - center) / 2, (.005 - leak) / .005, local_g]
        violation = np.maximum(-g, 0)
        objective = (.95 * w[self.i50] + .05 * w.mean()) / 100
        feasible = (np.all(c >= self.bc) and np.all(w <= self.bw) and c[self.i50] > .4
                    and np.all(sat <= .01) and np.all(center <= 2) and np.all(leak <= .005)
                    and (track != "local" or local >= .4))
        return dict(feasible=bool(feasible), width50_khz=float(w[self.i50]),
                    contrast50=float(c[self.i50]), local_contrast50=local,
                    min_contrast_gain=float((c - self.bc).min()),
                    min_width_gain_khz=float((self.bw - w).min()),
                    max_violation=float(violation.max()),
                    loss=float(objective + 2 * violation.max() + 5 * np.mean(violation**2))), g

    def smooth_constraints(self, x, track):
        record, pe = self.evaluate(x)
        n = len(self.amplitudes)
        monotone = self.monotone_local and track == "local"
        extra = 160 if monotone else 0
        if not record["valid"]:
            return np.full(6 * n + 2 + extra, -100.)
        c, depletion = centered_notch_features(self.detuning, pe)
        widths = (self.bw - .4) / 1000
        widths[self.i50] = min(widths[self.i50], x[-1] * .1)
        crossing = half_depth_residuals(self.detuning, depletion, c, widths)
        rows = record["rows"]
        sat = np.array([r["satellite_prominence"] for r in rows])
        center = np.array([abs(r["center_hz"]) / 1000 for r in rows])
        leak = np.array([r["final_leakage_max"] for r in rows])
        local = record["local50"].get("contrast")
        local = float(local) if local is not None and np.isfinite(local) else 0.
        target = self.target_contrast or (.403 if track == "primary" else .42)
        shape_constraints = []
        if monotone:
            # A rising central profile avoids discontinuously switching which
            # tiny local extremum is called the nearest shoulder during a step.
            radius = np.arange(0, .40001, .005)
            profiles = [np.interp(sign * radius, self.detuning, pe[self.i50])
                        for sign in (-1, 1)]
            shape_constraints = np.concatenate([np.diff(p) for p in profiles]) / .01
            local_g = (min(p[-1] - p[0] for p in profiles) - .401) / .02
        else:
            local_g = (local - .4) / .02 if track == "local" else 1.
        return np.r_[(c - self.bc - .001) / .02, crossing.ravel() / .05,
                     (c[self.i50] - target) / .02, (.01 - sat) / .01,
                     (2 - center) / 2, (.005 - leak) / .005,
                     local_g, shape_constraints]

    def export_ranked(self, track, name):
        ranked = []
        for record in self.records.values():
            quality, _ = self.quality(record, track)
            ranked.append({**record, "quality": quality, "loss": quality["loss"]})
        ranked.sort(key=lambda r: r["loss"])
        feasible = sorted((r for r in ranked if r["quality"]["feasible"]),
                          key=lambda r: r["quality"]["width50_khz"])
        save(self.directory / name, dict(candidates=feasible[:20], ranked=ranked[:40],
                                        evaluations=len(self.records), new_simulations=self.count))
        return ranked, feasible


def initial_population(seeds, size, rng, track):
    preferred = seeds[0] if track == "primary" else seeds[1]
    population = [preferred, seeds[1] if track == "primary" else seeds[0]]
    for i in range(size - 2):
        seed = seeds[i % len(seeds)]
        if i < size // 2:
            sigma = [.001, .004, .015, .05][i % 4]
            z = seed + rng.normal(0, sigma, 26)
        elif i < 3 * size // 4:
            fraction = rng.random()
            z = fraction * seeds[0] + (1 - fraction) * seeds[1] + rng.normal(0, .025, 26)
        else:
            z = seed + rng.normal(0, .12, 26)
            z[0] = rng.random()
            z[23:25] = rng.random(2)
        population.append(np.clip(z, 0, 1))
    return np.array(population)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("global", "local"), required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--track", choices=("primary", "local"), default="primary")
    parser.add_argument("--start", type=Path)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=110112)
    parser.add_argument("--extra-amplitudes", nargs="*", type=float, default=[])
    parser.add_argument("--fine", action="store_true")
    parser.add_argument("--target-contrast", type=float)
    parser.add_argument("--monotone-local", action="store_true")
    parser.add_argument("--trust-radius", type=float)
    args = parser.parse_args()
    directory = OUT / args.campaign
    amplitudes = GLOBAL_AMPS if args.stage == "global" else LOCAL_AMPS
    amplitudes = np.unique(np.r_[amplitudes, args.extra_amplitudes])
    detuning = np.linspace(-.5, .5, 401) if args.fine else (GLOBAL_DET if args.stage == "global" else LOCAL_DET)
    engine = Engine(directory, amplitudes, detuning, .25 if args.fine else .5,
                    args.budget, args.target_contrast, args.monotone_local)
    save(directory / "run.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    source_paths = [Path(__file__), *MODEL_PATHS, ROOT / "echospec/analysis/notch_constraints.py",
                    ROOT / "echospec/analysis/dip_metrics.py", ROOT / "scripts/optimize_flexible_50mhz_echo.py"]
    snapshot = directory / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in source_paths:
        (snapshot / path.name).write_bytes(path.read_bytes())
    seeds = [to_normalized(json.loads((SOURCE / name).read_text())["pulse"])
             for name in ("repaired_flexible_phase.json", "cleaner_repaired.json")]
    if args.start:
        start = json.loads(args.start.read_text())
        if "candidates" in start:
            start = (start["candidates"] or start["ranked"])[0]
        initial = to_normalized(start["pulse"])
    else:
        initial = seeds[0 if args.track == "primary" else 1]
    status = "budget exhausted"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        try:
            if args.stage == "global":
                population = initial_population(seeds, args.population, np.random.default_rng(args.seed), args.track)
                population[0] = initial
                save(directory / "initial_population.json", population.tolist())

                def loss(z):
                    record, _ = engine.evaluate(z)
                    return engine.quality(record, args.track)[0]["loss"]

                def callback(xk, convergence):
                    ranked, feasible = engine.export_ranked(args.track, "checkpoint.json")
                    print(json.dumps(dict(evaluations=len(engine.records), feasible=len(feasible),
                                          best=ranked[0]["quality"])), flush=True)
                    return False

                result = differential_evolution(loss, [(0, 1)] * 26, init=population,
                                                workers=pool.map, updating="deferred", polish=False,
                                                maxiter=args.generations, tol=0, atol=0,
                                                mutation=(.4, .95), recombination=.65,
                                                seed=args.seed, callback=callback)
            else:
                initial_record, _ = engine.evaluate(initial)
                initial_width = engine.quality(initial_record, args.track)[0]["width50_khz"]
                x0 = np.r_[initial, np.clip((initial_width + 2) / 100, .435, 1.8)]
                local_bounds = [(0, 1)] * 26
                if args.trust_radius:
                    local_bounds = [(max(0, z - args.trust_radius),
                                     min(1, z + args.trust_radius)) for z in initial]
                gradient_cache = {}

                def jacobian(x):
                    key = hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
                    if key not in gradient_cache:
                        base = engine.smooth_constraints(x, args.track)
                        points, steps = [], []
                        for j in range(27):
                            step = 2e-5 if x[j] < (1 if j < 26 else 1.8) - 2e-5 else -2e-5
                            point = x.copy()
                            point[j] += step
                            points.append(point)
                            steps.append(step)
                        values = list(pool.map(lambda p: engine.smooth_constraints(p, args.track), points))
                        gradient_cache[key] = np.column_stack([(v - base) / h for v, h in zip(values, steps, strict=True)])
                    return gradient_cache[key]

                def callback(x):
                    record, _ = engine.evaluate(x)
                    print(json.dumps(dict(evaluations=len(engine.records), target_width_khz=x[-1] * 100,
                                          quality=engine.quality(record, args.track)[0],
                                          minimum_smooth_constraint=float(engine.smooth_constraints(x, args.track).min()))), flush=True)
                    engine.export_ranked(args.track, "checkpoint.json")

                result = minimize(lambda x: x[-1], x0, jac=lambda x: np.r_[np.zeros(26), 1.],
                                  method="SLSQP", bounds=local_bounds + [(.435, 1.8)],
                                  constraints={"type": "ineq", "fun": lambda x: engine.smooth_constraints(x, args.track),
                                               "jac": jacobian}, callback=callback,
                                  options={"maxiter": args.iterations, "ftol": 2e-5, "disp": True})
            status = str(result.message)
        except EvaluationBudget:
            pass
    ranked, feasible = engine.export_ranked(args.track, "candidates.json")
    save(directory / "completion.json", dict(status=status, evaluations=len(engine.records),
                                             feasible=len(feasible), best=ranked[0]["quality"]))
    print("DONE", args.campaign, status, ranked[0]["quality"], flush=True)


if __name__ == "__main__":
    main()
