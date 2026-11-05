"""Bounded minimal-displacement repair using the final report frequency grid."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from echospec.simulation.polynomial_echo import PolynomialEchoPulse  # noqa: E402
from scripts.optimize_50mhz_echo import metrics, peak_scale  # noqa: E402
from scripts.optimize_decoupled_50mhz_echo import clean  # noqa: E402
from scripts.optimize_flexible_50mhz_echo import (  # noqa: E402
    BOUNDS,
    NAMES,
    encode,
    make_pulse,
)
from scripts.report_flexible_50mhz_echo import Cache, model_sources  # noqa: E402

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
AMPS = np.array([5., 6., 7., 10., 15., 16., 16.5, 17., 17.5, 18.,
                 20., 21.5, 30., 40., 50., 60.])
DENSE_AMPS = np.arange(5., 60.001, 0.5)
DET = np.linspace(-0.5, 0.5, 401)
ACTIVE = np.array([0, 1, 2, 3, 7, 10, 11, 16])
MAX_EVALUATIONS = 120
STEP = 2e-5


class EvaluationBudget(Exception):
    """Stop scheduling candidate simulations at the explicitly bounded budget."""


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def compare(rows: list[dict], baseline: list[dict]) -> tuple[dict, np.ndarray]:
    if not all(row["accepted"] for row in rows):
        return dict(feasible=False, repair_targets_pass=False, invalid=True), np.full(
            5*len(rows)+1, -100.0
        )
    contrast = np.array([row["contrast"] for row in rows])
    reference_contrast = np.array([row["contrast"] for row in baseline])
    width = np.array([row["fwhm_hz"]/1000 for row in rows])
    reference_width = np.array([row["fwhm_hz"]/1000 for row in baseline])
    satellite = np.array([row["satellite_prominence"] for row in rows])
    center = np.array([abs(row["center_hz"])/1000 for row in rows])
    leakage = np.array([row["final_leakage_max"] for row in rows])
    index50 = next(i for i, row in enumerate(rows) if row["rabi_mhz"] == 50.)
    gain = contrast-reference_contrast
    width_gain = reference_width-width
    inequalities = np.r_[100*(gain-0.0001), width_gain-0.05,
                         100*(contrast[index50]-0.4005), 100*(0.01-satellite),
                         (2-center)/2, 100*(0.005-leakage)]
    guards = (np.all(satellite <= 0.01) and np.all(center <= 2)
              and np.all(leakage <= 0.005))
    true_user = (np.all(gain >= 0) and np.all(width_gain >= 0)
                 and contrast[index50] > 0.4)
    repair_pass = (np.all(gain >= 0.0001) and np.all(width_gain >= 0.05)
                   and contrast[index50] > 0.4005 and guards)
    return dict(
        feasible=bool(true_user and guards), user_constraints_pass=bool(true_user),
        guardrails_pass=bool(guards), repair_targets_pass=bool(repair_pass),
        width50_khz=float(width[index50]), contrast50=float(contrast[index50]),
        minimum_contrast_gain=float(gain.min()), minimum_width_gain_khz=float(width_gain.min()),
        max_satellite=float(satellite.max()), max_center_khz=float(center.max()),
        max_leakage=float(leakage.max()), min_inequality=float(inequalities.min()),
        contrast_regression_amplitudes_mhz=[row["rabi_mhz"] for row, value in
                                          zip(rows, gain, strict=True) if value < 0],
        width_regression_amplitudes_mhz=[row["rabi_mhz"] for row, value in
                                       zip(rows, width_gain, strict=True) if value < 0],
    ), inequalities


def main() -> None:
    source = BASE / "rotating_adaptive/candidates.json"
    selected = json.loads(source.read_text())["candidates"][0]
    initial_parameters = encode(selected["pulse"])
    lower, upper = np.array(BOUNDS).T
    scales = upper-lower
    initial = ((initial_parameters-lower)/scales)[ACTIVE]
    local_bounds = list(zip(np.maximum(initial-0.03, 0),
                            np.minimum(initial+0.03, 1), strict=True))
    cache = Cache(BASE)
    original = PolynomialEchoPulse()
    baseline_populations, baseline_diagnostics, baseline_cache = cache.native(
        original, DENSE_AMPS/peak_scale(original), DET, dt_ns=0.25
    )
    dense_baseline = metrics(baseline_populations, DENSE_AMPS, DET)
    baseline = [dense_baseline[i] for i in np.searchsorted(DENSE_AMPS, AMPS)]
    source_paths = [Path(__file__), ROOT / "scripts/optimize_flexible_50mhz_echo.py",
                    ROOT / "scripts/report_flexible_50mhz_echo.py",
                    ROOT / "scripts/optimize_50mhz_echo.py",
                    ROOT / "echospec/analysis/dip_metrics.py",
                    ROOT / "echospec/simulation/flexible_echo.py",
                    ROOT / "echospec/simulation/decoupled_echo.py",
                    ROOT / "echospec/simulation/polynomial_echo.py",
                    ROOT / "echospec/simulation/qutrit_rk4.cpp"]
    configuration = dict(
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        initial_parameters=initial_parameters.tolist(), active_indices=ACTIVE.tolist(),
        active_names=[NAMES[i] for i in ACTIVE], normalized_local_bounds=local_bounds,
        amplitudes_mhz=AMPS.tolist(), dense_amplitudes_mhz=DENSE_AMPS.tolist(),
        detuning_mhz=DET.tolist(), dt_ns=0.25, t1_us=51.24, t2_us=7.31,
        baseline=baseline, baseline_cache=baseline_cache,
        objective="1e4 times squared displacement in original normalized bounds",
        repair_targets="C>=baseline+.0001; C50>.4005; W<=baseline-.05kHz; unchanged guards",
        final_acceptance="Zero-tolerance C/width regression, C50>.4, unchanged guards",
        max_candidate_evaluations=MAX_EVALUATIONS, max_iterations=5, workers=3,
        jacobian="Central differences, cached physical evaluations", step=STEP,
        source_hashes={str(path.relative_to(ROOT)):
                       hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
    )
    configuration = clean(configuration)
    config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    (OUT / "evaluations").mkdir(exist_ok=True)
    save(OUT / "configuration.json", configuration)
    snapshot = OUT / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in source_paths:
        (snapshot / (path.name+".source")).write_bytes(path.read_bytes())
    records, gradients = {}, {}
    lock = threading.Lock()
    submitted = 0

    def displacement(z):
        return float(1e4*np.sum((np.asarray(z)-initial)**2))

    def evaluate(z):
        nonlocal submitted
        z = np.asarray(z, dtype=float)
        key = hashlib.sha256(z.tobytes()).hexdigest()
        with lock:
            if key in records:
                return records[key]
            if submitted >= MAX_EVALUATIONS:
                raise EvaluationBudget
            submitted += 1
        parameters = initial_parameters.copy()
        parameters[ACTIVE] = lower[ACTIVE]+z*scales[ACTIVE]
        pulse = make_pulse(parameters)
        norm = peak_scale(pulse)
        populations, diagnostics, cache_name = cache.native(
            pulse, AMPS/norm, DET, dt_ns=0.25
        )
        rows = metrics(populations, AMPS, DET)
        quality, inequalities = compare(rows, baseline)
        record = dict(
            pulse=asdict(pulse), pulse_type="FlexibleEchoPulse", parameters=parameters.tolist(),
            normalized_active_parameters=z.tolist(), peak_scale=norm, rows=rows,
            quality=quality, inequalities=inequalities.tolist(), diagnostics=diagnostics,
            native_cache=cache_name, objective=displacement(z), configuration_sha256=config_hash,
        )
        save(OUT / "evaluations" / f"{key}.json", record)
        with lock:
            records[key] = record
            if len(records) % 10 == 0:
                best = max(records.values(), key=lambda r: min(r["inequalities"]))
                print(json.dumps(dict(evaluations=len(records), best=best["quality"])), flush=True)
        return record

    def callback(z):
        record = evaluate(z)
        print(json.dumps(dict(iteration_candidate=record["quality"],
                              displacement=record["objective"])), flush=True)

    status = "evaluation budget reached"
    with ThreadPoolExecutor(max_workers=3) as pool:
        def jacobian(z):
            z = np.asarray(z)
            key = hashlib.sha256(z.tobytes()).hexdigest()
            if key in gradients:
                return gradients[key]
            points = []
            for j in range(len(ACTIVE)):
                plus, minus = z.copy(), z.copy()
                plus[j] += STEP
                minus[j] -= STEP
                points.extend((plus, minus))
            values = list(pool.map(evaluate, points))
            derivative = np.column_stack([
                (np.array(values[2*j]["inequalities"])
                 - np.array(values[2*j+1]["inequalities"]))/(2*STEP)
                for j in range(len(ACTIVE))
            ])
            gradients[key] = derivative
            return derivative

        try:
            result = minimize(
                displacement, initial, jac=lambda z: 2e4*(np.asarray(z)-initial),
                method="SLSQP", bounds=local_bounds,
                constraints={"type": "ineq", "fun": lambda z: evaluate(z)["inequalities"],
                             "jac": jacobian}, callback=callback,
                options={"maxiter": 5, "ftol": 1e-10, "disp": True},
            )
            status = str(result.message)
        except EvaluationBudget:
            pass
    passing = sorted((r for r in records.values() if r["quality"]["repair_targets_pass"]),
                     key=lambda r: r["objective"])
    summary = dict(evaluations=len(records), optimizer_status=status,
                   repair_targets_pass_count=len(passing), configuration_sha256=config_hash)
    if not passing:
        summary["best_attempt"] = max(records.values(), key=lambda r: min(r["inequalities"]))
        summary["accepted"] = False
        save(OUT / "repair_summary.json", summary)
        print("NO REPAIR WITHIN BUDGET", summary["best_attempt"]["quality"], flush=True)
        return
    selected = passing[0]
    save(OUT / "selected_training_repair.json", selected)
    pulse = make_pulse(selected["parameters"])
    norm = peak_scale(pulse)
    dense, diagnostics, cache_name = cache.native(pulse, DENSE_AMPS/norm, DET, dt_ns=0.25)
    dense_rows = metrics(dense, DENSE_AMPS, DET)
    quality, inequalities = compare(dense_rows, dense_baseline)
    final = dict(
        **{key: value for key, value in selected.items()
           if key not in ("rows", "quality", "inequalities", "diagnostics", "native_cache")},
        name="repaired_flexible_phase", label="Repaired flexible phase", rows=dense_rows,
        quality=quality, accepted=quality["feasible"], inequalities=inequalities.tolist(),
        diagnostics=diagnostics, native_cache=cache_name,
        amplitudes_mhz=DENSE_AMPS.tolist(), detuning_mhz=DET.tolist(), dt_ns=0.25,
        baseline=dense_baseline, baseline_cache=baseline_cache,
        baseline_diagnostics=baseline_diagnostics, model_sources=model_sources(pulse),
        repair_training_record=selected,
    )
    save(BASE / "repaired_flexible_phase.json", final)
    summary.update(accepted=quality["feasible"], dense_quality=quality,
                   selected_objective=selected["objective"], native_dense_cache=cache_name)
    save(OUT / "repair_summary.json", summary)
    print("DENSE AUDIT", quality, flush=True)


if __name__ == "__main__":
    main()
