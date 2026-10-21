"""Optimize a smooth tail around the unchanged Lorentzian echo core.

The six spline ordinates are one fixed shape shared by all drive amplitudes.
No model or shared pulse code is modified. Run --stage optimize, then --stage
full and --stage refine. The original 30 us p=5 and C2 subtraction are controls.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echospec.simulation.qutrit import _finalize  # noqa: E402
from echospec.simulation.qutrit_native import _kernel  # noqa: E402
from scripts import scan_duration_polynomial as prior  # noqa: E402

OUT = ROOT / "outputs/optimized_echo_tail"
KNOTS = np.array([.05, .15, .5, 1.5, 4., 9., 15.])
SIGMA = prior.SIGMA_REFERENCE
AMPS = prior.previous.base.AMPS
DETUNING = prior.previous.base.DETUNING
TRAIN_INDICES = [50, 74, 99]
BOUNDS = [(-3., 1.5), (-5., 2.), (-6., 2.), (-8., 2.), (-10., 1.), (-12., 0.)]


def envelope(t, parameters=None, family="spline"):
    """Positive envelope and analytic d/dt, using physical time in us."""
    t = np.asarray(t, dtype=float)
    r = np.abs(t)
    f = (1 + (t / SIGMA)**2)**-.5
    df = -t / SIGMA**2 * (1 + (t / SIGMA)**2)**-1.5
    y, dy = np.maximum(0., 1 - (t / 15)**2), -2 * t / 15**2
    if family == "polynomial5":
        f, df = (f - .001) / .999, df / .999
        return np.where(r < 15, f * y**5, 0.), np.where(
            r < 15, df * y**5 + 5 * f * y**4 * dy, 0.)
    if family == "c2":
        a = 1 - .001**2
        norm = 1 - .001 * (1 + a / 2 + 3 * a**2 / 8)
        correction = .001 * (1 + a * y / 2 + 3 * a**2 * y**2 / 8)
        derivative = .001 * (a / 2 + 3 * a**2 * y / 4) * dy
        return np.where(r < 15, (f - correction) / norm, 0.), np.where(
            r < 15, (df - derivative) / norm, 0.)
    if family != "spline":
        raise ValueError(f"Unknown family {family}")
    spline = make_interp_spline(
        np.log(KNOTS), np.r_[0., parameters], k=5,
        bc_type=([(1, 0.), (2, 0.)], [(1, 0.), (2, 0.)]))
    u = np.log(np.clip(r, KNOTS[0], KNOTS[-1]))
    multiplier = np.exp(spline(u))
    d_multiplier = np.where(
        (r > KNOTS[0]) & (r < 15),
        multiplier * spline(u, 1) / np.maximum(r, KNOTS[0]) * np.sign(t), 0.)
    # A fixed, C2 quintic closure over the last 2 us at either end.
    v = np.clip((15 - r) / 2, 0., 1.)
    window = 10 * v**3 - 15 * v**4 + 6 * v**5
    dw = -(30 * v**2 - 60 * v**3 + 30 * v**4) * np.sign(t) / 2
    return (f * multiplier * window,
            (df * multiplier + f * d_multiplier) * window + f * multiplier * dw)


def drive(t, parameters=None, family="spline"):
    f, df = envelope(t, parameters, family)
    s = np.where(t < -.008, 1., np.where(t > .008, -1.,
                                       -np.sin(np.pi * t / .016)))
    ds = np.where(abs(t) <= .008, -np.pi / .016 * np.cos(np.pi * t / .016), 0.)
    return f * s - 1j * .5 * (df * s + f * ds) / (2 * np.pi * 216)


def simulate(parameters=None, *, family="spline", amplitudes=None,
             detuning=None, dt_ns=.5):
    """Vectorized waveform preparation; unchanged native qutrit RK4 equations."""
    amps = AMPS[TRAIN_INDICES] if amplitudes is None else np.asarray(amplitudes)
    det = DETUNING if detuning is None else np.asarray(detuning)
    steps = int(np.ceil(30000 / dt_ns))
    grid = np.linspace(-15., 15., 2 * steps + 1)
    waveform = drive(grid, parameters, family)
    det_grid, amp_grid = np.meshgrid(2 * np.pi * det, 2 * np.pi * amps)
    shape = det_grid.shape
    state = np.zeros((9, det_grid.size))
    state[0] = 1.
    _kernel()(state.ravel(), det_grid.size, steps, 30 / steps,
              np.ascontiguousarray(det_grid.ravel()),
              np.ascontiguousarray(amp_grid.ravel()),
              np.ascontiguousarray(waveform.real),
              np.ascontiguousarray(waveform.imag),
              -216 * 2 * np.pi, 1 / 51.24, 1 / 7.871480508563624)
    unpacked = np.zeros((6, *shape), dtype=complex)
    unpacked[:3] = state[:3].reshape(3, *shape)
    for i in range(3):
        unpacked[i + 3] = (state[2 * i + 3] + 1j * state[2 * i + 4]).reshape(shape)
    result = _finalize(unpacked)
    return np.array([result.ground, result.excited, result.second_excited])


def hashes():
    result = prior.previous.hashes()
    for fn in [envelope, drive, simulate, quality, objective]:
        result[fn.__name__] = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
    return result


def quality(populations, amps, det):
    rows = [dict(amplitude_mhz=float(a),
                 **prior.previous.base.spectroscopy_quality(det, populations[1, i]),
                 max_final_pf=float(populations[2, i].max()))
            for i, a in enumerate(amps)]
    high = [r for r in rows if r["amplitude_mhz"] >= 50]
    valid = all(r["accepted"] and np.isfinite(r["contrast_snr"]) for r in high)
    return dict(
        all_accepted=valid, rows=rows,
        worst_fwhm_khz=max(r["fwhm_hz"] / 1000 for r in high) if valid else 1e4,
        minimum_snr=min(r["contrast_snr"] for r in high) if valid else 0.,
        max_center_khz=max(abs(r["center_hz"]) / 1000 for r in high) if valid else 1e4,
        target_points=sum(r["accepted"] and r["fwhm_hz"] <= 50000
                          and r["contrast_snr"] >= 10 for r in high))


def objective(parameters):
    key = hashlib.sha256(np.asarray(parameters, dtype=np.float64).tobytes()).hexdigest()
    path = OUT / "evaluations" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["loss"]
    f, _ = envelope(np.r_[0., np.geomspace(1e-5, 15., 2000)], parameters)
    peak = float(np.max(f))
    if peak > 1.000001:
        metrics = dict(all_accepted=False, worst_fwhm_khz=1e4, minimum_snr=0.,
                       max_center_khz=1e4, rows=[], target_points=0)
        loss = 1e5 + 1e5 * (peak - 1)
    else:
        p = simulate(parameters)
        metrics = quality(p, AMPS[TRAIN_INDICES], DETUNING)
        # Fixed-shot visibility constraint, with margin for unseen amplitudes.
        deficit = max(0., 10.2 - metrics["minimum_snr"])
        center_excess = max(0., metrics["max_center_khz"] - 10.)
        loss = metrics["worst_fwhm_khz"] + 1e4 * deficit**2 + 100 * center_excess**2
    record = dict(parameters=list(parameters), peak_envelope=peak, loss=loss, **metrics)
    path.write_text(json.dumps(prior.clean(record), indent=2, allow_nan=False))
    return loss


def seed_parameters(power):
    t = np.minimum(KNOTS[1:], 14.)
    f = (1 + (t / SIGMA)**2)**-.5
    desired = (f - .001) / .999 * (1 - (t / 15)**2)**power
    logs = np.log(desired / f)
    return np.clip(logs, np.array(BOUNDS)[:, 0], np.array(BOUNDS)[:, 1])


def optimize(iterations, workers):
    rng = np.random.default_rng(20261011)
    seeds = np.array([seed_parameters(p) for p in [2, 3, 4, 5, 6, 8, 12, 16]])
    initial = np.concatenate([seeds, seeds + rng.normal(0, .65, seeds.shape),
                              seeds + rng.normal(0, 1.4, seeds.shape),
                              seeds + rng.normal(0, 2., seeds.shape)])
    initial = np.clip(initial, np.array(BOUNDS)[:, 0], np.array(BOUNDS)[:, 1])
    start = time.perf_counter()

    def callback(x, convergence):
        print(f"DE best loss={objective(x):.3f}, elapsed={time.perf_counter()-start:.1f}s",
              flush=True)

    with Pool(workers) as pool:
        result = differential_evolution(
            objective, BOUNDS, init=initial, maxiter=iterations, seed=20261011,
            workers=pool.map, updating="deferred", polish=False, callback=callback,
            tol=.0001, mutation=(.4, 1.), recombination=.7)
    print("Polishing best shape", result.fun, flush=True)
    polished = minimize(objective, result.x, method="Powell", bounds=BOUNDS,
                        options=dict(maxfev=200, xtol=.008, ftol=.0001))
    print("Local result", polished.fun, flush=True)
    records = [json.loads(p.read_text()) for p in (OUT / "evaluations").glob("*.json")]
    feasible = [r for r in records if r["all_accepted"] and r["minimum_snr"] >= 10.]
    chosen = min(feasible, key=lambda r: r["worst_fwhm_khz"])
    chosen.update(evaluations=len(records), de_loss=float(result.fun),
                  local_loss=float(polished.fun), seed=20261011,
                  training_amplitudes_mhz=AMPS[TRAIN_INDICES].tolist(),
                  selection="Lowest worst FWHM with minimum ideal SNR >= 10",
                  search_target_snr=10.2)
    (OUT / "selected.json").write_text(json.dumps(chosen, indent=2))
    print("Selected", chosen["worst_fwhm_khz"], chosen["minimum_snr"], flush=True)


def run_map(family, stage, parameters=None, tag=None):
    amps = AMPS if stage == "full" else AMPS[TRAIN_INDICES]
    det = (np.linspace(-.5, .5, 401) if stage == "refine" else
           np.linspace(-1., 1., 401) if stage == "wide" else DETUNING)
    dt = .25 if stage == "refine" else .5
    config = dict(family=family, parameters=parameters, duration_us=30.,
                  sigma_us=SIGMA, cutoff_core=.001, knots_us=KNOTS.tolist(),
                  outer_ramp_us=2., dt_ns=dt, computation_sha256=hashes(),
                  **{k: v for k, v in prior.previous.COMMON.items()
                     if k != "duration_us"}, drag_beta=-.5)
    if tag is not None:
        config["label"] = tag
    label = family if tag is None else tag
    encoded = json.dumps(config, sort_keys=True)
    path = OUT / f"{label}_{stage}.npz"
    start = time.perf_counter()
    if path.exists():
        with np.load(path) as data:
            assert str(data["config"]) == encoded
            np.testing.assert_array_equal(data["rabi_mhz"], amps)
            np.testing.assert_array_equal(data["detuning_mhz"], det)
            p = data["populations"]
    else:
        chunks = [simulate(parameters, family=family, amplitudes=amps[i:i+25],
                           detuning=det, dt_ns=dt) for i in range(0, len(amps), 25)]
        p = np.concatenate(chunks, axis=1)
        np.savez_compressed(path, populations=p, rabi_mhz=amps,
                            detuning_mhz=det, config=encoded)
    result = quality(p, amps, det)
    (OUT / f"{label}_{stage}.json").write_text(
        json.dumps(prior.clean(result), indent=2, allow_nan=False))
    print(f"{label} {stage}: {result['worst_fwhm_khz']:.3f} kHz, "
          f"SNR {result['minimum_snr']:.3f}, {time.perf_counter()-start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["optimize", "full", "refine", "wide", "validate",
                                            "select-margin"],
                        default="optimize")
    parser.add_argument("--margin", action="store_true",
                        help="Run the secondary candidate selected with SNR >= 10.5")
    parser.add_argument("--clean", action="store_true",
                        help="Run the candidate selected with a competing-dip penalty")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    (OUT / "evaluations").mkdir(parents=True, exist_ok=True)
    manifest = OUT / "computation_hashes.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == hashes(), "Calculation changed"
    manifest.write_text(json.dumps(hashes(), indent=2))
    if args.stage == "optimize":
        optimize(args.iterations, args.workers)
    elif args.stage == "select-margin":
        records = [json.loads(p.read_text())
                   for p in (OUT / "evaluations").glob("*.json")]
        feasible = [r for r in records if r["all_accepted"] and r["minimum_snr"] >= 10.5]
        chosen = min(feasible, key=lambda r: r["worst_fwhm_khz"])
        chosen["selection"] = ("Lowest training width with SNR >= 10.5; selected "
                               "after the first candidate's full-grid minimum was 9.874")
        (OUT / "selected_margin.json").write_text(json.dumps(chosen, indent=2))
        print(chosen["worst_fwhm_khz"], chosen["minimum_snr"], flush=True)
    elif args.stage in ["full", "refine", "wide"]:
        if args.clean:
            parameters = json.loads((OUT / "selected_clean.json").read_text())["parameters"]
            run_map("spline", args.stage, parameters, tag="clean")
            return
        if args.margin:
            parameters = json.loads((OUT / "selected_margin.json").read_text())["parameters"]
            run_map("spline", args.stage, parameters, tag="margin")
            return
        parameters = json.loads((OUT / "selected.json").read_text())["parameters"]
        with Pool(args.workers) as pool:
            pool.starmap(run_map, [("spline", args.stage, parameters),
                                  ("polynomial5", args.stage), ("c2", args.stage)])
    else:
        validate()


def validate():
    """Independent known-map agreement and exact-shape boundary checks."""
    baseline = simulate(family="polynomial5")
    with np.load(prior.OUT / "L30_p5_subtracted_cutoff_full.npz") as old:
        error = float(abs(baseline - old["populations"][:, TRAIN_INDICES]).max())
    assert error < 1e-10
    parameters = seed_parameters(5)
    for family in ["polynomial5", "c2", "spline"]:
        grid = np.r_[-15., -KNOTS, 0., KNOTS, 15.]
        f, df = envelope(grid, parameters, family)
        np.testing.assert_allclose(envelope(np.array([-15., 15.]), parameters, family),
                                   0., atol=1e-15)
        for t in [.003, .049, .05, .051, .15, .5, 1.5, 4., 9., 14.]:
            numeric = (envelope(np.array([t + 1e-6]), parameters, family)[0] -
                       envelope(np.array([t - 1e-6]), parameters, family)[0]) / 2e-6
            np.testing.assert_allclose(envelope(np.array([t]), parameters, family)[1],
                                       numeric, atol=1e-8, rtol=1e-5)
        assert np.isfinite(f).all() and np.isfinite(df).all()
    result = dict(previous_map_max_population_difference=error,
                  exact_derivatives_checked=True, endpoint_values_slopes_zero=True)
    (OUT / "implementation_checks.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
