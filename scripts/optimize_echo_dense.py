"""Optimize a single echo tail using every amplitude and the whole spectrum.

Physical waveform/model functions are imported unchanged from optimize_echo_tail.
All trials use 100 amplitudes; feasibility is required on 10.101--100 MHz.
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
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import optimize_echo_tail as model  # noqa: E402

OUT = ROOT / "outputs/dense_spectrum_optimization"
DETUNING = np.linspace(-.5, .5, 101)
OPERATING_MIN_MHZ = 10.
T2_WIDTH_MHZ = 1 / (np.pi * 7.31)
SEED = 20261013


def spectrum_metrics(populations, amplitudes, detuning):
    """Same linewidth/SNR, plus whole-spectrum shape and competing minima."""
    rows = []
    edges = abs(detuning) >= .4
    for amp, pe, pf in zip(amplitudes, populations[1], populations[2], strict=True):
        row = dict(amplitude_mhz=float(amp),
                   **model.prior.previous.base.spectroscopy_quality(detuning, pe),
                   max_final_pf=float(pf.max()))
        minima, props = find_peaks(-pe, prominence=0.)
        central = np.flatnonzero(abs(detuning[minima]) <= .02)
        other = np.ones(len(minima), dtype=bool)
        if central.size:
            nearest = central[np.argmin(abs(detuning[minima[central]]))]
            other[nearest] = False
        prominence = props["prominences"][other]
        row["satellite_prominence"] = float(prominence.max(initial=0.))
        row["satellite_detunings_mhz"] = detuning[minima[other]].tolist()
        row["satellite_prominences"] = prominence.tolist()
        background = np.polyval(np.polyfit(detuning[edges], pe[edges], 1), detuning)
        # Fit only the depth to a fixed, centered T2-width line; every frequency
        # sample contributes. This does not redefine the reported half-depth FWHM.
        template = 1 / (1 + (2 * detuning / T2_WIDTH_MHZ)**2)
        template -= np.polyval(np.polyfit(detuning[edges], template[edges], 1), detuning)
        template /= template[np.argmin(abs(detuning))]
        depth = background - pe
        scale = max(.02, float(depth.max()))
        fitted_depth = max(0., float(np.dot(depth, template) / np.dot(template, template)))
        row["whole_spectrum_error"] = float(np.mean(
            ((depth - fitted_depth * template) / scale)**2))
        # A single dip should rise away from the central basin on either side.
        left = pe[detuning < -.02][::-1]
        right = pe[detuning > .02]
        reversal = sum(float(np.maximum(-np.diff(side), 0.).sum())
                       for side in [left, right])
        row["total_outward_reversal"] = reversal
        rows.append(row)
    active = [r for r in rows if r["amplitude_mhz"] >= OPERATING_MIN_MHZ]
    valid = all(r["accepted"] and np.isfinite(r["contrast_snr"]) for r in active)
    return dict(rows=rows, all_accepted=valid,
                worst_fwhm_khz=max(r["fwhm_hz"] / 1000 for r in active) if valid else 1e4,
                minimum_snr=min(r["contrast_snr"] for r in active) if valid else 0.,
                max_center_khz=max(abs(r["center_hz"]) / 1000 for r in active) if valid else 1e4,
                max_satellite_prominence=max(r["satellite_prominence"] for r in active),
                max_outward_reversal=max(r["total_outward_reversal"] for r in active),
                mean_shape_error=float(np.mean([r["whole_spectrum_error"] for r in rows])),
                worst_shape_error=max(r["whole_spectrum_error"] for r in active))


def loss_from_metrics(metrics):
    if not metrics["all_accepted"]:
        return 1e6
    snr = max(0., 10.15 - metrics["minimum_snr"])
    satellite = max(0., metrics["max_satellite_prominence"] - .01)
    reversal = max(0., metrics["max_outward_reversal"] - .025)
    center = max(0., metrics["max_center_khz"] - 2.)
    return float(metrics["worst_fwhm_khz"] + 200 * metrics["mean_shape_error"] +
                 50 * metrics["worst_shape_error"] + 2e4 * snr**2 +
                 4e5 * satellite**2 + 1e4 * reversal**2 + 100 * center**2)


def computation_hashes():
    hashes = model.hashes()
    for fn in [spectrum_metrics, loss_from_metrics, objective]:
        hashes[fn.__name__] = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
    return hashes


def objective(parameters):
    key = hashlib.sha256(np.asarray(parameters, dtype=np.float64).tobytes()).hexdigest()
    path = OUT / "evaluations" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["loss"]
    envelope = model.envelope(np.r_[0., np.geomspace(1e-5, 15., 3000)], parameters)[0]
    peak = float(envelope.max())
    if peak > 1.000001:
        metrics = dict(all_accepted=False, worst_fwhm_khz=1e4, minimum_snr=0.,
                       max_satellite_prominence=1., rows=[])
        loss = 1e6 + 1e5 * (peak - 1)
    else:
        populations = model.simulate(parameters, amplitudes=model.AMPS,
                                     detuning=DETUNING, dt_ns=1.)
        metrics = spectrum_metrics(populations, model.AMPS, DETUNING)
        loss = loss_from_metrics(metrics)
        np.savez_compressed(path.with_suffix(".npz"), populations=populations,
                            rabi_mhz=model.AMPS, detuning_mhz=DETUNING)
    record = dict(parameters=list(parameters), peak_envelope=peak, loss=loss, **metrics)
    path.write_text(json.dumps(model.prior.clean(record), indent=2, allow_nan=False))
    return loss


def records():
    return [json.loads(p.read_text()) for p in (OUT / "evaluations").glob("*.json")]


def feasible(record):
    return (record["all_accepted"] and record["minimum_snr"] >= 10. and
            record["max_satellite_prominence"] <= .015 and
            record["max_center_khz"] <= 2.)


def checkpoint(stage, elapsed):
    trials = records()
    good = [r for r in trials if feasible(r)]
    best_loss = min(trials, key=lambda r: r["loss"])
    selected = min(good, key=lambda r: r["worst_fwhm_khz"]) if good else best_loss
    brief = {k: selected[k] for k in ["worst_fwhm_khz", "minimum_snr",
                                      "max_satellite_prominence", "loss"]}
    status = dict(stage=stage, elapsed_seconds=elapsed, evaluations=len(trials),
                  feasible_candidates=len(good), best=brief)
    (OUT / "progress.json").write_text(json.dumps(status, indent=2))
    (OUT / "selected_screen.json").write_text(json.dumps(
        dict(selected, selection_stage=stage, feasible=bool(good), evaluations=len(trials)),
        indent=2))
    print(json.dumps(status), flush=True)
    return selected


def optimize(iterations, workers):
    rng = np.random.default_rng(SEED)
    seeds = [model.seed_parameters(p) for p in [2, 3, 4, 5, 6, 8, 12]]
    seeds.append(np.array(json.loads((model.OUT / "selected_clean.json").read_text())[
        "parameters"]))
    seeds = np.array(seeds)
    initial = np.concatenate([seeds, seeds + rng.normal(0, .25, seeds.shape),
                              seeds + rng.normal(0, .65, seeds.shape)])
    bounds = np.array(model.BOUNDS)
    initial = np.clip(initial, bounds[:, 0], bounds[:, 1])
    start = time.perf_counter()
    generation = 0

    def callback(x, convergence):
        nonlocal generation
        generation += 1
        checkpoint(f"DE generation {generation}", time.perf_counter() - start)

    with Pool(workers) as pool:
        differential_evolution(objective, model.BOUNDS, init=initial,
                               maxiter=iterations, seed=SEED, workers=pool.map,
                               updating="deferred", polish=False, callback=callback,
                               mutation=(.35, .8), recombination=.75, tol=.0001)
        # Parallel deterministic coordinate refinement, still using all amplitudes.
        best = min(records(), key=lambda r: r["loss"])
        for step in [.12, .06, .03]:
            for sweep in range(2):
                x = np.array(best["parameters"])
                trials = [np.clip(x + sign * step * np.eye(6)[i], bounds[:, 0], bounds[:, 1])
                          for i in range(6) for sign in [-1, 1]]
                pool.map(objective, trials)
                best = min(records(), key=lambda r: r["loss"])
                checkpoint(f"coordinate {step:g}, sweep {sweep + 1}",
                           time.perf_counter() - start)
    checkpoint("complete", time.perf_counter() - start)


def validate_shortlist():
    trials = [r for r in records() if feasible(r)]
    if not trials:
        trials = [min(records(), key=lambda r: r["loss"])]
    by_width = sorted(trials, key=lambda r: r["worst_fwhm_khz"])
    by_loss = sorted(trials, key=lambda r: r["loss"])
    ranked = [r for pair in zip(by_loss, by_width, strict=True) for r in pair]
    # Preserve both the whole-spectrum optimum and the narrowest feasible line.
    # Distinct shapes avoid near-duplicate final tests.
    shortlist = []
    for trial in ranked:
        if all(np.linalg.norm(np.array(trial["parameters"]) - r["parameters"]) > .04
               for r in shortlist):
            shortlist.append(trial)
        if len(shortlist) == 4:
            break
    results = []
    for i, trial in enumerate(shortlist):
        path = OUT / f"candidate_{i}_full.npz"
        config = dict(parameters=trial["parameters"], dt_ns=.5,
                      computation_sha256=computation_hashes())
        encoded = json.dumps(config, sort_keys=True)
        if path.exists():
            with np.load(path) as data:
                assert str(data["config"]) == encoded
                populations = data["populations"]
        else:
            populations = model.simulate(trial["parameters"], amplitudes=model.AMPS,
                                         detuning=model.DETUNING, dt_ns=.5)
            np.savez_compressed(path, populations=populations, rabi_mhz=model.AMPS,
                                detuning_mhz=model.DETUNING, config=encoded)
        metrics = spectrum_metrics(populations, model.AMPS, model.DETUNING)
        metrics.update(parameters=trial["parameters"], candidate_index=i,
                       feasible=feasible(metrics), loss=loss_from_metrics(metrics))
        results.append(metrics)
        (OUT / f"candidate_{i}_full.json").write_text(json.dumps(
            model.prior.clean(metrics), indent=2, allow_nan=False))
        print("Validated", i, metrics["worst_fwhm_khz"], metrics["minimum_snr"],
              metrics["max_satellite_prominence"], flush=True)
    good = [r for r in results if r["feasible"]]
    selected = min(good or results, key=lambda r: r["loss"])
    selected.update(full_amplitude_points=100, operating_min_mhz=OPERATING_MIN_MHZ,
                    selected_after_full_grid_refinement=True, evaluations=len(records()))
    (OUT / "selected.json").write_text(json.dumps(model.prior.clean(selected), indent=2))
    narrow = min(good or results, key=lambda r: r["worst_fwhm_khz"])
    (OUT / "selected_narrow.json").write_text(json.dumps(model.prior.clean(narrow), indent=2))


def refine_narrow(workers):
    """Explore the width/selectivity tradeoff without demanding monotone wings.

The T2 template, SNR, satellite and center penalties are retained. This optional
branch removes only the separate outward-reversal penalty; every trial still
uses the full 100-amplitude spectrum and remains in the original trial log.
"""
    def secondary_loss(record):
        if not record["all_accepted"]:
            return record["loss"]
        return record["loss"] - 1e4 * max(0., record["max_outward_reversal"] - .025)**2

    start = time.perf_counter()
    initial_count = len(records())
    bounds = np.array(model.BOUNDS)
    with Pool(workers) as pool:
        for step in [.15, .075, .035, .015]:
            for sweep in range(2):
                best = min(records(), key=secondary_loss)
                x = np.array(best["parameters"])
                trials = [np.clip(x + sign * step * np.eye(6)[i], bounds[:, 0], bounds[:, 1])
                          for i in range(6) for sign in [-1, 1]]
                pool.map(objective, trials)
                checkpoint(f"narrow branch {step:g}, sweep {sweep + 1}",
                           time.perf_counter() - start)
    good = [r for r in records() if feasible(r)]
    best = min(good, key=lambda r: r["worst_fwhm_khz"])
    p = model.simulate(best["parameters"], amplitudes=model.AMPS,
                       detuning=model.DETUNING, dt_ns=.5)
    measured = spectrum_metrics(p, model.AMPS, model.DETUNING)
    measured.update(parameters=best["parameters"], full_map_filename="narrow_refined_full.npz",
                    feasible=feasible(measured), loss=loss_from_metrics(measured))
    config = json.dumps(dict(parameters=best["parameters"], dt_ns=.5,
                             computation_sha256=computation_hashes()), sort_keys=True)
    np.savez_compressed(OUT / "narrow_refined_full.npz", populations=p,
                        rabi_mhz=model.AMPS, detuning_mhz=model.DETUNING, config=config)
    (OUT / "narrow_refinement.json").write_text(json.dumps(dict(
        additional_trials=len(records()) - initial_count,
        removed_term="Outward reversal penalty only; retain whole-spectrum template, "
        "SNR, competing-dip and center penalties.",
        source_sha256=hashlib.sha256(inspect.getsource(refine_narrow).encode()).hexdigest(),
        refined_result=model.prior.clean(measured)), indent=2))
    prior = json.loads((OUT / "selected_narrow.json").read_text())
    if measured["feasible"] and measured["worst_fwhm_khz"] < prior["worst_fwhm_khz"]:
        (OUT / "selected_narrow_before_refinement.json").write_text(json.dumps(prior, indent=2))
        (OUT / "selected_narrow.json").write_text(json.dumps(model.prior.clean(measured), indent=2))
    print("Narrow refinement", measured["worst_fwhm_khz"], measured["minimum_snr"],
          measured["max_satellite_prominence"], flush=True)


def validate_narrow_margin():
    """Check the next trial if the narrowest misses selectivity on a finer grid."""
    first = json.loads((OUT / "narrow_refinement.json").read_text())["refined_result"]
    checked = [first]
    good = sorted([r for r in records() if feasible(r)], key=lambda r: r["worst_fwhm_khz"])
    for i, record in enumerate(good[:5]):
        if np.array_equal(record["parameters"], first["parameters"]):
            measured = first
        else:
            p = model.simulate(record["parameters"], amplitudes=model.AMPS,
                               detuning=model.DETUNING, dt_ns=.5)
            measured = spectrum_metrics(p, model.AMPS, model.DETUNING)
            measured.update(parameters=record["parameters"], feasible=feasible(measured),
                            full_map_filename=f"narrow_verified_{i}_full.npz")
            encoded = json.dumps(dict(parameters=record["parameters"], dt_ns=.5,
                                       computation_sha256=computation_hashes()), sort_keys=True)
            np.savez_compressed(OUT / measured["full_map_filename"], populations=p,
                                rabi_mhz=model.AMPS, detuning_mhz=model.DETUNING, config=encoded)
            checked.append(measured)
        if measured["feasible"]:
            previous = json.loads((OUT / "selected_narrow.json").read_text())
            if measured["worst_fwhm_khz"] < previous["worst_fwhm_khz"]:
                (OUT / "selected_narrow_before_refinement.json").write_text(json.dumps(previous, indent=2))
                (OUT / "selected_narrow.json").write_text(json.dumps(
                    model.prior.clean(measured), indent=2))
            print("Narrow candidate verified", measured["worst_fwhm_khz"],
                  measured["minimum_snr"], measured["max_satellite_prominence"], flush=True)
            break
    (OUT / "narrow_finalist_checks.json").write_text(json.dumps(
        model.prior.clean(checked), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["optimize", "validate", "narrow-refine",
                                            "validate-narrow"], default="optimize")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    (OUT / "evaluations").mkdir(parents=True, exist_ok=True)
    manifest = dict(computation_sha256=computation_hashes(), amplitudes_mhz=model.AMPS.tolist(),
                    screen_detuning_mhz=DETUNING.tolist(), screen_dt_ns=1., seed=SEED,
                    required_operating_range_mhz=[10., 100.], snr_target=10.15,
                    snr_selection_min=10., satellite_target=.01, satellite_selection_max=.015,
                    center_selection_max_khz=2., shape_target_fwhm_khz=T2_WIDTH_MHZ * 1000,
                    notes="All 100 amplitudes contribute to the spectrum objective; "
                    "SNR, FWHM and selectivity extrema use amplitudes >= 10 MHz.")
    path = OUT / "search_provenance.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Search definition changed"
    path.write_text(json.dumps(manifest, indent=2))
    if args.stage == "optimize":
        optimize(args.iterations, args.workers)
    elif args.stage == "narrow-refine":
        refine_narrow(args.workers)
    elif args.stage == "validate-narrow":
        validate_narrow_margin()
    else:
        validate_shortlist()


if __name__ == "__main__":
    main()
