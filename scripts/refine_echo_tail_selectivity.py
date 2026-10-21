"""Repeat tail optimization with a penalty for competing spectral dips.

The unconstrained pulse family and physical model are unchanged. This stage
adds a selectivity objective and retains every trial's population arrays.
"""

from __future__ import annotations

import argparse
import hashlib
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
from scripts import optimize_echo_tail as opt  # noqa: E402

OUT = opt.OUT / "selectivity"


def satellite_metrics(populations, detuning):
    """Prominence of competing Pe minima at least 50 kHz from resonance."""
    details = []
    for pe in populations[1]:
        indices, props = find_peaks(-pe, prominence=0.)
        mask = abs(detuning[indices]) >= .05
        locations = detuning[indices][mask]
        prominence = props["prominences"][mask]
        details.append(dict(max_prominence=float(prominence.max()) if prominence.size else 0.,
                            detuning_mhz=locations.tolist(),
                            prominence=prominence.tolist()))
    return dict(max_satellite_prominence=max(d["max_prominence"] for d in details),
                satellites=details)


def objective(parameters):
    key = hashlib.sha256(np.asarray(parameters, dtype=np.float64).tobytes()).hexdigest()
    path = OUT / "evaluations" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["loss"]
    f, _ = opt.envelope(np.r_[0., np.geomspace(1e-5, 15., 2000)], parameters)
    peak = float(f.max())
    if peak > 1.000001:
        metrics = dict(all_accepted=False, worst_fwhm_khz=1e4, minimum_snr=0.,
                       max_center_khz=1e4, rows=[], target_points=0,
                       max_satellite_prominence=1.)
        loss = 1e5 + 1e5 * (peak - 1)
    else:
        p = opt.simulate(parameters)
        metrics = opt.quality(p, opt.AMPS[opt.TRAIN_INDICES], opt.DETUNING)
        metrics.update(satellite_metrics(p, opt.DETUNING))
        deficit = max(0., 10.5 - metrics["minimum_snr"])
        excess = max(0., metrics["max_satellite_prominence"] - .015)
        center = max(0., metrics["max_center_khz"] - 10.)
        loss = (metrics["worst_fwhm_khz"] + 1e4 * deficit**2 +
                2e5 * excess**2 + 100 * center**2)
        np.savez_compressed(OUT / "evaluations" / f"{key}.npz", populations=p)
    record = dict(parameters=list(parameters), peak_envelope=peak, loss=loss, **metrics)
    path.write_text(json.dumps(opt.prior.clean(record), indent=2, allow_nan=False))
    return loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    (OUT / "evaluations").mkdir(parents=True, exist_ok=True)
    manifest = dict(physical_model=opt.hashes(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    training_amplitude_indices=opt.TRAIN_INDICES,
                    satellite_exclusion_mhz=.05, satellite_optimization_target=.015,
                    satellite_selection_cap=.025, selection_minimum_snr=10.)
    manifest_path = OUT / "provenance.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    manifest_path.write_text(json.dumps(manifest, indent=2))
    rng = np.random.default_rng(20261012)
    # Start with smooth controls and promising trials; retain diverse tail shapes.
    seeds = [opt.seed_parameters(p) for p in [3, 4, 5, 6, 8, 12, 16, 24]]
    old = [json.loads(p.read_text()) for p in (opt.OUT / "evaluations").glob("*.json")]
    old = sorted([r for r in old if r["all_accepted"] and r["minimum_snr"] >= 10],
                 key=lambda r: r["worst_fwhm_khz"])
    seeds.extend([np.array(r["parameters"]) for r in old[::max(1, len(old) // 8)][:8]])
    seeds = np.array(seeds)
    initial = np.concatenate([seeds, seeds + rng.normal(0, .55, seeds.shape)])
    initial = np.clip(initial, np.array(opt.BOUNDS)[:, 0], np.array(opt.BOUNDS)[:, 1])
    start = time.perf_counter()

    def callback(x, convergence):
        print(f"Selectivity loss={objective(x):.3f}, elapsed={time.perf_counter()-start:.1f}s",
              flush=True)

    with Pool(args.workers) as pool:
        result = differential_evolution(
            objective, opt.BOUNDS, init=initial, maxiter=args.iterations,
            seed=20261012, workers=pool.map, updating="deferred", polish=False,
            callback=callback, tol=.0001, mutation=(.4, 1.), recombination=.7)
    rows = [json.loads(p.read_text()) for p in (OUT / "evaluations").glob("*.json")]
    feasible = [r for r in rows if r["all_accepted"] and r["minimum_snr"] >= 10.
                and r["max_satellite_prominence"] <= .025]
    if not feasible:
        raise RuntimeError("No clean SNR-10 candidate found within this search")
    selected = min(feasible, key=lambda r: r["worst_fwhm_khz"])
    selected.update(evaluations=len(rows), generations=args.iterations, seed=20261012,
                    de_loss=float(result.fun), selection="Minimum worst FWHM with "
                    "SNR >= 10 and competing-dip prominence <= 0.025")
    (opt.OUT / "selected_clean.json").write_text(json.dumps(selected, indent=2))
    print("Selected clean", selected["worst_fwhm_khz"], selected["minimum_snr"],
          selected["max_satellite_prominence"], flush=True)


if __name__ == "__main__":
    main()
