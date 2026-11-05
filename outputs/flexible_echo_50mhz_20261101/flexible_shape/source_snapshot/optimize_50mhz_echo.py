"""Reproducible 50 MHz single-echo search with a contrast and sidelobe floor.

This writes a separate exploratory output directory; paper artifacts are unused.
The original qutrit Lindblad integrator and FWHM definition are reused unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.analysis.dip_metrics import dip_metrics  # noqa: E402
from echospec.simulation.polynomial_echo import (  # noqa: E402
    PolynomialEchoPulse,
    simulate_polynomial_echo,
)
from echospec.simulation.selective_echo import (  # noqa: E402
    PhaseEchoPulse,
    SelectiveEchoPulse,
)

OUT = ROOT / "outputs/single_echo_50mhz_20261101"
TRAIN_AMPS = np.array([40., 45., 50., 55., 60.])
DET = np.linspace(-.5, .5, 101)
T1, T2 = 51.24, 7.31
FAMILY = "selective"
NORMALIZATION = "nominal"


def peak_scale(pulse):
    """Actual complex-drive maximum, resolving the smoothed central peak."""
    half = pulse.duration_us / 2
    t = np.unique(np.r_[np.linspace(0, min(.25, half), 2049),
                         np.geomspace(min(.25, half), half, 1025)])
    t = np.r_[-t[:0:-1], t]
    values = abs(pulse.drive(t))
    index = int(np.argmax(values))
    if index == 0 or index == len(t) - 1:
        return float(values[index])
    result = minimize_scalar(lambda v: -abs(pulse.drive(v)),
                             bounds=(t[index-1], t[index+1]), method="bounded",
                             options={"xatol": 1e-13})
    return float(-result.fun)


def make_pulse(parameters, family):
    if family == "phase":
        length, logc, power, phase, logphase = parameters
        return PhaseEchoPulse(duration_us=length, cutoff=10**logc, power=power,
                              phase_rad=phase, phase_scale_us=10**logphase)
    if family == "polynomial":
        length, logc, power = parameters
        return PolynomialEchoPulse(duration_us=length, cutoff=10**logc, power=power)
    length, logs, logcap, logcore, power = parameters
    return SelectiveEchoPulse(duration_us=length, tail_scale_mhz=10**logs,
                              tail_cap_mhz=10**logcap, core_us=10**logcore,
                              edge_power=power)


def metrics(populations, amplitudes, detuning):
    rows = []
    for amp, pe, pf in zip(amplitudes, populations[1], populations[2], strict=True):
        row = dict(rabi_mhz=float(amp), **dip_metrics(detuning, pe))
        minima, props = find_peaks(-pe, prominence=0.)
        # Exclude only the nearest-to-zero minimum, never every central minimum.
        other = np.ones(len(minima), dtype=bool)
        if len(minima):
            other[np.argmin(abs(detuning[minima]))] = False
        row.update(satellite_prominence=float(props["prominences"][other].max(initial=0)),
                   final_leakage_max=float(pf.max()),
                   population_at_zero=float(np.interp(0, detuning, pe)))
        rows.append(row)
    return rows


def score(rows):
    if not all(r["accepted"] for r in rows):
        return 1e6
    widths = np.array([r["fwhm_hz"] / 1000 for r in rows])
    contrast = np.array([r["contrast"] for r in rows])
    satellites = max(r["satellite_prominence"] for r in rows)
    center = max(abs(r["center_hz"]) / 1000 for r in rows)
    # Every training amplitude must preserve absolute contrast >= 0.4.
    return (float(widths.max()) + 5e4 * float(np.maximum(.405 - contrast, 0).max())**2
            + 1e5 * max(0., satellites - .01)**2 + 50 * max(0., center - 2)**2)


def evaluate(parameters):
    pulse = make_pulse(parameters, FAMILY)
    peak = peak_scale(pulse)
    scale = peak if NORMALIZATION == "peak" else 1.
    populations, diagnostics = simulate_polynomial_echo(pulse, TRAIN_AMPS / scale, DET,
                                                        t1_us=T1, t2_us=T2, dt_ns=.5)
    rows = metrics(populations, TRAIN_AMPS, DET)
    loss = score(rows)
    record = dict(family=FAMILY, parameters=list(parameters), pulse=asdict(pulse),
                  loss=loss, rows=rows, diagnostics=diagnostics,
                  normalization=NORMALIZATION, raw_peak_scale=peak)
    key = hashlib.sha256(np.asarray(parameters, dtype=float).tobytes()).hexdigest()[:20]
    (OUT / "evaluations" / f"{FAMILY}_{key}.json").write_text(json.dumps(record, indent=2))
    return loss


def main():
    global FAMILY, NORMALIZATION, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["polynomial", "selective", "phase"], default="selective")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--normalization", choices=["nominal", "peak"], default="peak")
    args = parser.parse_args()
    FAMILY = args.family
    NORMALIZATION = args.normalization
    if NORMALIZATION == "peak":
        OUT = OUT / "peak"
    (OUT / "evaluations").mkdir(parents=True, exist_ok=True)
    if FAMILY == "polynomial":
        bounds = [(12, 60), (-4.3, -2.5), (1, 12)]
        seed = [30, -3, 5]
    elif FAMILY == "selective":
        bounds = [(8, 50), (-2.1, -.25), (-.7, .7), (-2.25, -1.0), (1.0, 5)]
        seed = [25, -1.1, 0, -1.82, 2]
    else:
        bounds = [(15, 55), (-4.0, -2.8), (1, 6), (-3.0, 3.0), (-1, 1)]
        seed = [35.47, -3.615, 1.72, 0, 0]
    # Threads share family settings and the read-only compiled native kernel.
    from multiprocessing.pool import ThreadPool

    counter = 0

    def callback(x, convergence):
        nonlocal counter
        counter += 1
        print(json.dumps(dict(generation=counter, loss=evaluate(x), parameters=list(x))),
              flush=True)

    with ThreadPool(args.workers) as pool:
        result = differential_evolution(evaluate, bounds, x0=seed, popsize=7,
                                        maxiter=args.iterations, seed=20261101,
                                        workers=pool.map, updating="deferred",
                                        polish=False, callback=callback, tol=.002)
    records = [json.loads(p.read_text()) for p in
               (OUT / "evaluations").glob(f"{FAMILY}_*.json")]
    best = min(records, key=lambda r: r["loss"])
    best["evaluations"] = len(records)
    best["optimizer_loss"] = float(result.fun)
    (OUT / f"{FAMILY}_selected.json").write_text(json.dumps(best, indent=2))
    print(json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
