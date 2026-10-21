"""Scan the exponent of the finite polynomial window at fixed 30 us duration.

Run: .venv/bin/python scripts/make_polynomial_power_scan.py --workers 2
All maps use 100 amplitudes, 201 detunings, and the existing beta=-0.5 model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import make_echo_edge_ramp_comparison as base  # noqa: E402
from scripts.make_outer_tail_ramp_scan import (  # noqa: E402
    MODEL_FILES,
    operating_summary,
    pareto_flags,
)

OUT = ROOT / "outputs/polynomial_power_30us_scan"
COMMON = dict(base.COMMON, duration_us=30.)
POWERS = [3., 4., 6., 8., 12.]


def name(cutoff, power):
    label = "hard" if power is None else f"p{power:g}"
    return f"c{cutoff:g}_{label}"


def hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in MODEL_FILES}


def run_case(cutoff, power, *, amplitudes=None, detuning=None,
             steps=30000, suffix=""):
    amps = base.AMPS if amplitudes is None else np.asarray(amplitudes)
    det = base.DETUNING if detuning is None else np.asarray(detuning)
    config = dict(**COMMON, cutoff=cutoff, drag_beta=-.5,
                  edge_window_kind="none" if power is None else "polynomial",
                  edge_ramp_us=0., edge_window_power=3. if power is None else power,
                  num_steps_per_half=steps)
    fingerprint = json.dumps(config, sort_keys=True)
    key = name(cutoff, power) + suffix
    path = OUT / f"{key}.npz"
    if path.exists():
        with np.load(path) as old:
            if (str(old["config"]) == fingerprint
                    and np.array_equal(old["rabi_mhz"], amps)
                    and np.array_equal(old["detuning_mhz"], det)):
                return key
        raise ValueError(f"Cached parameter mismatch: {path}")
    start = time.perf_counter()
    chunks = []
    for offset in range(0, len(amps), 25):
        result = base.simulate_qutrit_map(
            **config, rabi_mhz=amps[offset:offset+25], detuning_mhz=det,
            integration_backend="native")
        chunks.append(np.array([result.ground, result.excited, result.second_excited]))
    populations = np.concatenate(chunks, axis=1)
    np.savez_compressed(path, populations=populations, rabi_mhz=amps,
                        detuning_mhz=det, config=fingerprint,
                        integration_backend="native")
    print(f"{key}: {len(amps)} amplitudes, {time.perf_counter()-start:.1f} s",
          flush=True)
    return key


def inventory():
    cases = []
    for path in sorted(OUT.glob("c*.npz")):
        with np.load(path) as data:
            config = json.loads(str(data["config"]))
            if (config["num_steps_per_half"] != 30000
                    or not np.array_equal(data["rabi_mhz"], base.AMPS)
                    or not np.array_equal(data["detuning_mhz"], base.DETUNING)):
                continue
            power = (None if config["edge_window_kind"] == "none"
                     else config["edge_window_power"])
            cases.append(dict(case=path.stem, cutoff=config["cutoff"], power=power))
    return sorted(cases, key=lambda c: (c["cutoff"], c["power"] or 0))


def export():
    rows, summaries, baseline = [], [], []
    min_p, max_p, trace_error = 1., 0., 0.
    for case in inventory():
        with np.load(OUT / f"{case['case']}.npz") as data:
            p = data["populations"]
            cfg = json.loads(str(data["config"]))
            assert all(cfg[k] == v for k, v in COMMON.items())
            assert cfg["drag_beta"] == -.5 and p.shape == (3, 100, 201)
            assert np.isfinite(p).all()
            min_p, max_p = min(min_p, float(p.min())), max(max_p, float(p.max()))
            trace_error = max(trace_error, float(abs(p.sum(axis=0)-1).max()))
            group = [dict(**case, amplitude_mhz=float(amp),
                          **base.spectroscopy_quality(data["detuning_mhz"], p[1, i]),
                          max_final_pf=float(p[2, i].max()))
                     for i, amp in enumerate(data["rabi_mhz"])]
            rows.extend(group)
            summaries.append(dict(**case, **operating_summary(group),
                                  at_100_mhz=group[-1]))
            if case["power"] in [None, 3.]:
                label = "hard" if case["power"] is None else "polynomial"
                old_path = (ROOT / "outputs/echo_edge_ramp_30us_comparison" /
                            f"{base.case_name(case['cutoff'], label)}.npz")
                with np.load(old_path) as old:
                    np.testing.assert_array_equal(data["rabi_mhz"], old["rabi_mhz"])
                    np.testing.assert_array_equal(data["detuning_mhz"],
                                                  old["detuning_mhz"])
                    error = float(abs(old["populations"]-p).max())
                    assert error < 1e-10
                    baseline.append(dict(case=case["case"],
                                         previous=str(old_path.relative_to(ROOT)),
                                         max_population_difference=error))
    assert min_p >= 0 and max_p <= 1 and trace_error < 1e-10
    for c in base.CUTOFFS:
        group = [s for s in summaries if s["cutoff"] == c]
        for s, flag in zip(group, pareto_flags(group), strict=True):
            s["pareto"] = flag
    with (OUT / "metrics.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2))
    (OUT / "array_checks.json").write_text(json.dumps(dict(
        primary_maps=len(summaries), shape=[3, 100, 201], rows=len(rows),
        min_population=min_p, max_population=max_p, trace_error=trace_error,
        previous_baseline_agreement=baseline), indent=2))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--powers", type=float, nargs="+", default=POWERS)
    parser.add_argument("--cutoffs", type=float, nargs="+", default=base.CUTOFFS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    if any(not np.isfinite(p) or p < 3 for p in args.powers):
        parser.error("Powers must be finite and >= 3")
    if not set(args.cutoffs) <= set(base.CUTOFFS):
        parser.error("Use existing cutoffs .001, .003, .005")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "model_hashes.json"
    if manifest.exists() and json.loads(manifest.read_text()) != hashes():
        raise ValueError("Model changed: use a fresh output directory")
    manifest.write_text(json.dumps(hashes(), indent=2))
    if not args.metrics_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(run_case, c, p) for c in args.cutoffs
                    for p in [None, *sorted(set(args.powers))]]
            for job in as_completed(jobs):
                job.result()
    summaries = export()
    sources = hashes()
    for path in [Path(__file__), ROOT / "scripts/plot_polynomial_power_scan.py",
                 ROOT / "scripts/validate_polynomial_power_scan.py",
                 ROOT / "scripts/make_echo_edge_ramp_comparison.py",
                 ROOT / "scripts/make_outer_tail_ramp_scan.py",
                 ROOT / "echospec/analysis/dip_metrics.py",
                 ROOT / "echospec/analysis/spectroscopy_quality.py"]:
        if path.exists():
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    powers = sorted({s["power"] for s in summaries if s["power"] is not None})
    (OUT / "provenance.json").write_text(json.dumps(dict(
        parameters=dict(COMMON, drag_beta=-.5), powers=powers,
        cutoffs=sorted({s["cutoff"] for s in summaries}),
        amplitude_points=100, amplitude_range_mhz=[0, 100],
        detuning_points=201, detuning_range_mhz=[-.5, .5], dt_ns=.5,
        source_sha256=sources, python_version=sys.version,
        numpy_version=np.__version__,
        optimization="Maximize minimum ideal SNR/FWHM over 50.505 to 100 MHz; "
                     "also report worst width, minimum SNR and Pareto frontier. "
                     "This score is a heuristic, not frequency uncertainty.",
        normalization="Fixed duration, core width, nominal amplitude; no peak "
                      "or area normalization. Hard and p3 baselines rerun.",
        snr="Ideal binary Pe measurement, 500 shots at center and 250 each "
            "at +/-0.45 MHz; no readout error or technical noise.",
        fwhm="Half-depth of central Pe dip relative to a linear background "
             "from |detuning| >= 0.4 MHz; no parametric line fit.",
        reproduce_commands=[
            ".venv/bin/python scripts/make_polynomial_power_scan.py --powers "
            + " ".join(f"{p:g}" for p in powers),
            ".venv/bin/python scripts/validate_polynomial_power_scan.py",
            ".venv/bin/python scripts/plot_polynomial_power_scan.py",
        ],
    ), indent=2))
    for c in base.CUTOFFS:
        group = [s for s in summaries if s["cutoff"] == c]
        if group:
            best = max(group, key=lambda s: s["minimum_snr_per_mhz"])
            print("Best sampled", c, best["case"],
                  best["worst_fwhm_khz"], best["minimum_snr"], flush=True)


if __name__ == "__main__":
    main()
