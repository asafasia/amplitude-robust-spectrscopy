"""Check no-regression between the plotted amplitude samples."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.simulation.polynomial_echo import PolynomialEchoPulse  # noqa: E402
from scripts.optimize_50mhz_echo import metrics, peak_scale  # noqa: E402
from scripts.report_flexible_50mhz_echo import (  # noqa: E402
    assess,
    load_cases,
    save_json,
    sha,
)
from scripts.report_hybrid_echo import OUT, SOURCE, HybridCache  # noqa: E402


def main():
    executed_source_sha256 = sha(Path(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, default=OUT / "selected_candidates.json")
    parser.add_argument("--step", type=float, default=.1)
    parser.add_argument("--min", type=float, default=5.)
    parser.add_argument("--max", type=float, default=60.)
    parser.add_argument("--fine", action="store_true")
    parser.add_argument("--label", default="dense_band")
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cases, _ = load_cases(manifest)
    old = json.loads((SOURCE / "results.json").read_text())
    baseline_pulse = PolynomialEchoPulse(**old["cases"]["baseline"]["pulse"])
    amplitudes = np.round(np.arange(args.min, args.max + args.step / 2, args.step), 10)
    detuning = np.linspace(-.5, .5, 801 if args.fine else 401)
    dt = .125 if args.fine else .25
    cache = HybridCache(OUT)
    print("baseline", len(amplitudes), len(detuning), flush=True)
    base, diagnostics, base_source = cache.native(baseline_pulse, amplitudes / peak_scale(baseline_pulse), detuning, dt_ns=dt)
    baseline = metrics(base, amplitudes, detuning)
    settings = dict(amplitude_min_mhz=args.min, amplitude_max_mhz=args.max,
                    contrast_min_at50=.4, no_regression_width_khz_tolerance=0.,
                    no_regression_contrast_tolerance=0., max_center_khz=2.,
                    max_satellite_prominence=.01, max_final_leakage=.005)
    results = dict(amplitudes_mhz=amplitudes.tolist(), detuning_mhz=detuning.tolist(), dt_ns=dt,
                   baseline_cache=base_source, baseline_diagnostics=diagnostics,
                   baseline=baseline, cases={}, generator_sha256=executed_source_sha256)
    for name in args.names:
        pulse = cases[name]["pulse"]
        print(name, flush=True)
        norm = peak_scale(pulse)
        if args.chunk_size:
            chunks = [amplitudes[i:i + args.chunk_size] for i in range(0, len(amplitudes), args.chunk_size)]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                parts = list(pool.map(lambda a, pulse=pulse, norm=norm:
                                      cache.native(pulse, a / norm, detuning, dt_ns=dt), chunks))
            p = np.concatenate([part[0] for part in parts], axis=1)
            diag, source = [part[1] for part in parts], [part[2] for part in parts]
        else:
            p, diag, source = cache.native(pulse, amplitudes / norm, detuning, dt_ns=dt)
        rows = metrics(p, amplitudes, detuning)
        fine50 = next((r for r in rows if r["rabi_mhz"] == 50), None)
        if fine50 is None:
            p50, _, _ = cache.native(pulse, [50 / peak_scale(pulse)], np.linspace(-.5, .5, 801), dt_ns=.125)
            fine50 = metrics(p50, np.array([50.]), np.linspace(-.5, .5, 801))[0]
        q = assess(rows, baseline, settings, fine50)
        wg = np.array([(b["fwhm_hz"] - r["fwhm_hz"]) / 1000 for b, r in zip(baseline, rows, strict=True)])
        cg = np.array([r["contrast"] - b["contrast"] for b, r in zip(baseline, rows, strict=True)])
        margins = dict(width_gain_khz=float(wg.min()), width_gain_at_mhz=float(amplitudes[wg.argmin()]),
                       contrast_gain=float(cg.min()), contrast_gain_at_mhz=float(amplitudes[cg.argmin()]))
        results["cases"][name] = dict(pulse=cases[name]["selection_record"]["pulse"], rows=rows,
                                     constraints=q, margins=margins, diagnostics=diag, cache=source)
        print(json.dumps(dict(name=name, margins=margins, constraints=q)), flush=True)
        save_json(OUT / f"{args.label}.json", results)


if __name__ == "__main__":
    main()
