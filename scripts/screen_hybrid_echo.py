"""Held-out dense screening before expensive multilevel report validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.simulation.flexible_echo import FlexibleEchoPulse  # noqa: E402
from scripts.optimize_50mhz_echo import metrics, peak_scale  # noqa: E402
from scripts.report_flexible_50mhz_echo import (  # noqa: E402
    assess,
    local_shoulder_metrics,
    save_json,
    sha,
)
from scripts.report_hybrid_echo import OUT, SOURCE, HybridCache  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    candidate_text = args.candidate.read_text()
    record = json.loads(candidate_text)
    if "candidates" in record:
        record = (record["candidates"] or record["ranked"])[0]
    (OUT / "screens").mkdir(exist_ok=True)
    snapshot = OUT / "screens" / f"{args.name}_input.json"
    save_json(snapshot, record)
    pulse = FlexibleEchoPulse(**record["pulse"])
    norm, cache = peak_scale(pulse), HybridCache(OUT)
    with np.load(SOURCE / "validated_spectra.npz") as old:
        amplitudes, detuning = old["rabi_mhz"], old["detuning_mhz"]
        baseline = metrics(old["baseline"], amplitudes, detuning)
    population, diagnostics, source = cache.native(pulse, amplitudes / norm, detuning)
    rows = metrics(population, amplitudes, detuning)
    fine_detuning = np.linspace(-.5, .5, 801)
    fine, fine_diag, fine_source = cache.native(pulse, np.array([50 / norm]), fine_detuning, dt_ns=.125)
    fine50 = metrics(fine, np.array([50.]), fine_detuning)[0]
    local50 = local_shoulder_metrics(fine_detuning, fine[1, 0])
    settings = dict(amplitude_min_mhz=5., amplitude_max_mhz=60.,
                    contrast_min_at50=.4, no_regression_width_khz_tolerance=0.,
                    no_regression_contrast_tolerance=0., max_center_khz=2.,
                    max_satellite_prominence=.01, max_final_leakage=.005)
    quality = assess(rows, baseline, settings, fine50)
    wg = np.array([(b["fwhm_hz"] - r["fwhm_hz"]) / 1000 for b, r in zip(baseline, rows, strict=True)])
    cg = np.array([r["contrast"] - b["contrast"] for b, r in zip(baseline, rows, strict=True)])
    margins = dict(minimum_width_gain_khz=float(wg.min()), width_gain_amplitude_mhz=float(amplitudes[wg.argmin()]),
                   minimum_contrast_gain=float(cg.min()), contrast_gain_amplitude_mhz=float(amplitudes[cg.argmin()]))
    result = dict(name=args.name, pulse=record["pulse"], track=record.get("track"),
                  fine50=fine50, local50=local50, constraints=quality, margins=margins,
                  rows=rows, diagnostics=diagnostics, fine_diagnostics=fine_diag,
                  caches=[source, fine_source], candidate_source=str(args.candidate),
                  candidate_snapshot=str(snapshot), candidate_snapshot_sha256=sha(snapshot),
                  generator_sha256=sha(Path(__file__)))
    save_json(OUT / "screens" / f"{args.name}.json", result)
    print(json.dumps(dict(name=args.name, width50_khz=fine50["fwhm_hz"] / 1000,
                          contrast50=fine50["contrast"], local50=local50,
                          quality=quality, margins=margins)), flush=True)


if __name__ == "__main__":
    main()
