"""Numerical, equal-area, and shoulder controls for the selected outer tapers."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import make_outer_tail_ramp_scan as scan  # noqa: E402
from scripts.check_echo_edge_controls import shoulder_metrics  # noqa: E402

INDICES = [50, 74, 99]


def array_audit():
    max_trace_error, minimum, maximum = 0., 1., 0.
    inventory = scan.cases()
    for case in inventory:
        with np.load(case["path"]) as data:
            cfg = json.loads(str(data["config"]))
            for key, value in scan.base.COMMON.items():
                assert cfg[key] == value, (case["case"], key)
            assert cfg["drag_beta"] == -.5 and cfg["num_steps_per_half"] == 30000
            np.testing.assert_array_equal(data["rabi_mhz"], scan.base.AMPS)
            np.testing.assert_array_equal(data["detuning_mhz"], scan.base.DETUNING)
            p = data["populations"]
            assert p.shape == (3, 100, 201) and np.isfinite(p).all()
            max_trace_error = max(max_trace_error, float(abs(p.sum(axis=0)-1).max()))
            minimum, maximum = min(minimum, float(p.min())), max(maximum, float(p.max()))
    assert max_trace_error < 1e-9 and minimum >= -1e-9 and maximum <= 1+1e-9
    result = dict(primary_maps=len(inventory), shape_per_map=[3, 100, 201],
                  max_trace_error=max_trace_error, minimum_population=minimum,
                  maximum_population=maximum, all_parameters_and_grids_match=True)
    (scan.OUT / "array_checks.json").write_text(json.dumps(result, indent=2))


def validate_case(s):
    scan.configure()
    c, window, ramp = s["cutoff"], s["window"], s["ramp_us"]
    scan.base.run_case(c, window, "quintic", ramp,
                       amplitudes=scan.base.AMPS[INDICES],
                       detuning=np.linspace(-.5, .5, 401), steps=60000,
                       suffix="_refined")
    with np.load(scan.OUT / f"{s['case']}.npz") as data:
        coarse = data["populations"][:, INDICES]
        x = data["detuning_mhz"]
    with np.load(scan.OUT / f"{s['case']}_refined.npz") as data:
        fine = data["populations"]
        xf = data["detuning_mhz"]
    quality = scan.base.spectroscopy_quality
    old = [quality(x, y) for y in coarse[1]]
    time_only = [quality(x, y) for y in fine[1, :, ::2]]
    both = [quality(xf, y) for y in fine[1]]
    result = dict(case=s["case"], ramp_us=ramp, cutoff=c,
                  amplitudes_mhz=scan.base.AMPS[INDICES].tolist(),
                  refined_dt_ns=.25, refined_detuning_step_khz=2.5,
                  max_population_difference=float(abs(coarse-fine[:, :, ::2]).max()))
    for key, unit in [("fwhm_hz", "fwhm_change_hz"),
                      ("center_hz", "center_change_hz"),
                      ("contrast_snr", "snr_change")]:
        result[f"time_max_{unit}"] = max(abs(a[key]-b[key])
                                         for a, b in zip(old, time_only, strict=True))
        result[f"grid_max_{unit}"] = max(abs(a[key]-b[key])
                                         for a, b in zip(time_only, both, strict=True))
    area_rows = json.loads((scan.OUT / "waveform_metrics.json").read_text())
    ratio = next(a["area_fraction"] for a in area_rows if a["case"] == s["case"])
    match_label = window + "_area_matched_hard"
    scan.base.run_case(c, match_label, "none", 0.,
                       amplitudes=scan.base.AMPS[INDICES]*ratio, steps=60000)
    with np.load(scan.OUT / f"{scan.base.case_name(c, match_label)}.npz") as data:
        controls = [quality(data["detuning_mhz"], y) for y in data["populations"][1]]
    result["area_fraction"] = ratio
    result["equal_area_controls"] = [
        dict(amplitude_mhz=float(a), hard_amplitude_mhz=float(a*ratio),
             ramp_width_khz=r["fwhm_hz"]/1000, hard_width_khz=h["fwhm_hz"]/1000,
             ramp_snr=r["contrast_snr"], hard_snr=h["contrast_snr"])
        for a, r, h in zip(scan.base.AMPS[INDICES], both, controls, strict=True)]
    result["passed"] = bool(
        result["max_population_difference"] < 1e-4
        and result["time_max_fwhm_change_hz"] < 1000
        and result["grid_max_fwhm_change_hz"] < 1000
        and result["time_max_center_change_hz"] < 500
        and result["grid_max_center_change_hz"] < 500)
    (scan.OUT / f"validation_{s['case']}.json").write_text(json.dumps(result, indent=2))
    return result


def wide_check(s):
    scan.configure()
    c, w = s["cutoff"], s["window"]
    key = scan.base.run_case(c, w, s["kind"], s["ramp_us"],
                             amplitudes=scan.base.AMPS[[50, 99]],
                             detuning=np.linspace(-2, 2, 801), suffix="_wide")
    result = []
    with np.load(scan.OUT / f"{key}.npz") as data:
        for index, a in enumerate(data["rabi_mhz"]):
            for shoulder in [.3, .45, .6, .8, 1.]:
                metrics = shoulder_metrics(data["detuning_mhz"],
                                           data["populations"][1, index], shoulder)
                result.append(dict(case=s["case"], cutoff=c, window=w,
                                   amplitude_mhz=float(a), **metrics))
    return result


def main():
    scan.configure()
    array_audit()
    summaries = json.loads((scan.OUT / "summary.json").read_text())
    chosen = scan.choose_ramps(summaries)
    best = [next(s for s in summaries if s["case"] == r["best_tradeoff_case"])
            for r in chosen]
    for choice in chosen:
        best.append(next(s for s in summaries if s["cutoff"] == choice["cutoff"]
                         and s["kind"] == "quintic"
                         and s["ramp_us"] == choice["best_with_untouched_center_ramp_us"]))
    best = list({s["case"]: s for s in best}.values())
    validation, shoulders = [], []
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(validate_case, s) for s in best]
        for job in as_completed(jobs):
            result = job.result()
            validation.append(result)
            print(result["case"], "convergence", result["passed"], flush=True)
        wide = best + [s for s in summaries if s["kind"] in {"none", "polynomial"}]
        jobs = [pool.submit(wide_check, s) for s in wide]
        for job in as_completed(jobs):
            shoulders.extend(job.result())
    (scan.OUT / "validation.json").write_text(json.dumps(validation, indent=2))
    (scan.OUT / "shoulder_checks.json").write_text(json.dumps(shoulders, indent=2))
    if not all(r["passed"] for r in validation):
        raise RuntimeError("A convergence check exceeded the specified tolerance")


if __name__ == "__main__":
    main()
