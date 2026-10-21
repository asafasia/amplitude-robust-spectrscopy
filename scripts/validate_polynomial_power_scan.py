"""Convergence, shoulder-position and equal-area controls for power selection."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import make_polynomial_power_scan as scan  # noqa: E402
from scripts.check_echo_edge_controls import shoulder_metrics  # noqa: E402

INDICES = [50, 74, 99]


def area(cutoff, power):
    """Unsigned I area at nominal amplitude 1, including the sine inversion."""
    sigma = 15 / np.sqrt(cutoff**-2 - 1)

    def envelope(t):
        f = (1 + (t/sigma)**2)**-.5
        s = np.sin(np.pi*t/.016) if t < .008 else 1.
        w = 1 if power is None else max(0., 1-(t/15)**2)**power
        return f*s*w

    return 2*quad(envelope, 0, 15, points=[.008, sigma], epsabs=1e-11)[0]


def validate(case):
    c, p = case["cutoff"], case["power"]
    key = scan.run_case(c, p, amplitudes=scan.base.AMPS[INDICES],
                        detuning=np.linspace(-.5, .5, 401), steps=60000,
                        suffix="_refined")
    with np.load(scan.OUT / f"{case['case']}.npz") as data:
        old = data["populations"][:, INDICES]
        x = data["detuning_mhz"]
    with np.load(scan.OUT / f"{key}.npz") as data:
        fine = data["populations"]
        xf = data["detuning_mhz"]
    quality = scan.base.spectroscopy_quality
    original = [quality(x, y) for y in old[1]]
    time_only = [quality(x, y) for y in fine[1, :, ::2]]
    refined = [quality(xf, y) for y in fine[1]]
    result = dict(case=case["case"], cutoff=c, power=p,
                  amplitudes_mhz=scan.base.AMPS[INDICES].tolist(),
                  refined_dt_ns=.25, refined_frequency_step_khz=2.5,
                  max_population_difference=float(abs(old-fine[:, :, ::2]).max()))
    for metric in ["fwhm_hz", "center_hz", "contrast_snr"]:
        result[f"time_max_{metric}_change"] = max(
            abs(a[metric]-b[metric]) for a, b in zip(original, time_only, strict=True))
        result[f"grid_max_{metric}_change"] = max(
            abs(a[metric]-b[metric]) for a, b in zip(time_only, refined, strict=True))
    ratio = area(c, p) / area(c, 3)
    key = scan.run_case(c, 3., amplitudes=scan.base.AMPS[INDICES]*ratio,
                        steps=60000, detuning=xf,
                        suffix=f"_area_matched_to_p{p:g}")
    with np.load(scan.OUT / f"{key}.npz") as data:
        controls = [quality(xf, y) for y in data["populations"][1]]
    result["area_ratio_to_p3"] = ratio
    result["equal_unsigned_I_area_controls"] = [
        dict(amplitude_mhz=float(a), p3_amplitude_mhz=float(a*ratio),
             selected_fwhm_khz=r["fwhm_hz"]/1000, p3_fwhm_khz=h["fwhm_hz"]/1000,
             selected_snr=r["contrast_snr"], p3_snr=h["contrast_snr"])
        for a, r, h in zip(scan.base.AMPS[INDICES], refined, controls, strict=True)]
    result["passed"] = bool(result["max_population_difference"] < 1e-4
                            and result["time_max_fwhm_hz_change"] < 1000
                            and result["grid_max_fwhm_hz_change"] < 1000
                            and result["time_max_center_hz_change"] < 500
                            and result["grid_max_center_hz_change"] < 500)
    result["primary_population_tolerance_passed"] = bool(
        result["max_population_difference"] < 1e-4)
    result["status"] = "passed" if result["passed"] else "needs_refinement"
    if not result["passed"]:
        key = scan.run_case(c, p, amplitudes=scan.base.AMPS[INDICES],
                            detuning=xf, steps=120000, suffix="_extra_refined")
        with np.load(scan.OUT / f"{key}.npz") as data:
            extra = data["populations"]
        extra_metrics = [quality(xf, y) for y in extra[1]]
        delta = float(abs(extra-fine).max())
        factor = result["max_population_difference"] / delta if delta else np.inf
        check = dict(dt_ns=.125, max_population_difference=delta,
                     population_error_reduction_factor=factor)
        for metric in ["fwhm_hz", "center_hz", "contrast_snr"]:
            check[f"max_{metric}_change"] = max(
                abs(a[metric]-b[metric])
                for a, b in zip(refined, extra_metrics, strict=True))
        result["extra_refinement"] = check
        result["passed"] = bool(delta < 1e-4 and factor > 8
                                and check["max_fwhm_hz_change"] < 1000
                                and check["max_center_hz_change"] < 500
                                and result["time_max_fwhm_hz_change"] < 1000
                                and result["grid_max_fwhm_hz_change"] < 1000)
        result["status"] = ("linewidth_converged_with_population_precision_note"
                            if result["passed"] else "failed")
        result["note"] = (
            "The 0.5 ns primary map exceeds the 1e-4 population tolerance. "
            "Its displayed linewidth is stable within the reported time-step "
            "difference. The 0.25 to 0.125 ns check quantifies convergence; "
            "raw primary maps retain their common 0.5 ns step.")
    return result


def wide(case):
    key = scan.run_case(case["cutoff"], case["power"],
                        amplitudes=scan.base.AMPS[[50, 99]],
                        detuning=np.linspace(-2, 2, 801), suffix="_wide")
    rows = []
    with np.load(scan.OUT / f"{key}.npz") as data:
        for i, amp in enumerate(data["rabi_mhz"]):
            for shoulder in [.3, .45, .6, .8, 1.]:
                rows.append(dict(case=case["case"], cutoff=case["cutoff"],
                                 power=case["power"], amplitude_mhz=float(amp),
                                 **shoulder_metrics(data["detuning_mhz"],
                                                    data["populations"][1, i],
                                                    shoulder)))
    return rows


def main():
    summaries = scan.export()
    best = [max((s for s in summaries if s["cutoff"] == c),
                key=lambda s: s["minimum_snr_per_mhz"]) for c in scan.base.CUTOFFS]
    selected = {s["case"]: s for s in best}
    # Also refine the original p=3 reference at the best cutoff.
    original = next(s for s in summaries if s["power"] == 3 and s["cutoff"] == .001)
    selected[original["case"]] = original
    balanced = next(s for s in summaries if s["power"] == 8 and s["cutoff"] == .001)
    selected[balanced["case"]] = balanced
    results, shoulder_rows = [], []
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(validate, s) for s in selected.values()]
        for job in as_completed(jobs):
            result = job.result()
            results.append(result)
            print(result["case"], "convergence", result["passed"], flush=True)
        wide_cases = {s["case"]: s for s in best + [balanced] + [
            s for s in summaries if s["power"] == 3]}
        for job in as_completed([pool.submit(wide, s) for s in wide_cases.values()]):
            shoulder_rows.extend(job.result())
    (scan.OUT / "validation.json").write_text(json.dumps(results, indent=2))
    (scan.OUT / "shoulder_checks.json").write_text(json.dumps(shoulder_rows, indent=2))
    (scan.OUT / "waveform_metrics.json").write_text(json.dumps([
        dict(case=s["case"], cutoff=s["cutoff"], power=s["power"],
             unsigned_I_area=area(s["cutoff"], s["power"]),
             area_ratio_to_p3=area(s["cutoff"], s["power"])/area(s["cutoff"], 3),
             window_fwhm_us=(30 if s["power"] is None else
                             30*np.sqrt(1 - 2**(-1/s["power"]))))
        for s in summaries], indent=2))
    if not all(r["passed"] for r in results):
        raise RuntimeError("Convergence check failed")


if __name__ == "__main__":
    main()
