"""Time/grid convergence and explicitly area-matched hard-pulse controls."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echospec.analysis.spectroscopy_quality import spectroscopy_quality  # noqa: E402
from echospec.simulation.qutrit import simulate_qutrit_map  # noqa: E402
from scripts.make_echo_edge_ramp_comparison import (  # noqa: E402
    AMPS,
    OUT,
    WINDOWS,
    add_experiment_arguments,
    case_name,
    configure_experiment,
    integration_steps,
    run_case,
)


def configure_worker(duration_us, output_dir):
    global OUT
    OUT = configure_experiment(duration_us, output_dir)


def check_case(cutoff, window):
    _, kind, ramp = next(w for w in WINDOWS if w[0] == window)
    key = case_name(cutoff, window)
    indices = [50, 74, 99]
    run_case(cutoff, window, kind, ramp, amplitudes=AMPS[indices],
             detuning=np.linspace(-.5, .5, 401), steps=integration_steps(.25),
             suffix="_refined")
    with np.load(OUT/f"{key}.npz") as data:
        coarse = data["populations"][:, indices]
        x = data["detuning_mhz"]
    with np.load(OUT/f"{key}_refined.npz") as data:
        fine = data["populations"]
        xf = data["detuning_mhz"]
    old = [spectroscopy_quality(x, y) for y in coarse[1]]
    time_only = [spectroscopy_quality(x, y) for y in fine[1, :, ::2]]
    both = [spectroscopy_quality(xf, y) for y in fine[1]]
    result = dict(case=key, amplitudes_mhz=AMPS[indices].tolist(),
                  refined_dt_ns=.25, refined_detuning_step_khz=2.5,
                  max_population_difference=float(abs(coarse-fine[:, :, ::2]).max()))
    for metric in ["center_hz", "fwhm_hz", "contrast_snr"]:
        result[f"time_refinement_max_{metric}_change"] = max(
            abs(a[metric]-b[metric]) for a, b in zip(old, time_only, strict=True))
        result[f"grid_refinement_max_{metric}_change"] = max(
            abs(a[metric]-b[metric]) for a, b in zip(time_only, both, strict=True))
    areas = json.loads((OUT/"waveform_metrics.json").read_text())
    ratio = next(a["area_fraction"] for a in areas
                 if a["cutoff"] == cutoff and a["window"] == window)
    match_id = window + "_area_matched_hard"
    run_case(cutoff, match_id, "none", 0., amplitudes=AMPS[indices]*ratio,
             steps=integration_steps(.25))
    with np.load(OUT/f"{case_name(cutoff, match_id)}.npz") as data:
        hard = [spectroscopy_quality(data["detuning_mhz"], y)
                for y in data["populations"][1]]
    result["area_matched_comparison"] = [
        dict(window_amplitude_mhz=float(a), hard_amplitude_mhz=float(a*ratio),
             window_fwhm_khz=w["fwhm_hz"]/1000, hard_fwhm_khz=h["fwhm_hz"]/1000,
             window_snr=w["contrast_snr"], hard_snr=h["contrast_snr"],
             window_contrast=w["shoulder_contrast"], hard_contrast=h["shoulder_contrast"])
        for a, w, h in zip(AMPS[indices], both, hard, strict=True)]
    result["passed"] = bool(
        result["max_population_difference"] < 1e-4
        and result["time_refinement_max_fwhm_hz_change"] < 1000
        and result["grid_refinement_max_fwhm_hz_change"] < 1000
        and result["time_refinement_max_center_hz_change"] < 500
        and result["grid_refinement_max_center_hz_change"] < 500)
    (OUT/f"validation_{key}.json").write_text(json.dumps(result, indent=2))
    return result


def compare_backends():
    """Use saved NumPy references, or recompute a small independent grid."""
    results = []
    indices = [50, 74, 99]
    for window in ["hard", "ramp6", "polynomial"]:
        key = case_name(.001, window)
        with np.load(OUT/f"{key}.npz") as data:
            config = json.loads(str(data["config"]))
            detuning = data["detuning_mhz"]
            reference = data["populations"][:, indices]
            saved_native = ("integration_backend" in data
                            and str(data["integration_backend"]) == "native")
        if saved_native:
            detuning = detuning[::25]
            ref = simulate_qutrit_map(**config, rabi_mhz=AMPS[indices],
                                      detuning_mhz=detuning, integration_backend="numpy")
            reference = np.array([ref.ground, ref.excited, ref.second_excited])
        native = simulate_qutrit_map(**config, rabi_mhz=AMPS[indices],
                                     detuning_mhz=detuning, integration_backend="native")
        values = np.array([native.ground, native.excited, native.second_excited])
        error = float(abs(reference-values).max())
        results.append(dict(case=key, max_population_difference=error,
                            amplitudes_mhz=AMPS[indices].tolist(),
                            detuning_points=len(detuning), dt_ns=.5))
        if error >= 1e-8:
            raise RuntimeError(f"Native backend disagrees with NumPy: {key}: {error}")
    (OUT/"backend_agreement.json").write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+", help="cutoff:window, e.g. 0.001:polynomial")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--compare-backends", action="store_true")
    add_experiment_arguments(parser)
    args = parser.parse_args()
    configure_worker(args.duration_us, args.output_dir)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=configure_worker,
                             initargs=(args.duration_us, OUT)) as pool:
        jobs = []
        for entry in args.cases:
            cutoff, window = entry.split(":")
            jobs.append(pool.submit(check_case, float(cutoff), window))
        for job in as_completed(jobs):
            print(json.dumps(job.result()), flush=True)
    results = [json.loads(p.read_text()) for p in sorted(OUT.glob("validation_c*.json"))]
    (OUT/"validation.json").write_text(json.dumps(results, indent=2))
    if not all(r["passed"] for r in results):
        raise RuntimeError("A convergence check exceeded the specified tolerance")
    if args.compare_backends:
        compare_backends()


if __name__ == "__main__":
    main()
