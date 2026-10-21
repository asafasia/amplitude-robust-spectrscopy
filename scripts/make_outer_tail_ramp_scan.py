"""Scan fixed-duration outer-tail tapers with unchanged 30 us echo settings.

Run with .venv/bin/python scripts/make_outer_tail_ramp_scan.py --workers 2.
Additional --ramps values extend the saved scan without replacing old results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import make_echo_edge_ramp_comparison as base  # noqa: E402

OUT = ROOT / "outputs/outer_tail_30us_scan"
REFERENCE = ROOT / "outputs/echo_edge_ramp_30us_comparison"
COARSE_RAMPS = [.25, .5, 1., 2., 3., 4., 5., 6., 8., 10., 12., 14., 15.]
MODEL_FILES = ["echospec/simulation/qutrit.py", "echospec/simulation/edge_window.py",
               "echospec/simulation/qutrit_native.py", "echospec/simulation/qutrit_rk4.cpp"]


def configure():
    base.configure_experiment(30., OUT)


def model_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in MODEL_FILES}


def ramp_name(ramp):
    return f"ramp{ramp:g}"


def cases():
    """Only full, primary grids belong to the optimization dataset."""
    result = []
    for path in sorted(OUT.glob("c*.npz")):
        with np.load(path) as data:
            if (not np.array_equal(data["rabi_mhz"], base.AMPS)
                    or not np.array_equal(data["detuning_mhz"], base.DETUNING)):
                continue
            cfg = json.loads(str(data["config"]))
            if cfg["num_steps_per_half"] != 30000:
                continue
            kind = cfg["edge_window_kind"]
            ramp = cfg["edge_ramp_us"]
            label = {"none": "hard", "polynomial": "polynomial"}.get(
                kind, ramp_name(ramp))
            result.append(dict(case=path.stem, cutoff=cfg["cutoff"], window=label,
                               kind=kind, ramp_us=ramp, path=path))
    return sorted(result, key=lambda r: (r["cutoff"], r["kind"], r["ramp_us"]))


def operating_summary(rows, min_amplitude=50.):
    """Conservative score: invalid high-drive points disqualify a candidate."""
    selected = [r for r in rows if r["amplitude_mhz"] >= min_amplitude]
    if not selected:
        raise ValueError("No amplitudes in the requested operating range")
    valid = [bool(r["accepted"] and np.isfinite(r["fwhm_hz"])
             and r["fwhm_hz"] > 0 and np.isfinite(r["contrast_snr"])
             and r["contrast_snr"] >= 0) for r in selected]
    good = [r for r, ok in zip(selected, valid, strict=True) if ok]
    widths = [r["fwhm_hz"] / 1000 for r in good]
    snrs = [r["contrast_snr"] for r in good]
    complete = all(valid)
    return dict(
        sampled_points=len(selected), valid_points=sum(valid),
        points_passing_goal=sum(r["fwhm_hz"] <= 200000 and r["contrast_snr"] >= 10
                                for r in good),
        worst_fwhm_khz=max(widths) if complete else None,
        minimum_snr=min(snrs) if complete else None,
        minimum_snr_per_mhz=min(r["contrast_snr"] / (r["fwhm_hz"] / 1e6)
                                for r in good) if complete else 0.,
        median_snr_per_mhz=float(np.median([
            r["contrast_snr"] / (r["fwhm_hz"] / 1e6) for r in good])) if good else 0.,
        maximum_abs_center_khz=max(abs(r["center_hz"])/1000 for r in good)
        if complete else None,
    )


def pareto_flags(summaries):
    """Non-dominated in worst width and minimum SNR over the operating range."""
    flags = []
    for a in summaries:
        if a["worst_fwhm_khz"] is None:
            flags.append(False)
            continue
        dominated = any(
            b["worst_fwhm_khz"] is not None
            and b["worst_fwhm_khz"] <= a["worst_fwhm_khz"]
            and b["minimum_snr"] >= a["minimum_snr"]
            and (b["worst_fwhm_khz"] < a["worst_fwhm_khz"]
                 or b["minimum_snr"] > a["minimum_snr"])
            for b in summaries)
        flags.append(not dominated)
    return flags


def export_metrics():
    rows, summaries = [], []
    for case in cases():
        group = []
        with np.load(case["path"]) as data:
            for index, amp in enumerate(data["rabi_mhz"]):
                m = base.spectroscopy_quality(data["detuning_mhz"],
                                             data["populations"][1, index])
                group.append(dict(
                    **{k: v for k, v in case.items() if k != "path"},
                    amplitude_mhz=float(amp), **m,
                    max_final_pf=float(data["populations"][2, index].max())))
        rows.extend(group)
        summaries.append(dict(
            **{k: v for k, v in case.items() if k != "path"},
            **operating_summary(group),
            at_100_mhz=group[-1],
            full_nonzero_range=operating_summary(group, 1.),
        ))
    for c in base.CUTOFFS:
        subset = [s for s in summaries if s["cutoff"] == c]
        for summary, flag in zip(subset, pareto_flags(subset), strict=True):
            summary["pareto_over_all_shapes"] = flag
    if rows:
        with (OUT / "metrics.csv").open("w") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        (OUT / "summary.json").write_text(json.dumps(summaries, indent=2))
    return rows, summaries


def choose_ramps(summaries):
    result = []
    for c in base.CUTOFFS:
        group = [s for s in summaries if s["cutoff"] == c and s["kind"] == "quintic"]
        if not group:
            continue
        best = max(group, key=lambda s: s["minimum_snr_per_mhz"])
        width = min(group, key=lambda s: s["worst_fwhm_khz"]
                    if s["worst_fwhm_khz"] is not None else np.inf)
        snr = max(group, key=lambda s: s["minimum_snr"]
                  if s["minimum_snr"] is not None else -np.inf)
        overall = max((s for s in summaries if s["cutoff"] == c),
                      key=lambda s: s["minimum_snr_per_mhz"])
        local = max((s for s in group if s["ramp_us"] <= 6),
                    key=lambda s: s["minimum_snr_per_mhz"], default=None)
        strict = max((s for s in group if s["ramp_us"] < 15),
                     key=lambda s: s["minimum_snr_per_mhz"], default=None)
        result.append(dict(cutoff=c, best_tradeoff_ramp_us=best["ramp_us"],
                           narrowest_ramp_us=width["ramp_us"],
                           best_snr_ramp_us=snr["ramp_us"],
                           best_tradeoff_case=best["case"],
                           best_overall_case=overall["case"],
                           best_with_untouched_center_ramp_us=(
                               strict["ramp_us"] if strict else None),
                           best_local_upto_6us=local["ramp_us"] if local else None,
                           sampled_ramps_within_5percent_of_best=sorted(
                               s["ramp_us"] for s in group
                               if s["minimum_snr_per_mhz"] >=
                               .95*best["minimum_snr_per_mhz"])))
    return result


def run_one(cutoff, label, kind, ramp):
    configure()
    target = OUT / f"{base.case_name(cutoff, label)}.npz"
    previous = REFERENCE / target.name
    if not target.exists() and previous.exists():
        provenance = json.loads((REFERENCE / "provenance.json").read_text())
        for name, digest in model_hashes().items():
            if provenance["source_sha256"][name] != digest:
                raise ValueError(f"Reference model changed: {name}")
        with np.load(previous) as old:
            cfg = json.loads(str(old["config"]))
            expected = dict(**base.COMMON, cutoff=cutoff, drag_beta=-.5,
                            edge_window_kind=kind, edge_ramp_us=ramp,
                            num_steps_per_half=30000)
            assert cfg == expected
            np.testing.assert_array_equal(old["rabi_mhz"], base.AMPS)
            np.testing.assert_array_equal(old["detuning_mhz"], base.DETUNING)
        shutil.copyfile(previous, target)
    return base.run_case(cutoff, label, kind, ramp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ramps", nargs="+", type=float, default=COARSE_RAMPS)
    parser.add_argument("--cutoffs", nargs="+", type=float, default=base.CUTOFFS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    if any(not np.isfinite(r) or not 0 < r <= 15 for r in args.ramps):
        parser.error("Each ramp must be in (0, 15] us inside the 30 us pulse")
    if not set(args.cutoffs) <= set(base.CUTOFFS):
        parser.error("Use the existing cutoffs: 0.001, 0.003, 0.005")
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "model_hashes.json"
    if manifest.exists() and json.loads(manifest.read_text()) != model_hashes():
        raise ValueError("Model changed since cached scan; use a fresh output directory")
    manifest.write_text(json.dumps(model_hashes(), indent=2))
    if not args.metrics_only:
        windows = [("hard", "none", 0.), ("polynomial", "polynomial", 0.)]
        windows += [(ramp_name(r), "quintic", r) for r in sorted(set(args.ramps))]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(run_one, c, label, kind, ramp)
                    for c in args.cutoffs for label, kind, ramp in windows]
            for job in as_completed(jobs):
                print("Finished", job.result(), flush=True)
    rows, summaries = export_metrics()
    chosen = choose_ramps(summaries)
    (OUT / "selection.json").write_text(json.dumps(chosen, indent=2))
    hashes = model_hashes()
    for p in [Path(__file__), ROOT / "scripts/make_echo_edge_ramp_comparison.py",
              ROOT / "echospec/analysis/dip_metrics.py",
              ROOT / "echospec/analysis/spectroscopy_quality.py",
              ROOT / "scripts/plot_outer_tail_ramp_scan.py",
              ROOT / "scripts/validate_outer_tail_ramp_scan.py"]:
        if p.exists():
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    commands = []
    for c in base.CUTOFFS:
        ramps = sorted(s["ramp_us"] for s in summaries
                       if s["cutoff"] == c and s["kind"] == "quintic")
        commands.append(".venv/bin/python scripts/make_outer_tail_ramp_scan.py "
                        f"--cutoffs {c:g} --ramps " + " ".join(f"{r:g}" for r in ramps))
    commands.extend([
        ".venv/bin/python scripts/plot_outer_tail_ramp_scan.py",
        ".venv/bin/python scripts/validate_outer_tail_ramp_scan.py",
        ".venv/bin/python scripts/plot_outer_tail_ramp_scan.py",
    ])
    (OUT / "provenance.json").write_text(json.dumps(dict(
        parameters=base.COMMON, beta=-.5, cutoffs=base.CUTOFFS,
        amplitudes_mhz=base.AMPS.tolist(), detuning_mhz=base.DETUNING.tolist(),
        dt_ns=.5, steps_per_half=30000, metrics_rows=len(rows),
        scan=[{k: v for k, v in c.items() if k != "path"} for c in cases()],
        source_sha256=hashes, reference_directory=str(REFERENCE),
        reproduce_commands=commands,
        objective="Maximize the minimum (ideal contrast SNR / FWHM in MHz) across the 50 sampled amplitudes >=50 MHz. This is a heuristic sharpness score, not a frequency-uncertainty bound.",
        illustrative_targets=dict(maximum_fwhm_khz=200, minimum_snr=10),
        snr="1000 ideal binary Pe shots: 500 at dip center, 250 at each of +/-450 kHz. No readout error, drift or technical noise.",
        width="Nearest half-depth crossings relative to a linear background fitted at |detuning|>=400 kHz. Not a Lorentzian fit or a universal T2 limit.",
        taper="Quintic 10u^3-15u^4+6u^5 over the first/last R us inside the same 30 us. Central 30-2R us unchanged. R=15 is a limiting whole-pulse taper, not a localized tail.",
        normalization="No peak or area renormalization. DRAG differentiates the entire windowed I envelope.",
        caveat="Three-level RWA Lindblad model. Reported Pf is final population, not maximum transient leakage. Best means best sampled within this taper family and stated objective.",
    ), indent=2))
    print(json.dumps(chosen, indent=2), flush=True)


if __name__ == "__main__":
    main()
