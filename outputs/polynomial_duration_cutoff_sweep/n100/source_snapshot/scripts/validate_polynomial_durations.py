"""Save representative finer-time/frequency checks for the duration study.

The baseline simulation remains unchanged. Fine runs are atomic checkpoints;
--allow-partial can precompute them while the main study is still running.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")

from echospec.analysis.cutoff_metrics import cutoff_metrics  # noqa: E402
from echospec.analysis.duration_cutoffs import duration_directory  # noqa: E402
from echospec.simulation.polynomial_echo import (  # noqa: E402
    PolynomialEchoPulse,
    simulate_polynomial_echo,
)
from scripts.scan_polynomial_cutoffs import clean_json  # noqa: E402


def compare_metrics(detuning, original, fine, fine_detuning, shots):
    """Separate time-integration changes from frequency-sampling changes."""
    original_metrics = cutoff_metrics(detuning, original, shots=shots)
    time_metrics = cutoff_metrics(detuning, fine[::2], shots=shots)
    fine_metrics = cutoff_metrics(fine_detuning, fine, shots=shots)
    row = dict(
        original_status=original_metrics["status"],
        finer_time_status=time_metrics["status"],
        finer_frequency_status=fine_metrics["status"],
    )
    for name, before, after in [("time", original_metrics, time_metrics),
                                 ("frequency", time_metrics, fine_metrics)]:
        both = before["accepted"] and after["accepted"]
        for metric in ("fwhm_hz", "contrast_snr"):
            row[f"{name}_{metric}_absolute_change"] = (
                abs(after[metric] - before[metric]) if both else np.nan)
            row[f"{name}_{metric}_relative_change"] = (
                abs(after[metric] / before[metric] - 1) if both else np.nan)
    for name, metrics in [("original", original_metrics), ("fine", fine_metrics)]:
        for metric in ("fwhm_hz", "contrast_snr"):
            row[f"{name}_{metric}"] = metrics[metric]
    return row


def validate(root, *, allow_partial=False):
    root = Path(root).resolve()
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    config = manifest["common_configuration"]
    for source, digest in config["source_sha256"].items():
        if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Simulation source changed: {source}")
    destination = root / "validation"
    destination.mkdir(exist_ok=True)
    cutoffs = np.asarray(config["cutoffs"])
    amps = np.asarray(config["rabi_mhz"])
    det = np.asarray(config["detuning_mhz"])
    fine_det = np.empty(det.size * 2 - 1)
    fine_det[::2], fine_det[1::2] = det, (det[1:] + det[:-1]) / 2
    ci_list = sorted({0, int(np.argmin(abs(np.log(cutoffs / .001)))), len(cutoffs) - 1})
    ai_list = np.unique([int(np.argmin(abs(amps - a))) for a in (10, 50, 100)])
    rows, diagnostics, pending = [], [], []
    for duration in manifest["durations_us"]:
        directory = duration_directory(root, manifest, duration)
        for ci in ci_list:
            pulse = PolynomialEchoPulse(
                duration_us=duration, cutoff=float(cutoffs[ci]), power=config["power"],
                transition_us=config["transition_us"], drag_beta=config["drag_beta"],
                anharmonicity_mhz=config["anharmonicity_mhz"],
            )
            specification = dict(
                pulse=asdict(pulse), dt_ns=config["dt_ns"] / 2,
                t1_us=config["t1_us"], t2_us=config["t2_us"],
                rabi_mhz=amps[ai_list].tolist(), detuning_mhz=fine_det.tolist(),
                source_sha256=config["source_sha256"],
            )
            encoded = json.dumps(specification, sort_keys=True)
            path = destination / f"L{duration:g}us_cutoff_{ci:03d}_fine.npz"
            if not path.exists():
                population, diag = simulate_polynomial_echo(
                    pulse, amps[ai_list], fine_det, t1_us=config["t1_us"],
                    t2_us=config["t2_us"], dt_ns=config["dt_ns"] / 2,
                )
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, populations=population, configuration=encoded,
                                    diagnostics=json.dumps(diag), rabi_mhz=amps[ai_list],
                                    detuning_mhz=fine_det)
                temporary.replace(path)
            with np.load(path, allow_pickle=False) as saved:
                if str(saved["configuration"]) != encoded:
                    raise ValueError(f"Finer-grid cache settings changed: {path}")
                population = saved["populations"]
                diag = json.loads(str(saved["diagnostics"]))
            diagnostics.append(dict(duration_us=duration, cutoff_index=ci, **diag))
            baseline = directory / f"cutoff_{ci:03d}.npz"
            if not baseline.exists():
                pending.append(dict(duration_us=duration, cutoff_index=ci))
                print(f"Fine check cached; main case pending: L={duration:g}, c={cutoffs[ci]:.6g}", flush=True)
                continue
            with np.load(baseline, allow_pickle=False) as saved:
                actual = json.loads(str(saved["configuration"]))
                if (actual["pulse"] != asdict(pulse)
                        or {k: v for k, v in actual["config"].items() if k != "durations_us"} != config):
                    raise ValueError(f"Main-grid cache settings differ: {baseline}")
                np.testing.assert_array_equal(saved["rabi_mhz"], amps)
                np.testing.assert_array_equal(saved["detuning_mhz"], det)
                original = saved["populations"][:, ai_list, :]
            for j, ai in enumerate(ai_list):
                error = float(np.max(abs(original[:, j] - population[:, j, ::2])))
                rows.append(dict(duration_us=duration, cutoff_index=ci,
                                 cutoff=float(cutoffs[ci]), amplitude_mhz=float(amps[ai]),
                                 max_population_time_step_change=error,
                                 **compare_metrics(det, original[1, j], population[1, j],
                                                   fine_det, config["shots"])))
            print(f"Checked L={duration:g} us, c={cutoffs[ci]:.6g}; max Δpopulation={max(r['max_population_time_step_change'] for r in rows[-len(ai_list):]):.3g}", flush=True)
    if pending and not allow_partial:
        raise FileNotFoundError(f"Main simulation cases are pending: {pending}")
    if rows:
        with (destination / "convergence.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    maxima = {}
    for key in rows[0] if rows else []:
        if "change" in key:
            finite = [r[key] for r in rows if np.isfinite(r[key])]
            maxima[key] = max(finite) if finite else None
    result = dict(
        complete=not pending, created_utc=datetime.now(timezone.utc).isoformat(),
        duration_us=manifest["durations_us"], cutoff_indices=ci_list,
        cutoffs=cutoffs[ci_list].tolist(), amplitudes_mhz=amps[ai_list].tolist(),
        coarse_dt_ns=config["dt_ns"], finer_dt_ns=config["dt_ns"] / 2,
        coarse_detuning_points=len(det), finer_detuning_points=len(fine_det),
        compared_traces=len(rows), pending_main_cases=pending, maximum_changes=maxima,
        time_status_changes=sum(r["original_status"] != r["finer_time_status"] for r in rows),
        frequency_status_changes=sum(r["finer_time_status"] != r["finer_frequency_status"] for r in rows),
        scope="Three cutoffs (endpoints and nearest 0.001), three amplitudes (nearest 10, 50, 100 MHz), all lengths. Halve time step and double detuning intervals. This is a representative check, not a full-grid accuracy bound.",
        metric_comparison="Metric differences only compare pairs accepted at both resolutions. All status changes and original/fine metrics remain in convergence.csv.",
        diagnostics=diagnostics, source_sha256=config["source_sha256"],
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    path = destination / "convergence.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(clean_json(result), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps(maxima, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "outputs/polynomial_duration_cutoff_sweep/n100")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    validate(args.input_dir, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
