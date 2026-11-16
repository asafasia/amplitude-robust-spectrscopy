"""Load, validate, and compare matched polynomial cutoff sweeps."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

METRIC_KEYS = (
    "accepted", "fwhm_hz", "contrast_snr", "snr_per_mhz", "shoulder_contrast",
    "center_hz", "max_final_pf", "prominent_minima", "fwhm_grid_intervals",
    "fwhm_over_snr_hz", "fwhm_times_snr_hz", "shots_for_snr_10",
)


def check_matching_configuration(configuration, common, duration):
    """Reject mixing pulse, dynamics, grids, noise budgets, or source versions."""
    if duration not in configuration.get("durations_us", []):
        raise ValueError(f"Duration {duration:g} us is absent from the run configuration")
    actual = {k: v for k, v in configuration.items() if k != "durations_us"}
    if actual != common:
        changes = sorted(k for k in set(actual) | set(common)
                         if actual.get(k) != common.get(k))
        raise ValueError(f"Unmatched study settings: {', '.join(changes)}")


def duration_directory(root, manifest, duration):
    source = (manifest["baseline_copy"] if duration in manifest["reused_durations_us"]
              else manifest["new_runs"])
    return Path(root) / source / f"L{duration:.12g}us"


def robust_cutoff_summary(metrics, amplitudes, cutoffs, minimum_amplitude=10.0):
    """Choose one cutoff by max-min SNR/FWHM over the sampled drive range.

    Every active amplitude must be resolved. This is a sampled robustness
    score, not a global optimum or a guarantee of a unique spectral dip.
    """
    active = np.asarray(amplitudes) >= minimum_amplitude
    if not active.any():
        return None
    valid = np.asarray(metrics["accepted"], dtype=bool)[:, active]
    width = np.asarray(metrics["fwhm_hz"])[:, active]
    snr = np.asarray(metrics["contrast_snr"])[:, active]
    valid &= np.isfinite(width) & (width > 0) & np.isfinite(snr) & (snr > 0)
    eligible = valid.all(axis=1)
    if not eligible.any():
        return None
    score = np.full(len(cutoffs), -np.inf)
    score[eligible] = np.min(snr[eligible] / (width[eligible] / 1e6), axis=1)
    index = int(np.argmax(score))
    return dict(
        cutoff_index=index, cutoff=float(cutoffs[index]),
        sampled_amplitude_min_mhz=float(np.asarray(amplitudes)[active].min()),
        sampled_amplitude_max_mhz=float(np.asarray(amplitudes)[active].max()),
        worst_fwhm_khz=float(width[index].max() / 1000),
        minimum_snr=float(snr[index].min()),
        minimum_snr_per_mhz=float(score[index]),
        all_amplitudes_snr_at_least_10=bool(np.all(snr[index] >= 10)),
    )


def pareto_mask(width, snr):
    """Mark sampled points not dominated by lower width and higher SNR."""
    width, snr = np.asarray(width), np.asarray(snr)
    if width.shape != snr.shape or width.ndim != 1:
        raise ValueError("Require matching 1D width and SNR arrays")
    valid = np.isfinite(width) & np.isfinite(snr) & (width > 0) & (snr > 0)
    result = np.zeros(width.size, dtype=bool)
    best = -np.inf
    # Equal-width points with smaller SNR are dominated. Exact duplicates
    # of a frontier point are both retained.
    last_width, last_snr = None, None
    for index in np.lexsort((-snr, width)):
        if not valid[index]:
            continue
        duplicate = width[index] == last_width and snr[index] == last_snr
        if snr[index] > best or duplicate:
            result[index] = True
            best = max(best, snr[index])
            last_width, last_snr = width[index], snr[index]
    return result


def load_study(root, *, allow_partial=False):
    """Read only complete duration exports; retain all original precision.

    The browser gets all metric cells and the complete Pe cube as float32
    display copies; three-level float64 populations remain in each scan.npz.
    """
    root = Path(root)
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    common = manifest["common_configuration"]
    cutoffs = np.asarray(common["cutoffs"])
    amplitudes = np.asarray(common["rabi_mhz"])
    detuning = np.asarray(common["detuning_mhz"])
    cases, pending = [], []
    for duration in manifest["durations_us"]:
        directory = duration_directory(root, manifest, duration)
        path = directory / "scan.npz"
        if not path.exists() or not (directory / "summary.json").exists():
            if not allow_partial:
                raise FileNotFoundError(f"Incomplete duration export: {directory}")
            pending.append(duration)
            continue
        with np.load(path, allow_pickle=False) as saved:
            configuration = json.loads(str(saved["configuration"]))
            check_matching_configuration(configuration, common, duration)
            if float(saved["duration_us"]) != duration:
                raise ValueError(f"Wrong duration in {path}")
            for key, expected in [("cutoffs", cutoffs), ("rabi_mhz", amplitudes),
                                  ("detuning_mhz", detuning)]:
                np.testing.assert_array_equal(saved[key], expected)
            population = saved["populations"]
            if population.shape != (len(cutoffs), 3, len(amplitudes), len(detuning)):
                raise ValueError(f"Unexpected population layout in {path}")
            if not np.isfinite(population).all():
                raise ValueError(f"Nonfinite populations in {path}")
            trace_error = float(abs(population.sum(axis=1) - 1).max())
            if (trace_error > 2e-6 or population.min() < -2e-6
                    or population.max() > 1 + 2e-6):
                raise ValueError(f"Nonphysical populations in {path}")
            metrics = {key: saved[key] for key in METRIC_KEYS}
            for key, values in metrics.items():
                if values.shape != (len(cutoffs), len(amplitudes)):
                    raise ValueError(f"Wrong metric layout for {key} in {path}")
            accepted = metrics["accepted"].astype(bool)
            for key in ("fwhm_hz", "contrast_snr", "snr_per_mhz"):
                values = metrics[key][accepted]
                if not np.isfinite(values).all() or np.any(values <= 0):
                    raise ValueError(f"Invalid accepted {key} in {path}")
            # Float32 is for a display-only browser payload. Saved arrays and
            # analysis retain float64; no metrics are derived from this copy.
            population_maps = population[:, 1].astype("<f4")
        with (directory / "metrics.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != len(cutoffs) * len(amplitudes):
            raise ValueError(f"Incomplete metrics.csv in {directory}")
        for index, row in enumerate(rows):
            ci, ai = divmod(index, len(amplitudes))
            if (float(row["duration_us"]) != duration
                    or float(row["cutoff"]) != cutoffs[ci]
                    or float(row["amplitude_mhz"]) != amplitudes[ai]):
                raise ValueError(f"CSV coordinate mismatch in {directory}")
            if (row["accepted"] == "True") != accepted[ci, ai]:
                raise ValueError(f"CSV acceptance mismatch in {directory}")
            for key in ("fwhm_hz", "contrast_snr", "snr_per_mhz"):
                actual, expected = float(row[key]), metrics[key][ci, ai]
                if actual != expected and not (np.isnan(actual) and np.isnan(expected)):
                    raise ValueError(f"CSV {key} mismatch in {directory}")
        summary = json.loads((directory / "summary.json").read_text())
        cases.append(dict(
            duration_us=duration, directory=directory.relative_to(root).as_posix(),
            metrics=metrics, population_maps=population_maps, rows=rows,
            robust=robust_cutoff_summary(metrics, amplitudes, cutoffs),
            validation=dict(
                max_trace_error=trace_error,
                min_population=float(population.min()),
                max_population=float(population.max()),
                min_density_eigenvalue=min(x["min_density_eigenvalue"]
                                          for x in summary["diagnostics"]),
                accepted_traces=int(accepted.sum()), total_traces=int(accepted.size),
            ),
        ))
    if not cases:
        raise ValueError("No completed durations are available")
    return dict(manifest=manifest, common=common, cutoffs=cutoffs,
                amplitudes=amplitudes, detuning=detuning, cases=cases,
                pending=pending)
