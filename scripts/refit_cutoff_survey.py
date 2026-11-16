"""Refit the saved cutoff survey; leave all simulated populations unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from echospec.analysis.duration_cutoffs import duration_directory
from echospec.analysis.robust_dip_fit import FIT_SETTINGS, robust_dip_fit
from echospec.figures.coherence import REFERENCE_FWHM_HZ

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/polynomial_duration_cutoff_poly2_log_40_50/n100"
FIELDS = ("center_hz", "fwhm_hz", "contrast", "relative_rms", "relative_width_error",
          "radius_mhz", "reference_mhz", "y_offset", "y_scale")


def fit_cutoff(task):
    path, encoded, detuning, populations, original_mask = task
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            assert str(saved["configuration"]) == encoded
        return path
    count = len(populations)
    arrays = {key: np.full(count, np.nan) for key in FIELDS}
    arrays.update(accepted=np.zeros(count, dtype=bool),
                  status=np.full(count, "original_rejected", dtype="U32"),
                  parameters=np.full((count, 6), np.nan))
    for index in np.flatnonzero(original_mask):
        fit = robust_dip_fit(detuning, populations[index])
        arrays["accepted"][index] = fit["accepted"]
        arrays["status"][index] = fit["status"]
        for key in FIELDS:
            arrays[key][index] = fit.get(key, np.nan)
        if "parameters" in fit:
            arrays["parameters"][index] = fit["parameters"]
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, configuration=encoded, **arrays)
    temporary.replace(path)
    return path


def refinement_checks(source, arrays, lengths, cutoffs, amps):
    rows = []
    for path in sorted((source / "validation").glob("L*us_cutoff_*_fine.npz")):
        with np.load(path, allow_pickle=False) as saved:
            spec = json.loads(str(saved["configuration"]))
            duration = spec["pulse"]["duration_us"]
            li = int(np.flatnonzero(lengths == duration)[0])
            ci = int(np.argmin(abs(cutoffs - spec["pulse"]["cutoff"])))
            x, yy = saved["detuning_mhz"], saved["populations"][1]
            for index, amplitude in enumerate(saved["rabi_mhz"]):
                ai = int(np.flatnonzero(amps == amplitude)[0])
                timed = robust_dip_fit(x[::2], yy[index, ::2])
                refined = robust_dip_fit(x, yy[index])
                coarse_ok = bool(arrays["accepted"][li, ci, ai])
                # Preserve the original simulation acceptance mask at all refinements.
                original_ok = arrays["status"][li, ci, ai] != "original_rejected"
                row = dict(duration_us=float(duration), cutoff=float(cutoffs[ci]),
                           amplitude_mhz=float(amplitude), coarse_accepted=coarse_ok,
                           time_accepted=bool(original_ok and timed["accepted"]),
                           refined_accepted=bool(original_ok and refined["accepted"]))
                if coarse_ok and timed["accepted"] and refined["accepted"]:
                    row.update(
                        time_width_change_normalized=float(abs(
                            arrays["fwhm_hz"][li, ci, ai] - timed["fwhm_hz"]
                        ) / REFERENCE_FWHM_HZ),
                        frequency_width_change_normalized=float(abs(
                            timed["fwhm_hz"] - refined["fwhm_hz"]
                        ) / REFERENCE_FWHM_HZ),
                    )
                rows.append(row)
    local = []
    diagnostic = ROOT / "outputs/fig4_50us_linewidth_diagnostic"
    for index in (0, 6, 9, 10, 12, 13):
        path = diagnostic / f"fine_{index:02d}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as saved:
            x, y = saved["detuning_mhz"], saved["populations"][1, 0]
            spec = json.loads(str(saved["configuration"]))
        fit = robust_dip_fit(x, y)
        alternatives = [robust_dip_fit(x, y, window_factor=factor)
                        for factor in (1.2, 1.8)]
        local.append(dict(
            cutoff=spec["pulse"]["cutoff"], accepted=fit["accepted"],
            normalized_width=fit["fwhm_hz"] / REFERENCE_FWHM_HZ,
            contrast=fit["contrast"], relative_rms=fit["relative_rms"],
            window_factor_range=[1.2, 1.8],
            window_check_widths_normalized=[a["fwhm_hz"] / REFERENCE_FWHM_HZ
                                            for a in alternatives],
            window_check_accepted=[a["accepted"] for a in alternatives],
        ))
    return dict(
        representative_checks=rows, local_jump_checks=local,
        scope="Saved time/frequency refinements at three cutoffs and three drives "
              "for every duration, plus six refined spectra around the 50 us jump. "
              "Window sensitivity is a model-dependence check, not numerical convergence.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SOURCE)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    source = args.input_dir.resolve()
    destination = source / "robust_fits"
    destination.mkdir(exist_ok=True)
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    lengths = np.asarray(manifest["durations_us"])
    config = manifest["common_configuration"]
    cutoffs, amps = np.asarray(config["cutoffs"]), np.asarray(config["rabi_mhz"])
    x = np.asarray(config["detuning_mhz"])
    code = [Path(__file__), ROOT / "echospec/analysis/robust_dip_fit.py",
            ROOT / "echospec/analysis/dip_metrics.py"]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in code}
    datasets = []
    for length in lengths:
        path = duration_directory(source, manifest, length) / "scan.npz"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(path.relative_to(ROOT))] = digest
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["cutoffs"], cutoffs)
            np.testing.assert_array_equal(saved["rabi_mhz"], amps)
            np.testing.assert_array_equal(saved["detuning_mhz"], x)
            datasets.append((saved["populations"][:, 1].copy(), saved["accepted"].copy()))
    encoded = json.dumps(dict(source_sha256=hashes, settings=FIT_SETTINGS), sort_keys=True)
    signature = hashlib.sha256(encoded.encode()).hexdigest()[:16]
    cache = destination / "cache" / signature
    cache.mkdir(parents=True, exist_ok=True)
    tasks, paths = [], []
    for length, (population, accepted) in zip(lengths, datasets, strict=True):
        row_paths = []
        for ci in range(len(cutoffs)):
            path = cache / f"L{length:g}us_cutoff_{ci:03d}.npz"
            tasks.append((path, encoded, x, population[ci], accepted[ci]))
            row_paths.append(path)
        paths.append(row_paths)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fit_cutoff, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 25 == 0:
                print(f"Fitted {count}/{len(tasks)} cutoff maps", flush=True)
    arrays = {}
    for key in (*FIELDS, "accepted", "status", "parameters"):
        arrays[key] = np.stack([
            np.stack([np.load(path, allow_pickle=False)[key] for path in row])
            for row in paths
        ])
    for row, (_, original) in enumerate(datasets):
        assert not np.any(arrays["accepted"][row] & ~original)
    validation = refinement_checks(source, arrays, lengths, cutoffs, amps)
    np.savez_compressed(destination / "metrics.npz", durations_us=lengths,
                        cutoffs=cutoffs, rabi_mhz=amps, configuration=encoded, **arrays)
    metadata = dict(
        complete=True, source_sha256=hashes, settings=FIT_SETTINGS,
        durations_us=lengths.tolist(), accepted_counts=arrays["accepted"].sum(axis=(1, 2)).tolist(),
        statuses=dict(Counter(arrays["status"].ravel().tolist())),
        arrays_sha256=hashlib.sha256((destination / "metrics.npz").read_bytes()).hexdigest(),
        validation=validation,
        definition="FWHM and contrast of the locally fitted Gaussian component above "
                   "a quadratic background, with a smooth curvature-scaled window "
                   "and soft-L1 residual loss. Original rejected cells remain rejected. "
                   "Additional fit, physicality, sampling and identifiability checks apply.",
    )
    (destination / "provenance.json").write_text(json.dumps(metadata, indent=2,
                                                          allow_nan=False) + "\n")
    print(json.dumps({key: metadata[key] for key in ("accepted_counts", "statuses")}, indent=2))


if __name__ == "__main__":
    main()
