"""Replace the 45/60 us survey rows with new, matched 40/50 us simulations.

Run with .venv/bin/python scripts/extend_cutoff_duration_40_50.py --workers 6.
The original survey remains unchanged. Existing 10/20/30 us exports and their
refinement checkpoints are copied with checksum verification; only 40/50 us
are integrated. Repeating the command resumes atomic simulation checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-mpl-cache")

from echospec.analysis.duration_cutoffs import load_study  # noqa: E402
from scripts.repeat_extended_cutoff_poly2 import scan_arguments  # noqa: E402
from scripts.report_polynomial_durations import export_tables  # noqa: E402
from scripts.scan_polynomial_cutoffs import (  # noqa: E402
    main as scan,
)
from scripts.scan_polynomial_cutoffs import (  # noqa: E402
    parser as scan_parser,
)
from scripts.scan_polynomial_cutoffs import settings_from_args, write_json  # noqa: E402
from scripts.validate_polynomial_durations import validate  # noqa: E402

BASELINE = ROOT / "outputs/polynomial_duration_cutoff_poly2_log/n100"
OUTPUT = ROOT / "outputs/polynomial_duration_cutoff_poly2_log_40_50/n100"
REUSED = [10, 20, 30]
DURATIONS = [10, 20, 30, 40, 50]


def new_scan_arguments(output, workers):
    arguments = scan_arguments(output, workers)
    start = arguments.index("--durations-us") + 1
    stop = arguments.index("--power")
    arguments[start:stop] = ["40", "50"]
    return arguments


def copy_verified(source, destination, expected=None):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"Baseline checksum mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Existing copied data differ: {destination}")
    return digest


def prepare(output, workers):
    arguments = new_scan_arguments(output, workers)
    config = settings_from_args(scan_parser().parse_args(arguments))
    common = {k: v for k, v in config.items() if k != "durations_us"}
    baseline = json.loads((BASELINE / "dataset_manifest.json").read_text())
    if common != baseline["common_configuration"]:
        raise ValueError("New simulation settings or source differ from the baseline")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["common_configuration"] != common
                or manifest["durations_us"] != DURATIONS):
            raise ValueError("Existing output belongs to a different study")
    else:
        manifest = dict(
            schema_version=1, status="pending", common_configuration=common,
            durations_us=DURATIONS, reused_durations_us=REUSED,
            baseline_copy="baseline_copy", new_runs="simulations",
            baseline_source=str(BASELINE.relative_to(ROOT)),
            new_durations_us=[40, 50], expected_maps=500, expected_traces=50000,
            expected_population_grid_points=10050000,
            changes="Replace 45/60 us by newly integrated 40/50 us; identical pulse, "
                    "dissipation, grids, RK4 steps, and metric acceptance settings.",
            invocation=[sys.executable, "scripts/scan_polynomial_cutoffs.py", *arguments],
        )
    checksums = dict((name, digest) for digest, name in (
        line.split("  ", 1) for line in (BASELINE / "checksums.sha256").read_text().splitlines()
    ))
    reused_hashes = {}
    for duration in REUSED:
        directory = f"simulations/L{duration}us"
        names = ["scan.npz", "metrics.csv", "summary.json"]
        names += [f"cutoff_{index:03d}.npz" for index in (0, 40, 99)]
        for name in names:
            relative = f"{directory}/{name}"
            destination = output / "baseline_copy" / f"L{duration}us" / name
            reused_hashes[relative] = copy_verified(
                BASELINE / relative, destination, checksums.get(relative),
            )
        for path in (BASELINE / "validation").glob(f"L{duration}us_cutoff_*_fine.npz"):
            reused_hashes[str(path.relative_to(BASELINE))] = copy_verified(
                path, output / "validation" / path.name,
            )
    # Match argparse's floating duration representation in validator checkpoints.
    manifest["durations_us"] = [float(length) for length in DURATIONS]
    manifest["reused_source_sha256"] = reused_hashes
    write_json(manifest_path, manifest)
    sources = list(config["source_sha256"]) + [
        "scripts/extend_cutoff_duration_40_50.py",
        "scripts/repeat_extended_cutoff_poly2.py",
        "scripts/validate_polynomial_durations.py",
        "scripts/report_polynomial_durations.py",
        "echospec/analysis/duration_cutoffs.py",
        "scripts/make_supplemental_extended_cutoff.py",
        "scripts/make_main_duration_cutoff_scan.py",
        "echospec/figures/coherence.py",
        "echospec/figures/supplemental.py",
        "styles/supplemental.mplstyle",
    ]
    for source in sources:
        destination = output / "source_snapshot" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, destination)
    (output / "README.txt").write_text(
        "Matched three-level simulation survey for Figure 4 and Figure S8.\n"
        "Durations: 10, 20, 30, 40, 50 us. The 10/20/30 us rows and their\n"
        "fine-grid checkpoints are checksum-verified copies of the original\n"
        "power-2 logarithmic survey; 40/50 us are newly integrated.\n"
        "100 geometric cutoffs 1e-5 to 1; 100 geometric drives 1 to 100 MHz;\n"
        "201 detunings -0.5 to 0.5 MHz; beta=-0.5; 16 ns sine transition;\n"
        "T1=51.24 us, T2=7.31 us; alpha=-216 MHz; RK4 step <=0.25 ns.\n"
        "Drive is the nominal pre-smoothing scale, without peak normalization.\n"
        "Plot linewidths relative to Gamma=1/(pi*7.6 us); keep drive in MHz.\n"
        "Validation halves the time step and doubles the detuning intervals\n"
        "for 45 representative traces; it is not a full-grid error bound.\n"
        "Resume or reproduce from the repository root:\n"
        f".venv/bin/python scripts/extend_cutoff_duration_40_50.py --workers {workers} "
        f"--output-dir {output}\n"
        "Render the supplemental figure before the main figure:\n"
        "PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python "
        "scripts/make_supplemental_extended_cutoff.py\n"
        "PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python "
        "scripts/make_main_duration_cutoff_scan.py\n"
    )
    return arguments, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--prepare-only", action="store_true")
    stage.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    arguments, manifest = prepare(output, args.workers)
    if args.prepare_only:
        print(f"Prepared matched study: {output}")
        return

    def status(value, **extra):
        write_json(output / "status.json", dict(
            stage=value, updated_utc=datetime.now(timezone.utc).isoformat(),
            pid=os.getpid(), **extra,
        ))
        print(f"Stage: {value}", flush=True)

    try:
        if not args.finalize_only:
            status("simulating")
            scan(arguments)
        status("convergence_checks")
        validate(output)
        status("exporting")
        export_tables(output, load_study(output))
        paths = [output / "comparison_metrics.npz"]
        for duration in DURATIONS:
            directory = "baseline_copy" if duration in REUSED else "simulations"
            paths.append(output / directory / f"L{duration}us/scan.npz")
        (output / "checksums.sha256").write_text("".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(output)}\n" for path in paths
        ))
        manifest.update(status="complete", completed_durations_us=DURATIONS)
        write_json(output / "dataset_manifest.json", manifest)
        status("complete")
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
