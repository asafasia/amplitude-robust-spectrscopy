"""Repeat the last supplemental survey with p=2 and log-spaced 1–100 MHz drives.

Run: .venv/bin/python scripts/repeat_extended_cutoff_poly2.py --workers 6
Rerun the same command to resume atomic simulation checkpoints. All other
physics and grids match the original survey. Outputs remain separate from the
published figure and the old p=5 data.
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
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")
os.environ.setdefault("MPLBACKEND", "Agg")

from echospec.analysis.duration_cutoffs import load_study  # noqa: E402
from scripts.make_supplemental_extended_cutoff import main as plot  # noqa: E402
from scripts.report_polynomial_durations import export_tables  # noqa: E402
from scripts.scan_polynomial_cutoffs import (  # noqa: E402
    main as scan,
)
from scripts.scan_polynomial_cutoffs import (  # noqa: E402
    parser as scan_parser,
)
from scripts.scan_polynomial_cutoffs import (  # noqa: E402
    settings_from_args,
    write_json,
)
from scripts.validate_polynomial_durations import validate  # noqa: E402

OUTPUT = ROOT / "outputs/polynomial_duration_cutoff_poly2_log/n100"
STEM = "supp_extended_cutoff_duration_poly2_log"


def scan_arguments(output, workers):
    return [
        "--cutoff-count", "100", "--durations-us", "10", "20", "30", "45", "60",
        "--power", "2", "--transition-ns", "16", "--drag-beta", "-0.5",
        "--amplitude-spacing", "log", "--amplitude-min-mhz", "1",
        "--amplitude-max-mhz", "100", "--amplitude-points", "100",
        "--detuning-span-mhz", "0.5", "--detuning-points", "201",
        "--t1-us", "51.24", "--t2-us", "7.31", "--anharmonicity-mhz", "-216",
        "--dt-ns", "0.25", "--shots", "1000", "--workers", str(workers),
        "--output-dir", str(output / "simulations"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    arguments = scan_arguments(output, args.workers)
    config = settings_from_args(scan_parser().parse_args(arguments))
    common = {k: v for k, v in config.items() if k != "durations_us"}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["common_configuration"] != common:
            raise ValueError("Run settings/source changed; choose a fresh output directory")
    else:
        manifest = dict(
            schema_version=1, status="pending", common_configuration=common,
            durations_us=config["durations_us"], new_runs="simulations",
            reused_durations_us=[], baseline_copy=None,
            expected_maps=500, expected_traces=50000,
            expected_population_grid_points=10050000,
            comparison_source="outputs/polynomial_duration_cutoff_sweep/n100",
            changes="Polynomial p=5 -> 2; 100 linear 0–100 MHz amplitudes -> "
                    "100 geometric 1–100 MHz amplitudes (same displayed range). "
                    "No zero-drive point on the logarithmic grid. Other settings unchanged.",
            invocation=[sys.executable, "scripts/scan_polynomial_cutoffs.py", *arguments],
        )
        write_json(manifest_path, manifest)
    sources = list(config["source_sha256"]) + [
        "scripts/repeat_extended_cutoff_poly2.py",
        "scripts/make_supplemental_extended_cutoff.py",
        "scripts/validate_polynomial_durations.py",
        "scripts/report_polynomial_durations.py",
        "echospec/analysis/duration_cutoffs.py",
        "echospec/figures/coherence.py",
        "echospec/figures/supplemental.py", "styles/supplemental.mplstyle",
    ]
    for source in sources:
        destination = output / "source_snapshot" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, destination)
    (output / "README.txt").write_text(
        "Fresh three-level simulation survey, not hardware measurements.\n"
        + manifest["changes"] + "\n"
        "Durations 10, 20, 30, 45, 60 us; 100 geometric cutoffs 1e-5–1;\n"
        "201 detunings -0.5–0.5 MHz; beta=-0.5; 16 ns sine transition;\n"
        "T1=51.24 us, T2=7.31 us; alpha=-216 MHz; maximum RK4 step 0.25 ns.\n"
        "Amplitude is the nominal pre-smoothing scale; no peak normalization.\n"
        "status.json records the current stage. Each cutoff_###.npz is resumable.\n"
        "After simulation: 45 representative finer-grid traces, physicality checks,\n"
        "CSV/NPZ metrics and a separate paper-style PDF/PNG with provenance.\n"
        "Resume from the repository root using:\n"
        f".venv/bin/python scripts/repeat_extended_cutoff_poly2.py --workers {args.workers} "
        f"--output-dir {output}\n"
    )

    def status(stage, **extra):
        write_json(output / "status.json", dict(
            stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(),
            pid=os.getpid(), **extra,
        ))
        print(f"Stage: {stage}", flush=True)

    try:
        status("simulating")
        scan(arguments)
        status("convergence_checks")
        validate(output)
        status("exporting")
        study = load_study(output)
        export_tables(output, study)
        paths = [output / "comparison_metrics.npz"] + [
            output / "simulations" / f"L{length:g}us/scan.npz"
            for length in config["durations_us"]
        ]
        (output / "checksums.sha256").write_text("".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(output)}\n" for path in paths
        ))
        plot(["--input-dir", str(output), "--output-dir", str(ROOT / "figures/paper"),
              "--stem", STEM])
        manifest.update(status="complete", completed_durations_us=config["durations_us"])
        write_json(manifest_path, manifest)
        status("complete", figure=f"figures/paper/{STEM}.pdf",
               convergence="validation/convergence.json")
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
