"""Check one selected adaptive-rotation candidate on the saved dense amplitude grid."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from echospec.simulation.flexible_echo import FlexibleEchoPulse  # noqa: E402
from echospec.simulation.polynomial_echo import simulate_polynomial_echo  # noqa: E402
from scripts.optimize_50mhz_echo import metrics, peak_scale  # noqa: E402
from scripts.optimize_decoupled_50mhz_echo import clean  # noqa: E402
from scripts.optimize_flexible_50mhz_echo import DET, assess  # noqa: E402

OUT = Path(__file__).resolve().parent


def main() -> None:
    baseline_path = OUT / "theory_rotation_dense_audit.json"
    baseline_data = json.loads(baseline_path.read_text())
    baseline = baseline_data["baseline"]
    amplitudes = np.asarray(baseline_data["amplitudes_mhz"])
    assert np.array_equal(amplitudes, np.arange(5.0, 60.001, 0.5))
    assert np.array_equal(baseline_data["detuning_mhz"], DET)
    for path in ("echospec/simulation/polynomial_echo.py",
                 "echospec/simulation/qutrit_rk4.cpp",
                 "echospec/analysis/dip_metrics.py", "scripts/optimize_50mhz_echo.py"):
        assert (hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                == baseline_data["source_sha256"][path])

    source = OUT / "rotating_adaptive/candidates.json"
    candidates = json.loads(source.read_text())
    feasible = [r for r in candidates["candidates"] if r["quality"]["feasible"]]
    selected = feasible[0] if feasible else min(candidates["candidates"],
                                              key=lambda r: r["loss"])
    pulse = FlexibleEchoPulse(**selected["pulse"])
    norm = peak_scale(pulse)
    source_paths = [Path(__file__), ROOT / "scripts/optimize_flexible_50mhz_echo.py",
                    ROOT / "echospec/simulation/flexible_echo.py",
                    ROOT / "echospec/simulation/decoupled_echo.py",
                    ROOT / "echospec/simulation/polynomial_echo.py",
                    ROOT / "echospec/simulation/qutrit_rk4.cpp",
                    ROOT / "echospec/analysis/dip_metrics.py",
                    ROOT / "scripts/optimize_50mhz_echo.py"]
    source_hashes = {str(path.relative_to(ROOT)):
                     hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    populations, diagnostics = simulate_polynomial_echo(
        pulse, amplitudes/norm, DET, dt_ns=0.5
    )
    rows = metrics(populations, amplitudes, DET)
    quality, inequalities = assess(rows, baseline)
    comparisons = []
    for row, original in zip(rows, baseline, strict=True):
        width_delta = (row["fwhm_hz"]-original["fwhm_hz"])/1000
        contrast_delta = row["contrast"]-original["contrast"]
        checks = {
            "width_regression": width_delta > 0.05,
            "contrast_regression": contrast_delta < -0.0001,
            "satellite": row["satellite_prominence"] > 0.01,
            "center": abs(row["center_hz"]) > 2000,
            "leakage": row["final_leakage_max"] > 0.005,
            "contrast50": row["rabi_mhz"] == 50 and row["contrast"] <= 0.4,
        }
        comparisons.append(dict(
            amplitude_mhz=row["rabi_mhz"],
            failed=[name for name, failed in checks.items() if failed],
            width_regression_khz=width_delta, contrast_change=contrast_delta,
            satellite_prominence=row["satellite_prominence"],
            center_khz=row["center_hz"]/1000,
            final_leakage=row["final_leakage_max"],
        ))
    critical = dict(
        width=sorted(comparisons, key=lambda r: -r["width_regression_khz"])[:3],
        contrast=sorted(comparisons, key=lambda r: r["contrast_change"])[:3],
        satellite=sorted(comparisons, key=lambda r: -r["satellite_prominence"])[:3],
    )
    result = dict(
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        baseline_source=str(baseline_path),
        baseline_source_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        selected_training_feasible=bool(feasible), selected_record=selected,
        amplitudes_mhz=amplitudes.tolist(), detuning_mhz=DET.tolist(), dt_ns=0.5,
        peak_scale=norm, baseline=baseline, rows=rows, quality=quality,
        inequalities=inequalities.tolist(), diagnostics=diagnostics,
        violations=[r for r in comparisons if r["failed"]], critical=critical,
        source_hashes=source_hashes,
        validation_boundary="Native qutrit, coarse frequency grid, dense amplitude grid",
    )
    (OUT / "rotating_adaptive_dense_audit.json").write_text(
        json.dumps(clean(result), indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(OUT / "rotating_adaptive_dense_audit.npz",
                        amplitudes_mhz=amplitudes, detuning_mhz=DET,
                        populations=populations)
    print(json.dumps(clean(dict(quality=quality, rotation_mix=pulse.rotation_mix,
                                violations=result["violations"], critical=critical))),
          flush=True)


if __name__ == "__main__":
    main()
