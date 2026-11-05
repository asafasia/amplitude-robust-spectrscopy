"""Audit best feasible rotation seeds on a 0.5 MHz amplitude grid."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from echospec.simulation.flexible_echo import FlexibleEchoPulse  # noqa: E402
from echospec.simulation.polynomial_echo import (  # noqa: E402
    PolynomialEchoPulse,
    simulate_polynomial_echo,
)
from scripts.optimize_50mhz_echo import metrics, peak_scale  # noqa: E402
from scripts.optimize_decoupled_50mhz_echo import clean  # noqa: E402
from scripts.optimize_flexible_50mhz_echo import DET, assess  # noqa: E402

OUT = Path(__file__).resolve().parent


def main() -> None:
    amplitudes = np.arange(5.0, 60.001, 0.5)
    baseline_pulse = PolynomialEchoPulse()
    baseline_populations, baseline_diagnostics = simulate_polynomial_echo(
        baseline_pulse, amplitudes/peak_scale(baseline_pulse), DET, dt_ns=0.5
    )
    baseline = metrics(baseline_populations, amplitudes, DET)
    results, spectra = {}, {}
    for campaign in ("rotating_shape", "rotating_negative"):
        source = OUT / campaign / "candidates.json"
        data = json.loads(source.read_text())
        feasible = [r for r in data["candidates"] if r["quality"]["feasible"]]
        if not feasible:
            results[campaign] = dict(skipped="No training-feasible candidate")
            continue
        selected = feasible[0]
        pulse = FlexibleEchoPulse(**selected["pulse"])
        norm = peak_scale(pulse)
        populations, diagnostics = simulate_polynomial_echo(
            pulse, amplitudes/norm, DET, dt_ns=0.5
        )
        rows = metrics(populations, amplitudes, DET)
        quality, inequalities = assess(rows, baseline)
        violations = []
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
            if any(checks.values()):
                violations.append(dict(
                    amplitude_mhz=row["rabi_mhz"],
                    failed=[name for name, failed in checks.items() if failed],
                    width_regression_khz=width_delta,
                    contrast_change=contrast_delta,
                    satellite_prominence=row["satellite_prominence"],
                    center_khz=row["center_hz"]/1000,
                    final_leakage=row["final_leakage_max"],
                ))
        results[campaign] = dict(
            source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            selected_record=selected, peak_scale=norm, rows=rows, quality=quality,
            inequalities=inequalities.tolist(), diagnostics=diagnostics,
            violations=violations,
        )
        spectra[campaign] = populations
        print(json.dumps(clean(dict(campaign=campaign, quality=quality,
                                    violations=violations))), flush=True)
    sources = [Path(__file__), ROOT / "scripts/optimize_flexible_50mhz_echo.py",
               ROOT / "echospec/simulation/flexible_echo.py",
               ROOT / "echospec/simulation/decoupled_echo.py",
               ROOT / "echospec/simulation/polynomial_echo.py",
               ROOT / "echospec/simulation/qutrit_rk4.cpp",
               ROOT / "echospec/analysis/dip_metrics.py",
               ROOT / "scripts/optimize_50mhz_echo.py"]
    summary = dict(
        amplitudes_mhz=amplitudes.tolist(), detuning_mhz=DET.tolist(), dt_ns=0.5,
        baseline=baseline, baseline_diagnostics=baseline_diagnostics, campaigns=results,
        validation_boundary="Native qutrit, coarse frequency grid, dense amplitude grid",
        source_sha256={str(path.relative_to(ROOT)):
                       hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
    )
    (OUT / "theory_rotation_dense_audit.json").write_text(
        json.dumps(clean(summary), indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(OUT / "theory_rotation_dense_audit.npz",
                        amplitudes_mhz=amplitudes, detuning_mhz=DET,
                        baseline_populations=baseline_populations, **spectra)


if __name__ == "__main__":
    main()
