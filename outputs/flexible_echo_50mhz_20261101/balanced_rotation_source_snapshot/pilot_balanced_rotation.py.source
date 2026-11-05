"""Fit first-order area-balanced rotation seeds, then simulate five candidates."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from echospec.simulation.flexible_echo import FlexibleEchoPulse  # noqa: E402
from echospec.simulation.polynomial_echo import (  # noqa: E402
    PolynomialEchoPulse,
    simulate_polynomial_echo,
)
from scripts.optimize_50mhz_echo import metrics, peak_scale  # noqa: E402
from scripts.optimize_decoupled_50mhz_echo import clean  # noqa: E402
from scripts.optimize_flexible_50mhz_echo import (  # noqa: E402
    AMPS,
    DET,
    assess,
    encode,
)

OUT = Path(__file__).resolve().parent
EXTRA_ROTATIONS = (0.15, -0.3, 1.0, -1.0)


def quadrature(pulse: FlexibleEchoPulse, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate analytic waveforms on piecewise high-order Gauss-Legendre nodes."""
    half = pulse.duration_us / 2
    positive = np.array([0, pulse.transition_us / 2, pulse.core_us, 0.02, 0.05,
                         0.1, 0.2, 0.5, 1, 2, 4, 6, half-pulse.edge_width_us, half])
    positive = np.unique(positive[(positive >= 0) & (positive <= half)])
    edges = np.unique(np.r_[-positive, positive])
    nodes, weights = leggauss(order)
    radii = np.diff(edges) / 2
    centers = (edges[:-1] + edges[1:]) / 2
    return ((centers[:, None] + radii[:, None]*nodes).ravel(),
            (radii[:, None]*weights).ravel())


def complex_record(value: complex) -> dict:
    return dict(real=float(value.real), imaginary=float(value.imag),
                magnitude=float(abs(value)))


def audit_area(pulse: FlexibleEchoPulse, nodes: np.ndarray, weights: np.ndarray) -> dict:
    raw, derivative = pulse.raw_drive_and_derivative(nodes)
    full = pulse.drive(nodes)
    raw_area = weights @ raw
    full_area = weights @ full
    ends = pulse.drive(np.array([-pulse.duration_us/2, pulse.duration_us/2]))
    return dict(raw_area_us=complex_record(raw_area),
                complete_area_us=complex_record(full_area),
                derivative_integral=complex_record(weights @ derivative),
                full_minus_raw_area_us=complex_record(full_area-raw_area),
                integral_abs_raw_us=float(weights @ abs(raw)),
                endpoint_drive_abs=abs(ends).tolist(),
                quadrature_nodes=len(nodes))


def main() -> None:
    source_path = OUT / "smooth_base/best.json"
    source = json.loads(source_path.read_text())
    start = FlexibleEchoPulse(**source["pulse"])
    center_cancellation = (start.drag_beta * 1.875
                           / (2 * abs(start.anharmonicity_mhz) * start.transition_us))
    rotations = (center_cancellation, *EXTRA_ROTATIONS)
    fit_nodes, fit_weights = quadrature(start, 96)
    audit_nodes, audit_weights = quadrature(start, 192)
    baseline_pulse = PolynomialEchoPulse()
    baseline_populations, baseline_diagnostics = simulate_polynomial_echo(
        baseline_pulse, AMPS/peak_scale(baseline_pulse), DET, dt_ns=0.5
    )
    baseline = metrics(baseline_populations, AMPS, DET)
    paths = [Path(__file__), source_path,
             ROOT / "scripts/optimize_flexible_50mhz_echo.py",
             ROOT / "scripts/optimize_50mhz_echo.py",
             ROOT / "echospec/simulation/flexible_echo.py",
             ROOT / "echospec/simulation/decoupled_echo.py",
             ROOT / "echospec/simulation/polynomial_echo.py",
             ROOT / "echospec/simulation/qutrit_rk4.cpp",
             ROOT / "echospec/analysis/dip_metrics.py"]
    config = dict(
        initial_pulse=asdict(start), rotation_mixes=list(rotations),
        first_rotation_seed="Exact central DRAG cancellation for source geometry",
        fitted_fields="odd_phase_coefficients", coefficient_bounds=[-3.0, 3.0],
        odd_phase_scales_us=list(start.ODD_PHASE_SCALES_US),
        area_constraints="Real and imaginary raw-drive integrals equal zero",
        fit_method="bounded scipy least_squares; zero initial coefficients",
        fit_quadrature_order_per_segment=96, audit_quadrature_order_per_segment=192,
        fit_max_function_evaluations=400,
        amplitudes_mhz=AMPS.tolist(), detuning_mhz=DET.tolist(), dt_ns=0.5,
        t1_us=51.24, t2_us=7.31, candidate_simulations=5, baseline_simulations=1,
        baseline=baseline, baseline_diagnostics=baseline_diagnostics,
        quality_function="scripts.optimize_flexible_50mhz_echo.assess",
        constraints="C50>0.4; no FWHM/contrast regression vs original at5--60MHz",
        source_sha256={str(path.relative_to(ROOT)):
                       hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    )
    config_hash = hashlib.sha256(
        json.dumps(clean(config), sort_keys=True).encode()
    ).hexdigest()
    (OUT / "balanced_rotation_seed_configuration.json").write_text(
        json.dumps(clean(config), indent=2, allow_nan=False) + "\n"
    )
    records, populations_all = [], []
    for mix in rotations:
        seed = replace(start, rotation_mix=mix)
        before = audit_area(seed, audit_nodes, audit_weights)
        scale = before["integral_abs_raw_us"]

        def residual(coefficients, seed=seed, scale=scale):
            pulse = replace(seed, odd_phase_coefficients=tuple(coefficients))
            raw, _ = pulse.raw_drive_and_derivative(fit_nodes)
            integral = fit_weights @ raw
            return np.array([integral.real, integral.imag]) / scale

        result = least_squares(residual, np.zeros(3), bounds=(-3.0, 3.0),
                               max_nfev=400, ftol=1e-13, xtol=1e-13, gtol=1e-13)
        pulse = replace(seed, odd_phase_coefficients=tuple(result.x))
        after_fit_grid = audit_area(pulse, fit_nodes, fit_weights)
        after = audit_area(pulse, audit_nodes, audit_weights)
        norm = peak_scale(pulse)
        populations, diagnostics = simulate_polynomial_echo(
            pulse, AMPS/norm, DET, dt_ns=0.5
        )
        rows = metrics(populations, AMPS, DET)
        quality, inequalities = assess(rows, baseline)
        record = dict(
            name=f"area_balanced_eta_{mix:+.5f}", original_rotation_mix=mix,
            pulse=asdict(pulse), parameters=encode(asdict(pulse)).tolist(),
            peak_scale=norm, rows=rows, quality=quality, loss=quality["loss"],
            inequalities=inequalities.tolist(), diagnostics=diagnostics,
            area_fit=dict(success=bool(result.success), status=int(result.status),
                          message=str(result.message), nfev=int(result.nfev),
                          coefficients=result.x.tolist(), initial=before,
                          final=after, final_on_fit_grid=after_fit_grid,
                          normalized_raw_area_magnitude_us=(
                              after["raw_area_us"]["magnitude"]/norm),
                          normalized_complete_area_magnitude_us=(
                              after["complete_area_us"]["magnitude"]/norm)),
            configuration_sha256=config_hash,
        )
        records.append(record)
        populations_all.append(populations)
        print(json.dumps(clean(dict(rotation_mix=mix,
                                    coefficients=result.x.tolist(),
                                    raw_area_us=after["raw_area_us"],
                                    quality=quality))), flush=True)
    ranked = sorted(records, key=lambda record: record["loss"])
    summary = dict(
        candidates=ranked, evaluations=len(records), baseline_evaluations=1,
        feasible_count=sum(record["quality"]["feasible"] for record in records),
        interpretation=("First-order area-compensation prototype; not exact echo "
                        "cancellation or amplitude independence. Coarse native grid only."),
        configuration_sha256=config_hash,
    )
    (OUT / "balanced_rotation_seeds.json").write_text(
        json.dumps(clean(summary), indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(OUT / "balanced_rotation_seeds.npz", rotation_mixes=rotations,
                        amplitudes_mhz=AMPS, detuning_mhz=DET,
                        populations=np.asarray(populations_all),
                        baseline_populations=baseline_populations)
    print("DONE", len(records), "best", ranked[0]["original_rotation_mix"],
          ranked[0]["quality"], flush=True)


if __name__ == "__main__":
    main()
