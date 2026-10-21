"""Resolve the simulated echo spectral-center minimum versus DRAG beta.

Run with .venv/bin/python scripts/scan_drag_beta_center.py. No hardware access.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echospec.simulation.qutrit import simulate_qutrit_map  # noqa: E402

OUT = ROOT / "outputs/drag_beta_center_scan"
PARAMETERS = dict(
    duration_us=20.0, cutoff=0.005, echo=True, order=0.5,
    echo_transition_us=0.016, echo_transition_shape="sine",
    t1_us=51.24, t_phi_us=1 / (1 / 7.31 - 1 / (2 * 51.24)),
    anharmonicity_mhz=-216.0, stark_kappa_mhz_inv=0.0,
    integration_backend="native",
)


def centers_hz(detuning, populations):
    """Three-point parabolic refinement of each raw central minimum."""
    centers = []
    for row in populations:
        index = int(np.argmin(row))
        if index in (0, len(row) - 1):
            raise ValueError("Minimum at scan boundary")
        x = detuning[index - 1:index + 2] - detuning[index]
        a, b, _ = np.polyfit(x, row[index - 1:index + 2], 2)
        offset = -b / (2 * a)
        if a <= 0 or not x[0] <= offset <= x[-1]:
            raise ValueError("Invalid local parabolic minimum")
        centers.append((detuning[index] + offset) * 1e6)
    return np.asarray(centers)


def simulate(beta, detuning, rabi, steps):
    result = simulate_qutrit_map(
        **PARAMETERS, drag_beta=float(beta), detuning_mhz=detuning,
        rabi_mhz=rabi, num_steps_per_half=steps,
    )
    return np.array([result.ground, result.excited, result.second_excited])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    betas = np.unique(np.round(np.r_[np.linspace(-1, 0, 11),
                                    np.arange(-0.56, -0.439, 0.005)], 6))
    rabi = np.linspace(10, 42.72262747, 7)
    detuning = np.linspace(-0.06, 0.06, 121)
    populations, centers = [], []
    for beta in betas:
        pop = simulate(beta, detuning, rabi, 20000)
        populations.append(pop)
        centers.append(centers_hz(detuning, pop[1]))
        print(f"beta={beta:g} complete", flush=True)
    centers = np.asarray(centers)
    rms = np.sqrt(np.mean(centers**2, axis=1))
    best = int(np.argmin(rms))
    checks = []
    refined_detuning = np.linspace(-0.06, 0.06, 241)
    for j in range(best - 1, best + 2):
        fine = simulate(betas[j], refined_detuning, rabi, 40000)
        fine_centers = centers_hz(refined_detuning, fine[1])
        checks.append(dict(
            beta=float(betas[j]), rms_center_hz=float(np.sqrt(np.mean(fine_centers**2))),
            max_center_change_hz=float(np.max(abs(fine_centers - centers[j]))),
            max_population_change=float(np.max(abs(fine[:, :, ::2] - populations[j]))),
        ))
    np.savez_compressed(
        OUT / "scan.npz", beta=betas, rabi_mhz=rabi, detuning_mhz=detuning,
        populations=np.array(populations), centers_hz=centers, rms_center_hz=rms,
    )
    sources = [Path(__file__), ROOT / "echospec/simulation/qutrit.py",
               ROOT / "echospec/simulation/qutrit_native.py",
               ROOT / "echospec/simulation/qutrit_rk4.cpp"]
    provenance = dict(
        parameters=PARAMETERS, beta=betas.tolist(), rabi_mhz=rabi.tolist(),
        objective="RMS of seven raw central-dip positions about zero detuning",
        extraction="Three-point parabolic interpolation; no Gaussian fitting",
        detuning_mhz=[-0.06, 0.06], detuning_step_khz=1.0, time_step_ns=0.5,
        best_sampled_beta=float(betas[best]), best_rms_center_hz=float(rms[best]),
        beta_zero_rms_center_hz=float(rms[-1]),
        convergence=dict(time_step_ns=0.25, detuning_step_khz=0.5, checks=checks),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sources},
        reproduce=".venv/bin/python scripts/scan_drag_beta_center.py",
        limitation="Finite-grid simulation minimum for this pulse and center objective; not experimental optimization or minimum leakage",
    )
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({k: provenance[k] for k in
                      ["best_sampled_beta", "best_rms_center_hz", "convergence"]}, indent=2))


if __name__ == "__main__":
    main()
