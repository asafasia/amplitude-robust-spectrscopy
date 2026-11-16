"""Recompute the supplemental zero-DRAG midpoint comparison through 50 MHz.

Uses the original pulse and coherence parameters, without peak renormalization.
The saved 50 MHz slices are checked at half the time and detuning steps.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from echospec.paper_data import save_paper_dataset
from echospec.simulation.qutrit import simulate_qutrit_map
from scripts.make_sine_echo_transition_comparison import COMMON, WIDTHS

ROOT = Path(__file__).resolve().parents[1]
STEM = "08_midpoint_smoothing_50mhz"
RABI_MHZ = np.linspace(0.0, 50.0, 26)
DETUNING_MHZ = np.linspace(-0.5, 0.5, 201)
REFINED_DETUNING_MHZ = np.linspace(-0.5, 0.5, 401)
DT_NS = 0.125


def simulate(width_ns, rabi_mhz, detuning_mhz, dt_ns):
    """Return all three final populations with the unchanged pulse model."""
    result = simulate_qutrit_map(
        **COMMON,
        echo_transition_us=width_ns / 1000,
        rabi_mhz=rabi_mhz,
        detuning_mhz=detuning_mhz,
        num_steps_per_half=round(COMMON["duration_us"] * 1000 / (2 * dt_ns)),
        integration_backend="native",
    )
    return np.array([result.ground, result.excited, result.second_excited])


def run_case(width_ns):
    populations = simulate(width_ns, RABI_MHZ, DETUNING_MHZ, DT_NS)
    refined = simulate(
        width_ns, RABI_MHZ[-1:], REFINED_DETUNING_MHZ, DT_NS / 2
    )
    print(f"Completed {width_ns} ns transition and 50 MHz refinement", flush=True)
    return populations, refined


def leakage_metrics(populations, widths, detuning_mhz):
    """Keep the highest-drive leakage distinct from the whole-grid maximum."""
    return [
        {
            "transition_ns": int(width),
            "max_final_pf": float(pop[2].max()),
            "max_final_pf_at_highest_drive": float(pop[2, -1].max()),
            "highest_drive_maximum_detuning_mhz": float(
                detuning_mhz[np.argmax(pop[2, -1])]
            ),
        }
        for width, pop in zip(widths, populations, strict=True)
    ]


def main():
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_case, WIDTHS))
    populations = np.stack([result[0] for result in results])
    refined = np.stack([result[1] for result in results])
    checks = []
    for width, pop, fine in zip(WIDTHS, populations, refined, strict=True):
        delta = fine[:, 0, ::2] - pop[:, -1]
        max_error = float(np.max(np.abs(delta)))
        print(f"T={width} ns: refinement changes populations by {max_error:.3g}",
              flush=True)
        if max_error > 3e-6:
            raise RuntimeError(f"Unresolved time-step error at T={width}: {max_error}")
        checks.append({
            "transition_ns": width,
            "max_abs_population_change": max_error,
            "max_abs_final_pf_change": float(np.max(np.abs(delta[2]))),
            "refined_max_final_pf_at_50mhz": float(fine[2].max()),
            "detuning_refinement_change_in_max_pf": float(
                fine[2].max() - fine[2, 0, ::2].max()
            ),
        })
    sources = (
        "scripts/make_midpoint_leakage_50mhz.py",
        "scripts/make_sine_echo_transition_comparison.py",
        "echospec/simulation/qutrit.py",
        "echospec/simulation/qutrit_native.py",
        "echospec/simulation/qutrit_rk4.cpp",
    )
    metadata = {
        "parameters": COMMON,
        "transition_ns": list(WIDTHS),
        "dt_ns": DT_NS,
        "integration_backend": "native",
        "highest_drive_mhz": float(RABI_MHZ[-1]),
        "metrics": leakage_metrics(populations, WIDTHS, DETUNING_MHZ),
        "convergence": {
            "rabi_mhz": 50.0,
            "dt_ns": [DT_NS, DT_NS / 2],
            "detuning_step_mhz": [0.005, 0.0025],
            "checks": checks,
        },
        "grid_note": "26 drives from 0 to 50 MHz; 201 detunings from -0.5 to 0.5 MHz",
        "convention": "I=L*s without peak renormalization; beta=0, Q=0",
        "observable": "Final Pf; not maximum transient leakage",
        "population_order": ["ground", "excited", "second_excited"],
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in sources
        },
        "reproduce": (
            "PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python "
            "scripts/make_midpoint_leakage_50mhz.py"
        ),
    }
    paths = save_paper_dataset(
        STEM,
        category="numerical",
        arrays={
            "transition_ns": WIDTHS,
            "rabi_mhz": RABI_MHZ,
            "detuning_mhz": DETUNING_MHZ,
            "populations": populations,
            "refined_detuning_mhz": REFINED_DETUNING_MHZ,
            "refined_50mhz_populations": refined,
        },
        provenance=metadata,
    )
    for path in paths:
        print(path.relative_to(ROOT), flush=True)
    for metric in metadata["metrics"]:
        print(metric, flush=True)


if __name__ == "__main__":
    main()
