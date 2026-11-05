"""Recalculate Figure S7 for the current 20 us, 50 ns, p=2 pulse settings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from echospec.analysis.spectroscopy_quality import spectroscopy_quality
from echospec.paper_data import save_paper_dataset
from echospec.simulation.polynomial_echo import simulate_polynomial_echo
from scripts.make_current_drag_supplement import CurrentPulse

ROOT = Path(__file__).resolve().parents[1]
STEM = "07_current_edge_comparison"


def simulate_dataset():
    reference_path = ROOT / "paper/data/numerical/05_current_drag_comparison.json"
    reference = json.loads(reference_path.read_text())["provenance"]
    model = reference["model"]
    settings = dict(t1_us=model["t1_us"], t2_us=model["t2_us"])
    amplitudes = np.linspace(0, 100, 101)
    detuning = np.linspace(-0.5, 0.5, 201)
    arrays = dict(rabi_mhz=amplitudes, detuning_mhz=detuning)
    configs, diagnostics, checks = {}, {}, []
    for key, polynomial in [("drag", False), ("polynomial", True)]:
        pulse = CurrentPulse(polynomial=polynomial)
        assert pulse.duration_us == reference["pulse"]["duration_us"]
        assert pulse.cutoff == reference["pulse"]["cutoff"]
        assert pulse.transition_us * 1000 == reference["pulse"]["transition_ns"]
        assert pulse.power == reference["pulse"]["polynomial_power"]
        configs[key] = asdict(pulse)
        populations, diagnostic = simulate_polynomial_echo(
            pulse, amplitudes, detuning, dt_ns=0.5, **settings,
        )
        arrays[key], diagnostics[key] = populations, diagnostic
        print(f"{key}: full grid complete", flush=True)
        # Independently refine time and frequency at low, middle and high drive.
        selected = np.array([10, 50, 100])
        fine_detuning = np.linspace(-0.5, 0.5, 401)
        fine_time, _ = simulate_polynomial_echo(
            pulse, amplitudes[selected], detuning, dt_ns=0.25, **settings,
        )
        fine_frequency, _ = simulate_polynomial_echo(
            pulse, amplitudes[selected], fine_detuning, dt_ns=0.5, **settings,
        )
        arrays[f"{key}_refined_time_populations"] = fine_time
        arrays[f"{key}_refined_frequency_populations"] = fine_frequency
        for index, row in enumerate(selected):
            base = spectroscopy_quality(detuning, populations[1, row])
            timed = spectroscopy_quality(detuning, fine_time[1, index])
            frequency = spectroscopy_quality(fine_detuning, fine_frequency[1, index])
            assert base["accepted"] and timed["accepted"] and frequency["accepted"]
            checks.append(dict(
                case=key, rabi_mhz=float(amplitudes[row]),
                time_max_population_change=float(np.max(abs(
                    populations[:, row] - fine_time[:, index]))),
                time_fwhm_change_hz=abs(base["fwhm_hz"] - timed["fwhm_hz"]),
                frequency_fwhm_change_hz=abs(base["fwhm_hz"] - frequency["fwhm_hz"]),
                frequency_contrast_change=abs(base["contrast"] - frequency["contrast"]),
                frequency_snr_change=abs(base["contrast_snr"] - frequency["contrast_snr"]),
            ))
        print(f"{key}: independent time/frequency refinement complete", flush=True)
    arrays["refined_rabi_mhz"] = amplitudes[selected]
    arrays["refined_detuning_mhz"] = fine_detuning
    control, diagnostics["echo_control"] = simulate_polynomial_echo(
        CurrentPulse(polynomial=False, drag_beta=0), [100], detuning,
        dt_ns=0.5, **settings,
    )
    arrays["echo_control_100mhz"] = control[:, 0]
    assert max(c["time_max_population_change"] for c in checks) < 1e-4
    assert max(c["time_fwhm_change_hz"] for c in checks) < 100
    assert max(c["frequency_fwhm_change_hz"] for c in checks) < 2000
    sources = [
        "scripts/make_current_edge_supplement.py",
        "scripts/make_current_drag_supplement.py",
        "echospec/simulation/polynomial_echo.py",
        "echospec/simulation/qutrit_rk4.cpp",
        "echospec/analysis/dip_metrics.py",
        "echospec/analysis/spectroscopy_quality.py",
        str(reference_path.relative_to(ROOT)),
    ]
    provenance = dict(
        model=model, source_configs=configs, diagnostics=diagnostics,
        normalization="Each complete complex pulse has unit max|I-iQ|; amplitude is actual peak Rabi frequency",
        convention="I-iQ; Q=-beta*dI/dt/abs(alpha); numerical beta=-0.5",
        grid=dict(amplitudes=101, max_rabi_mhz=100, detunings=201,
                  detuning_window_mhz=[-0.5, 0.5], dt_ns=0.5),
        convergence=dict(checks=checks, refined_dt_ns=0.25,
                         refined_detuning_step_khz=2.5,
                         scope="Representative low, middle and high drive; separate time and detuning refinement"),
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        reproduce="PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_current_edge_supplement.py",
    )
    save_paper_dataset(STEM, category="numerical", arrays=arrays, provenance=provenance)
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    simulate_dataset()
