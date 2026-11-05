"""Bounded 24-candidate transition/core-phase pilot on an existing echo family."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from echospec.simulation.decoupled_echo import DecoupledEchoPulse  # noqa: E402
from echospec.simulation.polynomial_echo import simulate_polynomial_echo  # noqa: E402
from scripts.optimize_50mhz_echo import metrics, peak_scale  # noqa: E402
from scripts.optimize_decoupled_50mhz_echo import (  # noqa: E402
    AMPS,
    DET,
    clean,
    constraints,
)

OUT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "outputs/decoupled_echo_50mhz_20261101"


def main() -> None:
    source_path = PREVIOUS / "preserve_original_validation.json"
    baseline_path = PREVIOUS / "preserve_original_local/configuration.json"
    source = json.loads(source_path.read_text())
    baseline_config = json.loads(baseline_path.read_text())
    assert np.array_equal(baseline_config["amplitudes_mhz"], AMPS)
    assert np.array_equal(baseline_config["detuning_mhz"], DET)
    baseline = baseline_config["baseline"]
    start = DecoupledEchoPulse(**source["pulse"])
    paths = [
        Path(__file__), source_path, baseline_path,
        ROOT / "echospec/simulation/decoupled_echo.py",
        ROOT / "echospec/simulation/polynomial_echo.py",
        ROOT / "echospec/simulation/qutrit_rk4.cpp",
        ROOT / "scripts/optimize_50mhz_echo.py",
        ROOT / "scripts/optimize_decoupled_50mhz_echo.py",
        ROOT / "echospec/analysis/dip_metrics.py",
    ]
    config = dict(
        amplitude_mhz=AMPS.tolist(), detuning_mhz=DET.tolist(), dt_ns=0.5,
        t1_us=51.24, t2_us=7.31, initial_pulse=asdict(start), baseline=baseline,
        native_candidate_limit=24, baseline_reused_without_resimulation=True,
        first_stage_transition_ns=[4, 8, 16, 32, 64, 128, 256],
        first_stage_phase2_rad=[-0.8, 0.0, 0.8], phase2_scale_us=0.1,
        second_stage=(
            "Use narrowest feasible first-stage transition, otherwise lowest loss; "
            "probe phase2 +/-1.6 rad at100ns and +.8 rad at30ns."
        ),
        constraints="C50>0.4; no contrast/FWHM regression vs original across5--60MHz",
        feasibility_tolerances=dict(contrast=0.001, width_khz=0.25),
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    )
    config_hash = hashlib.sha256(
        json.dumps(clean(config), sort_keys=True).encode()
    ).hexdigest()
    (OUT / "pilot_transition_phase_configuration.json").write_text(
        json.dumps(clean(config), indent=2, allow_nan=False) + "\n"
    )
    records = []
    populations_all = []

    def probe(transition_ns: float, phase: float, phase_scale: float) -> None:
        assert len(records) < 24
        pulse = replace(start, transition_us=transition_ns / 1000,
                        phase2_rad=phase, phase2_scale_us=phase_scale)
        norm = peak_scale(pulse)
        populations, diagnostics = simulate_polynomial_echo(
            pulse, AMPS / norm, DET, dt_ns=0.5
        )
        rows = metrics(populations, AMPS, DET)
        quality = constraints(rows, baseline, contrast_at50_only=True)
        loss = (0.7 * quality["width50_khz"] + 0.3 * quality["max_width_khz"]
                + quality["violation"])
        record = dict(index=len(records), transition_ns=transition_ns,
                      phase2_rad=phase, phase2_scale_us=phase_scale,
                      pulse=asdict(pulse), peak_scale=norm, rows=rows,
                      quality=quality, loss=loss, diagnostics=diagnostics,
                      configuration_sha256=config_hash)
        records.append(record)
        populations_all.append(populations)
        print(json.dumps(clean(dict(index=record["index"], transition_ns=transition_ns,
                                    phase2_rad=phase, phase2_scale_us=phase_scale,
                                    quality=quality))), flush=True)

    for transition in config["first_stage_transition_ns"]:
        for phase in config["first_stage_phase2_rad"]:
            probe(transition, phase, 0.1)
    feasible = [record for record in records if record["quality"]["feasible"]]
    selected = (min(feasible, key=lambda row: row["quality"]["width50_khz"])
                if feasible else min(records, key=lambda row: row["loss"]))
    transition = selected["transition_ns"]
    probe(transition, -1.6, 0.1)
    probe(transition, 1.6, 0.1)
    probe(transition, 0.8, 0.03)
    feasible = sorted((r for r in records if r["quality"]["feasible"]),
                      key=lambda row: row["quality"]["width50_khz"])
    summary = dict(
        configuration_sha256=config_hash, evaluations=len(records),
        feasible_count=len(feasible), candidates=feasible[:8],
        best_feasible=feasible[0] if feasible else None,
        best_loss=min(records, key=lambda row: row["loss"]), records=records,
        validation_boundary="Coarse native pilot only; dense-grid validation required.",
    )
    (OUT / "pilot_transition_phase.json").write_text(
        json.dumps(clean(summary), indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(OUT / "pilot_transition_phase.npz", amplitudes_mhz=AMPS,
                        detuning_mhz=DET, populations=np.asarray(populations_all))
    print("DONE", len(records), "feasible", len(feasible), flush=True)


if __name__ == "__main__":
    main()
