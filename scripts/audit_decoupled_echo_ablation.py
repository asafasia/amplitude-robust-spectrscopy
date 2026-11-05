"""Ablate phase, tail weighting, and endpoint taper of the saved narrow trial."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.simulation.decoupled_echo import DecoupledEchoPulse  # noqa: E402
from echospec.simulation.polynomial_echo import simulate_polynomial_echo  # noqa: E402
from scripts.optimize_50mhz_echo import metrics, peak_scale  # noqa: E402
from scripts.optimize_decoupled_50mhz_echo import AMPS, DET, clean  # noqa: E402

OUT = ROOT / "outputs/decoupled_echo_50mhz_20261101"


def main():
    manifest = OUT / "selected_candidates.json"
    selection = json.loads(manifest.read_text())
    record = next(r for r in selection["candidates"] if r["name"] == "preserve_original")
    pulse = DecoupledEchoPulse(**record["pulse"])
    variants = {
        "selected": pulse,
        "phase_off": replace(pulse, phase_rad=0, phase2_rad=0),
        "unit_tail_gain": replace(pulse, tail_gain=1),
        "taper_over_whole_half": replace(pulse, edge_width_us=pulse.duration_us / 2),
    }
    rows, arrays = {}, {"rabi_mhz": AMPS, "detuning_mhz": DET}
    for name, variant in variants.items():
        norm = peak_scale(variant)
        population, diagnostics = simulate_polynomial_echo(variant, AMPS / norm, DET, dt_ns=.5)
        rows[name] = dict(pulse=asdict(variant), peak_scale=norm,
                          rows=metrics(population, AMPS, DET), diagnostics=diagnostics)
        arrays[name] = population
        print(name, [(r["rabi_mhz"], round(r["fwhm_hz"] / 1000, 3), round(r["contrast"], 5))
                     for r in rows[name]["rows"] if r["rabi_mhz"] in (5, 50, 60)], flush=True)
    sources = [Path(__file__), ROOT / "echospec/simulation/decoupled_echo.py",
               ROOT / "echospec/simulation/polynomial_echo.py",
               ROOT / "echospec/simulation/qutrit_rk4.cpp",
               ROOT / "echospec/analysis/dip_metrics.py", ROOT / "scripts/optimize_50mhz_echo.py"]
    metadata = dict(
        t1_us=51.24, t2_us=7.31, dt_ns=.5,
        comparison="Each variant is renormalized to the same actual complete-waveform peak.",
        limitation="One-at-a-time ablations are not reoptimized alternatives or causal proofs.",
        selection_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sources},
    )
    (OUT / "ablation.json").write_text(json.dumps(clean(dict(metadata=metadata, cases=rows)),
                                                 indent=2, allow_nan=False) + "\n")
    np.savez_compressed(OUT / "ablation.npz", **arrays)


if __name__ == "__main__":
    main()
