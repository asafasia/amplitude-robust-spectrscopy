"""Separate the effects of duration, cutoff, and taper power in the retuned echo."""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.simulation.polynomial_echo import (  # noqa: E402
    PolynomialEchoPulse,
    simulate_polynomial_echo,
)
from scripts.optimize_50mhz_echo import OUT, metrics, peak_scale  # noqa: E402


def main():
    source = OUT / "results.json"
    saved = json.loads(source.read_text())
    original = PolynomialEchoPulse(**saved["cases"]["baseline"]["pulse"])
    retuned = PolynomialEchoPulse(**saved["cases"]["retuned"]["pulse"])
    fields = ("duration_us", "cutoff", "power")
    cases = {}
    for switches in product((False, True), repeat=3):
        changed = [field for field, switch in zip(fields, switches, strict=True) if switch]
        name = "+".join(changed) if changed else "original"
        cases[name] = replace(original, **{field: getattr(retuned, field) for field in changed})
    cases.update(
        cutoff_halved=replace(retuned, cutoff=retuned.cutoff / 2),
        cutoff_fifth=replace(retuned, cutoff=retuned.cutoff / 5),
        transition_8ns=replace(retuned, transition_us=.008),
        transition_32ns=replace(retuned, transition_us=.032),
    )
    amplitudes = np.array([40., 50., 60.])
    detuning = np.linspace(-.5, .5, 401)

    def evaluate(item):
        name, pulse = item
        peak = peak_scale(pulse)
        populations, diagnostics = simulate_polynomial_echo(
            pulse, amplitudes / peak, detuning, dt_ns=.25)
        rows = metrics(populations, amplitudes, detuning)
        record = dict(name=name, pulse=asdict(pulse), core_sigma_ns=1000 * pulse.sigma_us,
                      nominal_scale_for_actual_50mhz=50 / peak, metrics=rows,
                      min_contrast_40_50_60=min(r["contrast"] for r in rows),
                      max_satellite_40_50_60=max(r["satellite_prominence"] for r in rows),
                      diagnostics=diagnostics)
        print(name, rows[1]["fwhm_hz"] / 1000, rows[1]["contrast"], flush=True)
        return record, populations

    with ThreadPoolExecutor(max_workers=3) as pool:
        evaluated = list(pool.map(evaluate, cases.items()))
    metadata = dict(
        purpose="Full factorial ablation of the three retuned parameters, plus local boundary checks",
        amplitude_convention="maximum absolute complete complex I-iQ drive",
        coherence_model="same three-level Lindblad model as the base study",
        t1_us=51.24, t2_us=7.31, dt_ns=.25,
        original_results_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__), ROOT / "scripts/optimize_50mhz_echo.py",
                                    ROOT / "echospec/simulation/polynomial_echo.py",
                                    ROOT / "echospec/simulation/qutrit_rk4.cpp")},
        validation_boundary="Diagnostic sweeps at three amplitudes, not a new optimization or hardware test",
    )
    (OUT / "retuning_parameter_audit.json").write_text(json.dumps(
        dict(metadata=metadata, cases=[r for r, _ in evaluated]), indent=2, allow_nan=False) + "\n")
    np.savez_compressed(OUT / "retuning_parameter_audit.npz", detuning_mhz=detuning,
                        rabi_mhz=amplitudes, **{r["name"]: p for r, p in evaluated})


if __name__ == "__main__":
    main()
