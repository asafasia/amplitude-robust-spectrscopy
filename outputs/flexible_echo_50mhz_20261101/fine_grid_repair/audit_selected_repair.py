"""Audit the stopped repair's selected positive-margin pulse, without optimization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from repair_flexible_phase import (
    BASE,
    DENSE_AMPS,
    DET,
    OUT,
    compare,
    make_pulse,
    metrics,
    model_sources,
    peak_scale,
    save,
)

from scripts.report_flexible_50mhz_echo import Cache


def main() -> None:
    source = OUT / "selected_training_repair.json"
    selected = json.loads(source.read_text())
    baseline_source = BASE / "baseline_validation.json"
    reference = json.loads(baseline_source.read_text())
    baseline = reference["rows"]
    assert np.array_equal([row["rabi_mhz"] for row in baseline], DENSE_AMPS)
    with np.load(BASE / "validated_spectra.npz") as saved:
        assert np.array_equal(saved["rabi_mhz"], DENSE_AMPS)
        assert np.array_equal(saved["detuning_mhz"], DET)
        recomputed = metrics(saved["baseline"], DENSE_AMPS, DET)
    for row, check in zip(baseline, recomputed, strict=True):
        assert row["contrast"] == check["contrast"]
        assert row["fwhm_hz"] == check["fwhm_hz"]
    pulse = make_pulse(selected["parameters"])
    norm = peak_scale(pulse)
    cache = Cache(BASE)
    populations, diagnostics, cache_name = cache.native(
        pulse, DENSE_AMPS/norm, DET, dt_ns=0.25
    )
    rows = metrics(populations, DENSE_AMPS, DET)
    quality, inequalities = compare(rows, baseline)
    final = dict(
        **{key: value for key, value in selected.items()
           if key not in ("rows", "quality", "inequalities", "diagnostics",
                          "native_cache", "peak_scale")},
        name="repaired_flexible_phase", label="Repaired flexible phase",
        rows=rows, quality=quality, accepted=quality["feasible"],
        inequalities=inequalities.tolist(), diagnostics=diagnostics,
        native_cache=cache_name, peak_scale=norm,
        amplitudes_mhz=DENSE_AMPS.tolist(), detuning_mhz=DET.tolist(), dt_ns=0.25,
        baseline=baseline, baseline_source=str(baseline_source),
        baseline_source_sha256=hashlib.sha256(baseline_source.read_bytes()).hexdigest(),
        model_sources=model_sources(pulse), repair_training_record=selected,
        selection_source=str(source),
        selection_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stopping_rule=("Stopped at parent instruction after positive training margins; "
                       "acceptance uses true zero regression, not relaxed tolerances."),
    )
    save(BASE / "repaired_flexible_phase.json", final)
    count = len(list((OUT / "evaluations").glob("*.json")))
    save(OUT / "repair_summary.json", dict(
        evaluations=count, accepted=quality["feasible"], dense_quality=quality,
        selected_objective=selected["objective"], native_dense_cache=cache_name,
        optimizer_status="Stopped explicitly after obtaining positive training margins",
        final_native_audits=1,
    ))
    print("DENSE AUDIT", quality, "CACHE", cache_name, flush=True)


if __name__ == "__main__":
    main()
