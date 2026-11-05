"""Separate finite-control and dissipative effects for frozen echo candidates.

Run with MPLCONFIGDIR=/tmp/echo50-mpl .venv/bin/python
scripts/diagnose_flexible_echo_linewidth.py. Existing study files are read-only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.analysis.dip_metrics import dip_metrics  # noqa: E402
from echospec.simulation.flexible_echo import FlexibleEchoPulse  # noqa: E402
from echospec.simulation.polynomial_echo import (  # noqa: E402
    PolynomialEchoPulse,
    simulate_polynomial_echo,
)
from scripts.report_decoupled_50mhz_echo import (  # noqa: E402
    clean,
    local_shoulder_metrics,
)

SOURCE = ROOT / "outputs/flexible_echo_50mhz_20261101"
OUT = ROOT / "outputs/flexible_echo_linewidth_diagnosis_20261101"
CASES = ("baseline", "cleaner_smooth", "flexible_phase")
LABELS = ("Original polynomial", "Smooth, higher contrast", "Narrow phase-shaped")
MODES = {
    "physical": (51.24, 7.31),
    "no_pure_dephasing": (51.24, 102.48),
    "negligible_dissipation": (1e9, 1e9),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    saved = json.loads((SOURCE / "results.json").read_text())
    with np.load(SOURCE / "validated_spectra.npz") as source_arrays:
        detuning = source_arrays["fine_detuning_mhz"].copy()
        physical = {name: source_arrays[f"{name}_fine50"].copy()
                    for name in CASES}
    classes = {"PolynomialEchoPulse": PolynomialEchoPulse,
               "FlexibleEchoPulse": FlexibleEchoPulse}
    arrays = {"detuning_mhz": detuning}
    records = {}

    def simulate(name, mode):
        case = saved["cases"][name]
        pulse = classes[case["pulse_type"]](**case["pulse"])
        t1, t2 = MODES[mode]
        # Coherent density matrices sit on the positivity boundary, requiring
        # finer steps than the dissipative cases. Keep the same physicality
        # tolerance and refine rather than clipping negative eigenvalues.
        dt_ns = .0625 if mode == "negligible_dissipation" else .125
        pop, diag = simulate_polynomial_echo(
            pulse, [50 / case["unit_waveform_peak"]], detuning,
            t1_us=t1, t2_us=t2, dt_ns=dt_ns)
        diag["requested_dt_ns"] = dt_ns
        return name, mode, pop[:, 0], diag

    def record(name, mode, pop, diag):
        pe = pop[1]
        arrays[f"{name}_{mode}"] = pop
        records.setdefault(name, {})[mode] = {
            "t1_us": MODES[mode][0], "t2_us": MODES[mode][1],
            "metrics": dip_metrics(detuning, pe),
            "local_shoulders": local_shoulder_metrics(detuning, pe),
            "population_at_zero": float(pe[np.argmin(abs(detuning))]),
            "diagnostics": diag,
        }
        m = records[name][mode]["metrics"]
        print(name, mode, m, flush=True)

    for name in CASES:
        record(name, "physical", physical[name], {"imported_dt_ns": .125})
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(simulate, name, mode) for name in CASES
                for mode in MODES if mode != "physical"]
        for future in as_completed(jobs):
            record(*future.result())
    metadata = {
        "cases": records,
        "pulse_records": {name: saved["cases"][name]["pulse"] for name in CASES},
        "actual_peak_mhz": 50,
        "dt_ns": {"physical": .125, "no_pure_dephasing": .125,
                  "negligible_dissipation": .0625},
        "note": "Frozen pulses; no reoptimization. Widths are not additive. "
                "Removing pure dephasing retains T1 and sets T2=2*T1. "
                "Negligible dissipation uses finite 1e9 us lifetimes.",
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
            Path(__file__), SOURCE / "results.json",
            SOURCE / "validated_spectra.npz",
            ROOT / "echospec/simulation/flexible_echo.py",
            ROOT / "echospec/simulation/polynomial_echo.py",
            ROOT / "echospec/simulation/qutrit_rk4.cpp")},
    }
    (OUT / "results.json").write_text(json.dumps(clean(metadata), indent=2) + "\n")
    np.savez_compressed(OUT / "spectra.npz", **arrays)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True,
                             layout="constrained")
    styles = (("physical", "With measured T1/T2", "C0"),
              ("no_pure_dephasing", "T1 only", "C1"),
              ("negligible_dissipation", "Dissipation off", "C2"))
    for ax, name, label in zip(axes, CASES, LABELS, strict=True):
        for mode, legend, color in styles:
            width = records[name][mode]["metrics"]["fwhm_hz"] / 1000
            ax.plot(detuning * 1000, arrays[f"{name}_{mode}"][1],
                    label=f"{legend}: {width:.1f} kHz", color=color)
        ax.set(title=label, xlabel="Detuning (kHz)", xlim=(-250, 250),
               ylim=(0, 1.02))
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Excited-state probability")
    fig.suptitle("Same pulse, same 50 MHz peak: switch dissipation on/off")
    fig.savefig(OUT / "linewidth_components.png", dpi=180)
    fig.savefig(OUT / "linewidth_components.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
