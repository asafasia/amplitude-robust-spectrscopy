"""DRAG results with q1 measurements and pulse-matched numerical controls.

Run with --simulate once to regenerate the numerical dataset; rendering alone
uses its portable export. No hardware access or experimental refitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from echospec.figures.coherence import REFERENCE_T2_STAR_S
from echospec.figures.drag_convention import (
    PAPER_DRAG_CONVENTION,
    SOURCE_DRAG_CONVENTION,
    polynomial_beta_to_paper,
)
from echospec.figures.supplemental import FIGURE_WIDTH, apply_supplemental_style
from echospec.paper_data import save_paper_dataset
from echospec.simulation.polynomial_echo import (
    PolynomialEchoPulse,
    simulate_polynomial_echo,
)
from scripts.make_latest_paper_measurements import (
    COLORS,
    F3,
    LABELS,
    load_export,
    series,
)
from scripts.scan_drag_beta_center import centers_hz

ROOT = Path(__file__).resolve().parents[1]
STEM = "05_current_drag_comparison"
NUMERICAL = ROOT / "paper/data/numerical"
OUT = ROOT / "figures/paper/supplemental"


@dataclass(frozen=True)
class CurrentPulse(PolynomialEchoPulse):
    """Current envelope shapes with unit complex-envelope peak."""

    duration_us: float = 20.0
    cutoff: float = 0.005
    power: float = 2.0
    transition_us: float = 0.05
    polynomial: bool = False

    def envelope(self, time_us):
        if self.polynomial:
            return super().envelope(time_us)
        t = np.asarray(time_us, dtype=float)
        base = 1 + (t / self.sigma_us) ** 2
        inside = np.abs(t) <= self.duration_us / 2
        return (
            np.where(inside, base ** -0.5, 0),
            np.where(inside, -t / self.sigma_us**2 * base ** -1.5, 0),
        )

    @cached_property
    def peak(self):
        grid = np.linspace(-self.duration_us / 2, self.duration_us / 2, 40001)
        index = int(np.argmax(np.abs(super().drive(grid))))
        assert 0 < index < grid.size - 1
        raw_drive = super().drive
        fit = minimize_scalar(
            lambda t: -abs(raw_drive(t)),
            bounds=(grid[index - 1], grid[index + 1]), method="bounded",
            options={"xatol": 1e-13},
        )
        assert fit.success
        return float(-fit.fun)

    def drive(self, time_us):
        return super().drive(time_us) / self.peak


def source_hashes():
    names = [
        "scripts/make_current_drag_supplement.py",
        "scripts/scan_drag_beta_center.py",
        "echospec/figures/drag_convention.py",
        "echospec/simulation/polynomial_echo.py",
        "echospec/simulation/qutrit_rk4.cpp",
        "paper/data/experimental/03_latest_q1_pulse_improvement.npz",
        "paper/data/experimental/03_latest_q1_pulse_improvement.json",
    ]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def simulate_dataset():
    _, experiment = load_export("03_latest_q1_pulse_improvement")
    t1_us = float(np.mean([
        fit["t1"] / 1000 for name, fit in experiment["calibrations"].items()
        if name.endswith("_t1") and fit.get("success", False)
    ]))
    t2_us = REFERENCE_T2_STAR_S * 1e6
    beta = np.unique(np.round(np.r_[np.linspace(-1, 0, 11),
                                    np.arange(-0.60, -0.399, 0.025)], 6))
    amplitudes = np.arange(10.0, 61, 10)
    detuning = np.linspace(-0.12, 0.12, 121)
    arrays = dict(beta=beta, scan_rabi_mhz=amplitudes, detuning_mhz=detuning)
    checks = []

    def run(pulse, rabi, delta, dt=0.5):
        pop, diagnostics = simulate_polynomial_echo(
            pulse, rabi, delta, t1_us=t1_us, t2_us=t2_us, dt_ns=dt,
        )
        return pop, centers_hz(delta, pop[1]), diagnostics

    for polynomial, label in [(False, "hard"), (True, "polynomial")]:
        populations, positions = [], []
        for b in beta:
            pulse = CurrentPulse(drag_beta=float(b), polynomial=polynomial)
            pop, center, _ = run(pulse, amplitudes, detuning)
            populations.append(pop)
            positions.append(center)
            print(f"{label} beta={b:g}: {np.sqrt(np.mean(center**2))/1000:.3f} kHz", flush=True)
        positions = np.asarray(positions)
        rms = np.sqrt(np.mean(positions**2, axis=1))
        arrays[f"{label}_populations"] = np.asarray(populations)
        arrays[f"{label}_centers_hz"] = positions
        arrays[f"{label}_rms_hz"] = rms
        best = int(np.argmin(rms))
        for index in range(max(0, best - 1), min(beta.size, best + 2)):
            fine_delta = np.linspace(-0.12, 0.12, 241)
            fine, center, diagnostics = run(
                CurrentPulse(drag_beta=float(beta[index]), polynomial=polynomial),
                amplitudes, fine_delta, 0.25,
            )
            checks.append(dict(
                shape=label, beta=float(beta[index]),
                max_center_change_hz=float(np.max(abs(center - positions[index]))),
                rms_hz=float(np.sqrt(np.mean(center**2))),
                max_population_change=float(np.max(abs(fine[:, :, ::2] - populations[index]))),
                diagnostics=diagnostics,
            ))
        print(f"{label} refinement complete", flush=True)
    arrays["curve_rabi_mhz"] = np.linspace(1, 60, 180)
    for name, polynomial, b in [("echo", False, 0), ("drag", False, -0.5),
                                ("poly", True, -0.5)]:
        pop, center, _ = run(CurrentPulse(drag_beta=b, polynomial=polynomial),
                             arrays["curve_rabi_mhz"], detuning)
        arrays[f"curve_{name}_populations"] = pop
        arrays[f"curve_{name}_centers_hz"] = center
        print(f"{name} center curve complete", flush=True)
    assert max(c["max_center_change_hz"] for c in checks) < 1
    provenance = dict(
        generator="scripts/make_current_drag_supplement.py",
        reproduction="PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_current_drag_supplement.py --simulate",
        pulse=dict(duration_us=20, cutoff=0.005, transition_ns=50, polynomial_power=2),
        model=dict(levels=3, t1_us=t1_us, t2_us=t2_us, anharmonicity_mhz=-216),
        drive="I-iQ; Q=-beta*dI/dt/abs(alpha); alpha is angular frequency",
        amplitude="Actual complex-envelope peak; each pulse divided by max|I-iQ|",
        objective="RMS central minimum about zero over six peak Rabi amplitudes 10-60 MHz",
        extraction="Three-point parabolic interpolation of direct Pe minimum",
        dt_ns=0.5, detuning_step_khz=2,
        convergence=dict(dt_ns=0.25, detuning_step_khz=1, checks=checks),
        best_sampled={label: dict(
            beta=float(beta[np.argmin(arrays[f'{label}_rms_hz'])]),
            rms_hz=float(np.min(arrays[f'{label}_rms_hz'])),
        ) for label in ["hard", "polynomial"]},
        experimental_beta_optimization="Not supplied; selected acquisition beta is not inferred from this simulation",
        source_sha256=source_hashes(),
    )
    save_paper_dataset(STEM, category="numerical", arrays=arrays, provenance=provenance)
    return arrays, provenance


def paper_scan_view(simulation, metadata):
    """Relabel the legacy scan without changing populations or sample order."""
    displayed = dict(simulation)
    displayed["beta"] = polynomial_beta_to_paper(simulation["beta"])
    provenance = deepcopy(metadata)
    for result in provenance["best_sampled"].values():
        result["beta"] = float(polynomial_beta_to_paper(result["beta"]))
    for check in provenance["convergence"]["checks"]:
        check["beta"] = float(polynomial_beta_to_paper(check["beta"]))
    provenance["drive"] = PAPER_DRAG_CONVENTION
    provenance["source_beta_convention"] = SOURCE_DRAG_CONVENTION
    provenance["beta_conversion"] = (
        "beta=-beta_source; sample order, waveforms and populations unchanged"
    )
    return displayed, provenance


def render():
    with np.load(NUMERICAL / f"{STEM}.npz") as saved:
        sim = {key: saved[key].copy() for key in saved.files}
    meta = json.loads((NUMERICAL / f"{STEM}.json").read_text())["provenance"]
    sim, meta = paper_scan_view(sim, meta)
    data, exp_meta = load_export("03_latest_q1_pulse_improvement")
    apply_supplemental_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH, 2.7), layout="constrained")
    objective, simulated, measured = axes
    for key, color, label in [("hard", COLORS[1], "DRAG only"),
                               ("polynomial", COLORS[2], "+ polynomial")]:
        objective.plot(sim["beta"], sim[f"{key}_rms_hz"] / 1000,
                       color=color, marker=".", ms=4, label=label)
        best = meta["best_sampled"][key]
        objective.scatter(best["beta"], best["rms_hz"] / 1000,
                          marker="*", s=65, color=color, zorder=4)
    objective.axvline(0.5, color="0.5", ls=":", lw=0.7)
    objective.set(yscale="log", xlabel=r"DRAG coefficient $\beta$",
                  ylabel="RMS center offset (kHz)", title="(a)  Numerical beta scan")
    objective.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.02, 0.1),
                     handlelength=0.8, handletextpad=0.4, borderpad=0.2)
    displayed = dict(sim)
    common = np.logical_and.reduce([
        series(data, name, "accepted") & (series(data, name, "peak_rabi_mhz") > 0)
        & (series(data, name, "peak_rabi_mhz") <= 60) for name in F3
    ])
    displayed["experiment_common_accepted"] = common
    statistics = []
    for index, (name, key) in enumerate(zip(F3, ["echo", "drag", "poly"], strict=True)):
        center = sim[f"curve_{key}_centers_hz"] / 1000
        simulated.plot(sim["curve_rabi_mhz"], center, color=COLORS[index], label=LABELS[index])
        rabi = series(data, name, "peak_rabi_mhz")[common]
        centers = series(data, name, "center_hz")[common] / 1000
        measured.errorbar(rabi, centers - centers.mean(),
                          yerr=series(data, name, "center_error_hz")[common] / 1000,
                          color=COLORS[index], lw=0.65, marker=".", ms=2,
                          elinewidth=0.3, label=LABELS[index])
        displayed[f"experiment_{key}_rabi_mhz"] = rabi
        displayed[f"experiment_{key}_center_variation_khz"] = centers - centers.mean()
        displayed[f"experiment_{key}_center_error_khz"] = series(data, name, "center_error_hz")[common] / 1000
        statistics.append(dict(pulse=name, points=int(common.sum()),
                               rms_khz=float(np.std(centers))))
    simulated.set(title="(b)  Simulated centers", ylabel="Dip center (kHz)")
    measured.set(title="(c)  Measured centers", ylabel="Center minus mean (kHz)")
    for ax in [simulated, measured]:
        ax.set(xlabel=r"$\Omega_{\rm pk}/(2\pi)$ (MHz)", xlim=(0, 60),
               ylim=(-100, 100))
        ax.axhline(0, color="0.6", lw=0.5)
        ax.legend(fontsize=7, loc="lower left")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(OUT / f"drag_beta_optimization.{ext}", dpi=300)
    plt.close(fig)
    np.savez_compressed(OUT / "drag_beta_optimization.npz", **displayed)
    provenance = dict(
        panels={
            "a": "Numerical RMS resonance-center offset versus DRAG coefficient",
            "b": "Simulated resonance centers versus peak drive",
            "c": "Measured resonance centers minus each curve's mean versus peak drive",
        },
        center_panels_ylim_khz=[-100, 100],
        simulated_scan=meta,
        experiment=dict(source_session=exp_meta["source_session"],
                        source_dataset="paper/data/experimental/03_latest_q1_pulse_improvement.npz",
                        duration_us=20, cutoff=0.005, transition_ns=50, polynomial_power=2,
                        acquisition_beta=[0, 1.2557575757575759, 1.2557575757575759],
                        beta_convention=PAPER_DRAG_CONVENTION,
                        shots=5000, statistics=statistics,
                        processing="Saved Gaussian centers on common accepted nonzero rows up to 60 MHz; one mean subtracted per pulse; no smoothing",
                        optimization_scan="Deferred by user; not present in this figure"),
        source_sha256={**source_hashes(), **{
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [NUMERICAL / f"{STEM}.npz", NUMERICAL / f"{STEM}.json"]
        }},
        reproduce="PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_current_drag_supplement.py",
    )
    (OUT / "drag_beta_optimization_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(dict(best_sampled=meta["best_sampled"], experiment=statistics), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()
    if args.simulate:
        simulate_dataset()
    render()
