"""Export five measured Fig. 2 slices and simulate their exact drive settings.

Reproduce with PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python
scripts/make_supplemental_spectrum_slices.py. Only constant vertical offsets
are removed for display; populations and all alignment values are archived.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from echospec.figures.supplemental import FIGURE_WIDTH, apply_supplemental_style
from echospec.simulation.qutrit import simulate_qutrit_map

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/paper/supplemental"
STEM = "s02_spectrum_slices"
MAP_STEM = "figures/paper/02_central_spectroscopy_experiment_simulation_variant"
TARGETS_MHZ = np.array([3.0, 10.0, 20.0, 30.0, 40.0])
PROTOCOLS = ("root", "echo")
COLORS = ("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#7D5BA6")
STACK_SPACING = 0.55


def prepare_data() -> tuple[dict, dict]:
    """Select actual acquired rows without drive interpolation or fit selection."""
    map_path = ROOT / f"{MAP_STEM}.npz"
    meta_path = ROOT / f"{MAP_STEM}_provenance.json"
    source_meta = json.loads(meta_path.read_text())
    settings = source_meta["simulation_settings"]
    arrays = {"target_rabi_mhz": TARGETS_MHZ}
    checks = {}
    selections = {}
    with np.load(map_path, allow_pickle=False) as maps:
        for protocol in PROTOCOLS:
            prefix = f"narrow_{protocol}_experiment"
            drives = maps[f"{prefix}_rabi_mhz"]
            indices = np.abs(drives[:, None] - TARGETS_MHZ).argmin(axis=0)
            if np.unique(indices).size != 5:
                raise ValueError("Five distinct measured amplitudes are required")
            x = maps[f"{prefix}_detuning_mhz"]
            selected = drives[indices]
            experiment = maps[f"{prefix}_population"][indices].copy()
            print(f"Simulating {protocol}: {selected} MHz", flush=True)
            result = simulate_qutrit_map(
                **settings, echo=protocol == "echo", detuning_mhz=x,
                rabi_mhz=selected,
            )
            # Every displayed amplitude, sampled detunings: numerical accuracy,
            # independently of the experiment/model agreement.
            check_indices = np.array([0, 125, 225, 250, 275, 375, 500])
            fine = simulate_qutrit_map(
                **dict(settings, num_steps_per_half=2 * settings["num_steps_per_half"]),
                echo=protocol == "echo", detuning_mhz=x[check_indices],
                rabi_mhz=selected,
            )
            error = float(np.max(np.abs(
                result.excited[:, check_indices] - fine.excited
            )))
            if error > 1e-4:
                raise RuntimeError(f"{protocol}: step-halving error {error}")
            exp_means = experiment.mean(axis=1)
            sim_means = result.excited.mean(axis=1)
            exp_centered = experiment - exp_means[:, None]
            sim_centered = result.excited - sim_means[:, None]
            offsets = np.arange(5) * STACK_SPACING
            arrays.update({
                f"{protocol}_detuning_mhz": x,
                f"{protocol}_amplitude_indices": indices,
                f"{protocol}_rabi_mhz": selected,
                f"{protocol}_experiment_raw": experiment,
                f"{protocol}_simulation_pe": result.excited,
                f"{protocol}_simulation_pf": result.second_excited,
                f"{protocol}_simulation_pg": result.ground,
                f"{protocol}_experiment_mean": exp_means,
                f"{protocol}_simulation_mean": sim_means,
                f"{protocol}_display_offsets": offsets,
                f"{protocol}_experiment_display": exp_centered + offsets[:, None],
                f"{protocol}_simulation_display": sim_centered + offsets[:, None],
            })
            checks[protocol] = error
            selections[protocol] = {
                "source_run": source_meta["experimental_inputs"][f"narrow_{protocol}"],
                "amplitude_indices": indices.tolist(),
                "actual_rabi_mhz": selected.tolist(),
                "removed_experiment_means": exp_means.tolist(),
                "removed_simulation_means": sim_means.tolist(),
                "offset_only_rmse": np.sqrt(np.mean(
                    (exp_centered - sim_centered) ** 2, axis=1
                )).tolist(),
            }
    paths = [map_path, meta_path, Path(__file__)]
    paths += [ROOT / "echospec/simulation" / name for name in
              ("qutrit.py", "qutrit_native.py", "qutrit_rk4.cpp")]
    metadata = {
        "generator": str(Path(__file__).relative_to(ROOT)),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        "selection": "Nearest acquired rows to 3, 10, 20, 30, 40 MHz; no fit-based selection",
        "simulation_settings": settings,
        "simulation_grid": "Exact measured drives and all 501 measured detunings",
        "selections": selections,
        "display": {
            "formula": "y = raw - mean(raw over all 501 detunings) + j * 0.55",
            "applies_to": "Experiment and simulation independently; j=0..4",
            "gain": 1.0, "frequency_shift_mhz": 0.0,
            "experimental_smoothing": "None; every measured point is displayed",
            "warning": "Relative population, not absolute probability or calibrated readout correction",
        },
        "step_halving_max_pe_error": checks,
        "numerical_check": "All five amplitudes at seven detunings; not physical-model validation",
    }
    return arrays, metadata


def build_figure(data: dict) -> plt.Figure:
    """Two stacked-spectrum panels with one common vertical population scale."""
    apply_supplemental_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH, 5.0), sharex=True, sharey=True,
        layout="constrained",
    )
    for ax, protocol, title in zip(
        axes, PROTOCOLS, ("(a)  Root-Lorentzian", "(b)  Echo-root-Lorentzian"),
        strict=True,
    ):
        x = data[f"{protocol}_detuning_mhz"]
        for j, color in enumerate(COLORS):
            ax.scatter(
                x, data[f"{protocol}_experiment_display"][j], s=3.0,
                color=color, alpha=0.7, linewidths=0, zorder=2,
            )
            ax.plot(
                x, data[f"{protocol}_simulation_display"][j],
                color="0.15", lw=1.0, zorder=3,
            )
            # Labels share a fixed position between neighboring baselines.
            ax.text(
                -0.47, data[f"{protocol}_display_offsets"][j] + 0.25,
                rf"${data[f'{protocol}_rabi_mhz'][j]:.2f}$ MHz",
                fontsize=8, color=color, va="bottom",
            )
        ax.axvline(0, color="0.65", ls=":", lw=0.7, zorder=0)
        ax.set(
            title=title, xlabel=r"Detuning $\Delta/(2\pi)$ (MHz)",
            xlim=(-0.5, 0.5), ylim=(-0.45, 2.7), yticks=[],
            xticks=[-0.4, -0.2, 0, 0.2, 0.4],
        )
        # A shared 0.2-population scale bar retains the contrast comparison.
        ax.plot([0.43, 0.43], [-0.27, -0.07], color="0.2", lw=1.3)
        ax.text(0.40, -0.17, "0.2", va="center", ha="right", fontsize=8)
    axes[0].set_ylabel("Relative population (vertically offset)")
    fig.legend(
        handles=[
            Line2D([], [], color="0.15", lw=1.0, label="Simulation"),
            Line2D([], [], color="0.35", marker="o", ls="none", markersize=3,
                   label="Experiment"),
        ], loc="outside upper center", ncol=2,
    )
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, metadata = prepare_data()
    np.savez_compressed(OUT / f"{STEM}.npz", **arrays)
    (OUT / f"{STEM}_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    figure = build_figure(arrays)
    for extension in ("pdf", "png", "svg"):
        figure.savefig(OUT / f"{STEM}.{extension}", dpi=300)
    plt.close(figure)
    print(json.dumps(metadata["selections"], indent=2), flush=True)


if __name__ == "__main__":
    main()
