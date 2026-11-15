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
from matplotlib import patheffects
from matplotlib.lines import Line2D

from echospec.figures.coherence import REFERENCE_T2_STAR_S
from echospec.figures.supplemental import FIGURE_WIDTH, apply_supplemental_style
from echospec.simulation.qutrit import simulate_qutrit_map

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/paper/supplemental"
STEM = "s02_spectrum_slices"
MAP_STEM = "paper/data/experimental/02_latest_q1_spectroscopy"
MAPS = {"root": "A3_narrow_no_echo", "echo": "A4_narrow_echo"}
TARGETS_MHZ = np.array([3.0, 5.0, 10.0, 15.0, 20.0])
PROTOCOLS = ("root", "echo")
SLICE_COLOR = "#0072B2"
# Match the population colors in main-text Fig. 2.
MAP_COLORMAP = "magma"
MAP_COLOR_LIMITS = (0.0, 0.8)
STACK_SPACING = 0.55


def select_rows(drives, targets):
    """Select distinct measured rows within the available amplitude domain."""
    drives, targets = np.asarray(drives), np.asarray(targets)
    if (drives.ndim != 1 or targets.ndim != 1 or not drives.size
            or not np.isfinite(drives).all() or not np.isfinite(targets).all()
            or np.any(np.diff(drives) <= 0)):
        raise ValueError("Require finite targets and a strictly increasing drive axis")
    if np.any(targets < drives[0] - 1e-10) or np.any(targets > drives[-1] + 1e-10):
        raise ValueError("Requested slice is outside the measured amplitude range")
    indices = np.abs(drives[:, None] - targets).argmin(axis=0)
    if np.unique(indices).size != targets.size:
        raise ValueError("Distinct measured amplitudes are required")
    return indices


def prepare_data() -> tuple[dict, dict]:
    """Select actual acquired rows without drive interpolation or fit selection."""
    map_path = ROOT / f"{MAP_STEM}.npz"
    meta_path = ROOT / f"{MAP_STEM}.json"
    source_meta = json.loads(meta_path.read_text())["provenance"]
    pulse = source_meta["maps"][MAPS["root"]]["parameters"]
    t1_us = np.mean([
        fit["t1"] / 1000 for name, fit in source_meta["calibrations"].items()
        if name.endswith("_t1") and fit.get("success", False)
    ])
    t2_us = REFERENCE_T2_STAR_S * 1e6
    settings = dict(
        duration_us=pulse["lorentzian_length_in_ns"] / 1000,
        cutoff=pulse["cutoff"], order=0.5, t1_us=float(t1_us),
        t_phi_us=float(1 / (1 / t2_us - 1 / (2 * t1_us))),
        anharmonicity_mhz=-216.0, echo_transition_us=0.0, drag_beta=0.0,
        edge_window_kind="none", num_steps_per_half=20000,
        integration_backend="native",
    )
    arrays = {"target_rabi_mhz": TARGETS_MHZ}
    checks = {}
    selections = {}
    with np.load(map_path, allow_pickle=False) as maps:
        for protocol in PROTOCOLS:
            name = MAPS[protocol]
            acquired = source_meta["maps"][name]["parameters"]
            assert acquired["lorentzian_length_in_ns"] / 1000 == settings["duration_us"]
            assert acquired["cutoff"] == settings["cutoff"]
            assert acquired["echo"] == (protocol == "echo")
            assert acquired["echo_transition_time_ns"] == 0
            assert acquired["drag_beta"] == 0 and acquired["edge_smoothing"] == "none"
            prefix = name.lower()
            drives = maps[f"{prefix}_nominal_rabi_mhz"]
            indices = select_rows(drives, TARGETS_MHZ)
            x = maps[f"{prefix}_detuning_hz"] / 1e6
            selected = drives[indices]
            experiment = maps[f"{prefix}_state"][indices].copy()
            print(f"Simulating {protocol}: {selected} MHz", flush=True)
            result = simulate_qutrit_map(
                **settings, echo=protocol == "echo", detuning_mhz=x,
                rabi_mhz=selected,
            )
            # Every displayed amplitude, sampled detunings: numerical accuracy,
            # independently of the experiment/model agreement.
            check_indices = np.abs(x[:, None] - np.array(
                [-0.5, -0.25, -0.05, 0, 0.05, 0.25, 0.5]
            )).argmin(axis=0)
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
                "source_run": source_meta["maps"][name]["source_raw"],
                "pulse_parameters": acquired,
                "measured_detuning_points": int(x.size),
                "convergence_detuning_mhz": x[check_indices].tolist(),
                "amplitude_indices": indices.tolist(),
                "actual_rabi_mhz": selected.tolist(),
                "removed_experiment_means": exp_means.tolist(),
                "removed_simulation_means": sim_means.tolist(),
                "offset_only_rmse": np.sqrt(np.mean(
                    (exp_centered - sim_centered) ** 2, axis=1
                )).tolist(),
            }
    paths = [map_path, meta_path, Path(__file__),
             ROOT / "echospec/figures/coherence.py"]
    paths += [ROOT / "echospec/simulation" / name for name in
              ("qutrit.py", "qutrit_native.py", "qutrit_rk4.cpp")]
    metadata = {
        "generator": str(Path(__file__).relative_to(ROOT)),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        "selection": "Nearest acquired rows to 3, 5, 10, 15, 20 MHz in current Fig. 2 narrow maps; no fit-based selection",
        "simulation_settings": settings,
        "reference_t2_star_us": t2_us,
        "source_session": source_meta["source_session"],
        "amplitude_convention": "Nominal Rabi frequency, identical to main-text Fig. 2",
        "simulation_grid": "Exact measured drives and all 200 measured detunings",
        "selections": selections,
        "display": {
            "formula": "y = raw - mean(raw over all measured detunings) + j * 0.55",
            "applies_to": "Experiment and simulation independently; j=0..4",
            "gain": 1.0, "frequency_shift_mhz": 0.0,
            "experimental_smoothing": "None; every measured point is displayed",
            "warning": "Relative population, not absolute probability or calibrated readout correction",
        },
        "step_halving_max_pe_error": checks,
        "numerical_check": "All five amplitudes at seven detunings; not physical-model validation",
    }
    return arrays, metadata


def measurement_map_context(data: dict, maps: dict) -> dict:
    """Require the overview maps to contain the exact archived experimental cuts."""
    context = {}
    for protocol in PROTOCOLS:
        prefix = MAPS[protocol].lower()
        detuning = maps[f"{prefix}_detuning_hz"] / 1e6
        drives = maps[f"{prefix}_nominal_rabi_mhz"]
        population = maps[f"{prefix}_state"]
        indices = data[f"{protocol}_amplitude_indices"]
        if population.shape != (drives.size, detuning.size):
            raise ValueError(f"{protocol}: map must have drive-by-detuning axes")
        if not (
            np.array_equal(detuning, data[f"{protocol}_detuning_mhz"])
            and np.array_equal(drives[indices], data[f"{protocol}_rabi_mhz"])
            and np.array_equal(population[indices], data[f"{protocol}_experiment_raw"])
        ):
            raise ValueError(f"{protocol}: overview map does not match archived slices")
        context[protocol] = (detuning, drives, population)
    return context


def rendering_metadata() -> dict:
    """Describe the current presentation separately from the simulated arrays."""
    paths = [ROOT / f"{MAP_STEM}.npz", Path(__file__),
             ROOT / "styles/supplemental.mplstyle"]
    return {
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        "layout": "Measured maps above the corresponding stacked spectra",
        "maps": "Full acquired maps; population colors matched to main-text Fig. 2",
        "map_colormap": MAP_COLORMAP,
        "map_color_limits": list(MAP_COLOR_LIMITS),
        "slice_guides": "Exact acquired drive values, matching blue traces and labels",
        "experiment": "Blue connected points; no interpolation or smoothing",
        "simulation": "Dashed charcoal curves; archived simulation unchanged",
        "scale_bar": "Capped vertical bar spanning a population difference of 0.2",
        "accent_color": SLICE_COLOR,
    }


def build_figure(data: dict, maps: dict) -> plt.Figure:
    """Measured map context above connected spectra on one population scale."""
    context = measurement_map_context(data, maps)
    apply_supplemental_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH, 6.1), layout="constrained")
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1, 1, 0.035), height_ratios=(1.35, 4.1),
        hspace=0.035, wspace=0.045,
    )
    axes = []
    map_axes = []
    for column, (protocol, title) in enumerate(zip(
        PROTOCOLS, ("Root-Lorentzian", "Echo-root-Lorentzian"), strict=True,
    )):
        map_ax = fig.add_subplot(
            grid[0, column], sharey=map_axes[0] if map_axes else None,
        )
        ax = fig.add_subplot(
            grid[1, column], sharex=map_ax, sharey=axes[0] if axes else None,
        )
        axes.append(ax)
        map_axes.append(map_ax)
        map_x, map_drives, population = context[protocol]
        mesh = map_ax.pcolormesh(
            map_x, map_drives, population, shading="nearest", cmap=MAP_COLORMAP,
            vmin=MAP_COLOR_LIMITS[0], vmax=MAP_COLOR_LIMITS[1], rasterized=True,
        )
        map_ax.set(
            title=f"({chr(97 + column)})  {title}",
            ylim=(0, 20.7), yticks=data[f"{protocol}_rabi_mhz"],
            yticklabels=[f"{drive:.1f}" for drive in data[f"{protocol}_rabi_mhz"]],
        )
        map_ax.tick_params(axis="x", labelbottom=False)
        map_ax.tick_params(axis="y", labelsize=7, labelcolor=SLICE_COLOR)
        for drive in data[f"{protocol}_rabi_mhz"]:
            guide = map_ax.axhline(drive, color=SLICE_COLOR, ls="--", lw=0.8)
            guide.set_path_effects([
                patheffects.Stroke(linewidth=1.5, foreground="white"),
                patheffects.Normal(),
            ])
        x = data[f"{protocol}_detuning_mhz"]
        for j in range(len(data[f"{protocol}_rabi_mhz"])):
            ax.plot(
                x, data[f"{protocol}_experiment_display"][j],
                color=SLICE_COLOR, lw=0.7, marker="o", ms=1.45,
                markeredgewidth=0, zorder=2,
            )
            ax.plot(
                x, data[f"{protocol}_simulation_display"][j],
                color="0.2", lw=1.0, ls="--", zorder=3,
            )
            ax.text(
                -0.47, data[f"{protocol}_display_offsets"][j] + 0.25,
                rf"${data[f'{protocol}_rabi_mhz'][j]:.1f}$ MHz",
                fontsize=8, color=SLICE_COLOR, va="bottom",
            )
        ax.axvline(0, color="0.75", ls=":", lw=0.6, zorder=0)
        ax.set(
            title=f"({chr(99 + column)})",
            xlabel=r"Detuning $\Delta/(2\pi)$ (MHz)",
            xlim=(-0.5, 0.5), ylim=(-0.45, 2.7), yticks=[],
            xticks=[-0.4, -0.2, 0, 0.2, 0.4],
        )
        ax.errorbar(
            0.43, -0.25, yerr=0.1, fmt="none", ecolor="0.2",
            elinewidth=0.9, capsize=3, capthick=0.9,
        )
        ax.text(0.39, -0.25, r"$\Delta P=0.2$", va="center", ha="right", fontsize=7)
    map_axes[0].set_ylabel(r"$\Omega_0/(2\pi)$ (MHz)")
    colorbar = fig.colorbar(
        mesh, cax=fig.add_subplot(grid[0, 2]), ticks=np.linspace(*MAP_COLOR_LIMITS, 5),
    )
    colorbar.ax.set_title(r"$P_e$", fontsize=8, pad=6)
    axes[0].set_ylabel("Relative population (vertically offset)")
    fig.legend(
        handles=[
            Line2D([], [], color="0.2", lw=1.0, ls="--", label="Simulation"),
            Line2D([], [], color=SLICE_COLOR, marker="o", lw=0.7, markersize=2.5,
                   label="Experiment"),
        ], loc="outside upper center", ncol=2,
    )
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, metadata = prepare_data()
    np.savez_compressed(OUT / f"{STEM}.npz", **arrays)
    metadata["rendering"] = rendering_metadata()
    (OUT / f"{STEM}_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with np.load(ROOT / f"{MAP_STEM}.npz", allow_pickle=False) as maps:
        figure = build_figure(arrays, maps)
    for extension in ("pdf", "png", "svg"):
        figure.savefig(OUT / f"{STEM}.{extension}", dpi=300)
    plt.close(figure)
    print(json.dumps(metadata["selections"], indent=2), flush=True)


if __name__ == "__main__":
    main()
