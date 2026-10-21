"""Eight-panel broad/narrow experimental and simulated spectroscopy comparison.

Produces the eight-panel layout adopted as the manuscript's Figure 2.
Simulation pulse duration and cutoff match the displayed experimental runs;
coherence and anharmonicity use the paper's primary-qubit model parameters.
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from make_main_central_spectroscopy_experiment import RUNS, load_measurement, validate
from matplotlib.lines import Line2D

from echospec.figures import FigureVariant, apply_figure_style, save_figure
from echospec.simulation.qutrit import simulate_qutrit_map

ROOT = Path(__file__).resolve().parents[1]
STEM = "02_central_spectroscopy_experiment_simulation_variant"
OUTPUT = ROOT / "figures/paper"
CACHE = ROOT / "outputs/figure2_matched_simulation"
DOMAINS = ("broad", "narrow")
PROTOCOLS = (("root", "Root-Lorentzian"), ("echo", "Echo-root-Lorentzian"))


def main():
    measurements = {key: load_measurement(run) for key, run in RUNS.items()}
    validate(measurements)
    source = ROOT / "paper/coherence_parameters.tex"
    coherence = source.read_text()

    def parameter(name):
        match = re.search(r"\\newcommand\{\\" + name + r"\}\{([-\d.]+)\}", coherence)
        if match is None:
            raise ValueError(f"Missing paper parameter {name}")
        return float(match.group(1))

    reference = measurements[("broad", "root")]
    rabi_max = float(reference.rabi_mhz[-1])
    rabi = np.unique(np.r_[reference.rabi_mhz[::2], rabi_max])
    grids = {
        domain: np.linspace(-span, span, 401)
        for domain, span in (("broad", 25), ("narrow", 0.5))
    }
    settings = {
        "duration_us": reference.duration_us,
        "cutoff": reference.cutoff,
        "order": 0.5,
        "t1_us": parameter("MeasuredTOne"),
        "t_phi_us": parameter("EffectiveTPhi"),
        "anharmonicity_mhz": parameter("MeasuredAnharmonicity"),
        "num_steps_per_half": 12800,
        "integration_backend": "native",
    }
    for domain in DOMAINS:
        for protocol, _ in PROTOCOLS:
            m = measurements[(domain, protocol)]
            assert np.isclose(m.duration_us, settings["duration_us"])
            assert np.isclose(m.cutoff, settings["cutoff"])

    # A small deterministic convergence check includes both map boundaries and
    # the resonance, at low/mid/high drive, before producing the dense maps.
    check_detuning = np.array([-25, -0.5, -0.1, 0, 0.1, 0.5, 25])
    check_rabi = np.array([3, 20, rabi_max])
    convergence = {}
    for protocol, _ in PROTOCOLS:
        common = dict(
            settings,
            echo=protocol == "echo",
            detuning_mhz=check_detuning,
            rabi_mhz=check_rabi,
        )
        coarse = simulate_qutrit_map(**common)
        fine = simulate_qutrit_map(**dict(common, num_steps_per_half=25600))
        error = float(np.max(np.abs(coarse.excited - fine.excited)))
        if error > 1e-4:
            raise RuntimeError(f"{protocol}: convergence error {error}")
        convergence[protocol] = error
    print("Step-halving checks:", convergence, flush=True)

    solver_hashes = {
        name: hashlib.sha256(
            (ROOT / "echospec/simulation" / name).read_bytes()
        ).hexdigest()
        for name in ("qutrit.py", "qutrit_native.py", "qutrit_rk4.cpp")
    }
    cache_signature = hashlib.sha256(
        json.dumps(
            {"settings": settings, "solver": solver_hashes}, sort_keys=True
        ).encode()
        + rabi.tobytes()
        + b"".join(g.tobytes() for g in grids.values())
    ).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)

    def simulate(domain, protocol):
        path = CACHE / f"{domain}_{protocol}_{cache_signature}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                return {key: cached[key] for key in cached.files}
        print(f"Simulating {domain} {protocol}: {rabi.size} x 401 points", flush=True)
        result = simulate_qutrit_map(
            **settings,
            detuning_mhz=grids[domain],
            rabi_mhz=rabi,
            echo=protocol == "echo",
        )
        values = {
            "pe": result.excited,
            "pf": result.second_excited,
            "pg": result.ground,
        }
        np.savez_compressed(path, **values)
        print(f"Completed {domain} {protocol}", flush=True)
        return values

    simulations = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(simulate, domain, protocol): (domain, protocol)
            for domain in DOMAINS
            for protocol, _ in PROTOCOLS
        }
        for future in as_completed(futures):
            simulations[futures[future]] = future.result()

    apply_figure_style(FigureVariant.PAPER)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": None,
        }
    )
    # Each block has four touching panels. Compress the vertical layout by
    # 10% while preserving the width and font sizes.
    width, height, panel = 7.1, 3.8, 1.4
    vertical_scale = 0.9
    fig = plt.figure(figsize=(width, height * vertical_scale))
    block_lefts = (0.53, 3.77)
    bottom = 0.35
    shown = {}
    panel_metadata = {}
    t2_us = parameter("EffectiveTTwo")
    slope = np.sqrt(settings["t1_us"] / t2_us)
    coherence_hwhm_mhz = 1 / (2 * np.pi * t2_us)
    for group, domain in enumerate(DOMAINS):
        for row, kind in enumerate(("experiment", "simulation")):
            for col, (protocol, title) in enumerate(PROTOCOLS):
                x0 = block_lefts[group] + col * panel
                y0 = bottom + (1 - row) * panel
                ax = fig.add_axes(
                    [x0 / width, y0 / height, panel / width, panel / height]
                )
                m = measurements[(domain, protocol)]
                if kind == "experiment":
                    x, y, z = m.detuning_mhz, m.rabi_mhz, m.excited
                else:
                    x, y, z = grids[domain], rabi, simulations[(domain, protocol)]["pe"]
                mesh = ax.pcolormesh(
                    x,
                    y,
                    z,
                    shading="auto",
                    cmap="magma",
                    vmin=0,
                    vmax=0.8,
                    rasterized=True,
                )
                span = 25 if domain == "broad" else 0.5
                if domain == "broad":
                    ref_y = np.linspace(0, min(rabi_max, span / slope), 200)
                    for sign in (-1, 1):
                        ax.plot(sign * slope * ref_y, ref_y, color="#ee3333", lw=0.45)
                else:
                    for sign in (-1, 1):
                        ax.axvline(
                            sign * coherence_hwhm_mhz,
                            color="white",
                            lw=0.45,
                            ls=(0, (3, 15)),
                        )
                ax.set(
                    xlim=(-span, span),
                    ylim=(0, rabi_max),
                    yticks=[0, 20, 40],
                    xticks=[-20, 0, 20] if domain == "broad" else [-0.3, 0, 0.3],
                )
                ax.tick_params(
                    direction="in",
                    top=True,
                    right=True,
                    labelbottom=row == 1,
                    labelleft=col == 0,
                    length=2.5,
                    pad=2,
                )
                boundary = ax.spines["bottom" if row == 0 else "top"]
                boundary.set_color("0.65")
                boundary.set_linewidth(0.8)
                if row == 0 and col == 0:
                    ax.get_yticklabels()[0].set_bbox(
                        {"facecolor": "white", "edgecolor": "none", "pad": 0.2}
                    )
                if row == 1:
                    ax.set_xlabel(r"$\Delta/2\pi$ (MHz)", labelpad=2)
                if col == 0:
                    ax.set_ylabel(r"$\Omega_0/2\pi$ (MHz)", labelpad=2)
                label = chr(97 + group * 4 + row * 2 + col)
                text = ax.text(
                    0.04,
                    0.92,
                    f"({label})",
                    color="white",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                )
                text.set_path_effects([pe.withStroke(linewidth=1, foreground="black")])
                if row == 0:
                    ax.set_title(title, fontsize=7, pad=4)
                key = f"{domain}_{protocol}_{kind}"
                shown.update(
                    {
                        f"{key}_detuning_mhz": x,
                        f"{key}_rabi_mhz": y,
                        f"{key}_population": z,
                    }
                )
                panel_metadata[label] = {
                    "kind": kind,
                    "domain": domain,
                    "protocol": protocol,
                }
        fig.text(
            (block_lefts[group] + panel) / width,
            3.53 / height,
            "Broad spectrum" if domain == "broad" else "Narrow spectrum",
            ha="center",
            fontsize=10,
        )

    separator_y = (bottom + panel) / height
    fig.add_artist(
        Line2D(
            [0.01, 0.99],
            [separator_y, separator_y],
            transform=fig.transFigure,
            color="0.65",
            linewidth=0.8,
            zorder=-1,
        )
    )

    for row, label in enumerate(("Experiment", "Simulation")):
        fig.text(
            0.012,
            (bottom + (1.5 - row) * panel) / height,
            label,
            rotation=90,
            va="center",
            fontsize=9,
            color="0.35",
            fontstyle="italic",
        )
    cb_ax = fig.add_axes([0.38, 3.48 / height, 0.22, 0.07 / height])
    cb = fig.colorbar(mesh, cax=cb_ax, orientation="horizontal", ticks=[0, 0.4, 0.8])
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(labelsize=7, length=2, pad=1)
    fig.text(0.615, 3.515 / height, r"$P_e$", va="center", fontsize=8)
    # Record the interpretation alongside the figure instead of implying a fit
    # or readout correction: displayed simulations are raw |e> populations.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = save_figure(
        fig, STEM, variant=FigureVariant.PAPER, formats=("pdf", "png", "svg"), dpi=350
    )
    plt.close(fig)
    np.savez_compressed(OUTPUT / f"{STEM}.npz", **shown)
    provenance = {
        "generator": "scripts/make_figure2_experiment_simulation_variant.py",
        "included_in_manuscript": True,
        "panels": panel_metadata,
        "figure_size_inches": [width, height * vertical_scale],
        "panel_size_inches": [panel, panel * vertical_scale],
        "row_separator": {"color": "0.65", "linewidth_pt": 0.8},
        "color_limits": [0, 0.8],
        "display_saturation_above": 0.8,
        "simulation_settings": settings,
        "solver_hashes": solver_hashes,
        "cache_signature": cache_signature,
        "step_halving_max_population_error": convergence,
        "experimental_inputs": {
            f"{domain}_{protocol}": RUNS[(domain, protocol)]
            for domain in DOMAINS
            for protocol, _ in PROTOCOLS
        },
        "comparison": "Raw measured populations versus simulated |e> populations; "
        "no gain, offset, or frequency fits; no readout model",
        "red_lines": "Broad-spectrum panels only: linear steady-state Bloch "
        "half-width, not a finite-pulse fit",
        "coherence_reference": {
            "panels": ["e", "f", "g", "h"],
            "t2_us": t2_us,
            "detuning_mhz": [-coherence_hwhm_mhz, coherence_hwhm_mhz],
            "fwhm_khz": 2 * coherence_hwhm_mhz * 1000,
            "definition": "Vertical lines at +/-1/(2*pi*T2); "
            "separation is the coherence-limited FWHM 1/(pi*T2)",
            "color": "white",
            "linewidth_pt": 0.45,
            "linestyle": "dashed",
            "dash_pattern": [3, 15],
        },
        "simulation_parameters_source": "paper/coherence_parameters.tex; "
        "duration and cutoff from experimental runs",
        "notes": "Two-state discrimination does not independently resolve |f>; "
        "this is a model comparison, not a demonstrated quantitative fit.",
    }
    (OUTPUT / f"{STEM}_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    for path in paths:
        print(path, flush=True)


if __name__ == "__main__":
    main()
