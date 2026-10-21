"""Four measured maps with the constant-drive linear broadening reference.

Standalone alternative only: the current Figure 2 and manuscript are untouched.
Red lines are the high-drive asymptotes of the steady-state Bloch half width,
not a fit to the finite-duration constant-pulse acquisition.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from make_main_central_spectroscopy_experiment import RUNS, load_measurement, validate

from echospec.figures import FigureVariant, apply_figure_style, save_figure

ROOT = Path(__file__).resolve().parents[1]
STEM = "02_central_spectroscopy_broadening_variant"


def main() -> None:
    measurements = {key: load_measurement(run) for key, run in RUNS.items()}
    validate(measurements)
    coherence_path = ROOT / "paper/coherence_parameters.tex"
    coherence = coherence_path.read_text()

    def parameter(name: str) -> float:
        match = re.search(r"\\newcommand\{\\" + name + r"\}\{([\d.]+)\}", coherence)
        if match is None:
            raise ValueError(f"Missing coherence parameter: {name}")
        return float(match.group(1))

    t1_us, t2_us = parameter("MeasuredTOne"), parameter("EffectiveTTwo")
    # Gamma_f = sqrt(1 + Omega^2 T1 T2)/(pi T2) is the full width in Hz.
    # Therefore its linear, high-drive HALF-width is
    # |Delta|/(2 pi) = (Omega/(2 pi)) sqrt(T1/T2).
    slope = np.sqrt(t1_us / t2_us)
    rabi_min = max(float(m.rabi_mhz[0]) for m in measurements.values())
    rabi_max = min(float(m.rabi_mhz[-1]) for m in measurements.values())
    color_min, color_max = 0.0, 0.6

    apply_figure_style(FigureVariant.PAPER)
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": None,
        }
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(3.38, 3.65),
        sharey=True,
        layout="compressed",
    )
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.0)
    red = "#ee3333"
    fig.suptitle("Constant-drive linear half-width (red)", fontsize=7)
    arrays = {"reference_t1_us": t1_us, "reference_t2_us": t2_us}
    inputs = {}
    for row, domain in enumerate(("broad", "narrow")):
        for col, (protocol, title) in enumerate(
            (
                ("root", "Root-Lorentzian"),
                ("echo", "Echo-root-Lorentzian"),
            )
        ):
            ax = axes[row, col]
            m = measurements[(domain, protocol)]
            mesh = ax.pcolormesh(
                m.detuning_mhz,
                m.rabi_mhz,
                m.excited,
                cmap="magma",
                vmin=color_min,
                vmax=color_max,
                shading="auto",
                rasterized=True,
            )
            # Sample each line inside this panel's actual window. In the narrow
            # maps the reference naturally exits near the bottom; do not stretch
            # or rescale it to make it visible over the full amplitude range.
            max_detuning = max(abs(m.detuning_mhz[0]), abs(m.detuning_mhz[-1]))
            reference_rabi = np.linspace(0, min(rabi_max, max_detuning / slope), 250)
            halfwidth = reference_rabi * slope
            for sign in (-1, 1):
                ax.plot(sign * halfwidth, reference_rabi, color=red, lw=0.45)
            ax.set_box_aspect(1)
            ax.set(
                xlim=(m.detuning_mhz[0], m.detuning_mhz[-1]),
                ylim=(rabi_min, rabi_max),
                yticks=[0, 20, 40],
                xticks=[-20, 0, 20] if row == 0 else [-0.4, 0, 0.4],
            )
            label = ax.text(
                0.035,
                0.94,
                f"({chr(97 + 2 * row + col)})",
                transform=ax.transAxes,
                va="top",
                color="white",
                fontweight="bold",
            )
            label.set_path_effects(
                [
                    path_effects.withStroke(linewidth=1.25, foreground="black"),
                ]
            )
            if row == 0:
                ax.set_title(title)
            key = f"{domain}_{protocol}"
            arrays.update(
                {
                    f"{key}_detuning_mhz": m.detuning_mhz,
                    f"{key}_rabi_mhz": m.rabi_mhz,
                    f"{key}_population": m.excited,
                    f"{key}_reference_rabi_mhz": reference_rabi,
                    f"{key}_reference_halfwidth_mhz": halfwidth,
                }
            )
            inputs[key] = {
                "run": RUNS[(domain, protocol)],
                "population_sha256": hashlib.sha256(m.excited.tobytes()).hexdigest(),
            }

    fig.supxlabel(r"$\Delta/2\pi$ (MHz)", fontsize=7)
    fig.supylabel(r"$\Omega_0/2\pi$ (MHz)", fontsize=7)
    colorbar = fig.colorbar(
        mesh,
        ax=axes,
        location="top",
        pad=0.04,
        fraction=0.035,
        aspect=35,
        ticks=[0.0, 0.2, 0.4, 0.6],
    )
    colorbar.set_label(r"Measured $P_e$")
    saved = save_figure(
        fig,
        STEM,
        variant=FigureVariant.PAPER,
        formats=("pdf", "png", "svg"),
        dpi=350,
        bbox_inches=None,
    )
    plt.close(fig)
    output = saved[0].with_suffix("")
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    provenance = {
        "generator": "scripts/make_figure2_broadening_variant.py",
        "included_in_manuscript": False,
        "figure_size_inches": [3.38, 3.65],
        "panel_box_aspect": 1,
        "reference_line_width_pt": 0.45,
        "inputs": inputs,
        "processing": "Unchanged populations; display color scale fixed to 0--0.6",
        "display_saturation": "Populations above 0.6 use the upper endpoint color",
        "color_limits": [color_min, color_max],
        "reference": {
            "model": "Steady-state Bloch high-drive asymptote; not a finite-pulse fit",
            "halfwidth_mhz": "(Omega_0 / (2*pi) in MHz) * sqrt(T1/T2)",
            "t1_us": t1_us,
            "t2_us": t2_us,
            "slope": float(slope),
            "coherence_source": str(coherence_path.relative_to(ROOT)),
            "coherence_sha256": hashlib.sha256(coherence_path.read_bytes()).hexdigest(),
            "leaves_broad_window_at_rabi_mhz": float(25 / slope),
            "leaves_narrow_window_at_rabi_mhz": float(0.5 / slope),
        },
    }
    output.with_name(f"{STEM}_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    for path in saved:
        print(path)
    print(json.dumps(provenance["reference"], indent=2))


if __name__ == "__main__":
    main()
