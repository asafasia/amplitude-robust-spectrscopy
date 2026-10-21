"""Build Figure 3(a-f) from the 2026-08-30 q6 data and qutrit simulations."""

from __future__ import annotations

# Local source imports follow the repository path setup.
# ruff: noqa: E402
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(
    os.environ.get("AMPLITUDE_ROBUST_ROOT", Path(__file__).resolve().parents[1])
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echospec.analysis.trace_alignment import fit_population_offset
from echospec.figures import FigureVariant, apply_figure_style

DATA_PATH = Path(
    os.environ.get(
        "FIG3_TRACE_DATA",
        ROOT / "figures/paper/03_lorentzian_echo_slices_data.csv",
    )
)
OUTPUT_DIR = Path(os.environ.get("FIG3_OUTPUT_DIR", ROOT / "figures/paper"))
OUTPUT_STEM = "03_lorentzian_echo_slices"
TARGET_RABI_MHZ = (3.0, 20.0, 40.0)

PROTOCOLS = (
    (
        "current_noecho_experiment",
        "current_noecho_three_level_simulation",
        "Root-Lorentzian",
        "#00838f",
        "#00474e",
    ),
    (
        "current_echo_experiment",
        "current_echo_three_level_simulation",
        "Echo-root-Lorentzian",
        "#6a1b9a",
        "#350b4e",
    ),
)


def load_traces() -> dict[tuple[str, float], tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, float], list[tuple[float, float]]] = defaultdict(list)
    with DATA_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped[(row["series"], float(row["target_rabi_mhz"]))].append(
                (float(row["detuning_mhz"]), float(row["excited_probability"]))
            )

    traces = {}
    for key, values in grouped.items():
        ordered = np.asarray(sorted(values), dtype=float)
        traces[key] = (ordered[:, 0], ordered[:, 1])
    return traces


def main() -> None:
    traces = load_traces()
    apply_figure_style(FigureVariant.PAPER)
    plt.rcParams.update(
        {
            "figure.figsize": (3.38, 4.45),
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(3.38, 4.45),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panel_labels = iter("abcdef")
    fits = []
    aligned_rows = []

    for row, target_rabi in enumerate(TARGET_RABI_MHZ):
        for column, (
            experiment_key,
            simulation_key,
            title,
            marker_color,
            simulation_color,
        ) in enumerate(PROTOCOLS):
            axis = axes[row, column]
            sim_x, sim_y = traces[(simulation_key, target_rabi)]
            exp_x, exp_y = traces[(experiment_key, target_rabi)]
            panel = next(panel_labels)
            fit = fit_population_offset(exp_x, exp_y, sim_x, sim_y)
            fits.append(
                {
                    "panel": panel,
                    "target_rabi_mhz": target_rabi,
                    "simulation_series": simulation_key,
                    **fit,
                }
            )
            aligned_y = sim_y + fit["offset"]
            aligned_rows.extend(
                (simulation_key, target_rabi, x, raw, shifted)
                for x, raw, shifted in zip(sim_x, sim_y, aligned_y, strict=True)
            )
            axis.plot(sim_x, aligned_y, color=simulation_color, lw=0.8, zorder=2)
            axis.scatter(
                exp_x,
                exp_y,
                s=1.0,
                color=marker_color,
                edgecolors="none",
                linewidths=0.0,
                alpha=1.0,
                zorder=3,
            )
            axis.axvline(0.0, color="0.45", lw=0.55, ls="--", zorder=0)
            axis.set_xlim(-0.5, 0.5)
            axis.set_ylim(0.0, 0.82)
            axis.set_box_aspect(1.0)
            axis.text(
                0.04,
                0.94,
                f"({panel})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 0.5,
                },
            )
            if row == 0:
                axis.set_title(title)
            if row == 2:
                axis.set_xlabel(r"$\Delta/2\pi$ (MHz)")
            if column == 0:
                axis.set_ylabel(r"$P_e$")
                axis.text(
                    0.96,
                    0.94,
                    rf"$\Omega_0/2\pi={target_rabi:g}$ MHz",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.0,
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        figure.savefig(
            OUTPUT_DIR / f"{OUTPUT_STEM}.{extension}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(figure)
    with (OUTPUT_DIR / f"{OUTPUT_STEM}_aligned_simulation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "series",
                "target_rabi_mhz",
                "detuning_mhz",
                "original_simulation_pe",
                "aligned_simulation_pe",
            ]
        )
        writer.writerows(aligned_rows)
    source_provenance = ROOT / "figures/paper" / f"{OUTPUT_STEM}_provenance.json"
    provenance = json.loads(source_provenance.read_text())
    provenance["per_panel_simulation_alignment"] = {
        "model": "simulation + offset; fixed gain and detuning",
        "objective": "Unweighted sum of squared residuals over all 501 measured points per panel, -0.5 to +0.5 MHz",
        "estimator": "Mean of experiment minus simulation interpolated at measured detunings",
        "experimental_reference": "Existing globally aligned experimental CSV; original global gain/offset retained",
        "source_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "maximum_absolute_offset": max(abs(fit["offset"]) for fit in fits),
        "fits": fits,
        "warning": "Display alignment, not a calibrated readout-assignment correction or proof of its cause",
    }
    (OUTPUT_DIR / f"{OUTPUT_STEM}_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(fits, indent=2))
    print(OUTPUT_DIR / f"{OUTPUT_STEM}.pdf")


if __name__ == "__main__":
    main()
