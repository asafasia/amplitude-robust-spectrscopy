"""Build a Figure 3 variant with half the original q1 anharmonicity."""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echospec.figures import FigureVariant, apply_figure_style
from echospec.simulation.qutrit import simulate_qutrit_map


TRACE_PATH = ROOT / "figures/paper/03_lorentzian_echo_slices_data.csv"
OUTPUT_DIR = ROOT / "figures/paper"
OUTPUT_STEM = os.environ.get(
    "FIG3_OUTPUT_STEM", "03_lorentzian_echo_slices_half_anharmonicity"
)
TARGET_RABI_MHZ = (2.5, 10.0, 25.0)
ORIGINAL_ANHARMONICITY_MHZ = -217.106667324065
ANHARMONICITY_MHZ = float(
    os.environ.get("FIG3_ANHARMONICITY_MHZ", str(ORIGINAL_ANHARMONICITY_MHZ / 2.0))
)
T1_US = 27.1558023040541
T2_STAR_US = 6.49786215784872
DURATION_US = 20.0
CUTOFF = 0.005
ORDER = 0.5
DETUNING_MHZ = np.linspace(-0.5, 0.5, 50)
SIM_DETUNING_SIGN = float(os.environ.get("FIG3_SIM_DETUNING_SIGN", "1"))
SIM_DETUNING_MHZ = SIM_DETUNING_SIGN * DETUNING_MHZ
RABI_GRID_MHZ = np.linspace(0.0, 25.0, 50)
POINT_SIZE = float(os.environ.get("FIG3_POINT_SIZE", "2.5"))

PROTOCOLS = (
    ("current_noecho_experiment", "Root-Lorentzian", "#00838f", False),
    ("current_echo_experiment", "Echo-root-Lorentzian", "#6a1b9a", True),
)


def load_experiment_traces() -> dict[tuple[str, float], tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, float], list[tuple[float, float]]] = defaultdict(list)
    with TRACE_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["series"].endswith("_experiment"):
                grouped[(row["series"], float(row["target_rabi_mhz"]))].append(
                    (float(row["detuning_mhz"]), float(row["excited_probability"]))
                )
    return {
        key: (values[:, 0], values[:, 1])
        for key, values in (
            (key, np.asarray(sorted(points), dtype=float))
            for key, points in grouped.items()
        )
    }


def main() -> None:
    t_phi_us = 1.0 / (1.0 / T2_STAR_US - 1.0 / (2.0 * T1_US))
    experiments = load_experiment_traces()
    simulations = {}
    for name, _, _, echo in PROTOCOLS:
        print(f"Simulating {name} with alpha={ANHARMONICITY_MHZ:.9g} MHz", flush=True)
        simulations[name] = simulate_qutrit_map(
            duration_us=DURATION_US,
            detuning_mhz=SIM_DETUNING_MHZ,
            rabi_mhz=RABI_GRID_MHZ,
            t1_us=T1_US,
            t_phi_us=t_phi_us,
            anharmonicity_mhz=ANHARMONICITY_MHZ,
            num_steps_per_half=30000,
            cutoff=CUTOFF,
            echo=echo,
            order=ORDER,
        )

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
    figure, axes = plt.subplots(3, 2, figsize=(3.38, 4.45), sharex=True, sharey=True, constrained_layout=True)
    panel_labels = iter("abcdef")
    output_rows = []
    for row, target_rabi in enumerate(TARGET_RABI_MHZ):
        grid_index = int(np.argmin(abs(RABI_GRID_MHZ - target_rabi)))
        actual_rabi = float(RABI_GRID_MHZ[grid_index])
        for column, (experiment_key, title, color, echo) in enumerate(PROTOCOLS):
            axis = axes[row, column]
            exp_x, exp_y = experiments[(experiment_key, target_rabi)]
            result = simulations[experiment_key]
            sim_y = result.excited[grid_index]
            total_sim_y = sim_y + result.second_excited[grid_index]
            axis.plot(DETUNING_MHZ, sim_y, color=color, lw=1.25, zorder=2, label=r"$P_e$")
            axis.plot(DETUNING_MHZ, total_sim_y, color=color, lw=1.0, ls="--", zorder=2, label=r"$P_e+P_f$")
            axis.scatter(exp_x, exp_y, s=POINT_SIZE, color=color, edgecolors="none", linewidths=0.0, zorder=3)
            axis.axvline(0.0, color="0.45", lw=0.55, ls="--", zorder=0)
            axis.set_xlim(-0.5, 0.5)
            axis.set_ylim(0.0, 0.82)
            axis.set_box_aspect(1.0)
            axis.text(0.04, 0.94, f"({next(panel_labels)})", transform=axis.transAxes, ha="left", va="top", fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5})
            if row == 0:
                axis.set_title(title)
                if column == 0:
                    axis.legend(loc="upper right", bbox_to_anchor=(1.0, 0.76), fontsize=5.2, frameon=False, handlelength=1.8)
            if row == 2:
                axis.set_xlabel(r"Drive detuning $(f_d-f_{01})$ (MHz)")
            if column == 0:
                axis.set_ylabel(r"$P_e$")
                axis.text(0.96, 0.94, rf"$\Omega_0/2\pi={target_rabi:g}$ MHz", transform=axis.transAxes, ha="right", va="top", fontsize=6.0)
            for x, y, total_y in zip(DETUNING_MHZ, sim_y, total_sim_y):
                output_rows.append((f"half_anharmonicity_{'echo' if echo else 'noecho'}_three_level_simulation", target_rabi, actual_rabi, x, y))
                output_rows.append((f"half_anharmonicity_{'echo' if echo else 'noecho'}_three_level_total_simulation", target_rabi, actual_rabi, x, total_y))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        figure.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)

    with (OUTPUT_DIR / f"{OUTPUT_STEM}_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("series", "target_rabi_mhz", "actual_rabi_mhz", "detuning_mhz", "excited_probability"))
        for (series, target), (x, y) in sorted(experiments.items()):
            for xi, yi in zip(x, y):
                writer.writerow((series, target, "", xi, yi))
        writer.writerows(output_rows)
    (OUTPUT_DIR / f"{OUTPUT_STEM}_provenance.json").write_text(
        "{\n"
        f'  "figure": "{OUTPUT_STEM}",\n'
        '  "experimental_data": "03_lorentzian_echo_slices_data.csv",\n'
        f'  "anharmonicity_mhz": {ANHARMONICITY_MHZ:.15g},\n'
        f'  "original_anharmonicity_mhz": {ORIGINAL_ANHARMONICITY_MHZ:.15g},\n'
        f'  "t1_us": {T1_US:.15g}, "t2_star_us": {T2_STAR_US:.15g},\n'
        f'  "duration_us": {DURATION_US:g}, "cutoff": {CUTOFF:g}, "simulation_steps_per_half": 30000\n'
        "}\n",
        encoding="utf-8",
    )
    print(OUTPUT_DIR / f"{OUTPUT_STEM}.png")


if __name__ == "__main__":
    main()
