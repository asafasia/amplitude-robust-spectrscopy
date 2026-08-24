"""Compare several Figure 3 echo slices with multilevel simulations."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from echospec.figures import FigureVariant, apply_figure_style
from echospec.simulation.multilevel import simulate_multilevel_map
from echospec.simulation.qutrit import simulate_qutrit_map
from make_latest_data_pulse_parameter_simulation import load_measurement


OUTPUT_DIR = Path(
    os.environ.get(
        "FIG3_MULTI_OUTPUT_DIR",
        ROOT / "outputs/figure3_multi_rabi_qutrit_cutoff_0p001_L20us",
    )
)
EXPERIMENT_RUN_ID = "14-02-28-518579"

RABI_MHZ = np.asarray([2.5, 5.0, 10.0, 15.0, 20.0, 25.0])
DETUNING_MHZ = np.linspace(-0.5, 0.5, 50)
DURATION_US = float(os.environ.get("FIG3_MULTI_SIM_DURATION_US", "20.0"))
CUTOFF = float(os.environ.get("FIG3_MULTI_SIM_CUTOFF", "0.001"))
ORDER = 0.5
T1_US = 27.1558023040541
T2_STAR_US = 6.49786215784872
ANHARMONICITY_MHZ = -217.106667324065
STEPS_PER_HALF = 30_000
LEVELS = int(os.environ.get("FIG3_MULTI_LEVELS", "3"))
OUTPUT_STEM = f"figure3_echo_{LEVELS}level_multi_rabi_experiment_comparison"


def main() -> None:
    t_phi_us = 1.0 / (1.0 / T2_STAR_US - 1.0 / (2.0 * T1_US))
    common = {
        "duration_us": DURATION_US,
        "detuning_mhz": DETUNING_MHZ,
        "rabi_mhz": RABI_MHZ,
        "t1_us": T1_US,
        "t_phi_us": t_phi_us,
        "anharmonicity_mhz": ANHARMONICITY_MHZ,
        "num_steps_per_half": STEPS_PER_HALF,
        "cutoff": CUTOFF,
        "echo": True,
        "order": ORDER,
    }
    if LEVELS == 3:
        result = simulate_qutrit_map(**common)
        populations = np.stack(
            (result.ground, result.excited, result.second_excited)
        )
    else:
        result = simulate_multilevel_map(levels=LEVELS, **common)
        populations = result.populations
    simulated_primary_excited = populations[1]
    simulated_excited = populations[1:].sum(axis=0)

    exp_detuning, exp_rabi, exp_map, exp_parameters = load_measurement(
        EXPERIMENT_RUN_ID
    )
    comparisons = []
    output_rows = []
    for sim_index, target_rabi in enumerate(RABI_MHZ):
        exp_index = int(np.argmin(abs(exp_rabi - target_rabi)))
        measured = exp_map[exp_index]
        interpolated = np.interp(
            exp_detuning, DETUNING_MHZ, simulated_excited[sim_index]
        )
        interpolated_primary = np.interp(
            exp_detuning,
            DETUNING_MHZ,
            simulated_primary_excited[sim_index],
        )
        residual = measured - interpolated
        metrics = {
            "target_rabi_mhz": float(target_rabi),
            "experimental_rabi_mhz": float(exp_rabi[exp_index]),
            "experimental_rabi_error_mhz": float(
                exp_rabi[exp_index] - target_rabi
            ),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(abs(residual))),
            "pearson_correlation": float(
                np.corrcoef(measured, interpolated)[0, 1]
            ),
            "experimental_minimum_probability": float(measured.min()),
            "experimental_minimum_detuning_mhz": float(
                exp_detuning[np.argmin(measured)]
            ),
            "simulation_minimum_probability": float(
                simulated_excited[sim_index].min()
            ),
            "simulation_minimum_detuning_mhz": float(
                DETUNING_MHZ[np.argmin(simulated_excited[sim_index])]
            ),
            "max_highest_level_population": float(
                populations[-1, sim_index].max()
            ),
        }
        comparisons.append((metrics, measured, residual))
        output_rows.extend(
            zip(
                np.full(exp_detuning.size, target_rabi),
                np.full(exp_detuning.size, exp_rabi[exp_index]),
                exp_detuning,
                measured,
                interpolated_primary,
                interpolated,
                residual,
            )
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / f"{OUTPUT_STEM}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "target_rabi_mhz",
                "experimental_rabi_mhz",
                "detuning_mhz",
                "experimental_excited_probability",
                "simulated_p_1",
                "simulated_total_excited_probability",
                "experiment_minus_simulation",
            )
        )
        writer.writerows(output_rows)

    provenance = {
        "description": "Multi-Rabi Figure 3 experiment/multilevel comparison",
        "experiment_run_id": EXPERIMENT_RUN_ID,
        "experiment_cutoff": float(exp_parameters["cutoff"]),
        "experiment_duration_us": float(
            exp_parameters["lorentzian_length_in_ns"]
        )
        / 1000.0,
        "simulation_cutoff": CUTOFF,
        "duration_us": DURATION_US,
        "target_rabi_mhz": RABI_MHZ.tolist(),
        "detuning_min_mhz": float(DETUNING_MHZ[0]),
        "detuning_max_mhz": float(DETUNING_MHZ[-1]),
        "simulation_detuning_points": int(DETUNING_MHZ.size),
        "experimental_detuning_points": int(exp_detuning.size),
        "t1_us": T1_US,
        "t2_star_us": T2_STAR_US,
        "t_phi_us": t_phi_us,
        "anharmonicity_mhz": ANHARMONICITY_MHZ,
        "steps_per_half": STEPS_PER_HALF,
        "simulation_levels": LEVELS,
        "simulation_observable": " + ".join(
            f"p_{level}" for level in range(1, LEVELS)
        ),
        "comparisons": [item[0] for item in comparisons],
    }
    (OUTPUT_DIR / f"{OUTPUT_STEM}.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    apply_figure_style(FigureVariant.PAPER)
    total_excited_label = (
        r"$P_e+P_f$"
        if LEVELS == 3
        else "$" + "+".join(rf"P_{level}" for level in range(1, LEVELS)) + "$"
    )
    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 4.7),
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
        }
    )
    figure, axes = plt.subplots(
        2, 3, sharex=True, sharey=True, constrained_layout=True
    )
    for sim_index, (axis, target_rabi) in enumerate(zip(axes.flat, RABI_MHZ)):
        metrics, measured, _ = comparisons[sim_index]
        axis.plot(
            DETUNING_MHZ,
            simulated_primary_excited[sim_index],
            color="#6a1b9a",
            lw=1.25,
            label=r"$P_e$",
        )
        axis.plot(
            DETUNING_MHZ,
            simulated_excited[sim_index],
            color="#ef6c00",
            lw=1.0,
            ls="--",
            label=total_excited_label,
        )
        axis.scatter(
            exp_detuning,
            measured,
            s=3.5,
            color="#00838f",
            edgecolors="none",
            label="Experiment",
            zorder=3,
        )
        axis.axvline(0.0, color="0.45", lw=0.55, ls="--")
        axis.set_title(
            rf"{target_rabi:g} MHz; "
            rf"$r={metrics['pearson_correlation']:.2f}$, "
            rf"RMSE={metrics['rmse']:.3f}"
        )
        axis.set_xlim(-0.5, 0.5)
        axis.set_ylim(0.0, 0.82)
    for axis in axes[-1]:
        axis.set_xlabel(r"$\Delta/2\pi$ (MHz)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Excited probability")
    axes[0, 0].legend(frameon=False, fontsize=6.0, loc="lower left")
    figure.suptitle(
        rf"Echo-root-Lorentzian: {LEVELS}-level sim. "
        rf"$L={DURATION_US:g}\,\mu s$, "
        rf"$c={CUTOFF:g}$; experiment "
        rf"$L={float(exp_parameters['lorentzian_length_in_ns']) / 1000.0:g}"
        rf"\,\mu s$, $c={float(exp_parameters['cutoff']):g}$",
        fontsize=8.5,
    )
    for extension in ("pdf", "png"):
        figure.savefig(
            OUTPUT_DIR / f"{OUTPUT_STEM}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)

    for metrics, _, _ in comparisons:
        print(
            f"{metrics['target_rabi_mhz']:g} MHz "
            f"(exp {metrics['experimental_rabi_mhz']:.6f}): "
            f"RMSE={metrics['rmse']:.6f}, "
            f"r={metrics['pearson_correlation']:.6f}"
        )
    print(OUTPUT_DIR / f"{OUTPUT_STEM}.png")


if __name__ == "__main__":
    main()
