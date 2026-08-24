"""Simulate the Figure 3 echo-root-Lorentzian sweep at 15 MHz."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from echospec.figures import FigureVariant, apply_figure_style
from echospec.simulation.qutrit import simulate_qutrit_map
from make_latest_data_pulse_parameter_simulation import load_measurement


OUTPUT_DIR = Path(
    os.environ.get(
        "FIG3_15MHZ_OUTPUT_DIR",
        ROOT / "outputs/figure3_15mhz_qutrit_1d",
    )
)
OUTPUT_STEM = "figure3_echo_qutrit_15mhz_1d"
COMPARISON_STEM = "figure3_echo_qutrit_15mhz_experiment_comparison"
EXPERIMENT_RUN_ID = "14-02-28-518579"

# Physical parameters from Figure 3 provenance and the matched q1 profile.
REFERENCE_DURATION_US = 20.0
REFERENCE_CUTOFF = 0.005
CUTOFF = float(os.environ.get("FIG3_SIM_CUTOFF", "0.005"))
ORDER = 0.5
REFERENCE_TAU_US = (REFERENCE_DURATION_US / 2.0) / np.sqrt(
    REFERENCE_CUTOFF ** (-1.0 / ORDER) - 1.0
)
DURATION_US = float(
    os.environ.get(
        "FIG3_SIM_DURATION_US",
        str(REFERENCE_DURATION_US),
    )
)
RABI_MHZ = 15.0
T1_US = 27.1558023040541
T2_STAR_US = 6.49786215784872
ANHARMONICITY_MHZ = -217.106667324065

# Match Figure 3's simulated detuning sweep.  The denser temporal grid is used
# by the repository's direct RK4 qutrit implementation to resolve |e>-|f> motion.
DETUNING_MHZ = np.linspace(-0.5, 0.5, 50)
STEPS_PER_HALF = 30_000


def main() -> None:
    t_phi_us = 1.0 / (1.0 / T2_STAR_US - 1.0 / (2.0 * T1_US))
    result = simulate_qutrit_map(
        duration_us=DURATION_US,
        detuning_mhz=DETUNING_MHZ,
        rabi_mhz=np.asarray([RABI_MHZ]),
        t1_us=T1_US,
        t_phi_us=t_phi_us,
        anharmonicity_mhz=ANHARMONICITY_MHZ,
        num_steps_per_half=STEPS_PER_HALF,
        cutoff=CUTOFF,
        echo=True,
        order=ORDER,
    )

    ground = result.ground[0]
    excited = result.excited[0]
    second_excited = result.second_excited[0]
    total_excited = excited + second_excited

    exp_detuning, exp_rabi, exp_excited_map, exp_parameters = load_measurement(
        EXPERIMENT_RUN_ID
    )
    exp_index = int(np.argmin(abs(exp_rabi - RABI_MHZ)))
    exp_actual_rabi = float(exp_rabi[exp_index])
    exp_excited = exp_excited_map[exp_index]
    simulated_at_experiment = np.interp(
        exp_detuning, DETUNING_MHZ, total_excited
    )
    residual = exp_excited - simulated_at_experiment
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(abs(residual)))
    correlation = float(np.corrcoef(exp_excited, simulated_at_experiment)[0, 1])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / f"{OUTPUT_STEM}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("detuning_mhz", "p_g", "p_e", "p_f"))
        writer.writerows(zip(DETUNING_MHZ, ground, excited, second_excited))

    provenance = {
        "description": "Three-level Figure 3 echo-root-Lorentzian line cut",
        "model": "three-level transmon Lindblad/RK4",
        "pulse_shape": "echo_root_lorentzian",
        "rabi_mhz": RABI_MHZ,
        "duration_us": DURATION_US,
        "cutoff": CUTOFF,
        "order": ORDER,
        "tau_us": (DURATION_US / 2.0)
        / np.sqrt(CUTOFF ** (-1.0 / ORDER) - 1.0),
        "reference_tau_us": REFERENCE_TAU_US,
        "detuning_min_mhz": float(DETUNING_MHZ[0]),
        "detuning_max_mhz": float(DETUNING_MHZ[-1]),
        "detuning_points": int(DETUNING_MHZ.size),
        "t1_us": T1_US,
        "t2_star_us": T2_STAR_US,
        "t_phi_us": t_phi_us,
        "anharmonicity_mhz": ANHARMONICITY_MHZ,
        "steps_per_half": STEPS_PER_HALF,
    }
    (OUTPUT_DIR / f"{OUTPUT_STEM}_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    with (OUTPUT_DIR / f"{COMPARISON_STEM}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "detuning_mhz",
                "experimental_excited_probability",
                "simulated_p_e_plus_p_f",
                "experiment_minus_simulation",
            )
        )
        writer.writerows(
            zip(exp_detuning, exp_excited, simulated_at_experiment, residual)
        )

    comparison = {
        "description": "Figure 3 experiment versus three-level 15 MHz line cut",
        "experiment_run_id": EXPERIMENT_RUN_ID,
        "target_rabi_mhz": RABI_MHZ,
        "experimental_rabi_mhz": exp_actual_rabi,
        "experimental_rabi_error_mhz": exp_actual_rabi - RABI_MHZ,
        "experimental_points": int(exp_detuning.size),
        "simulation_observable": "p_e + p_f",
        "simulation_cutoff": CUTOFF,
        "simulation_duration_us": DURATION_US,
        "simulation_tau_us": (DURATION_US / 2.0)
        / np.sqrt(CUTOFF ** (-1.0 / ORDER) - 1.0),
        "rmse": rmse,
        "mae": mae,
        "pearson_correlation": correlation,
        "experimental_minimum": {
            "detuning_mhz": float(exp_detuning[np.argmin(exp_excited)]),
            "excited_probability": float(exp_excited.min()),
        },
        "simulation_minimum": {
            "detuning_mhz": float(DETUNING_MHZ[np.argmin(total_excited)]),
            "excited_probability": float(total_excited.min()),
        },
        "experiment_parameters": {
            "duration_us": float(exp_parameters["lorentzian_length_in_ns"])
            / 1000.0,
            "cutoff": float(exp_parameters["cutoff"]),
            "echo": bool(exp_parameters["echo"]),
            "shots": int(exp_parameters["num_shots"]),
        },
    }
    (OUTPUT_DIR / f"{COMPARISON_STEM}.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )

    apply_figure_style(FigureVariant.PAPER)
    figure, axis = plt.subplots(figsize=(3.38, 2.25), constrained_layout=True)
    axis.plot(DETUNING_MHZ, excited, lw=1.4, label=r"$P_e$")
    axis.plot(DETUNING_MHZ, second_excited, lw=1.1, label=r"$P_f$")
    axis.axvline(0.0, color="0.45", lw=0.6, ls="--")
    axis.set(
        xlabel=r"$\Delta/2\pi$ (MHz)",
        ylabel="Final population",
        xlim=(-0.5, 0.5),
        ylim=(0.0, None),
        title=(
            r"Echo-root-Lorentzian, $\Omega_0/2\pi=15$ MHz"
            + rf", $c={CUTOFF:g}$, $L={DURATION_US:.4g}\,\mu s$"
        ),
    )
    axis.legend(frameon=False)
    for extension in ("pdf", "png"):
        figure.savefig(
            OUTPUT_DIR / f"{OUTPUT_STEM}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)

    comparison_figure, (comparison_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(3.38, 3.35),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (3.0, 1.0)},
    )
    comparison_axis.plot(
        DETUNING_MHZ,
        total_excited,
        color="#6a1b9a",
        lw=1.4,
        label=r"Three-level $P_e+P_f$",
    )
    comparison_axis.scatter(
        exp_detuning,
        exp_excited,
        s=5.0,
        color="#00838f",
        edgecolors="none",
        label=rf"Experiment ({exp_actual_rabi:.4f} MHz)",
        zorder=3,
    )
    comparison_axis.axvline(0.0, color="0.45", lw=0.6, ls="--")
    comparison_axis.set_ylabel("Excited probability")
    comparison_axis.set_ylim(0.0, 0.82)
    comparison_axis.legend(frameon=False, fontsize=6.0)
    comparison_axis.set_title(
        rf"Echo-root-Lorentzian, target $\Omega_0/2\pi=15$ MHz, "
        rf"sim. $c={CUTOFF:g}$, $L={DURATION_US:.4g}\,\mu s$"
    )

    residual_axis.axhline(0.0, color="0.45", lw=0.6, ls="--")
    residual_axis.plot(exp_detuning, residual, color="#455a64", lw=0.9)
    residual_axis.set(
        xlabel=r"$\Delta/2\pi$ (MHz)",
        ylabel="Exp.−sim.",
        xlim=(-0.5, 0.5),
    )
    for extension in ("pdf", "png"):
        comparison_figure.savefig(
            OUTPUT_DIR / f"{COMPARISON_STEM}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(comparison_figure)

    minimum_index = int(np.argmin(excited))
    zero_index = int(np.argmin(abs(DETUNING_MHZ)))
    print(OUTPUT_DIR / f"{OUTPUT_STEM}.png")
    print(
        f"min P_e={excited[minimum_index]:.9f} "
        f"at {DETUNING_MHZ[minimum_index]:.9f} MHz"
    )
    print(
        f"P_e nearest zero={excited[zero_index]:.9f} "
        f"at {DETUNING_MHZ[zero_index]:.9f} MHz"
    )
    print(f"max P_f={second_excited.max():.9f}")
    print(f"experimental Rabi={exp_actual_rabi:.9f} MHz")
    print(f"comparison RMSE={rmse:.9f}, MAE={mae:.9f}, r={correlation:.9f}")


if __name__ == "__main__":
    main()
