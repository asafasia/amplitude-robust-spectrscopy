"""Generate the square, one-column AC-Stark shift figure for the Letter."""

# ruff: noqa: E402

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".codex_tmp" / "mpl"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from echospec.figures import FigureVariant, apply_figure_style, save_figure
from echospec.simulation.qutrit import simulate_qutrit_map

RABI_MHZ = np.linspace(0.0, 60.0, 31)
# q1 OPX1000 two-photon spectroscopy:
# alpha/(2*pi) = 2*(f02/2-f01) = 2*(4.159106667-4.267106667) GHz.
ANHARMONICITY_MHZ = -216.0
ROOT_DETUNING_MHZ = np.linspace(-0.25, 0.25, 1001)
ROOT_FEATURE_HALF_WINDOW_MHZ = 0.22
ROOT_SYMMETRY_OFFSET_MAX_MHZ = 0.10
ROOT_SYMMETRY_CENTER_BOUND_MHZ = (
    ROOT_FEATURE_HALF_WINDOW_MHZ - ROOT_SYMMETRY_OFFSET_MAX_MHZ
)
ROOT_MAX_DISPLAYED_CENTER_MHZ = 0.025
ROOT_DURATION_US = 10.0
ROOT_CUTOFF = 0.002
ROOT_ORDER = 0.5
ROOT_T1_US = 51.24
ROOT_T_PHI_US = 7.87
ROOT_STEPS_PER_HALF = 7000
ROOT_DATA_PATH = (
    PROJECT_ROOT / "figures/paper/04_main_ac_stark_center_map_root.npz"
)
ECHO_DATA_PATH = (
    PROJECT_ROOT
    / "data/generated/accumulated_phase_duration_sweep/20us/results.npz"
)
CONSTANT_COLOR = "#c62828"
LORENTZIAN_COLOR = "#00838f"
ECHO_COLOR = "#6a1b9a"
CORRECTED_ECHO_COLOR = "#ef6c00"


def root_phase_average_center_mhz(rabi_mhz: float) -> float:
    """Leading-order center prior for the root-Lorentzian reference."""
    sigma_us = (ROOT_DURATION_US / 2) / np.sqrt(
        ROOT_CUTOFF ** (-1 / ROOT_ORDER) - 1
    )
    mean_square_envelope = (
        2
        * sigma_us
        / ROOT_DURATION_US
        * np.arctan(ROOT_DURATION_US / (2 * sigma_us))
    )
    return float(
        mean_square_envelope * (-rabi_mhz**2 / (2 * ANHARMONICITY_MHZ))
    )


def root_symmetry_center_mhz(
    detuning_mhz: np.ndarray,
    trace: np.ndarray,
    expected_center_mhz: float,
) -> float:
    """Locate the root-pulse spectral symmetry axis."""
    spline = CubicSpline(detuning_mhz, trace)
    offsets = np.linspace(0.002, ROOT_SYMMETRY_OFFSET_MAX_MHZ, 197)
    scale = max(float(np.ptp(trace)), 1e-12)

    def objective(center_mhz: float) -> float:
        odd_component = spline(center_mhz + offsets) - spline(
            center_mhz - offsets
        )
        asymmetry = float(np.mean((odd_component / scale) ** 2))
        prior = 1e-3 * (
            (center_mhz - expected_center_mhz)
            / ROOT_SYMMETRY_CENTER_BOUND_MHZ
        ) ** 2
        return asymmetry + prior

    grid = np.linspace(
        -ROOT_SYMMETRY_CENTER_BOUND_MHZ,
        ROOT_SYMMETRY_CENTER_BOUND_MHZ,
        481,
    )
    costs = np.asarray([objective(center) for center in grid])
    best = int(np.argmin(costs))
    lower = grid[max(0, best - 1)]
    upper = grid[min(grid.size - 1, best + 1)]
    if lower == upper:
        return float(grid[best])
    return float(
        minimize_scalar(
            objective,
            bounds=(float(lower), float(upper)),
            method="bounded",
            options={"xatol": 1e-10},
        ).x
    )


def root_reference_centers() -> np.ndarray:
    """Extract the original root-Lorentzian reference from its paper cache."""
    cache_is_current = False
    if ROOT_DATA_PATH.exists():
        with np.load(ROOT_DATA_PATH, allow_pickle=False) as data:
            cache_is_current = (
                "anharmonicity_mhz" in data
                and float(data["anharmonicity_mhz"]) == ANHARMONICITY_MHZ
                and float(data["duration_us"]) == ROOT_DURATION_US
                and float(data["cutoff"]) == ROOT_CUTOFF
                and np.array_equal(data["detuning_mhz"], ROOT_DETUNING_MHZ)
                and np.array_equal(data["rabi_mhz"], RABI_MHZ)
            )
            if cache_is_current:
                excitation = np.asarray(data["excitation"], dtype=float)

    if not cache_is_current:
        result = simulate_qutrit_map(
            duration_us=ROOT_DURATION_US,
            detuning_mhz=ROOT_DETUNING_MHZ,
            rabi_mhz=RABI_MHZ,
            t1_us=ROOT_T1_US,
            t_phi_us=ROOT_T_PHI_US,
            anharmonicity_mhz=ANHARMONICITY_MHZ,
            num_steps_per_half=ROOT_STEPS_PER_HALF,
            cutoff=ROOT_CUTOFF,
            echo=False,
            order=ROOT_ORDER,
        )
        excitation = result.excited + result.second_excited
        np.savez_compressed(
            ROOT_DATA_PATH,
            detuning_mhz=ROOT_DETUNING_MHZ,
            rabi_mhz=RABI_MHZ,
            excitation=excitation,
            duration_us=ROOT_DURATION_US,
            cutoff=ROOT_CUTOFF,
            order=ROOT_ORDER,
            t1_us=ROOT_T1_US,
            t_phi_us=ROOT_T_PHI_US,
            anharmonicity_mhz=ANHARMONICITY_MHZ,
            steps_per_half=ROOT_STEPS_PER_HALF,
        )

    centers = np.zeros_like(RABI_MHZ)
    central = np.abs(ROOT_DETUNING_MHZ) <= ROOT_FEATURE_HALF_WINDOW_MHZ
    x = ROOT_DETUNING_MHZ[central]
    for index, row in enumerate(excitation):
        if index == 0:
            continue
        centers[index] = root_symmetry_center_mhz(
            x,
            row[central],
            root_phase_average_center_mhz(float(RABI_MHZ[index])),
        )
    centers[np.abs(centers) > ROOT_MAX_DISPLAYED_CENTER_MHZ] = np.nan
    return centers


def dressed_resonance_center_mhz(rabi_mhz: np.ndarray) -> np.ndarray:
    """Return the exact dressed g-e center of the three-level Hamiltonian."""
    local_detuning = np.linspace(-30.0, 15.0, 9001)
    centers = np.zeros_like(rabi_mhz, dtype=float)
    for index, omega in enumerate(rabi_mhz):
        hamiltonian = np.zeros((local_detuning.size, 3, 3), dtype=float)
        hamiltonian[:, 1, 1] = -local_detuning
        hamiltonian[:, 2, 2] = -2.0 * local_detuning + ANHARMONICITY_MHZ
        hamiltonian[:, 0, 1] = hamiltonian[:, 1, 0] = omega / 2.0
        hamiltonian[:, 1, 2] = hamiltonian[:, 2, 1] = omega / np.sqrt(2.0)
        eigenvalues = np.linalg.eigvalsh(hamiltonian)
        centers[index] = local_detuning[
            np.argmin(eigenvalues[:, 2] - eigenvalues[:, 1])
        ]
    return centers


def accumulated_phase_echo_centers() -> tuple[np.ndarray, np.ndarray]:
    """Load matched uncorrected and corrected echo-root spectral centers."""
    if not ECHO_DATA_PATH.exists():
        raise FileNotFoundError(
            "Run scripts/make_accumulated_phase_duration_report.py first: "
            f"{ECHO_DATA_PATH} is missing."
        )
    with np.load(ECHO_DATA_PATH, allow_pickle=False) as data:
        if float(data["duration_us"]) != 20.0:
            raise ValueError("Figure 5 requires the 20 us duration cache.")
        if float(data["cutoff"]) != 0.001:
            raise ValueError("Figure 5 requires cutoff c=0.001.")
        if float(data["drag_beta"]) != 0.0:
            raise ValueError("Figure 5 requires beta=0.")
        if float(data["anharmonicity_mhz"]) != ANHARMONICITY_MHZ:
            raise ValueError(
                "Figure 5 accumulated-phase data use the wrong anharmonicity."
            )
        fit_rabi_mhz = np.asarray(data["fit_rabi_mhz"], dtype=float)
        plain_centers_mhz = np.asarray(data["plain_centers_mhz"], dtype=float)
        corrected_centers_mhz = np.asarray(
            data["corrected_centers_mhz"], dtype=float
        )

    display = fit_rabi_mhz <= RABI_MHZ[-1]
    matched_rabi_mhz = np.concatenate(([0.0], fit_rabi_mhz[display]))
    if not np.array_equal(matched_rabi_mhz, RABI_MHZ):
        raise ValueError("The accumulated-phase cache does not match the Figure 5 grid.")
    return (
        np.concatenate(([0.0], plain_centers_mhz[display])),
        np.concatenate(([0.0], corrected_centers_mhz[display])),
    )


def main() -> None:
    apply_figure_style(FigureVariant.PAPER)

    dressed_center = dressed_resonance_center_mhz(RABI_MHZ)
    root_center = root_reference_centers()
    echo_center, corrected_echo_center = accumulated_phase_echo_centers()

    figure, axis = plt.subplots(figsize=(3.35, 3.35), constrained_layout=True)
    axis.plot(
        RABI_MHZ,
        dressed_center,
        "o-",
        color=CONSTANT_COLOR,
        ms=2.6,
        zorder=2,
        label="constant",
    )
    root_line, = axis.plot(
        RABI_MHZ,
        root_center,
        "s-",
        color=LORENTZIAN_COLOR,
        ms=4.2,
        zorder=3,
        label="root",
    )
    echo_line, = axis.plot(
        RABI_MHZ,
        echo_center,
        "^-",
        color=ECHO_COLOR,
        ms=3.4,
        zorder=4,
        label="echo-root",
    )
    corrected_echo_line, = axis.plot(
        RABI_MHZ,
        corrected_echo_center,
        "s-",
        color=CORRECTED_ECHO_COLOR,
        ms=2.0,
        zorder=5,
        label="corrected echo-root",
    )
    axis.axhline(0.0, color="0.5", lw=0.7)
    axis.set_xlabel(r"$\Omega_0/2\pi$ (MHz)")
    axis.set_ylabel(r"$f_{01}$ shift (MHz)")
    axis.set_xlim(-1.0, 61.0)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=4.8, ncol=2, loc="upper left")

    inset = axis.inset_axes([0.42, 0.39, 0.55, 0.39])
    inset.plot(
        RABI_MHZ,
        1e3 * root_center,
        "s-",
        color=root_line.get_color(),
        ms=1.8,
    )
    inset.plot(
        RABI_MHZ,
        1e3 * echo_center,
        "^-",
        color=echo_line.get_color(),
        ms=1.8,
    )
    inset.plot(
        RABI_MHZ,
        1e3 * corrected_echo_center,
        "s-",
        color=corrected_echo_line.get_color(),
        ms=1.8,
    )
    inset.axhline(0.0, color="0.5", lw=0.6)
    inset.set_xlim(0.0, float(RABI_MHZ[-1]))
    inset.set_ylim(-10.0, 10.0)
    inset.set_ylabel(r"$f_{01}$ shift (kHz)", fontsize=5.2)
    inset.set_title(r"central $\pm10$ kHz zoom", fontsize=5.5)
    inset.tick_params(labelsize=4.8)
    inset.grid(alpha=0.2)

    saved = save_figure(
        figure,
        "04_main_ac_stark_shifts_square",
        variant=FigureVariant.PAPER,
        formats=("pdf", "png", "svg"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    print("Saved:", *(path.relative_to(PROJECT_ROOT) for path in saved), sep="\n  ")


if __name__ == "__main__":
    main()
