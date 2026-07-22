"""Generate the square, one-column AC-Stark shift figure for the Letter."""

# ruff: noqa: E402

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".codex_tmp" / "mpl"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from echospec.figures import FigureVariant, apply_figure_style, save_figure
from echospec.simulation.qutrit import simulate_qutrit_map

RABI_MHZ = np.linspace(0.0, 60.0, 31)
# A sub-kilohertz central grid is required to resolve the shaped-pulse centers
# over the experimental 0--15.32 MHz sweep.  The former 50-kHz grid made the
# weak-drive extrema numerically ill-defined and forced a 20-MHz plotting cut.
CENTER_DETUNING_MHZ = np.linspace(-0.25, 0.25, 1001)
FEATURE_HALF_WINDOW_MHZ = 0.22
SYMMETRY_OFFSET_MAX_MHZ = 0.10
SYMMETRY_CENTER_BOUND_MHZ = FEATURE_HALF_WINDOW_MHZ - SYMMETRY_OFFSET_MAX_MHZ
EXPERIMENTAL_RABI_MAX_MHZ = 15.316126673091269
MAX_DISPLAYED_SHAPED_CENTER_MHZ = 0.025
SHAPED_DURATION_US = 10.0
T1_US = 51.24
T_PHI_US = 7.87
ANHARMONICITY_MHZ = -200.0
SHAPED_STEPS_PER_HALF = 7000
SHAPED_CUTOFF = 0.002
SHAPED_ORDER = 0.5
CONSTANT_COLOR = "#c62828"
LORENTZIAN_COLOR = "#00838f"
ECHO_COLOR = "#6a1b9a"


def shaped_phase_average_center_mhz(rabi_mhz: float) -> float:
    """Leading-order center prior from the time-averaged squared envelope."""
    sigma_us = (SHAPED_DURATION_US / 2) / np.sqrt(
        SHAPED_CUTOFF ** (-1 / SHAPED_ORDER) - 1
    )
    mean_square_envelope = (
        2
        * sigma_us
        / SHAPED_DURATION_US
        * np.arctan(SHAPED_DURATION_US / (2 * sigma_us))
    )
    return float(
        mean_square_envelope * (-rabi_mhz**2 / (2 * ANHARMONICITY_MHZ))
    )


def symmetry_center_mhz(
    detuning_mhz: np.ndarray,
    trace: np.ndarray,
    expected_center_mhz: float,
) -> float:
    """Locate the spectral symmetry axis without selecting a local fringe."""
    spline = CubicSpline(detuning_mhz, trace)
    offsets = np.linspace(0.002, SYMMETRY_OFFSET_MAX_MHZ, 197)
    scale = max(float(np.ptp(trace)), 1e-12)

    def objective(center_mhz: float) -> float:
        odd_component = spline(center_mhz + offsets) - spline(center_mhz - offsets)
        asymmetry = float(np.mean((odd_component / scale) ** 2))
        # The very weak-drive trace is nearly flat.  A small perturbative prior
        # resolves only numerical ties; spectral asymmetry dominates once a
        # feature is visible.
        prior = 1e-3 * (
            (center_mhz - expected_center_mhz) / SYMMETRY_CENTER_BOUND_MHZ
        ) ** 2
        return asymmetry + prior

    grid = np.linspace(
        -SYMMETRY_CENTER_BOUND_MHZ,
        SYMMETRY_CENTER_BOUND_MHZ,
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


def shaped_feature_centers(*, echo: bool) -> np.ndarray:
    """Extract a central feature position for one shaped protocol.

    Both protocols use the symmetry axis of a fine central-frequency scan.
    This definition remains stable when coherent sub-fringes make a single
    local maximum or minimum ambiguous.
    """
    protocol = "echo_root" if echo else "root"
    cache_path = (
        PROJECT_ROOT
        / "figures/paper"
        / f"04_main_ac_stark_center_map_{protocol}.npz"
    )
    if cache_path.exists():
        with np.load(cache_path) as cached:
            cached_detuning = np.asarray(cached["detuning_mhz"], dtype=float)
            cached_rabi = np.asarray(cached["rabi_mhz"], dtype=float)
            if np.array_equal(cached_detuning, CENTER_DETUNING_MHZ) and np.array_equal(
                cached_rabi, RABI_MHZ
            ):
                excitation = np.asarray(cached["excitation"], dtype=float)
            else:
                cache_path.unlink()
                return shaped_feature_centers(echo=echo)
    else:
        result = simulate_qutrit_map(
            duration_us=SHAPED_DURATION_US,
            detuning_mhz=CENTER_DETUNING_MHZ,
            rabi_mhz=RABI_MHZ,
            t1_us=T1_US,
            t_phi_us=T_PHI_US,
            anharmonicity_mhz=ANHARMONICITY_MHZ,
            num_steps_per_half=SHAPED_STEPS_PER_HALF,
            cutoff=SHAPED_CUTOFF,
            echo=echo,
            order=SHAPED_ORDER,
        )
        excitation = result.excited + result.second_excited
        np.savez_compressed(
            cache_path,
            detuning_mhz=CENTER_DETUNING_MHZ,
            rabi_mhz=RABI_MHZ,
            excitation=excitation,
        )
    centers = np.zeros_like(RABI_MHZ)
    central = np.abs(CENTER_DETUNING_MHZ) <= FEATURE_HALF_WINDOW_MHZ
    x = CENTER_DETUNING_MHZ[central]
    for index, row in enumerate(excitation):
        if index == 0:
            continue
        y = row[central]
        centers[index] = symmetry_center_mhz(
            x,
            y,
            shaped_phase_average_center_mhz(float(RABI_MHZ[index])),
        )
    return centers


def main() -> None:
    apply_figure_style(FigureVariant.PAPER)

    dressed_center = dressed_resonance_center_mhz(RABI_MHZ)
    root_center = shaped_feature_centers(echo=False)
    echo_center = shaped_feature_centers(echo=True)
    root_center[np.abs(root_center) > MAX_DISPLAYED_SHAPED_CENTER_MHZ] = np.nan
    echo_center[np.abs(echo_center) > MAX_DISPLAYED_SHAPED_CENTER_MHZ] = np.nan

    figure, axis = plt.subplots(figsize=(3.35, 3.35), constrained_layout=True)
    axis.plot(
        RABI_MHZ,
        dressed_center,
        "o-",
        color=CONSTANT_COLOR,
        ms=2.5,
        label="constant",
    )
    root_line, = axis.plot(
        RABI_MHZ,
        root_center,
        "s-",
        color=LORENTZIAN_COLOR,
        ms=2.8,
        label="root",
    )
    echo_line, = axis.plot(
        RABI_MHZ,
        echo_center,
        "^-",
        color=ECHO_COLOR,
        ms=2.8,
        label="echo-root",
    )
    axis.axhline(0.0, color="0.5", lw=0.7)
    axis.axvspan(
        0.0,
        EXPERIMENTAL_RABI_MAX_MHZ,
        color="0.92",
        zorder=0,
        label="measured sweep",
    )
    axis.set_xlabel(r"$\Omega_0/2\pi$ (MHz)")
    axis.set_ylabel(r"$f_{01}$ shift (MHz)")
    axis.set_xlim(-1.0, 61.0)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=5.2, ncol=2, loc="upper left")

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
    inset.axhline(0.0, color="0.5", lw=0.6)
    inset.axvspan(0.0, EXPERIMENTAL_RABI_MAX_MHZ, color="0.92", zorder=0)
    shaped_centers_khz = 1e3 * np.concatenate([root_center, echo_center])
    inset_lower = min(-25.0, 5.0 * np.floor(np.nanmin(shaped_centers_khz) / 5.0) - 5.0)
    inset_upper = max(25.0, 5.0 * np.ceil(np.nanmax(shaped_centers_khz) / 5.0) + 5.0)
    inset.set_xlim(0.0, float(RABI_MHZ[-1]))
    inset.set_ylim(inset_lower, inset_upper)
    inset.set_ylabel(r"$f_{01}$ shift (kHz)", fontsize=5.2)
    inset.set_title("full shaped-pulse range", fontsize=5.5)
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
