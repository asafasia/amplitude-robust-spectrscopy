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
from scipy.optimize import curve_fit, minimize_scalar

from echospec.figures import FigureVariant, apply_figure_style, save_figure
from echospec.simulation.qutrit import simulate_qutrit_map

RABI_MHZ = np.linspace(0.0, 60.0, 31)
# A sub-kilohertz central grid is required to resolve the shaped-pulse centers
# over the experimental 0--15.32 MHz sweep.  The former 50-kHz grid made the
# weak-drive extrema numerically ill-defined and forced a 20-MHz plotting cut.
CENTER_DETUNING_MHZ = np.linspace(-0.25, 0.25, 1001)
FEATURE_HALF_WINDOW_MHZ = 0.12
EXPERIMENTAL_RABI_MAX_MHZ = 15.316126673091269
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


def gaussian_feature(
    detuning_mhz: np.ndarray,
    offset: float,
    amplitude: float,
    center_mhz: float,
    sigma_mhz: float,
    polarity: float,
) -> np.ndarray:
    """Smooth peak or dip used to locate a central shaped-pulse feature."""
    return offset + polarity * amplitude * np.exp(
        -0.5 * ((detuning_mhz - center_mhz) / sigma_mhz) ** 2
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


def shaped_feature_centers(*, echo: bool, minimum: bool) -> np.ndarray:
    """Extract a central feature position for one shaped protocol.

    Both protocols are fit on a fine central-frequency grid.  Using the same
    smooth local estimator for the peak and dip prevents a scalar optimizer
    from jumping to neighboring coherent fringes at weak drive.
    """
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
    centers = np.zeros_like(RABI_MHZ)
    central = np.abs(CENTER_DETUNING_MHZ) <= FEATURE_HALF_WINDOW_MHZ
    x = CENTER_DETUNING_MHZ[central]
    step = float(np.median(np.diff(x)))
    polarity = -1.0 if minimum else 1.0
    for index, row in enumerate(excitation):
        if index == 0:
            continue
        y = row[central]
        edge_count = max(5, x.size // 8)
        offset0 = float(np.median(np.r_[y[:edge_count], y[-edge_count:]]))
        extremum_index = int(np.argmin(y) if minimum else np.argmax(y))
        amplitude0 = max(polarity * (float(y[extremum_index]) - offset0), 1e-8)
        try:
            values, _ = curve_fit(
                lambda detuning, offset, amplitude, center, sigma: gaussian_feature(
                    detuning, offset, amplitude, center, sigma, polarity
                ),
                x,
                y,
                p0=[offset0, amplitude0, float(x[extremum_index]), 0.04],
                bounds=(
                    [-0.2, 0.0, -FEATURE_HALF_WINDOW_MHZ, step],
                    [1.2, 1.2, FEATURE_HALF_WINDOW_MHZ, FEATURE_HALF_WINDOW_MHZ],
                ),
                maxfev=50_000,
            )
        except (RuntimeError, ValueError):
            centers[index] = np.nan
        else:
            centers[index] = values[2]
    return centers


def main() -> None:
    apply_figure_style(FigureVariant.PAPER)

    dressed_center = dressed_resonance_center_mhz(RABI_MHZ)
    root_center = shaped_feature_centers(echo=False, minimum=False)
    echo_center = shaped_feature_centers(echo=True, minimum=True)

    figure, axis = plt.subplots(figsize=(3.35, 3.35), constrained_layout=True)
    axis.plot(
        RABI_MHZ,
        dressed_center,
        "o-",
        color=CONSTANT_COLOR,
        ms=2.5,
        label=r"constant: dressed $f_{01}$",
    )
    root_line, = axis.plot(
        RABI_MHZ,
        root_center,
        "s-",
        color=LORENTZIAN_COLOR,
        ms=2.8,
        label="root: fitted center",
    )
    echo_line, = axis.plot(
        RABI_MHZ,
        echo_center,
        "^-",
        color=ECHO_COLOR,
        ms=2.8,
        label="echo-root: minimum",
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
    inset.set_xlim(0.0, EXPERIMENTAL_RABI_MAX_MHZ)
    inset.set_ylim(-5.0, 5.0)
    inset.set_ylabel(r"$f_{01}$ shift (kHz)", fontsize=5.2)
    inset.set_title("measured-range zoom", fontsize=5.5)
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
