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
CENTER_DETUNING_MHZ = np.linspace(-10.0, 10.0, 401)
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


def gaussian_peak(
    detuning_mhz: np.ndarray,
    offset: float,
    amplitude: float,
    center_mhz: float,
    sigma_mhz: float,
) -> np.ndarray:
    """Smooth envelope used to locate the central root-Lorentzian ridge."""
    return offset + amplitude * np.exp(
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

    The ordinary root-Lorentzian trace contains coherent sub-fringes, so a
    bounded scalar maximizer can jump between local peaks.  Its center is
    instead obtained from a Gaussian fit to the central spectral envelope,
    matching the center estimator used for the measured amplitude sweep.  The
    echo-root trace has a unique central depletion and retains spline-refined
    minimum extraction.
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
    central = np.abs(CENTER_DETUNING_MHZ) <= 0.5
    x = CENTER_DETUNING_MHZ[central]
    step = float(np.median(np.diff(x)))
    for index, row in enumerate(excitation):
        if index == 0:
            continue
        if not minimum:
            y = row[central]
            edge_count = max(3, x.size // 5)
            offset0 = float(np.median(np.r_[y[:edge_count], y[-edge_count:]]))
            amplitude0 = max(float(y.max()) - offset0, 1e-3)
            try:
                values, _ = curve_fit(
                    gaussian_peak,
                    x,
                    y,
                    p0=[offset0, amplitude0, 0.0, 0.12],
                    bounds=(
                        [-0.2, 0.0, -0.2, step / 2.0],
                        [1.2, 1.2, 0.2, 0.5],
                    ),
                    maxfev=20_000,
                )
            except (RuntimeError, ValueError):
                centers[index] = np.nan
            else:
                centers[index] = values[2]
            continue
        spline = CubicSpline(CENTER_DETUNING_MHZ, row)
        centers[index] = minimize_scalar(
            lambda value, curve=spline: float(curve(value)),
            bounds=(-0.5, 0.5),
            method="bounded",
            options={"xatol": 1e-9},
        ).x
    return centers


def main() -> None:
    apply_figure_style(FigureVariant.PAPER)

    dressed_center = dressed_resonance_center_mhz(RABI_MHZ)
    root_center = shaped_feature_centers(echo=False, minimum=False)
    echo_center = shaped_feature_centers(echo=True, minimum=True)
    stable_shaped = RABI_MHZ >= 20.0

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
        RABI_MHZ[stable_shaped],
        root_center[stable_shaped],
        "s-",
        color=LORENTZIAN_COLOR,
        ms=2.8,
        label="root: fitted center",
    )
    echo_line, = axis.plot(
        RABI_MHZ[stable_shaped],
        echo_center[stable_shaped],
        "^-",
        color=ECHO_COLOR,
        ms=2.8,
        label="echo-root: minimum",
    )
    axis.axhline(0.0, color="0.5", lw=0.7)
    axis.set_xlabel(r"$\Omega_0/2\pi$ (MHz)")
    axis.set_ylabel(r"$f_{01}$ shift (MHz)")
    axis.set_xlim(-1.0, 61.0)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=5.2, ncol=2, loc="upper left")

    inset = axis.inset_axes([0.42, 0.39, 0.55, 0.39])
    inset.plot(
        RABI_MHZ[stable_shaped],
        1e3 * root_center[stable_shaped],
        "s-",
        color=root_line.get_color(),
        ms=1.8,
    )
    inset.plot(
        RABI_MHZ[stable_shaped],
        1e3 * echo_center[stable_shaped],
        "^-",
        color=echo_line.get_color(),
        ms=1.8,
    )
    inset.axhline(0.0, color="0.5", lw=0.6)
    inset.set_xlim(20.0, 60.0)
    inset.set_ylim(-25.0, 25.0)
    inset.set_ylabel(r"$f_{01}$ shift (kHz)", fontsize=5.2)
    inset.set_title("shaped-pulse zoom", fontsize=5.5)
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
