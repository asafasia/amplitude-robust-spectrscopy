from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from click import Tuple
import numpy as np
from numpy.typing import ArrayLike


# =========================
# Utilities
# =========================


# =========================
# FWHM from Gaussian fit
# =========================


def find_intersection_x(
    x: ArrayLike,
    y1: ArrayLike,
    y2: ArrayLike,
    *,
    bracket: tuple[float, float] | None = None,
    kind: str = "cubic",
) -> float:
    """
    Find x such that y1(x) == y2(x) using interpolation + brentq.

    If bracket is None, tries to auto-bracket around the first sign change.
    Returns the x-coordinate of the crossing.
    """
    x, y1 = _as_sorted_xy(x, y1)
    _, y2 = _as_sorted_xy(x, y2)

    f1 = interp1d(x, y1, kind=kind, fill_value="extrapolate")
    f2 = interp1d(x, y2, kind=kind, fill_value="extrapolate")

    def g(xx: float) -> float:
        return float(f1(xx) - f2(xx))

    if bracket is None:
        # Search for sign change over the data grid
        gg = (y1 - y2)
        s = np.sign(gg)
        idx = np.where(np.diff(s) != 0)[0]
        if idx.size == 0:
            raise RuntimeError(
                "No sign change found; provide a bracket=(x_left, x_right).")
        i = int(idx[0])
        bracket = (float(x[i]), float(x[i + 1]))

    return float(brentq(g, bracket[0], bracket[1]))


def z_to_populations(z: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """
    Convert <sigma_z> to populations (P_g, P_e).

    Parameters
    ----------
    z : float or ndarray
        Expectation value of Pauli-Z, in [-1, 1].

    Returns
    -------
    P_g, P_e : same type as z
        Ground and excited state populations.
    """
    z = np.asarray(z)
    P_e = (1.0 - z) / 2.0
    P_g = (1.0 + z) / 2.0
    return (P_g, P_e)


def populations_to_z(P_g: ArrayLike, P_e: ArrayLike) -> ArrayLike:
    """
    Convert populations to <sigma_z>.

    Parameters
    ----------
    P_g, P_e : float or ndarray
        Ground and excited state populations.

    Returns
    -------
    z : same type
        Expectation value of Pauli-Z.
    """
    P_g = np.asarray(P_g)
    P_e = np.asarray(P_e)
    return P_g - P_e


def _non_linear_sweep(
    detunings: ArrayLike,
    non_linear: bool,
) -> np.ndarray:

    center = 0

    detunings = np.asarray(detunings)

    if non_linear:
        max_dev = np.max(np.abs(detunings - center))
        scaled = (detunings - center) / max_dev  # Scale to [-1, 1]
        non_linear_scaled = np.sign(
            scaled) * (np.abs(scaled) ** 3)  # Cubic scaling
        detunings = non_linear_scaled * max_dev + center  # Rescale back

    return detunings


if __name__ == "__main__":
    # Simple test
    import matplotlib.pyplot as plt
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d

    x = np.linspace(-5, 5, 100)

    x1 = _non_linear_sweep(x, non_linear=True)

    plt.plot(np.ones(len(x1)), x1, '.')
