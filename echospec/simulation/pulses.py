from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, brentq

from enum import Enum


class PulseType(str, Enum):
    LORENTZIAN = "lorentzian"
    GAUSSIAN = "gaussian"
    SQUARE = "square"


pulse_type: PulseType = PulseType.LORENTZIAN


@dataclass(slots=True, frozen=True)
class PulseArgs:
    pulse_length: float
    cutoff: float = 1e-4
    order: float = 0.5
    zeroed_pulse: bool = False


def square_half(t: ArrayLike, args: PulseArgs) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return 2.0 * np.heaviside(t, 0.0) - 1.0


def gaussian_envelope(t: ArrayLike, args: PulseArgs) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    L = args.pulse_length
    c = args.cutoff
    if not (0 < c < 1):
        raise ValueError("cutoff must be in (0, 1).")
    sigma = np.sqrt(-((L / 2) ** 2) / (2 * np.log(c)))
    return np.exp(-0.5 * (t / sigma) ** 2)


def gaussian_half(t: ArrayLike, args: PulseArgs) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    L = args.pulse_length
    env = gaussian_envelope(t, args)
    return -np.sin(2 * np.pi / L * t) * env


def lorentzian_envelope(t: ArrayLike, args: PulseArgs) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    n = args.order
    c = args.cutoff
    L = args.pulse_length
    if not (0 < c < 1):
        raise ValueError("cutoff must be in (0, 1).")
    if n <= 0:
        raise ValueError("n must be > 0.")

    sigma = ((L / 2) ** 2 / ((1 / c ** (1 / n)) - 1)) ** 0.5
    env = (1.0 + (t / sigma) ** 2) ** (-n)
    return env - c if args.zeroed_pulse else env


def echo_lorentzian_envelope(t: ArrayLike, args: PulseArgs) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return (1.0 - 2.0 * np.heaviside(t, 0.0)) * lorentzian_envelope(t, args)


PulseFunc = Callable[[ArrayLike, PulseArgs], np.ndarray]

_PULSE_REGISTRY: dict[PulseType, dict[bool, PulseFunc | None]] = {
    PulseType.LORENTZIAN: {
        False: lorentzian_envelope,
        True: echo_lorentzian_envelope,
    },
    PulseType.GAUSSIAN: {
        False: gaussian_envelope,
        True: gaussian_half
    },
    PulseType.SQUARE: {
        False: None,
        True: None,
    },
}


def choose_pulse(
    pulse_type: PulseType,
    eco_pulse: bool,

) -> PulseFunc | None:
    """
    Return the correct pulse envelope function.

    Returns
    -------
    PulseFunc | None
        None means time-independent (static) drive.
    """
    try:
        pulse = _PULSE_REGISTRY[pulse_type][eco_pulse]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported pulse configuration: "
            f"pulse_type={pulse_type}, eco_pulse={eco_pulse}"
        ) from exc

    return pulse


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    args = PulseArgs(pulse_length=1.0, cutoff=1e-2,
                     order=0.5, zeroed_pulse=True)
    tlist = np.linspace(-0.6, 0.6, 1000)

    pulse = choose_pulse(PulseType.LORENTZIAN, eco_pulse=False)
    eco_pulse = choose_pulse(PulseType.LORENTZIAN, eco_pulse=True)

    plt.plot(tlist, pulse(tlist, args), label="Lorentzian")
    plt.plot(tlist, eco_pulse(tlist, args), label="Echo Lorentzian")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Pulse Envelopes")
    plt.legend()
    plt.grid()
    plt.show()
