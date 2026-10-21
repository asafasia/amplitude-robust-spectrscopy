"""Even finite-support windows and exact physical-time derivatives."""

from __future__ import annotations

import numpy as np


def edge_window(time_us, duration_us, *, kind="none", ramp_us=0.0, power=3.0):
    """Return a window and d(window)/dt, with time and duration in microseconds.

    ``quintic`` tapers each outer ``ramp_us`` of the existing duration using
    10u^3 - 15u^4 + 6u^5. ``polynomial`` uses (1 - (2t/L)^2)^power
    throughout, with finite power >= 3 (default 3).
    Both windows and their first two derivatives vanish at the endpoints.
    No peak or area normalization is applied. ``none`` preserves hard edges.
    """
    if not np.isfinite(duration_us) or duration_us <= 0:
        raise ValueError("duration_us must be finite and positive")
    if kind not in {"none", "quintic", "polynomial"}:
        raise ValueError("edge window must be none, quintic, or polynomial")
    if not np.isfinite(ramp_us) or ramp_us < 0:
        raise ValueError("ramp_us must be finite and nonnegative")
    if kind == "quintic" and not 0 < ramp_us <= duration_us / 2:
        raise ValueError("quintic ramp_us must lie in (0, duration_us/2]")
    if kind != "quintic" and ramp_us != 0:
        raise ValueError("ramp_us is only used by the quintic window")
    if not np.isfinite(power) or power < 3:
        raise ValueError("power must be finite and at least 3")
    if kind != "polynomial" and power != 3:
        raise ValueError("power is only used by the polynomial window")
    half = duration_us / 2
    distance = half - abs(time_us)
    if kind == "none":
        return (1.0 if distance >= 0 else 0.0), 0.0
    if distance <= 0:
        return 0.0, 0.0
    if kind == "polynomial":
        x = time_us / half
        return ((1 - x * x) ** power,
                -2 * power * x * (1 - x * x) ** (power - 1) / half)
    if distance >= ramp_us:
        return 1.0, 0.0
    u = distance / ramp_us
    value = u**3 * (10 - 15 * u + 6 * u * u)
    derivative = -np.sign(time_us) * 30 * u * u * (1 - u) ** 2 / ramp_us
    return float(value), float(derivative)
