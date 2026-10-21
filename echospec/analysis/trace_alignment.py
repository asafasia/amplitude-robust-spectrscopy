"""Additive alignment of simulation traces to measured population samples."""

import numpy as np


def fit_population_offset(exp_x, exp_y, sim_x, sim_y):
    """Minimize unweighted squared residuals with a single additive offset.

    Interpolate simulation values at measurement coordinates; reject extrapolation.
    The gain and frequency coordinates are fixed.
    """
    exp_x, exp_y, sim_x, sim_y = (
        np.asarray(values, dtype=float) for values in (exp_x, exp_y, sim_x, sim_y)
    )
    for x, y in ((exp_x, exp_y), (sim_x, sim_y)):
        if x.ndim != 1 or x.size < 2 or x.shape != y.shape:
            raise ValueError("Trace coordinates and values must be matching 1D arrays")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Trace values must be finite")
        if np.any(np.diff(x) <= 0):
            raise ValueError("Trace coordinates must be strictly increasing")
    if exp_x[0] < sim_x[0] or exp_x[-1] > sim_x[-1]:
        raise ValueError("Simulation must cover all experimental coordinates")
    residual = exp_y - np.interp(exp_x, sim_x, sim_y)
    offset = float(np.mean(residual))
    return {
        "offset": offset,
        "rmse_before": float(np.sqrt(np.mean(residual**2))),
        "rmse_after": float(np.sqrt(np.mean((residual - offset) ** 2))),
        "sample_count": int(exp_x.size),
    }
