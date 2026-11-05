"""Fringe spacing and frequency estimation with explicit alias diagnostics.

Frequencies are cyclic MHz. A sinusoid fitted against frequency determines an
effective time in microseconds, not a temporal oscillation frequency in MHz.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _arrays(x, y, variance):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 8:
        raise ValueError("Need matching 1D arrays with at least eight samples")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Samples must be finite")
    if not np.all(np.diff(x) > 0):
        raise ValueError("The frequency axis must increase strictly")
    var = np.ones_like(x) if variance is None else np.asarray(variance, dtype=float)
    if var.shape != x.shape or not np.all(np.isfinite(var)) or np.any(var <= 0):
        raise ValueError("Variance must be positive, finite, and match the data")
    return x, y, var


def _local_minima(grid, evaluate):
    costs = np.array([evaluate(value)[0] for value in grid])
    tolerance = 1e-12*max(1., float(np.max(abs(costs))))
    # Zero fitted contrast produces a flat profile. Do not repeatedly refine
    # every point of that plateau as if it were a distinct informative minimum.
    indices = np.flatnonzero(
        (costs[1:-1] <= costs[:-2]+tolerance)
        & (costs[1:-1] <= costs[2:]+tolerance)
        & ((costs[1:-1] < costs[:-2]-tolerance)
           | (costs[1:-1] < costs[2:]-tolerance))
    )+1
    minima = [(float(grid[0]), float(costs[0])),
              (float(grid[-1]), float(costs[-1]))]
    for i in indices:
        optimum = minimize_scalar(lambda z: evaluate(z)[0],
                                   bounds=(grid[i-1], grid[i+1]), method="bounded",
                                   options={"xatol": 1e-12})
        if not optimum.success:
            raise RuntimeError("Profile minimum refinement failed")
        minima.append((float(optimum.x), float(optimum.fun)))
    return costs, sorted(minima, key=lambda row: row[1])


def fit_fringe_spacing(frequency_mhz, probability, *,
                       effective_time_bounds_us=(8., 18.), grid_points=501,
                       variance=None):
    """Fit a + b*x + c*cos(2*pi*q*x) + d*sin(2*pi*q*x).

    Positive q is the effective time in us. The inferred trough location is
    defined only modulo 1/q; it must not be reported as the bare resonance.
    """
    x, y, var = _arrays(frequency_mhz, probability, variance)
    lo, hi = effective_time_bounds_us
    if not 0 < lo < hi or grid_points < 5:
        raise ValueError("Require ordered positive time bounds and >=5 grid points")
    weight = 1/np.sqrt(var)

    def evaluate(q):
        design = np.column_stack([np.ones_like(x), x,
                                  np.cos(2*np.pi*q*x), np.sin(2*np.pi*q*x)])
        beta = np.linalg.lstsq(design*weight[:, None], y*weight, rcond=None)[0]
        predicted = design @ beta
        return float(np.sum((predicted-y)**2/var)), (beta, predicted)

    grid = np.linspace(lo, hi, grid_points)
    costs, minima = _local_minima(grid, evaluate)
    q, cost = minima[0]
    beta, prediction = evaluate(q)[1]
    spacing = 1/q
    phase_center = (np.arctan2(beta[3], beta[2])+np.pi)/(2*np.pi*q)
    phase_center = (phase_center+spacing/2) % spacing-spacing/2
    candidates = phase_center+spacing*np.arange(
        np.ceil((x[0]-phase_center)/spacing),
        np.floor((x[-1]-phase_center)/spacing)+1,
    )
    return dict(effective_time_us=q, spacing_mhz=spacing,
                phase_center_modulo_spacing_mhz=float(phase_center),
                candidate_troughs_mhz=candidates, coefficients=beta,
                prediction=prediction, residual=y-prediction, cost=cost,
                rms_probability=float(np.sqrt(np.mean((y-prediction)**2))),
                search_time_us=grid, search_cost=costs)


def fit_frequency_shift(frequency_mhz, probability, response, *,
                        shift_bounds_mhz=(-.4, .4), grid_points=241,
                        variance=None):
    """Fit offset + gain*response(frequency - resonance_shift), gain in [0,2].

    ``response`` must predict Pe versus detuning for known pulse/device
    parameters. Each trial profiles out readout offset and positive gain.
    Search/refine every sampled minimum; return competing aliases explicitly.
    Standard errors are local Gaussian estimates, conditional on this model,
    and are provided only when measurement variances are supplied. They do
    not account for model error, nuisance device parameters, or other minima.
    """
    has_variance = variance is not None
    x, y, var = _arrays(frequency_mhz, probability, variance)
    lo, hi = shift_bounds_mhz
    if not lo < hi or grid_points < 5:
        raise ValueError("Require ordered shift bounds and >=5 grid points")
    weight = 1/var
    mean_y = np.average(y, weights=weight)

    def evaluate(shift):
        model = np.asarray(response(x-shift), dtype=float)
        if model.shape != y.shape or not np.all(np.isfinite(model)):
            raise ValueError("Response must return finite probabilities matching x")
        mean_model = np.average(model, weights=weight)
        centered = model-mean_model
        norm = np.sum(weight*centered**2)
        gain = (float(np.clip(np.sum(weight*centered*(y-mean_y))/norm, 0., 2.))
                if norm > 0 else 0.)
        offset = float(mean_y-gain*mean_model)
        predicted = offset+gain*model
        return float(np.sum((predicted-y)**2/var)), (offset, gain, predicted)

    grid = np.linspace(lo, hi, grid_points)
    costs, minima = _local_minima(grid, evaluate)
    shift, cost = minima[0]
    offset, gain, prediction = evaluate(shift)[1]
    stderr = np.nan
    if has_variance and lo < shift < hi and 0 < gain < 2:
        step = 1e-6
        derivative = gain*(np.asarray(response(x-shift-step))
                           - np.asarray(response(x-shift+step)))/(2*step)
        jacobian = np.column_stack([derivative, np.ones_like(x), response(x-shift)])
        information = jacobian.T @ (weight[:, None]*jacobian)
        if np.linalg.matrix_rank(information) == 3:
            stderr = float(np.sqrt(np.linalg.inv(information)[0, 0]))
    return dict(resonance_shift_mhz=shift, offset=offset, gain=gain,
                prediction=prediction, residual=y-prediction, cost=cost,
                reduced_chi_squared=cost/(len(x)-3) if has_variance else np.nan,
                local_standard_error_mhz=stderr,
                rms_probability=float(np.sqrt(np.mean((y-prediction)**2))),
                search_shift_mhz=grid, search_cost=costs,
                minima=np.asarray(minima),
                at_search_boundary=bool(shift in (lo, hi)))
