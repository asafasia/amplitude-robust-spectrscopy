"""Local Gaussian dip fits that do not select individual half-depth crossings."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .dip_metrics import dip_metrics

FIT_SETTINGS = dict(
    model="Gaussian dip plus quadratic background", loss="soft_l1", f_scale=0.03,
    window_factor=1.5, minimum_radius_mhz=0.12, maximum_radius_mhz=0.5,
    curvature_offsets_mhz=[-0.01, -0.005, 0.0, 0.005, 0.01],
    minimum_contrast=0.001, minimum_width_intervals=6,
    maximum_relative_rms=0.10, maximum_relative_width_error=0.25,
)
GAUSSIAN_K = 4 * np.log(2)


def _profile(v, parameters):
    amplitude, center, width, b0, b1, b2 = parameters
    return (b0 + b1 * v + b2 * v**2
            - amplitude * np.exp(-GAUSSIAN_K * ((v - center) / width)**2))


def evaluate_dip_fit(detuning_mhz, fit):
    """Evaluate the complete fitted spectrum, including its local background."""
    v = (np.asarray(detuning_mhz) - fit["reference_mhz"]) / fit["radius_mhz"]
    return fit["y_offset"] + fit["y_scale"] * _profile(v, fit["parameters"])


def robust_dip_fit(detuning_mhz, population, *, window_factor=1.5):
    """Fit a central dip with a smooth local window and quadratic background.

    The window radius is 1.5 times a curvature-based Gaussian width, clipped
    to 0.12--0.5 MHz. Its residual taper exp[-0.5*((x-x0)/radius)^8] avoids
    discontinuities when samples enter the fit window. Two independent width
    seeds are optimized; no neighboring spectrum supplies a fitted value.
    Width and contrast describe the Gaussian component, not the broad pedestal.
    The returned width error diagnoses parameter identifiability under residual
    model error; it is not a calibrated experimental uncertainty.
    """
    x, y = np.asarray(detuning_mhz, dtype=float), np.asarray(population, dtype=float)
    if (x.ndim != 1 or x.shape != y.shape or x.size < 9
            or not np.isfinite(x).all() or not np.all(np.diff(x) > 0)
            or not np.isfinite(window_factor) or window_factor <= 0):
        raise ValueError("Require matching increasing 1D arrays and a positive window factor")
    invalid = dict(
        accepted=False, status="no_resolved_dip", center_hz=np.nan,
        fwhm_hz=np.nan, contrast=np.nan, relative_rms=np.nan,
        relative_width_error=np.nan, radius_mhz=np.nan,
    )
    if not np.isfinite(y).all() or y.min() < -2e-6 or y.max() > 1 + 2e-6:
        return invalid
    raw = dip_metrics(x, y)
    if not raw["accepted"]:
        return invalid
    reference = raw["center_hz"] / 1e6
    offsets = np.asarray(FIT_SETTINGS["curvature_offsets_mhz"])
    if reference + offsets[0] < x[0] or reference + offsets[-1] > x[-1]:
        return invalid
    curvature = 2 * np.polyfit(offsets, np.interp(reference + offsets, x, y), 2)[0]
    if curvature <= 0:
        return invalid
    seed_width = np.sqrt(8 * np.log(2) * raw["contrast"] / curvature)
    radius = float(np.clip(window_factor * seed_width,
                           FIT_SETTINGS["minimum_radius_mhz"],
                           FIT_SETTINGS["maximum_radius_mhz"]))
    v = (x - reference) / radius
    weights = np.exp(-0.5 * v**8)
    selected = weights > 1e-8
    v, yy, weights = v[selected], y[selected], weights[selected]
    offset = float(yy.min())
    scale = max(float(np.ptp(yy)), FIT_SETTINGS["minimum_contrast"])
    yn = (yy - offset) / scale

    def residual(parameters):
        return (_profile(v, parameters) - yn) * weights

    def jacobian(parameters):
        amplitude, center, width = parameters[:3]
        delta = v - center
        g = np.exp(-GAUSSIAN_K * (delta / width)**2)
        columns = (-g, -2 * GAUSSIAN_K * amplitude * g * delta / width**2,
                   -2 * GAUSSIAN_K * amplitude * g * delta**2 / width**3,
                   np.ones_like(v), v, v**2)
        return np.column_stack(columns) * weights[:, None]

    results = []
    for guess in (seed_width, seed_width / 2):
        initial = [1, 0, float(np.clip(guess / radius, .006, 1.9)), 1, 0, 0]
        result = least_squares(
            residual, initial, jac=jacobian,
            bounds=([0, -.3, .005, -2, -2, -3], [5, .3, 2, 4, 2, 3]),
            loss=FIT_SETTINGS["loss"], f_scale=FIT_SETTINGS["f_scale"],
            max_nfev=150,
        )
        results.append(result)
    result = min(results, key=lambda item: item.cost)
    parameters = result.x
    amplitude, shift, width, b0, b1, b2 = parameters
    center = reference + shift * radius
    contrast = amplitude * scale
    error = residual(parameters)
    relative_rms = float(np.sqrt(np.sum(error**2) / np.sum(weights**2))
                         / max(amplitude, 1e-15))
    jac = jacobian(parameters)
    singular_values = np.linalg.svd(jac, compute_uv=False)
    well_conditioned = singular_values[-1] > singular_values[0] * 1e-6
    covariance = np.linalg.pinv(jac.T @ jac, rcond=1e-12)
    covariance *= np.sum(error**2) / max(np.sum(weights**2) - 6, 1)
    relative_width_error = float(np.sqrt(max(covariance[2, 2], 0)) / width)
    background_center = offset + scale * (b0 + b1 * shift + b2 * shift**2)
    fwhm = width * radius
    status = "accepted"
    if not result.success or np.any(result.active_mask):
        status = "fit_boundary_or_failure"
    elif not FIT_SETTINGS["minimum_contrast"] <= contrast <= 1:
        status = "invalid_fitted_contrast"
    elif not 0 <= background_center - contrast <= background_center <= 1:
        status = "unphysical_local_fit"
    elif fwhm < FIT_SETTINGS["minimum_width_intervals"] * np.max(np.diff(x)):
        status = "undersampled_fit"
    elif (abs(shift) + width / 2 >= 1 or center - fwhm / 2 < x[0]
          or center + fwhm / 2 > x[-1]):
        status = "half_width_outside_window"
    elif relative_rms > FIT_SETTINGS["maximum_relative_rms"]:
        status = "poor_local_fit"
    elif (not well_conditioned
          or relative_width_error > FIT_SETTINGS["maximum_relative_width_error"]):
        status = "unidentified_width"
    return dict(
        accepted=status == "accepted", status=status, center_hz=float(center * 1e6),
        fwhm_hz=float(fwhm * 1e6), contrast=float(contrast),
        relative_rms=relative_rms, relative_width_error=relative_width_error,
        radius_mhz=radius, reference_mhz=float(reference), y_offset=offset,
        y_scale=scale, parameters=parameters.tolist(),
        seed_width_mhz=float(seed_width), nfev=int(result.nfev),
    )
