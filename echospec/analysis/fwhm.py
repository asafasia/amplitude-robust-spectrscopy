from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, brentq
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt  # local import: keeps module light


def _as_sorted_xy(x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(
            f"x and y must have same length. Got {x.size} and {y.size}.")
    idx = np.argsort(x)
    return x[idx], y[idx]


def _gaussian(x: np.ndarray, A: float, mu: float, sigma: float, d: float) -> np.ndarray:
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + d


def _lorentzian(x: np.ndarray, A: float, mu: float, gamma: float, d: float) -> np.ndarray:
    return A * (gamma ** 2) / ((x - mu) ** 2 + gamma ** 2) + d


def fwhm_gaussian_fit(
    x: ArrayLike,
    y: ArrayLike,
    *,
    echo: bool = False,
    smooth_sigma_pts: float = 1.0,
    plot: bool = False,
    ax=None,
) -> tuple[float, float, float, np.ndarray]:
    """
    Fit a (possibly inverted) Gaussian and return:
      mu, fwhm, snr_like, popt

    - echo=True: expects a dip; internally flips it to fit a positive Gaussian.
    - smooth_sigma_pts: gaussian_filter1d sigma in *points* (not physical units).
    - snr_like here is |A| (fit amplitude).
    """
    x, y = _as_sorted_xy(x, y)
    interp_func = interp1d(
        x, y, kind="linear", fill_value="extrapolate")
    x = np.linspace(np.min(x), np.max(x), num=500)  # uniform grid
    y = interp_func(x)

    # Smooth (optional)
    if smooth_sigma_pts and smooth_sigma_pts > 0:
        y_s = gaussian_filter1d(y, smooth_sigma_pts)
    else:
        y_s = y

    # If echo (dip), flip around baseline so the dip becomes a peak
    if echo:
        baseline = np.median(y_s)
        y_fit = baseline - y_s
    else:
        y_fit = y_s

    # Ensure we are fitting a positive-ish peak; if not, just flip sign
    if np.max(y_fit) < -np.min(y_fit):
        y_fit = -y_fit

    A0, mu0, sigma0, d0 = _robust_initial_guess(x, y_fit)

    # # Bounds: wide + physically meaningful
    # # sigma must be > 0, mu within range, amplitude unconstrained sign is ok but we already made peak positive
    eps = np.finfo(float).eps
    lower = (-np.inf, x.min(), eps, -np.inf)
    upper = (np.inf, x.max(), (x.max() - x.min()) * 10 + eps, np.inf)

    popt, pcov = curve_fit(
        _lorentzian,
        x,
        y_fit,
        p0=(A0, mu0, sigma0, d0),
        bounds=(lower, upper),
        maxfev=20_000,
    )

    A, mu, sigma, d = popt
    # fwhm = float(2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(sigma))
    fwhm = float(2.0 * abs(sigma))
    snr_like = float(abs(A))

    if plot:
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(x, y, ".", label="raw")
        ax.plot(x, y_s, "-", label="smoothed")
        ax.plot(x, (np.median(y_s) - _gaussian(x, *popt))
                if echo else _lorentzian(x, *popt), "--", label="fit (mapped)")
        ax.axvline(mu, linestyle="--")
        ax.axvline(mu - fwhm / 2, linestyle="--")
        ax.axvline(mu + fwhm / 2, linestyle="--")
        ax.set_title(f"FWHM = {fwhm:.3g}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    return float(mu), abs(fwhm), snr_like, popt


def _robust_initial_guess(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    # baseline ~ median, amplitude ~ peak-to-baseline, mu ~ argmax, sigma ~ 1/6 span
    d0 = float(np.median(y))
    i0 = int(np.argmax(y))
    mu0 = float(x[i0])*0
    A0 = float(y[i0] - d0)
    sigma0 = float((x.max() - x.min()) / 6) if x.max() > x.min() else 1.0
    sigma0 = max(sigma0, np.finfo(float).eps)
    return A0, mu0, sigma0, d0
