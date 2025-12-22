from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


# --- Define models ---
def lorentzian(x, A, x0, gamma, y0):
    # gamma is FWHM for this parametrization
    return y0 + A * (0.5 * gamma) ** 2 / ((x - x0) ** 2 + (0.5 * gamma) ** 2)


def gaussian(x, A, x0, sigma, y0):
    return y0 + A * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


# Use Voigt if needed (scipy.special.wofz) — omitted here for brevity


# --- Example pipeline function ---
def estimate_linewidth(
    x,
    y,
    model="lorentzian",
    smooth=False,
    sg_window=11,
    sg_poly=3,
    n_bootstrap=500,
    rng_seed=0,
):
    """
    x, y : 1D arrays
    model : 'lorentzian' or 'gaussian'
    smooth : whether to apply Savitzky-Golay (light)
    """
    # 0) Basic checks
    assert x.ndim == 1 and y.ndim == 1 and len(x) == len(y)
    order = 0  # baseline poly order used below (if you want to remove baseline)

    # 1) Optional smoothing (keep window << expected FWHM in points)
    y_proc = y.copy()
    if smooth:
        # Ensure window length is odd and <= len(y)
        if sg_window >= len(y):
            sg_window = len(y) - (1 - len(y) % 2)
        if sg_window % 2 == 0:
            sg_window += 1
        y_proc = savgol_filter(y_proc, sg_window, sg_poly)

    # 2) Rough initial guesses
    A0 = np.max(y_proc) - np.min(y_proc)
    x0_0 = x[np.argmax(y_proc)]
    y0_0 = np.min(y_proc)
    # estimate width roughly: width at half height by simple interpolation
    half = y0_0 + A0 / 2
    # find nearest indices around half
    try:
        left_idx = np.where(y_proc[: np.argmax(y_proc)] <= half)[0][-1]
        right_idx = np.where(y_proc[np.argmax(y_proc) :] <= half)[0][0] + np.argmax(
            y_proc
        )
        gamma0 = x[right_idx] - x[left_idx]
        if gamma0 <= 0:
            gamma0 = (x[-1] - x[0]) / 20
    except Exception:
        gamma0 = (x[-1] - x[0]) / 20

    # 3) Choose model
    if model.lower() == "lorentzian":
        fit_func = lorentzian
        p0 = [A0, x0_0, gamma0, y0_0]
    elif model.lower() == "gaussian":
        fit_func = gaussian
        p0 = [A0, x0_0, gamma0 / 2.355, y0_0]  # sigma ~ FWHM/2.355
    else:
        raise ValueError("model must be 'lorentzian' or 'gaussian'")

    # 4) Curve fit
    try:
        popt, pcov = curve_fit(fit_func, x, y, p0=p0, maxfev=5000)
    except RuntimeError:
        popt, pcov = curve_fit(
            fit_func,
            x,
            y,
            p0=p0,
            maxfev=5000,
            bounds=(
                [-np.inf, -np.inf, 1e-12, -np.inf],
                [np.inf, np.inf, np.inf, np.inf],
            ),
        )

    # Extract parameters and nominal uncertainties
    perr = np.sqrt(np.diag(pcov))

    if model.lower() == "lorentzian":
        A_fit, x0_fit, gamma_fit, y0_fit = popt
        gamma_err = perr[2]
    else:  # gaussian - convert sigma to FWHM
        A_fit, x0_fit, sigma_fit, y0_fit = popt
        gamma_fit = 2.355 * sigma_fit
        gamma_err = 2.355 * perr[2]

    # 5) Bootstrap residual resampling to get robust uncertainties
    rng = np.random.default_rng(rng_seed)
    boot_gammas = []
    residuals = y - fit_func(x, *popt)
    for _ in range(n_bootstrap):
        resampled = fit_func(x, *popt) + rng.choice(
            residuals, size=len(residuals), replace=True
        )
        try:
            pb, _ = curve_fit(fit_func, x, resampled, p0=popt, maxfev=5000)
            if model.lower() == "lorentzian":
                boot_gammas.append(pb[2])
            else:
                boot_gammas.append(2.355 * pb[2])
        except Exception:
            continue
    boot_gammas = np.array(boot_gammas)

    # summarize
    result = {
        "popt": popt,
        "pcov": pcov,
        "perr": perr,
        "gamma_fit": float(gamma_fit),
        "gamma_err_cov": float(gamma_err),
        "gamma_boot_mean": (
            float(np.mean(boot_gammas)) if len(boot_gammas) > 0 else None
        ),
        "gamma_boot_std": (
            float(np.std(boot_gammas, ddof=1)) if len(boot_gammas) > 1 else None
        ),
        "bootstrap_samples": len(boot_gammas),
        "model": model,
    }
    return result


# --- Usage example ---
x = np.linspace(-1, 1, 101)
y = lorentzian(x, A=1.0, x0=0.02, gamma=0.1, y0=0.01) + 0.05 * np.random.randn(len(x))
res = estimate_linewidth(x, y, model="lorentzian", smooth=False, n_bootstrap=300)
print("FWHM:", res["gamma_fit"], "+/-", res["gamma_boot_std"])

plt.plot(x, y, label="data")

x = np.linspace(-1, 1, 101)
y = lorentzian(x, A=1.0, x0=0.02, gamma=0.1, y0=0.01) + 0.05 * np.random.randn(len(x))
res = estimate_linewidth(x, y, model="lorentzian", smooth=True, n_bootstrap=300)
print("FWHM:", res["gamma_fit"], "+/-", res["gamma_boot_std"])

plt.plot(x, y, label="data")
plt.plot(x, lorentzian(x, *res["popt"]), label="fit", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
