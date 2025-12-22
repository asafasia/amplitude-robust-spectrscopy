from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


def FWHM(X, Y):
    plt.plot(X, Y)
    X = np.array(X)
    Y = np.array(Y)
    Y = gaussian_filter(Y, 5)
    plt.plot(X, Y)
    if Y[len(Y) // 2] > Y[len(Y) // 2 + 1]:
        Y = 1 - Y
    f = interp1d(X, Y, kind="quadratic")
    peaks, _ = find_peaks(Y, height=0)
    x_peaks = X[peaks]
    y_peaks = Y[peaks]

    if len(x_peaks) >= 2:
        closest_indices = np.argsort(np.abs(x_peaks))[:2]
        x_min, x_max = np.sort(x_peaks[closest_indices])
        mask = (X >= x_min) & (X <= x_max)
        x_shrunk = X[mask]
        y_shrunk = Y[mask]
        X = x_shrunk
        Y = y_shrunk
    else:
        print("Not enough peaks found around x=0.")
    y_max = max(Y)
    y_min = Y[X == 0]
    y_min = f(0)
    y_half = (y_max + y_min) / 2
    Y -= y_half
    Y = abs(Y)
    x_dense = np.linspace(-X[0], X[1], 1000)

    def g(x):
        return np.abs(f(x) - y_half)

    y_dense = g(x_dense)
    min_index = np.argmin(y_dense)
    x_min = x_dense[min_index]
    fwhm = abs(x_min * 2)
    snr = abs(y_max - y_min)

    plt.axvline(x=fwhm, color="k", linestyle="--")
    return fwhm, snr


def FWHM(X, Y, plot=False, echo=False):

    i = 40

    X = np.array(X)[i:-i]
    Y = np.array(Y)[i:-i]

    # X = np.array(X)
    # Y = np.array(Y)

    if echo:

        def gaussian(x, A, mu, sigma, d):
            return -A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + d

    else:

        def gaussian(x, A, mu, sigma, d):
            return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + d

    if plot:
        plt.plot(X, Y)

    X = np.array(X)
    Y = np.array(Y)

    Y = gaussian_filter(Y, 1)

    if plot:
        plt.plot(X, Y)

    A0 = max(Y) - min(Y)
    mu0 = 0
    # sigma0 = (max(X) - min(X)) / 6  # Initial guess for standard deviation
    sigma0 = 40e3
    d0 = np.mean(Y)

    initial_guess = [A0, mu0, sigma0, d0]
    bounds = ([0, min(X), 0, -1], [1, max(X), max(X), 1])

    # print(initial_guess)

    try:

        popt, pcov = curve_fit(gaussian, X, Y, p0=initial_guess, bounds=bounds)

        mu = popt[1]
        sigma = popt[2]
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma  # FWHM for Gaussian
        snr = abs(popt[0]) / abs(popt[3])
        # print("FWHM (from fit):", fwhm)

        if plot:
            plt.plot(X, gaussian(X, *popt), "r--", label="Gaussian Fit")
            plt.axvline(x=fwhm / 2 + mu, color="k", linestyle="--")
            plt.axvline(x=-fwhm / 2 + mu, color="k", linestyle="--")
            plt.axvline(x=mu, color="k", linestyle="--")
            plt.show()
        # print(fwhm)

        if fwhm > 0.12 * MHz or abs(mu) > max(abs(X)) / 3:
            # print("FWHM is larger than the range of X values.")
            return 0, 0, 0
        return mu, fwhm, snr
    except Exception as e:
        print("Error in FWHM calculation:", e)
        return 0, 0, 0


def FWHM_2D(X, Y, Z, echo=False):
    fwhms = []
    snrs = []
    mus = []
    for zi in Z:
        mu, fwhm, snr = FWHM(X, zi, echo=echo)
        fwhms.append(fwhm)
        snrs.append(snr)
        mus.append(mu)

    return np.array(mus), np.array(fwhms), np.array(snrs)


if __name__ == "__main__":

    import os

    echo = False

    file = "data/2025-09-14/"

    files = os.listdir(file)
    files = [f for f in files if f.endswith(".json")]

    for f in files:

        name = f

        file = os.path.join("data/2025-09-14/", f)

        import json

        with open(file, "r") as f:
            data = json.load(f)

        x = np.array(data["sweep_parameters"]["detuning"])
        y = np.array(data["sweep_parameters"]["amplitudes"])
        z = np.array(data["measured_data"]["states"])

        # for zi in z[::20]:
        #     # plt.plot(x, zi)
        #     FWHM(x, zi, plot=True, echo=echo)
        #     # plt.show()

        mus, fwhms, snrs = FWHM_2D(x, y, z, echo=echo)
        plt.title(name)
        plt.pcolormesh(x, y, z, shading="auto")

        fwhms[fwhms == 0] = None
        mus[mus == 0] = None
        plt.plot(mus + fwhms / 2, y, "go-")
        plt.plot(mus - fwhms / 2, y, "go-")
        plt.plot(mus, y, "ro-")
        plt.xlabel("Detuning (MHz)")
        plt.ylabel("Amplitude (MHz.)")
        plt.colorbar(label="Signal Intensity")
        plt.xlim(x[0], x[-1])
        plt.show()
