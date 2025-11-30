from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

from .args import *
from scipy.optimize import root_scalar, fsolve
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import minimize, curve_fit


# def FWHM(X, Y):
#     # plt.show()
#     X = np.array(X)
#     Y = np.array(Y)
#     # plt.plot(X/2/pi/MHz,Y)
#     Y = gaussian_filter(Y, 2)

#     if Y[len(Y) // 2] > Y[len(Y) // 2 + 1]:
#         Y = 1 - Y

#     f = interp1d(X, Y, kind="quadratic")

#     # plt.show()
#     # plt.plot(X, Y)

#     # if peak rotate data

#     # find_peaks
#     peaks, _ = find_peaks(Y, height=0)
#     x_peaks = X[peaks]
#     y_peaks = Y[peaks]
#     # plt.plot(X/2/pi/MHz,Y)
#     # plt.plot(x_peaks/2/pi/MHz, y_peaks,'.')
#     # plt.show()

#     # Calculate FWHM
#     # Check if there are at least two peaks close to x=0
#     if len(x_peaks) >= 2:
#         closest_indices = np.argsort(np.abs(x_peaks))[
#             :2
#         ]  # Get indices of the two closest peaks
#         x_min, x_max = np.sort(
#             x_peaks[closest_indices]
#         )  # Get x-range between those peaks
#         # Extract portion of data between the two peaks
#         mask = (X >= x_min) & (X <= x_max)
#         x_shrunk = X[mask]
#         y_shrunk = Y[mask]
#         X = x_shrunk
#         Y = y_shrunk
#     else:
#         print("Not enough peaks found around x=0.")
#     y_max = max(Y)
#     y_min = Y[X == 0]
#     y_min = f(0)

#     y_half = (y_max + y_min) / 2
#     Y -= y_half
#     Y = abs(Y)

#     # desecete
#     # fwhm = X[np.argmin(Y)] * 2
#     #
#     x_dense = np.linspace(-X[0], X[1], 1000)

#     # conti
#     def g(x):
#         return np.abs(f(x) - y_half)

#     y_dense = g(x_dense)

#     min_index = np.argmin(y_dense)

#     x_min = x_dense[min_index]

#     fwhm = abs(x_min * 2)

#     snr = abs(y_max - y_min)

#     # plt.plot(X, Y)

#     return fwhm, snr


def FWHM(X, Y, plot=False, echo=False):

    # i = 40

    # X = np.array(X)[i:-i]
    # Y = np.array(Y)[i:-i]

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

    A0 = max(Y) - min(Y)
    mu0 = 0
    # sigma0 = (max(X) - min(X)) / 6  # Initial guess for standard deviation
    sigma0 = 40e3
    d0 = np.min(Y)

    initial_guess = [A0, mu0, sigma0, d0]
    bounds = ([0, min(X), 0, -1], [1, max(X), max(X), 1])

    if plot:
        plt.plot(X, Y)
        plt.axhline(y=A0, color="g", linestyle="--", label="A0")
        plt.axhline(y=d0, color="b", linestyle="--", label="d0")
        plt.axvline(x=mu0, color="y", linestyle="--", label="mu0")
        plt.axvline(x=mu0 + sigma0, color="c", linestyle="--", label="sigma0")
        plt.axvline(x=mu0 - sigma0, color="c", linestyle="--")
        plt.legend()

    try:

        popt, pcov = curve_fit(gaussian, X, Y, p0=initial_guess, bounds=bounds)

        mu = popt[1]
        sigma = popt[2]
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma  # FWHM for Gaussian
        snr = abs(popt[0])

        print("FWHM (calculated):", fwhm)

        # print("FWHM (from fit):", fwhm)

        if plot:
            plt.plot(X, gaussian(X, *popt), "r--", label="Gaussian Fit")
            plt.axvline(x=fwhm / 2 + mu, color="k", linestyle="--")
            plt.axvline(x=-fwhm / 2 + mu, color="k", linestyle="--")
            plt.axvline(x=mu, color="k", linestyle="--")
            plt.show()

        return mu, abs(fwhm), abs(snr)
    except Exception as e:
        print("Error in FWHM calculation:", e)
        return 0, 0, 0


def cos(t, w):
    return np.cos(w * t)


def gaussian(t, args):
    cutoff = args["cutoff"]
    pulse_length = args["pulse_length"]

    sigma = np.sqrt(-((pulse_length / 2) ** 2) / (2 * np.log(cutoff)))
    return np.exp(-(t**2) / (2 * sigma**2))


def gaussian_s(t, a, b, c, d):
    return a * np.exp(-((t - b) ** 2) / (2 * c**2)) + d


def gaussian_half(t, args):
    cutoff = args["cutoff"]
    pulse_length = args["pulse_length"]

    sigma = np.sqrt(-((pulse_length / 2) ** 2) / (2 * np.log(cutoff)))
    pulse = np.exp(-(t**2) / (2 * sigma**2))
    return -np.sin(2 * np.pi / pulse_length * t) * pulse


def lorentzian(t, args):
    n = args["n"]
    cutoff = args["cutoff"]
    pulse_length = args["pulse_length"]
    sigma = ((pulse_length / 2) ** 2 / ((1 / cutoff ** (1 / n)) - 1)) ** (1 / 2)

    if args["zeroed_pulse"]:
        return (1 + (t / sigma) ** 2) ** (-n) - cutoff
    else:
        return 1 / (1 + (t / sigma) ** 2) ** n

        # return np.abs(t/10e-6) ** -1


def lorentzian_half(t, args):
    pulse_length = args["pulse_length"]
    pulse = lorentzian(t, args)
    # return pulse
    return (1 - 2 * np.heaviside(t, 0)) * pulse

    # return np.tanh(10 * t / pulse_length) * pulse

    # return -np.sin(10 * np.pi / pulse_length * t) * pulse

    # n = 4
    # s = -1
    # pulse_s = 0
    # for i in np.linspace(-n // 2, n // 2, n - 1):
    #     phase = np.sign(i) * pulse_length * i ** 2 / n ** 2
    #
    #     pulse_s += s * 2 * np.heaviside(t + phase, 0)
    #     s *= -1
    #
    # pulse_s = 1 + pulse_s
    # return pulse_s * pulse


def square_half(t, args):
    pulse_length = args["pulse_length"]
    # return -np.sin(np.pi / pulse_length * t)

    return 2 * np.heaviside(t, 0) - 1
    # return -np.sin((10*np.pi / pulse_length * t))
    # n = 8
    # s = -1
    # pulse_s = 0
    # for i in np.linspace(-n // 2, n // 2, n - 1):
    #     phase = np.sign(i) * pulse_length * i ** 2 / n ** 2
    #
    #     pulse_s += s * 2 * np.heaviside(t + phase, 0)
    #     s *= -1
    #
    # pulse_s = 1 + pulse_s
    # return pulse_s


def double(t, args):
    pulse_length = args["pulse_length"]
    sigma = pulse_length / 10
    return gaussian_s(t, 1, 1, 1, 1)


def find_per(x, v1, v2):
    interp1 = interp1d(x, v1, kind="cubic")
    interp2 = interp1d(x, v2, kind="cubic")

    def g(x):
        return interp1(x) - interp2(x)

    # plt.plot(x/MHz, interp1(x) - interp2(x), '.')

    root = fsolve(g, 0)

    print(f"Period = {root[0] / MHz:.3f} MHz")

    return interp1(root)
