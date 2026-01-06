from matplotlib import pyplot as plt

from echospec.simulation.fwhm import Torreys_solution

import numpy as np
from scipy.optimize import brentq
from echospec.utils.parameters import *

from echospec.utils.units import Units as u

from numpy import pi

span = 10 * 2 * pi * u.MHz
w_rabi = 0.001 * 2 * pi * u.MHz / 2


def find_fwhm_vs_rabi(T1, T2, w_rabi, plot=False):
    detunings = np.linspace(-span / 2, span / 2, 61)
    y = Torreys_solution(detunings, w_rabi, T1, T2)
    half_max = max(y) / 2

    def half_max_func(x):
        return Torreys_solution(x, w_rabi, T1, T2) - half_max

    x_left = brentq(half_max_func, -100 * 2 * pi * u.MHz, 0)
    x_right = brentq(half_max_func, 0, 100 * 2 * pi * u.MHz)

    if plot:
        print(x_left, x_right)
        plt.axvline(x_left / 2 / pi / u.MHz, linestyle='-.', color='gray')
        plt.axvline(x_right / 2 / pi / u.MHz, linestyle='-.', color='gray')
        plt.axhline(half_max, linestyle='-.', color='gray')
        plt.plot(detunings / 2 / pi / u.MHz, y,
                 label="Torrey's solution", color='blue')
        plt.show()

    fwhm = (x_right - x_left) / 2 / pi
    snr = max(y)

    return fwhm, snr


def find_fwhm_vec_vs_rabi(T1, T2):

    ws = np.logspace(-3, 3, 500) * 2 * pi * u.kHz * 5
    fwhms = []
    snrs = []
    for w in ws:
        fwhm, snr = find_fwhm_vs_rabi(T1, T2, w, False)

        fwhms.append(fwhm)
        snrs.append(snr)

    fwhms = np.array(fwhms)
    snrs = np.array(snrs)

    return ws, fwhms, snrs


def plot(ax, ws, fwhms, snrs) -> None:

    T1 = params.T1
    T2 = params.T2
    T2_limit = 2 * T1 * T2 / (2 * T1 + T2)

    scatter = ax.scatter(ws / 2 / pi / u.MHz, fwhms / T2_limit,
                         c=snrs, label='Bloch FWHM', vmin=0, vmax=0.5)

    x = ws / 2 / pi / u.MHz
    y = fwhms / T2_limit / snrs
    ax.plot(ws / 2 / pi / u.    MHz, fwhms / T2_limit / snrs, label='FWHM/SNR')

    ws_min = x[np.argmin(y)]
    s_min = y[np.argmin(y)]

    # plt.plot(ws_min, s_min, 'ro', label=f'min = {s_min:.2f}')

    plt.xlabel("Rabi amplitude [MHz]")
    plt.ylabel("FWHM [T2 limit units]")
    plt.colorbar(scatter, ax=ax, label='SNR', pad=0.04)

    # ax2 = ax.twinx()
    # ax2.plot(ws / 2 / pi / MHz, sigma,color='r')
    # ax2.set_ylabel(r"$\sigma$ [kHz]", color='r')
    # ax2.set_yticks([1e3,1e4,1e5,1e6])  # Set y-axis tick color to red

    plt.grid()

    # plt.axhline(y=1, color='black')
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    ax.set_yscale('log')
    ax.set_xscale('log')
    amps = np.logspace(-3, 1) * 2 * pi * u.MHz
    lin = amps / np.sqrt(T2 / T1) / T2_limit * 2
    ax.plot(amps / u.MHz / 2 / pi, lin / 2 / pi, color='gray', linestyle='--')
    ax.set_xlim([1e-3, 1e1])
    ax.set_ylim([1e0, 1e3])
    ax.legend()
    #
    # ax2.tick_params(axis='y', colors='r')
    #
    # ax2.set_ylim([1e3,1e7])

    #
    # plt.ylim([1, 1e3])


if __name__ == "__main__":

    params = Parameters()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ws, fwhms, snrs = find_fwhm_vec_vs_rabi(T1=params.T1, T2=params.T2)
    plot(ax, ws, fwhms, snrs)

    print(fwhms)
