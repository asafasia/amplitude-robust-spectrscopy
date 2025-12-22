from matplotlib import pyplot as plt

from simi.fwhm import Torreys_solution

import numpy as np
from scipy.optimize import brentq
from simi.args import *

span = 10 * 2 * pi * MHz
w_rabi = 0.001 * 2 * pi * MHz / 2


def find_fwhm_vs_rabi(T1, T2, w_rabi, plot=False):
    detunings = np.linspace(-span / 2, span / 2, 61)
    y = Torreys_solution(detunings, w_rabi, T1, T2)
    half_max = max(y) / 2

    def half_max_func(x):
        return Torreys_solution(x, w_rabi, T1, T2) - half_max

    x_left = brentq(half_max_func, -100 * 2 * pi * MHz, 0)
    x_right = brentq(half_max_func, 0, 100 * 2 * pi * MHz)

    if plot:
        print(x_left, x_right)
        plt.axvline(x_left / 2 / pi / MHz, linestyle='-.', color='gray')
        plt.axvline(x_right / 2 / pi / MHz, linestyle='-.', color='gray')
        plt.axhline(half_max, linestyle='-.', color='gray')
        plt.plot(detunings / 2 / pi / MHz, y,
                 label="Torrey's solution", color='blue')
        plt.show()

    fwhm = (x_right - x_left) / 2 / pi
    snr = max(y)

    return fwhm, snr


def find_fwhm_vec_vs_rabi(ax, T1=qubit_args['T1'], T2=qubit_args['T_dephasing']):
    T2 = (1 / (T_dephasing) + 1 / (2 * T1)) ** -1

    ws = np.logspace(-3, 3, 500) * 2 * pi * KHz * 5
    fwhms = []
    snrs = []
    for w in ws:
        fwhm, snr = find_fwhm_vs_rabi(T1, T2, w, False)

        fwhms.append(fwhm)
        snrs.append(snr)

    fwhms = np.array(fwhms)
    snrs = np.array(snrs)
    sigma = fwhms / snrs / 100

    print(rf' $\sigma$ = {min(sigma):.1f} Hz')

    T2_limit = 1 / T2 / pi

    scatter = ax.scatter(ws / 2 / pi / MHz, fwhms / T2_limit,
                         c=snrs, label='Bloch FWHM', vmin=0, vmax=0.5)

    x = ws / 2 / pi / MHz
    y = fwhms / T2_limit / snrs
    ax.plot(ws / 2 / pi / MHz, fwhms / T2_limit / snrs, label='FWHM/SNR')

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
    amps = np.logspace(-3, 1) * 2 * pi * MHz
    lin = amps / np.sqrt(T2 / T1) / T2_limit * 2
    ax.plot(amps / MHz / 2 / pi, lin / 2 / pi, color='gray', linestyle='--')
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
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    find_fwhm_vec_vs_rabi(ax)
    plt.show()
