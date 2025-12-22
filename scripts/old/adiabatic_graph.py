from matplotlib import pyplot as plt

from args import *
from utils import lorentzian


def A(detuning, w, plot=False):
    l = qubit_args['pulse_length']
    c = qubit_args['cutoff']
    n = qubit_args['n']
    ts = np.linspace(-l, l, 10000)
    #
    pulse_vec = 10 * lorentzian(ts, qubit_args) * np.heaviside(ts + l / 2, 0) * np.heaviside(-ts + l / 2, 0)
    #
    # Theta = 0.5 * np.arctan(pulse_vec / detuning)
    #
    # dTheta = np.diff(Theta) / np.diff(ts)
    #
    # E = np.sqrt(pulse_vec ** 2 + detuning ** 2)
    # #
    # A = dTheta / E[:-1]
    #
    # max_A = np.max(A)

    T = ((l / 2) ** 2 / ((1 / c ** (1 / n)) - 1)) ** (1 / 2)

    A_max = np.sqrt(2 * np.log(2 * w ** 2 / detuning ** 2)) / (3 * np.sqrt(3) * detuning * T)
    print(A_max)

    # if plot:
    #     plt.plot(ts / us, pulse_vec / 2 / pi / MHz)
    #     plt.show()
    return A_max


A(0.01, 1 * 2 * pi * MHz, plot=True)
A(0.01, 0.1 * 2 * pi * MHz, plot=True)
A(0.01, 0.01 * 2 * pi * MHz, plot=True)

plt.show()
# #
# detunings = np.linspace(-qubit_args['detuning_span'], qubit_args['detuning_span'], 610)
# w_rabis = np.linspace(0, qubit_args['rabi_frequency'], 10)
# A_matrix = []
#
# for w in w_rabis:
#     print(f'w_rabi = {w / 2 / pi / MHz:.1f} MHz')
#     As = [A(detuning, w) for detuning in detunings]
#     A_matrix.append(As)
#     #
#     plt.plot(detunings / pi / T2_limit, As)
#
# plt.show()
#
# plt.pcolormesh(detunings / 2 / pi / MHz, w_rabis / 2 / pi / MHz, A_matrix)
#
# # plt.ylim(0, 1)
# # plt.axvline(x=1, color='gray', linestyle='-.')
# # plt.axvline(x=-1, color='gray', linestyle='-.')
#
# plt.show()
