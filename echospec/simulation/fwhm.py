from matplotlib import pyplot as plt
from qutip import *
from numpy import pi
import numpy as np
from echospec.utils.units import Units as u

from echospec.utils.parameters import Parameters


def spectroscopy(detuning, params: Parameters):
    T1 = params.T1
    T_dephasing = params.T_dephasing
    a = destroy(2)
    H0 = -detuning * sigmaz() / 2
    H1 = params.rabi_frequency * sigmax() / 2
    H = H0 + H1
    ts = np.linspace(0, params.pulse_length, 12030)
    psi0 = basis(2, 0)
    c_opts = [np.sqrt(1 / T1) * a, np.sqrt(2 / T_dephasing) * a.dag() * a]
    result = mesolve(H, psi0, ts, c_opts, [a.dag() * a], args={})
    return result


def Torreys_solution(d, w_rabi, T1, T2):
    a = 1 + (T2 * d) ** 2
    b = 1 + (T2 * d) ** 2 + (w_rabi) ** 2 * T1 * T2
    return -a / b / 2 + 1 / 2


def run_par(detuning, sim_args):
    result = spectroscopy(detuning, sim_args)
    y = result.expect[0][-1]
    return y


def plot_spectroscopy(detunings, w_rabi, params: Parameters) -> None:

    params.rabi_frequency = w_rabi
    params.pulse_length = 200 * u.us
    params.cutoff = 0.999
    params.eco_pulse = False
    vec = parallel_map(run_par, detunings, task_args=(params,))
    plt.plot(detunings / u.MHz / 2 / pi, vec,
             label=f'square ({w_rabi / u.MHz / 2 / pi:.4f} MHz)')
    plt.axhline(y=max(vec), color='gray', linestyle='-.')
    plt.axhline(y=max(vec) / 2, color='gray', linestyle='-.')
    plt.legend()


def plot(ax, detunings, w_rabi, color):
    params.rabi_frequency = w_rabi
    vec_torreys = Torreys_solution(
        detunings, params.rabi_frequency, params.T1, params.T2)
    vec = parallel_map(run_par, detunings, task_args=(params,))
    ax.plot(detunings / 2 / pi / u.MHz, vec, color=color,
            linewidth=4, label=rf'{w_rabi/2/pi/u.MHz:.3f} MHz')
    # ax.plot(detunings / 2 / pi / MHz, vec_torreys, '.', label=f'{w_rabi/2/pi/MHz:.3f} MHz')
    return


def lorentzian(x, a, b, c):
    return a / (1 + (x / b) ** 2) + c


if __name__ == "__main__":

    params = Parameters()
    span = 10 * 2 * pi * u.MHz
    detunings = np.linspace(-span / 2, span / 2, 61)

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.axvline(x=params.T2_limit / 2 / u.MHz, linestyle='-.', color='gray')
    plt.axvline(x=-params.T2_limit / 2 / u.MHz, linestyle='-.', color='gray')

    plot(w_rabi=0.1 * 2 * pi * u.MHz, detunings=detunings, ax=ax, color='C0')

    plt.xlabel("Detuning (MHz)")
    plt.ylabel("Population")
    # plt.legend()
    plt.savefig("torrey.pdf", dpi=300, bbox_inches='tight')
    plt.show()
