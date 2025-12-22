# from matplotlib import pyplot as plt
# from scipy.optimize import curve_fit
# from scipy.optimize import minimize_scalar

# from simulation.args import *
# from qutip import *


# def spectroscopy(detuning, args):
#     H0 = -detuning * sigmaz() / 2
#     H1 = args['w_rabi'] * sigmax() / 2
#     H = H0 + H1
#     ts = np.linspace(0, args['pulse_length'], 12030)
#     psi0 = basis(2, 0)
#     c_opts = [np.sqrt(1 / T1) * a, np.sqrt(2 / T_dephasing) * a.dag() * a]
#     result = mesolve(H, psi0, ts, c_opts, [a.dag() * a], args={})
#     return result


# def Torreys_solution(d, w_rabi, T1, T2):
#     a = 1 + (T2 * d) ** 2
#     b = 1 + (T2 * d) ** 2 + (w_rabi) ** 2 * T1 * T2
#     return -a / b / 2 + 1 / 2


# def run_par(detuning, sim_args):
#     result = spectroscopy(detuning, sim_args)
#     y = result.expect[0][-1]
#     return y


# def plot_spectroscopy(detunings, w_rabi):
#     qubit_args['w_rabi'] = w_rabi
#     qubit_args['pulse_length'] = 200 * us
#     qubit_args['cutoff'] = 0.999
#     qubit_args['echo'] = False
#     vec = parallel_map(run_par, detunings, task_args=(qubit_args,))
#     # plt.plot(detunings / MHz / 2 / pi, vec,
#     #          label=f'square ({w_rabi / MHz / 2 / pi:.4f} MHz)')
#     # plt.axhline(y=max(vec), color='gray', linestyle='-.')
#     # plt.axhline(y=max(vec) / 2, color='gray', linestyle='-.')
#     # plt.legend()


# def plot(ax, detunings, w_rabi, color):
#     qubit_args['w_rabi'] = w_rabi
#     vec_torreys = Torreys_solution(detunings, qubit_args['w_rabi'], T1, T2)
#     vec = parallel_map(run_par, detunings, task_args=(qubit_args,))
#     ax.plot(detunings / 2 / pi / MHz, vec, color=color,
#             linewidth=4, label=rf'{w_rabi/2/pi/MHz:.3f} MHz')
#     # ax.plot(detunings / 2 / pi / MHz, vec_torreys, '.', label=f'{w_rabi/2/pi/MHz:.3f} MHz')
#     return


# def lorentzian(x, a, b, c):
#     return a / (1 + (x / b) ** 2) + c


# if __name__ == "__main__":
#     span = 10 * 2 * pi * MHz
#     detunings = np.linspace(-span / 2, span / 2, 61)

#     fig, ax = plt.subplots(figsize=(8, 6))

#     plt.axvline(x=T2_limit / 2 / MHz, linestyle='-.', color='gray')
#     plt.axvline(x=-T2_limit / 2 / MHz, linestyle='-.', color='gray')

#     plot(w_rabi=0.1 * 2 * pi * MHz, detunings=detunings, ax=ax, color='C0')

#     plt.xlabel("Detuning (MHz)")
#     plt.ylabel("Population")
#     # plt.legend()
#     plt.savefig("torrey.pdf", dpi=300, bbox_inches='tight')
#     plt.show()
