from scipy.optimize import curve_fit

from utils import *
from matplotlib import pyplot as plt

w0 = qubit_args["qubit_frequency"]
Omega = qubit_args["rabi_frequency"]
alpha = qubit_args["anharmonicity"]

detuning = 2 * pi * MHz

qubit_Hamiltonian = detuning * a.dag() * a / 2 + alpha * a.dag() * a.dag() * a * a / 2

H = [qubit_Hamiltonian]

psi0 = (basis(N_dim, 0) + basis(N_dim, 1)).unit()

tlist = np.linspace(0, 50 * us, 100000)
t_dense = np.linspace(0, tlist[-1], 1000)
c_ops = [
    np.sqrt(1 / qubit_args["T1"]) * a,
    np.sqrt(2 / qubit_args["T_dephasing"]) * a.dag() * a
]

sim_args = {
    'w': w0
}

result = mesolve(H, psi0, tlist, c_ops, [sigmax()], args=sim_args)

expect = result.expect[0]

plt.plot(tlist, result.expect[0])


def exp_cos(t, a, b, c, d, e):
    return a * np.exp(-b * t) * np.cos(c * t + d) + e


fit_args = curve_fit(exp_cos, tlist, result.expect[0], p0=[1, 1, 1, 1, 1])

plt.plot(t_dense, exp_cos(t_dense, *fit_args[0]), 'r--', label=f'fit, T2 = {1 / fit_args[0][1] / us:.4f} us')
plt.legend()
plt.show()
