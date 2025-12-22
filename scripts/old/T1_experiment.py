from scipy.optimize import curve_fit

from utils import *
from matplotlib import pyplot as plt

w0 = qubit_args["qubit_frequency"]
Omega = qubit_args["rabi_frequency"]
alpha = qubit_args["anharmonicity"]

detuning = 2 * pi * MHz

qubit_Hamiltonian = detuning * a.dag() * a / 2 + alpha * a.dag() * a.dag() * a * a / 2

H = [qubit_Hamiltonian]

psi0 = basis(N_dim, 1)

tlist = np.linspace(0, 100 * us, 100)


t_dense = np.linspace(0, tlist[-1], 1000)
c_ops = [
    np.sqrt(1 / qubit_args["T1"]) * a,
    np.sqrt(2 / qubit_args["T_dephasing"]) * a.dag() * a
]

sim_args = {
    'w': w0
}

result = mesolve(H, psi0, tlist, c_ops, [a.dag()*a], args=sim_args)

expect = result.expect[0]

plt.plot(tlist/us, result.expect[0])


def exp_cos(t, a, b, c):
    return a * np.exp(-b * t) + c


fit_args = curve_fit(exp_cos, tlist, result.expect[0], p0=[1, 1/T1, 0])

plt.plot(t_dense/us, exp_cos(t_dense, *fit_args[0]), 'r--', label=f'fit, T1 = {1 / fit_args[0][1] / us:.4f} us')
plt.legend()
plt.show()
