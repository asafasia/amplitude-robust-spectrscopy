import numpy as np
from qutip import *
from matplotlib import pyplot as plt

gama1 = 2
gama2 = 3
wq = 1
wd = 0

detuning = wq - wd
ts = np.linspace(0, 10, 1000)


def density_BR(t):
    alpha = 1 / np.sqrt(2)
    beta = 1 / np.sqrt(2)
    r11 = 1 + (alpha ** 2 - 1) * np.exp(-gama1 * t)
    r22 = beta ** 2 * np.exp(-gama1 * t)

    r12 = alpha * beta.conjugate() * np.exp(-gama2 * t)
    r21 = r12.conjugate()

    return np.array([[r11, r12], [r21, r22]])


rho = Qobj(density_BR(0))
e1s = []
e2s = []
sigmax_expected = []

for t in ts:
    rho = Qobj(density_BR(t))
    e1, e2 = rho.eigenenergies()
    rho12 = rho[0][1]
    e1s.append(e1)
    e2s.append(e2)
    sigmax_expected.append()

plt.plot(ts, e1s, label="eigen value 1")

plt.plot(ts, e2s, label="eigen value 2")

plt.plot(ts, rho12)

plt.legend()
plt.title('Eigenvalues of the density matrix')
plt.show()
