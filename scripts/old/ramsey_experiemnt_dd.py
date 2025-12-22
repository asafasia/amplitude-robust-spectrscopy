from matplotlib import pyplot as plt
from qutip import *
import numpy as np
from qutip.nonmarkov.heom import DrudeLorentzBath
from qutip.nonmarkov.heom import HEOMSolver
from numpy import pi as pi, real
from scipy.optimize import curve_fit


def run(ttotal):
    w = 1
    a = destroy(2)
    V = 1
    delt = 20
    lam = 0.0005
    gamma = 0.005
    T = .05
    Nk = 5

    H0 = w * sigmaz() / 2
    H1 = sigmax() / 2

    psi0 = (basis(2, 0) + basis(2, 1)).unit()
    rho0 = psi0 * psi0.dag()

    # ttotal = 200.
    tlist = np.linspace(0, ttotal, 1000)

    def cos(t):
        return np.cos(w * t)

    def drive(t):
        tp = pi / V

        N = int(ttotal / (tp + delt))
        RF = 0.
        for n in np.arange(1, N + 1):
            if t >= n * delt + (n - 1) * tp:
                if t <= n * delt + n * tp:
                    RF += V
        return RF

    def drive_1(t):
        return V * (np.heaviside(t - ttotal / 2 + pi / V, 0) - np.heaviside(t - ttotal / 2 - pi / V, 0))

    def final_pulse(t):
        return drive(t)

    def power_density(w):
        return w * 2 * lam * gamma / ((gamma ** 2 + w ** 2))

    H = [H0, [H1, drive_1]]

    Q = sigmaz()

    bath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=Nk, tag="bath1")

    max_depth = 5  # maximum hierarchy depth to retain
    optionsODE = Options(nsteps=1500, store_states=True, rtol=1e-12, atol=1e-12, max_step=1 / 20)

    solver = HEOMSolver(H0, bath, max_depth=max_depth, options=optionsODE)

    result = solver.run(rho0, tlist)

    rhos = expect(result.states, sigmax())
    return rhos[-1]


def decay(tlist, a, b, c, d, e):
    return a * np.exp(-1 *1/b ** 2 * tlist ** 2/2) * np.cos(c * tlist + d) + e


tlist = np.linspace(0.1, 200, 300)

vec = parallel_map(run, tlist, progress_bar=True)
plt.plot(tlist, vec)
try:
    popt = curve_fit(decay, tlist, vec, p0=[1, 1, 1, 0, 0])[0]
    plt.plot(tlist, decay(tlist, *popt[0]), label=f'decay = {popt[1]}')
except:
    print('Could not fit the curve')

plt.show()

# plt.plot(tlist, drive_1(tlist))
# plt.plot(tlist, rhos)
# plt.show()
