from matplotlib import pyplot as plt
from qutip import *
import numpy as np
from qutip.nonmarkov.heom import DrudeLorentzBath
from qutip.nonmarkov.heom import HEOMSolver
from numpy import pi as pi, real

ttotal = 200.


def run(ttotal):
    a = destroy(2)
    w = 0.01 * 2 * np.pi
    V = 0.11
    delt = 20
    lam = 0.0005
    gamma = 0.005
    T = .05
    Nk = 5

    H0 = w * sigmaz() / 2
    H1 = sigmax() / 2

    psi0 = (basis(2, 0) + basis(2, 1)).unit()
    rho0 = psi0 * psi0.dag()

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
        return (np.heaviside(t - ttotal / 2 + pi / V / 2, 1) - np.heaviside(t - ttotal / 2 - pi / V / 2, 1)) * V

    def final_pulse(t):
        return drive_1(t)

    def power_density(w):
        return w * 2 * lam * gamma / ((gamma ** 2 + w ** 2))

    H = [H0, [H1, final_pulse]]
    H = H0 + H1

    Q = sigmaz()

    bath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=Nk, tag="bath1")

    max_depth = 5  # maximum hierarchy depth to retain
    optionsODE = Options(nsteps=1500, store_states=True, rtol=1e-12, atol=1e-12, max_step=1 / 20)
    solver = HEOMSolver(H, bath, max_depth=max_depth, options=optionsODE)

    result = solver.run(rho0, tlist)
    rhos = expect(result.states, sigmax())
    rhos = real(rhos)
    return rhos[-1]


tlist = np.linspace(100, 10000, 201)
vec = parallel_map(run, tlist, task_args=(), progress_bar=True)
plt.plot(tlist, vec)
plt.show()

