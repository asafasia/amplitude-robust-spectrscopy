import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def lorentzian(x, x0, tau):
    return 1 / (1 + ((x - x0) / tau) ** 2)


def dual_lorentzian(x, x0, d, tau):
    return lorentzian(x, x0 - d, tau)/2 + lorentzian(x, x0 + d, tau)/2


x = np.linspace(-20, 20, 1000)

x0 = 0
tau = 1
d = 2

ds = np.linspace(0.2, 5, 100)

errors =[]
for d in ds:
    noise = 0.1
    y1 = lorentzian(x, x0, tau) + np.random.normal(0, noise, len(x))

    y2 = dual_lorentzian(x, x0, d, tau) + np.random.normal(0, noise, len(x))

    # plt.plot(x, y1)

    # plt.plot(x, y2, label='y1')

    popt1, pcov1 = curve_fit(lorentzian, x, y1)
    popt2, pcov2 = curve_fit(dual_lorentzian, x, y2)

    perr1 = np.sqrt(np.diag(pcov1))
    perr2 = np.sqrt(np.diag(pcov2))

    print(perr1)
    print(perr2)
    #
    # plt.plot(x, lorentzian(x, *popt1))
    # plt.plot(x, dual_lorentzian(x, *popt2), label='y2')
    #
    # plt.show()

    errors.append(perr2[1])

plt.plot(ds,errors)


plt.show()
