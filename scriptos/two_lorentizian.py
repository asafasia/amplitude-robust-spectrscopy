import numpy as np
from matplotlib import pyplot as plt

from utils import FWHM

d = 2


def lorentzian(x, x0, tau):
    return 1 / (1 + ((x - x0) / tau) ** 2)


def super_position(d, plot=False):
    x = np.linspace(-10, 10, 1000)

    y1 = lorentzian(x, d / 2, 1)

    y2 = lorentzian(x, -d / 2, 1)

    y = y1 + y2

    fwhm, snr = FWHM(x, y)

    if plot:
        plt.plot(x, y1, label='x0 = d/2')
        plt.plot(x, y2, label='x0 = -d/2')
        plt.plot(x, y, label='Total')
        plt.axvline(x=fwhm / 2, color='r', label='FWHM')
        plt.axvline(x=-fwhm / 2, color='r', label='FWHM')

        plt.show()
    return fwhm, snr


ds = np.linspace(0, 5, 500)

snrs = []
fwhms = []

for d in ds:
    fwhm, snr = super_position(d, plot=False)
    fwhms.append(fwhm)
    snrs.append(snr)

fwhm = np.array(fwhm, )
snrs = np.array(snrs, )

plt.plot(ds, fwhms, label='fwhm')
plt.plot(ds, snrs, label='snr')

plt.plot(ds, fwhms / snrs, label='sigma')

plt.ylim([0, 10])

plt.legend()

plt.axvline(x=2)
plt.show()
