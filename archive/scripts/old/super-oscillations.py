import numpy as np
from numpy import pi


n = 24


def F(t, a):
    return (np.cos(t / n)+ 1j *a* np.sin(t / n)) ** n

N = 1
a = 1.5
t = np.linspace(-2*N * np.pi, 2*N * np.pi, 1000)

Fs = F(t, a)

import matplotlib.pyplot as plt

# plt.plot(t/pi, np.abs(Fs)**2)

# plt.plot(t/pi, np.cos(t), 'k--', label='cos(t)')
plt.plot(t/pi, np.real(Fs))
# plt.ylim([-1.1, 1.1])
plt.show()
