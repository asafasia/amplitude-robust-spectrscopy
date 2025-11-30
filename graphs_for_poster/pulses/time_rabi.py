import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from graphs_for_poster.open_hdf5 import get_data_from_file

folder = '/Users/asafsolonnikov/Documents/GitHub/Power-Broadening-2/graphs_for_poster/pulses/'

file_path = folder + 'time-rabi__1.hdf5'
data, args, exp_args = get_data_from_file(file_path, dim=1)
t, y = data

#
# t = np.linspace(0, 10, 100)
# y = 2 * np.sin(2 * np.pi * t) * np.exp(-t / 5) + np.random.normal(0, 0.1, t.shape) - 1

fig = plt.figure(figsize=(5, 2.2))
ax = fig.add_subplot(111)

T1 = 31e-6


def exp_decay(t, A, T2, omega, C):
    return A * np.exp(-t / T2) * np.cos(2 * np.pi * omega * t + np.pi) + C


plt.plot(t[0:-1:2]*1e6, y[0:-1:2], '.',color='C4')
args_time_rabi = curve_fit(exp_decay, t, y, p0=[1 / 2, T1, 1 / 0.24e-5, np.mean(y)])[0]
plt.plot(t*1e6, exp_decay(t, *args_time_rabi), color='purple', linestyle='--', label=f'$T_1$ = {args_time_rabi[1]:.2f}')

plt.ylabel('Excited state\n population', fontsize=15)
plt.xlabel('Delay [us]', fontsize=15)
plt.setp(ax.spines.values(), lw=2)
plt.tight_layout()
plt.title('Rabi Oscillations')
# plt.ylim([0.4, 0.74])
plt.tight_layout()
plt.savefig('rabi.pdf', format='pdf', transparent=True)
plt.show()
