import numpy as np
from matplotlib import pyplot as plt, rcParams
from scipy.optimize import curve_fit

from graphs_for_poster.open_hdf5 import get_data_from_file


def exp_decay(t, A, T1, C):
    return A * np.exp(-t / T1) + C


def exp_cos_decay(t, A, T2, C, omega, phi):
    return A * np.exp(-t / T2) * np.cos(2 * np.pi * omega * t) + C

rcParams.update({
    'text.usetex': True,
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (15, 5),
    'axes.linewidth': 2.5,  # Width of the box (axes) lines
    'grid.linewidth': 0.8  # Width of the grid lines
})

folder = '/Users/asafsolonnikov/Documents/GitHub/Power-Broadening-2/graphs_for_poster/t1/'
file = 'T1.hdf5'

T1 = 30e-6
T2 = 10e-6

file_path = folder + file

data, args, exp_args = get_data_from_file(file_path, dim=1)
x, y = data

args_T1 = curve_fit(exp_decay, x, y, p0=[1, T1, 0])[0]



fig, axes = plt.subplots(1, 2, figsize=(7, 3))


axes[0].plot(x * 1e6, y, 'o', markersize=6)
axes[0].plot(
    x * 1e6, exp_decay(x, *args_T1),
    color='C0',
    linestyle='--',
    label=f'$T_1$ = {args_T1[1] * 1e6:.2f} us',
    linewidth=2
)

#


# %%

# fig = plt.figure(figsize=(5, 4))
# ax = fig.add_subplot(111)


file = 'T2-ramsey__2.hdf5'
file_path = folder + file
data, args, exp_args = get_data_from_file(file_path, dim=1)

x, y = data

y = y / (max(y) - min(y)) * 0.92
y = y - np.mean(y) + 0.5
args_T2 = curve_fit(exp_cos_decay, x, y, p0=[0.5, T2 / 4, 0.5, 0.25e6, 0])[0]


axes[1].plot(x * 1e6, y, 'go')

axes[1].plot(
    x * 1e6, exp_cos_decay(x, *args_T2),
    color='g',
    linestyle='--',
    label=f'$T_2$ = {args_T2[1] * 1e6:.2f} us'
)
axes[1].set_xlabel('Delay [us]', fontsize=20)
axes[0].set_xlabel('Delay [us]', fontsize=20)
axes[0].set_ylabel('Excited state\n probability', fontsize=20)

axes[0].legend(fontsize=14)
axes[1].legend(fontsize=14)
plt.tight_layout()
#
plt.savefig('t1_t2.svg', format='svg', transparent=True)
plt.show()
