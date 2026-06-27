from matplotlib import pyplot as plt

from args import us, MHz, T1, T2
from graphs_for_poster.open_hdf5 import get_data_from_file, list_hdf5_files
from matplotlib import rcParams
import numpy as np

# folder = '/graphs_for_poster/lorentzian/data/'
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

folder = '/Users/asafsolonnikov/Documents/GitHub/Power-Broadening-2/graphs_for_poster/lorentzian2/data/'

list_of_files = list_hdf5_files(folder)

d = np.load(folder + 'lorentzian_d.npy')
a = np.load(folder + 'lorentzian_a.npy')
m = np.load(folder + 'lorentzian_m.npy')

fig, axes = plt.subplots(2, 3, figsize=(9, 5))
index = [32, 53, 52]

axes[0, 0].plot(d / 2 / np.pi / 1e6, m[index[0]])
axes[1, 0].pcolor(d / 2 / np.pi / 1e6, a / 2 / np.pi / MHz, m, cmap='plasma')
axes[1, 0].axhline(y=a[index[0]] / 2 / np.pi / 1e6, linestyle='--', color='y', label='slice')
axes[0, 0].set_title(f'pulse length = 10 us', fontsize=12)
axes[1, 0].set_xlabel('Detuning [MHz]')

for i in range(1, 3):
    file_path = list_of_files[i - 1]
    data, args, exp_args = get_data_from_file(file_path, dim=2)

    x, y, matrix = data

    # axes[i, j].plot(x, y)
    # axes[i, j].set_aspect('equal')  # Set the aspect ratio to 'equal'
    axes[0, i].plot(x / 1e6, matrix[index[i]])

    c1 = axes[1, i].pcolor(x / MHz, y, matrix, cmap='plasma')
    # axes[0, i].axvline(x=1 / T2 / 2 / np.pi / 1e6, linestyle='-.', color='g', label='slice')
    # axes[0, i].axvline(x=-1 / T2 / 2 / np.pi / 1e6, linestyle='-.', color='g', label='slice')
    axes[1, i].axhline(y=y[index[i]], linestyle='--', color='y', label='slice')

    axes[0, i].set_title(f'pulse length = {exp_args["pulse_length"] / 1e3} us', fontsize=12)
    axes[1, i].set_xlabel('Detuning [MHz]')
    axes[1, 0].set_ylabel('Drive \nAmplitude [MHz]')
    axes[0, 0].set_ylabel('Excited State \nProbability')
    axes[0, 2].set_xlim([-0.3, 0.3])
    axes[1, 2].set_xlim([-0.3, 0.3])
    axes[0, 2].axvline(x=1 / T2 / 2 / np.pi / 1e6, linestyle='--', color='g')
    axes[0, 2].axvline(x=-1 / T2 / 2 / np.pi / 1e6, linestyle='--', color='g', label='T2 limit')
    axes[1, 2].axvline(x=1 / T2 / 2 / np.pi / 1e6, linestyle='--', color='g')
    axes[1, 2].axvline(x=-1 / T2 / 2 / np.pi / 1e6, linestyle='--', color='g', label='T2 limit')
    # axes[0, 2].legend()
    axes[1, i].linewidth = 5

cbar = fig.colorbar(c1, ax=axes, orientation='vertical', fraction=0.02, pad=0.1)
cbar.set_label('excited state probability', fontsize=12)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('lorentzian2.svg', format='svg', transparent=True)
plt.show()
