import numpy as np
from matplotlib import pyplot as plt

from args import us, MHz, T2
from graphs_for_poster.open_hdf5 import get_data_from_file, list_hdf5_files
from matplotlib import rcParams

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

folder = '/Users/asafsolonnikov/Documents/GitHub/Power-Broadening-2/graphs_for_poster/lorentzian-eco/data/'

list_of_files = list_hdf5_files(folder)[1:]

d = np.load(folder + 'lorentzian_d.npy')
a = np.load(folder + 'lorentzian_a.npy')
m = np.load(folder + 'lorentzian_m.npy')
print(d)
print(d.shape, a.shape, m.shape)

fig, axes = plt.subplots(3, 2, figsize=(8, 9))

indexes = [25, 40, 10]
axes[0, 1].pcolor(d / 1e3 / 2 / np.pi, a / 2 / np.pi / 1e6, m, cmap='inferno')
axes[0, 0].plot(d / 1e3 / 2 / np.pi, m[indexes[0]])
axes[0, 1].axhline(y=a[indexes[0]] / 2 / np.pi / 1e6, linestyle='--', color='b', label='slice')
axes[0, 1].axvline(x=-1 / T2 / 2 / np.pi / 1e3, linestyle='-.', color='g', label='T2 limit')
axes[0, 1].axvline(x=1 / T2 / 2 / np.pi / 1e3, linestyle='-.', color='g')
axes[0, 0].axvline(x=-1 / T2 / 2 / np.pi / 1e3, linestyle='-.', color='g', label='T2 limit')
axes[0, 0].axvline(x=1 / T2 / 2 / np.pi / 1e3, linestyle='-.', color='g')
axes[0, 1].legend()
for i in range(1, 3):
    file_path = list_of_files[i - 1]
    data, args, exp_args = get_data_from_file(file_path, dim=2)
    x, y, matrix = data

    # axes[i, j].plot(x, y)
    # axes[i, j].set_aspect('equal')  # Set the aspect ratio to 'equal'
    index = indexes[i]
    c1 = axes[i, 1].pcolor(x / 1e3, y, matrix, cmap='inferno')
    axes[i, 0].plot(x / 1e3, matrix[index])
    axes[i, 1].axhline(y=y[index], linestyle='--', color='b', label='slice')

    # axes[i, 1].set_title(f'cutoff = {exp_args["cutoff"] * 100} \%')
    axes[2, 1].set_xlabel('Detuning [kHz]')
    axes[2, 0].set_xlabel('Detuning [kHz]')

    axes[1, 1].set_ylabel('Amplitude [MHz]')
    axes[i, 1].linewidth = 5
    axes[1, 0].set_ylabel('Excited state probability')
    axes[i, 0].axvline(x=1 / T2 / 2 / np.pi / 1e3, linestyle='-.', color='g', label='slice')
    axes[i, 0].axvline(x=-1 / T2 / 2 / np.pi / 1e3, linestyle='-.', color='g', label='slice')
    fig.text(0.834, 0.83-i*0.315, f'pulse length = {exp_args['pulse_length'] / 1e3} us', va='center', rotation=270, fontsize=12)

cbar = fig.colorbar(c1, ax=axes, orientation='vertical', fraction=0.025, pad=0.1)
cbar.set_label('excited state probability', fontsize=12)
fig.text(0.834, 0.83,f'pulse length = {exp_args['pulse_length']/1e3} us', va='center', rotation=270, fontsize=12)

# fig.subplots_adjust(bottom=0.)  # Adjust bottom margin to make space for colorbar

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('lorentzian-eco.svg', format='svg', transparent=True)

plt.show()
