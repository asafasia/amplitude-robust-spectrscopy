import numpy as np
from matplotlib import pyplot as plt

from args import us, MHz
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

folder = '/Users/asafsolonnikov/Documents/GitHub/Power-Broadening-2/graphs_for_poster/lorentzian/data/'

list_of_files = list_hdf5_files(folder)

print(list_of_files)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for i in range(3):
    file_path = list_of_files[i]
    data, args, exp_args = get_data_from_file(file_path, dim=2)
    x, y, matrix = data
    X, Y = np.meshgrid(x, y)
    # axes[i, j].plot(x, y)
    # axes[i, j].set_aspect('equal')  # Set the aspect ratio to 'equal'

    c1 = axes[i].pcolormesh(X / MHz, Y, matrix, shading='auto')
    axes[i].set_title(f'cutoff = {exp_args["cutoff"] * 100} \%')
    axes[i].set_xlabel('Detuning [MHz]')
    axes[0].set_ylabel('Amplitude [MHz]')
    axes[i].linewidth = 5

cbar = fig.colorbar(c1, ax=axes, orientation='vertical', fraction=0.03, pad=0.1)
cbar.set_label('excited state probability', fontsize=12)
plt.tight_layout(rect=[0, 0, 0.85, 1])

plt.savefig('lorentzian1.svg', format='svg', transparent=True)

plt.show()
