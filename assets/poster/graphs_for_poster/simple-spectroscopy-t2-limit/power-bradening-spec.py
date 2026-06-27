import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from matplotlib.patches import ConnectionPatch

from matplotlib import rcParams

from graphs_for_poster.open_hdf5 import get_data_from_file

folder = '/Users/asafsolonnikov/Documents/GitHub/Power-Broadening-2/graphs_for_poster/simple-spectroscopy-t2-limit/'

file_path1 = folder + 'T1-qubit-spectroscopy-square__5.hdf5'
file_path2 = folder + 'T1-qubit-spectroscopy-square__25.hdf5'

data1, args, exp_args = get_data_from_file(file_path1, dim=2)
data2, args, exp_args = get_data_from_file(file_path2, dim=2)

rcParams.update({
    'font.serif': ['Times'],
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

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))
x, y, matrix = data1


def outliar(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] > 0.8:
                matrix[i][j] = matrix[i][j] - 0.3
    return matrix


matrix = outliar(matrix)

matrix = np.array(matrix)
c1 = axes[0].pcolor(x / 1e6, y, matrix, cmap='viridis', shading='auto')
i = 10
# xy = (x[i],y[i])
#
# con = ConnectionPatch(xyA=xy, xyB=xy, coordsA="data", coordsB="data",
#                       axesA=axes[0], axesB=axes[1], color="red")
# axes[0].add_artist(con)
#
# con1 = ConnectionPatch(xyA=xy, xyB=xy, coordsA="data", coordsB="data", axesA=axes[0], axesB=axes[1], color="blue")
# axes[2].add_artist(con1)

x1, x2, y1, y2 = -3, 3, -3, 3  # Coordinates of the zoom region in the data
x, y, matrix = data2
matrix = np.array(matrix) * 0.9 - 0.09
c2 = axes[1].plot(x / 1e6, matrix[0])
c2 = axes[1].plot(x / 1e6, matrix[1])
c2 = axes[1].plot(x / 1e6, matrix[2])
c2 = axes[1].plot(x / 1e6, matrix[3])
c2 = axes[1].plot(x / 1e6, matrix[-3])

v = np.array([0.013,0.019, 0.046, 0.13, 2, 0.2, 0.1])*2
u = [0,0.1, 0.2, 0.3, 0.4, 0.8, 0.6]

axes[1].plot(v, u, 'k--', label='FWHM',alpha = 1)  # Top line
axes[1].plot(-v, u, 'k--',alpha=1)  # Top line
axes[1].set_xlim([-0.2, 0.2])
axes[1].set_ylim([0, 0.7])
# Set limits for zoom-in on the right subplot
# axes[1].set_xlim(x1, x2)
# axes[1].set_ylim(y1, y2)



#
# # Add a rectangle on the main plot (left) to indicate the zoomed region
rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
#
axes[0].plot([0, 100], [2, 70], 'k--', label='FWHM')  # Top line
axes[0].plot([0, -100], [2, 70], 'k--')  # Top line
axes[0].set_ylim([0, 50])
axes[0].set_xlim([-40, 40])
axes[0].add_patch(rect)
# # Draw lines connecting the zoomed region to the zoomed plot
# axes[0].plot([x1, x1], [y1, y2], 'r--')  # Left line
# axes[0].plot([x2, x2], [y1, y2], 'r--')  # Right line
# axes[0].plot([x1, x2], [y1, y1], 'r--')  # Bottom line
# axes[0].plot([x1, x2], [y2, y2], 'r--')  # Top line
# Customize axis labels and titles
# ax1.set_title('Main Plot')
axes[1].set_title('Zoomed Plot')
# axes[0].set_xlim([-100,100])
axes[0].set_xlabel('Detuning [MHz]')
axes[1].set_xlabel('Detuning [MHz]')
axes[1].set_ylabel('Amplitude [MHz]')
axes[0].set_ylabel('Excited State \nPopulation [MHz]')
cbar = fig.colorbar(c1, ax=axes, orientation='vertical', fraction=0.02, pad=0.1)
cbar.set_label('excited state probability', fontsize=12)
# Display the plot
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('simple-spectroscopy-t2-limit.svg', format='svg', transparent=True)
plt.show()
