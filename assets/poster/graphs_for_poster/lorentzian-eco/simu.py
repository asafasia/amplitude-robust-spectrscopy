import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from args import MHz, T2, T1
from graphs_for_poster.open_hdf5 import get_data_from_file


fig = plt.figure(figsize=(3.5, 3))
ax = fig.add_subplot(111)

folder = '/Users/asafsolonnikov/Documents/GitHub/Power-Broadening-2/'
folder_simu = '/Users/asafsolonnikov/Documents/GitHub/Power-Broadening-2/graphs_for_poster/lorentzian-eco/data/'
path = folder + 'spec.npy'

data, args, exp_args = get_data_from_file(folder_simu + 'for_simu.hdf5', dim=1)

print(data)
x, y = data

x = x / 1e3 - 9
y = y + 0.015
# y_smooth = gaussian_filter1d(y, sigma=1)
y_smooth = savgol_filter(y, 50, 7)


detunings, v = np.load(path)
plt.title('Simulation vs Data', fontsize=17)

plt.plot(detunings / 2 / np.pi / 1e3, v, linewidth=2,label='Simulation',color='b')
plt.plot(x, y, label='Raw', color='black', linewidth=1.5)
plt.plot(x, y_smooth,color = '#DC143C', label='Smoothed', linewidth=2)

plt.axvline(x=1 / T2 / 1e3 / 2 / np.pi, color='g', linestyle='-.', label='T2')
plt.axvline(x=-1 / T2 / 1e3 / 2 / np.pi, color='g', linestyle='-.', )
plt.axvline(x=1 / T1 / 1e3 / 2 / np.pi, color='r', linestyle='--', label='T1')
plt.axvline(x=-1 / T1 / 1e3 / 2 / np.pi, color='r', linestyle='--')
plt.xlim([-35, 35])
plt.ylim([0.27, 0.32])
plt.xlabel('Detuning [kHz]', fontsize=14)
plt.ylabel('Excited state\n probability', fontsize=14)
plt.legend(ncol=2)
plt.setp(ax.spines.values(), lw=2)
plt.tight_layout()
plt.savefig('lorentzian-eco-simu.svg', format='svg', transparent=True)
plt.show()
