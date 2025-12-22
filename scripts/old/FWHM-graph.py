from save_helper import Result
from matplotlib import pyplot as plt
import os
from simi.utils import *


def get_files_in_folder(folder_path):
    files = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            files.append(file)
    return files


folder_path = 'data/files/'

file_names = get_files_in_folder(folder_path)

for file_name in file_names:
    a = Result.load(folder_path + file_name)

    amplitudes, FWHMs = np.array(a.data['amplitudes']), np.array(a.data['FWHMs'])
    plt.plot(amplitudes[1:] / MHz/2/pi, abs(FWHMs[1:]) / 2 / pi / MHz, label=a.metadata['cutoff'])

plt.axhline(y=1 / T2 / 2 / MHz, color='r',label = 'T2 limit')

plt.legend()
plt.xlabel('Rabi Amplitude [MHz]')
plt.ylabel('FWHM [MHz]')
plt.show()
