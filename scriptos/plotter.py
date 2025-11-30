from save_helper import Result
from matplotlib import pyplot as plt
import os
from utils import *
import ipywidgets as widgets


def get_files_in_folder(folder_path):
    files = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            files.append(file)
    return files


folder_path = 'data/files/'

file_names = get_files_in_folder(folder_path)

cutoff_pulse_lengths = {}

for file_name in file_names:
    data = Result.load(folder_path + file_name)
    cutoff = data.metadata['cutoff']
    pulse_length = data.metadata['pulse_length']
    if cutoff not in cutoff_pulse_lengths:
        cutoff_pulse_lengths[cutoff] = [pulse_length]
    else:
        if pulse_length not in cutoff_pulse_lengths[cutoff]:
            cutoff_pulse_lengths[cutoff].append(pulse_length)

cutoff_slider = widgets.FloatSlider([1,2,3,4], description='Cutoff:')
pulse_length_slider = widgets.FloatSlider(min=min(cutoff_pulse_lengths[min(cutoff_pulse_lengths.keys())]), max=max(cutoff_pulse_lengths[min(cutoff_pulse_lengths.keys())]), step=0.1, description='Pulse Length:')



# data = read_json(file_path)
    # cutoff = data['metadata']['cutoff']
    # pulse_length = data['metadata']['pulse_length']
    # if cutoff not in cutoff_pulse_lengths:
    #     cutoff_pulse_lengths[cutoff] = [pulse_length]
    # else:
    #     if pulse_length not in cutoff_pulse_lengths[cutoff]:
    #         cutoff_pulse_lengths[cutoff].append(pulse_length)
    #



# for file_name in file_names:
#     a = Result.load(folder_path + file_name)
#
#     amplitudes, FWHMs = np.array(a.data['amplitudes']), np.array(a.data['FWHMs'])
#     plt.plot(amplitudes[1:] / MHz/2/pi, abs(FWHMs[1:]) / 2 / pi / MHz, label=a.metadata['cutoff'])
#
# plt.axhline(y=1 / T2 / 2 / MHz, color='r',label = 'T2 limit')
#
# plt.legend()
# plt.xlabel('Rabi Amplitude [MHz]')
# plt.ylabel('FWHM [MHz]')
# plt.show()
