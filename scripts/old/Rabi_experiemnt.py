from matplotlib import pyplot as plt

from run_simulation import single_run_change_dict
from args import *

detunings = np.linspace(0, 1 / T1, 2)

vec = []

amplitudes = np.linspace(0.01, 100,100) * 2 * pi * MHz
for amp in amplitudes:
    qubit_args['rabi_frequency'] = amp  # Convert to angular frequency
    vec.append(single_run_change_dict(0, qubit_args))

plt.plot(amplitudes / MHz / pi / 2, vec, label=f'detuning = {0 / MHz} MHz')
plt.xlabel("Amplitude [MHz]")
plt.ylabel("Excited state population")
# plt.axhline(y=0.5, color='k')
plt.legend()
plt.show()
