from run_simulation import *
from args import *


def objective_function_1d(amp, length):
    change_dict = {
        'rabi_frequency': amp,
        'pulse_length': length,
    }

    vec1 = double_run_change_dict(0, change_dict)
    vec2 = double_run_change_dict(1 / T2 / 2 / pi, change_dict)

    return abs(vec1 - vec2)


amps = np.linspace(0, 100, 100) * 2 * pi * MHz
cutoffs = np.linspace(0.001, 0.1, 20)
lengths = np.linspace(1, 50, 40) * us
z = []

for i, length in enumerate(lengths):
    print(f'{i}/{len(lengths)}')
    z.append(parallel_map(objective_function_1d, amps, task_args=(length,)))

z = np.array(z).T

plt.plot(amps / MHz / 2 / pi, z)
plt.show()

x, y = np.meshgrid(amps, lengths)
plt.xlabel('Rabi frequency [MHz]')
plt.ylabel('pulse length [us]')

plt.pcolormesh(x / MHz / 2 / np.pi, y / us, z.T)

print('max: ', np.max(z))

# bounds = [(0, amps[-1])]
# result = dual_annealing(objective_function_1d, bounds)
#
# plt.axvline(x=result.x / MHz / 2 / pi, color='r', label='Optimal amplitude')
cbar = plt.colorbar()
#
plt.show()
