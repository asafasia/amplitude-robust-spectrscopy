from scipy.optimize import curve_fit

from run_simulation import *
from utils import *

t1 = T1
t2 = T2

if qubit_args['pulse_type'] == 'lorentzian':
    half_pulse = True
else:
    half_pulse = False

plt.title(
    f'pulse type is {qubit_args['pulse_type']}, eco is {qubit_args['eco_pulse']} \n pulse length = {qubit_args["pulse_length"] / us:0.3f} us \n cutoff = {qubit_args["cutoff"]} , amplitude = {qubit_args['rabi_frequency'] / 2 / pi / MHz:.3f} MHz \n $T_1$ = {T1 / us:.2f} us, $T_d$ = {T_dephasing / us:.2f} us -> $T_2$ = {T2 / us:.1f} us')

vec = parallel_map(single_run_change_dict, detunings,task_args=(qubit_args,), progress_bar=True)

x = detunings / MHz / 2 / np.pi
y = vec

plt.plot(x, y, label=f'data', color='C2')

T1_limit = 1 / T1 / MHz / pi
T2_limit = 1 / T2 / MHz / pi

mask = (x > -T1_limit) & (x < T1_limit)
x_restricted = x[mask]
print(x_restricted)
y_restricted = np.array([y[i] for i in range(len(y)) if mask[i]])
# params, _ = curve_fit(harmonic_approx, x_restricted, y_restricted)
# a, c = params

# plt.scatter(x, y, label="Original Data", color="blue")
# plt.scatter(x_restricted, y_restricted, label="Restricted Data", color="orange")
# plt.plot(x, harmonic_approx(x, *params), label="Harmonic Approximation", color="red")

plt.xlabel('Detuning [MHz]')
plt.xlim([x[0], x[-1]])
plt.axvline(x=T2_limit / 2, color='r', linestyle='-.', label='T2')
plt.axvline(x=-T2_limit / 2, color='r', linestyle='-.', )
if detunings[-1] < 2 * MHz:
    plt.axvline(x=T1_limit / 2, color='C4', linestyle='--', label='T1')
    plt.axvline(x=-T1_limit / 2, color='C4', linestyle='--')

plt.legend()
plt.tight_layout()
# plt.ylim([0, 0.5])


fwhm, _ = FWHM(detunings, vec)

plt.axvline(fwhm / 2 / pi / MHz / 2)
plt.show()

np.save('spec1', np.array([detunings, vec]))

