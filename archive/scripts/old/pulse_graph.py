import numpy as np
from scipy.interpolate import interp1d
# from args import *
import matplotlib.pyplot as plt
from utils import *

plt.subplot(2, 1, 1)

dt = 5 * ns

T = 0 * us
pulse_length = qubit_args['pulse_length']


def square_pulse(t, args):
    return 1


tlist = np.arange(-qubit_args['pulse_length'] / 2, qubit_args['pulse_length'] / 2, dt)
extended_tlist = np.arange(-T / 2 - qubit_args['pulse_length'] / 2, qubit_args['pulse_length'] / 2 + T / 2, dt)

if qubit_args['pulse_type'] == 'square':
    if not qubit_args['eco_pulse']:
        pulse = square_pulse
    else:
        pulse = square_half
elif qubit_args['pulse_type'] == 'gaussian':
    if not qubit_args['eco_pulse']:
        pulse = gaussian
    else:
        pulse = gaussian_half
elif qubit_args['pulse_type'] == 'lorentzian':
    if not qubit_args['eco_pulse']:
        pulse = lorentzian
    else:
        pulse = lorentzian_half
else:
    raise ValueError('Invalid pulse type')


def new_pulse(t, T, pulse):
    return np.heaviside(t + pulse_length / 2, 0) * np.heaviside(-t + pulse_length / 2, 0) * pulse(t, qubit_args)


pulse = new_pulse(extended_tlist, T, pulse)

# plt.figure(figsize=(10, 6))
plt.plot(extended_tlist / us, pulse, label='lorentzian half pulse')

plt.axvline(x=qubit_args['pulse_length'] / 2 / us, color='r')
plt.axhline(y=qubit_args['cutoff'], linestyle='--', color='b', label=f'cutoff =  {qubit_args['cutoff']}')
# plt.axhline(y=0, color='k')
plt.title('Pulse Shape')
plt.xlabel('time [us]')
plt.ylabel('normalized pulse amplitude')
plt.axhline(y=0, color='k')
# plt.ylim([-1.1, 1.1])
# plt.xlim([-20, 20])
plt.tight_layout()
plt.legend(loc='upper right')

subplot = plt.subplot(2, 1, 2)

N = len(extended_tlist)
dt = extended_tlist[1] - extended_tlist[0]

dft = np.fft.fft(pulse)
t_pulse = np.fft.ifft(dft)

dft = np.concatenate((dft[N // 2:], dft[:N // 2]))
dft_freq = np.fft.fftfreq(N, dt)
dft_freq = np.concatenate((dft_freq[N // 2:], dft_freq[:N // 2]))

inter = interp1d(dft_freq, np.abs(dft) / len(dft), kind='cubic')

plt.plot(dft_freq / MHz, np.abs(dft) / len(dft), '.', label='Fourier Transform', color='C4')

# FWHM = FWHM(dft_freq, np.abs(dft) / len(dft), False)
#
# print(f'FWHM = {FWHM / MHz} MHz')

new_dt_freq = np.linspace(dft_freq[0], dft_freq[-1], 1000000)
plt.plot(new_dt_freq / MHz, inter(new_dt_freq), label='Interpolated', color='C3')

# plt.axvline(x=FWHM / 2 / MHz, color='r', label='FWHM')
plt.xlim([-3, 3])
plt.title('Fourier Transform')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()

# W = wigner(Qobj(pulse), tlist, tlist)


# f_pulse = np.ones(new_dt_freq)

# t_pulse = np.fft.ifft(dft)



plt.figure()
plt.plot(tlist/us, t_pulse.real, label='Reconstructed pulse')
# plt.xlim([-0.1, 0.1])
plt.show()


plt.figure()
W = wigner(Qobj(pulse), tlist, new_dt_freq,parfor =True)
plt.contourf(tlist, new_dt_freq, W, 100)
plt.show()