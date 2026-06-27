import numpy as np
from utils import *
from matplotlib import rcParams

# Enable LaTeX for text rendering
rcParams.update({
    'text.usetex': True,
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (15, 5),
    'axes.linewidth': 3,  # Width of the axes' lines
    'grid.linewidth': 0.8  # Width of the grid lines
})


def sqaure_pulse(t, args):
    return 1


def eco_pulse(t, args):
    return (1 - 2 * np.heaviside(t, 0))


dt = 1 * ns
T = 1 * us
tlist = np.arange(-qubit_args['pulse_length'] / 2, qubit_args['pulse_length'] / 2, dt)
extended_tlist = np.arange(-T / 2 - qubit_args['pulse_length'] / 2, qubit_args['pulse_length'] / 2 + T / 2, dt)
pulse_length = qubit_args['pulse_length']


def new_pulse(t, T, pulse):
    return np.heaviside(t + pulse_length / 2, 0) * np.heaviside(-t + pulse_length / 2, 0) * pulse(t, qubit_args)


eco = False
plt.figure(figsize=(5, 3))

if not eco:
    plt.title('Pulse Examples',fontsize=23)
    y1 = new_pulse(extended_tlist, T, lorentzian)
    y2 = gaussian(tlist, qubit_args)
    y3 = new_pulse(extended_tlist, qubit_args, sqaure_pulse)

    plt.plot(extended_tlist / us, y3, label='square', color='#1f77b4', linewidth=5)
    plt.plot(tlist / us, y2, label='gaussian', linestyle='-.', color='purple', linewidth=5)
    plt.plot(extended_tlist / us, y1, linestyle='--', color='#2ca02c', linewidth=5, label=r'Lorentzian Pulse')
    plt.ylim([0, 1.5])

    plt.plot([-1, -1], [0, 0.1], color='r', linestyle='-.', linewidth=7)
    plt.plot([1, 1], [0, 0.1], color='r', linestyle='-.', linewidth=7)
    plt.legend(ncol=3,loc='upper right',fontsize=10,frameon=False)
    plt.savefig('pulse.svg', format='svg', transparent=True)
    plt.xlabel('Time [a.u]', fontsize=28)
    plt.ylabel('Amplitude', fontsize=28)
    plt.tight_layout()
    plt.savefig('pulse.svg', format='svg', transparent=True)


else:
    plt.title('Echo Pulse',fontsize=23, )

    y4 = new_pulse(extended_tlist, T, lorentzian)*eco_pulse(extended_tlist, qubit_args)
    plt.ylim([-1.5, 1.5])
    plt.axhline(y=0,color='gray', linestyle='-.', linewidth=2)
    plt.plot(extended_tlist / us, y4, label='Echo Pulse\n(Lorentzian)', linestyle='-', color='C4', linewidth=5)

    plt.legend(fontsize=12,)
    plt.xlabel('Time [a.u]', fontsize=28)
    plt.ylabel('Amplitude', fontsize=28)
    plt.tight_layout()
    plt.savefig('pulse_echo.svg', format='svg', transparent=True)




plt.show()
