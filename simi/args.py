import numpy as np
from numpy import pi
from qutip import *

N_dim = 2
GHz = 1e9
MHz = 1e6
KHz = 1e3
us = 1e-6
ns = 1e-9


a = destroy(N_dim)


qubit_args = {
    "qubit_frequency": 5 * GHz * 2 * pi,
    "anharmonicity": 200 * MHz * 2 * pi,
    "T1": 30 * us,
    "T_dephasing": 20 * us,
    "detuning_span": 0.5 * 2 * pi * MHz,
    "rabi_frequency": 50 * 2 * pi * MHz,
    "pulse_length": 40 * us,
    "cutoff": 1e-4,
    "n": 1 / 2,
    "pulse_type": "lorentzian",
    "eco_pulse": True,
    "sigma": 1,
    "flat_top": 0 * us,
    "floor": False,
    "rabi_points": 51,
    "detuning_points": 101,
    "length_points": 21,
    "zeroed_pulse": False,
    "full_state": False,
}

T1 = qubit_args["T1"]
T_dephasing = qubit_args["T_dephasing"]
T2 = (1 / (T_dephasing) + 1 / (2 * T1)) ** -1

print(f"T1: {T1 / us:.2f} us")
print(f"T2: {T2 / us:.2f} us")
T2_limit = 1 / (np.pi * T2)

print(f"T2 limit: {T2_limit / KHz:.2f} kHz")


broadening_condition = (
    qubit_args["rabi_frequency"] /
    (np.pi * np.sqrt(T2 / T1)) * qubit_args["cutoff"]
)
print("broadening condition:")
# print(broadening_condition, "Hz")
print(broadening_condition / 1e3, "kHz")
# print(broadening_condition / 1e6, "MHz")
