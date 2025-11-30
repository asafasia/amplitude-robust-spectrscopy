from .utils import *


class Hamiltonian:

    def __init__(
        self,
        rotated_frame=False,
        anharmonicity=0,
        detuning=0,
        rabi_frequency=None,
        pulse_type="gaussian",
        eco_pulse=False,
    ):

        self.anharmonicity = anharmonicity
        self.rotated_frame = rotated_frame
        self.pulse_type = pulse_type
        self.detuning = detuning
        self.w0 = qubit_args["qubit_frequency"]
        self.eco_pulse = eco_pulse
        self.omega = rabi_frequency

    def get_hamiltonian(
        self,
    ):
        qubit_Hamiltonian = (
            self.detuning * a.dag() * a
            + self.anharmonicity / 2 * a.dag() * a.dag() * a * a
        )
        pulse_Hamiltonian = self.omega * (a.dag() + a) / 2


        if self.pulse_type == "square":
            if not self.eco_pulse:
                pulse = None
            else:
                pulse = square_half
        elif self.pulse_type == "gaussian":
            if self.eco_pulse:
                pulse = gaussian_half
            else:
                pulse = gaussian
        elif self.pulse_type == "lorentzian":
            if self.eco_pulse:
                pulse = lorentzian_half
            else:
                pulse = lorentzian
        else:
            raise ValueError("Invalid pulse type")

        if self.pulse_type == "square":
            if not self.eco_pulse:
                H = qubit_Hamiltonian + pulse_Hamiltonian
            else:

                H = [qubit_Hamiltonian, [pulse_Hamiltonian, pulse]]

            return H
        else:

            H = [qubit_Hamiltonian, [pulse_Hamiltonian, pulse]]
            return H
