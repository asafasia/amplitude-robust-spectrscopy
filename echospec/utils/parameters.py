import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING
from echospec.utils.units import Units as u

if TYPE_CHECKING:
    from echospec.simulation.pulses import PulseType


@dataclass
class Parameters:
    """Simulation settings.

    ``detuning`` is the angular drive detuning ``omega_d - omega_01``.  The
    rotating-frame number term is consequently ``-detuning * n``.
    """

    qubit_frequency: float = 5 * u.GHz * 2 * u.pi2
    detuning: float = 0 * 2 * u.pi2 * u.MHz
    anharmonicity: float = 200 * u.MHz * 2 * u.pi2
    T1: float = 50 * u.us
    T_dephasing: float = 7 * u.us
    detuning_span: float = 0.5 * 2 * u.pi2 * u.MHz
    rabi_frequency: float = 50 * 2 * u.pi2 * u.MHz
    pulse_length: float = 40 * u.us
    cutoff: float = 1e-4
    order: float = 1 / 2
    pulse_type: "PulseType | None" = None
    eco_pulse: bool = False
    sigma: float = 1
    flat_top: float = 0 * u.us
    floor: bool = False
    rabi_points: int = 51
    detuning_points: int = 101
    length_points: int = 21
    zeroed_pulse: bool = False
    full_state: bool = False

    def __post_init__(self):
        """Initialize pulse_type with default if not provided."""
        if self.pulse_type is None:
            from echospec.simulation.pulses import PulseType
            object.__setattr__(self, 'pulse_type', PulseType.LORENTZIAN)

    @property
    def T2(self) -> float:
        """Total coherence time T2 based on T1 and T_dephasing."""
        if self.T1 == 0 or self.T_dephasing == 0:
            return 0.0
        return 1 / (1 / self.T_dephasing + 1 / (2 * self.T1))

    @property
    def gamma_relaxation(self) -> float:
        """Rate of energy relaxation (1/T1)."""
        return 1 / self.T1

    @property
    def gamma_dephasing(self) -> float:
        """Rate of pure dephasing (1/T_phi)."""
        return 1 / self.T_dephasing

    @property
    def gamma_extinction(self) -> float:
        """Total transverse relaxation rate (1/T2)."""
        return 1 / self.T2

    @property
    def T2_limit(self) -> float:
        """Return the coherence-limited cyclic-frequency FWHM, 1/(pi*T2)."""
        return 1 / self.T2/np.pi

    @property
    def broadening_condition(self) -> float:
        """Calculate the broadening condition."""
        return (
            self.rabi_frequency /
            (np.pi * np.sqrt(self.T2 / self.T1)) * self.cutoff
        )


if __name__ == "__main__":
    # Example usage of Parameters class
    params = Parameters()
    print(f"Qubit Frequency: {params.qubit_frequency}")
    print(f"T2: {params.T2}")
    print(f"Gamma Relaxation: {params.gamma_relaxation}")
    print(f"Gamma Dephasing: {params.gamma_dephasing}")
    print(f"Gamma Extinction: {params.gamma_extinction}")
    print(f"T2 Limit: {params.T2_limit}")
    print(f"Broadening Condition: {params.broadening_condition}")
