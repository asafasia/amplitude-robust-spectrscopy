from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np


def load_bundle(path):
    path = Path(path)
    bundle = np.load(path, allow_pickle=True)
    return SimpleNamespace(
        path=path,
        raw=bundle,
        metadata=_load_json(bundle, "metadata.json"),
        parameters=_load_json(bundle, "parameters.json"),
        profile_qubits=_load_json(bundle, "profile__qubits.json").get("qubits", {}),
        profile_pulses=_load_json(bundle, "profile__pulses.json").get("pulses", {}),
    )


def extract_qubit_variables(path, qubit_index=0):
    bundle = load_bundle(path)
    raw = bundle.raw

    qubits = raw["data__qubit"]
    qubit_name = str(qubits[qubit_index])
    qubit_profile = bundle.profile_qubits[qubit_name]
    qubit_f01_hz = qubit_profile["frequencies_hz"]["qubit_f01"]

    detuning_hz = raw["data__detuning"]
    drive_frequency_ghz = detuning_to_drive_frequency_ghz(detuning_hz, qubit_f01_hz)
    state = raw["data__state"]
    result = state[qubit_index]

    amp_prefactor = raw["data__amp_prefactor"] if "data__amp_prefactor" in raw.files else None
    pi_pulse_name, pi_pulse = get_pi_pulse(bundle, qubit_name)
    rabi_amp_mhz = None
    if amp_prefactor is not None and pi_pulse is not None:
        rabi_amp_mhz = amp_prefactor_to_rabi_amp_mhz(
            amp_prefactor,
            pi_pulse["amplitude"],
            pi_pulse["length_ns"],
        )

    return SimpleNamespace(
        bundle=bundle,
        metadata=bundle.metadata,
        parameters=bundle.parameters,
        profile_qubits=bundle.profile_qubits,
        profile_pulses=bundle.profile_pulses,
        state=state,
        result=result,
        qubits=qubits,
        qubit_index=qubit_index,
        qubit_name=qubit_name,
        qubit_profile=qubit_profile,
        qubit_f01_hz=qubit_f01_hz,
        detuning_hz=detuning_hz,
        drive_frequency_ghz=drive_frequency_ghz,
        amp_prefactor=amp_prefactor,
        pi_pulse_name=pi_pulse_name,
        pi_pulse=pi_pulse,
        rabi_amp_mhz=rabi_amp_mhz,
    )


def detuning_to_drive_frequency_ghz(detuning_hz, qubit_f01_hz):
    """Convert conventional drive detuning, ``f_d - f_01``, to GHz."""
    return (qubit_f01_hz + detuning_hz) / 1e9


def get_pi_pulse(bundle, qubit_name):
    qubit_profile = bundle.profile_qubits.get(qubit_name, {})
    pulse_name = qubit_profile.get("operations", {}).get("x180")
    pulse = bundle.profile_pulses.get(qubit_name, {}).get(pulse_name)
    return pulse_name, pulse


def amp_prefactor_to_rabi_amp_mhz(amp_prefactor, pi_amp, pi_length_ns):
    pi_rabi_mhz = 1000 / (2 * pi_length_ns)
    return np.asarray(amp_prefactor) / pi_amp * pi_rabi_mhz


def _load_json(bundle, key):
    if key not in bundle.files:
        return {}
    return json.loads(bundle[key].item())
