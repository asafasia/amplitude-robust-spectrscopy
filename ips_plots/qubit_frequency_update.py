from pathlib import Path
import json

import numpy as np
from scipy.optimize import curve_fit

from bundle_utils import extract_qubit_variables


def lorentzian(x, offset, amplitude, center, gamma):
    return offset + amplitude / (1 + ((x - center) / gamma) ** 2)


def extract_f10_from_latest_echo(bundle_path):
    data = extract_qubit_variables(bundle_path)
    best_index = np.unravel_index(np.nanargmax(data.result), data.result.shape)
    best_amp_index = best_index[1]
    frequency_ghz = data.drive_frequency_ghz
    state_slice = data.result[:, best_amp_index]

    offset0 = float(np.nanmin(state_slice))
    amplitude0 = float(np.nanmax(state_slice) - offset0)
    center0 = float(frequency_ghz[int(np.nanargmax(state_slice))])
    gamma0 = max(float((frequency_ghz.max() - frequency_ghz.min()) / 50), 1e-5)

    fit_params, _ = curve_fit(
        lorentzian,
        frequency_ghz,
        state_slice,
        p0=[offset0, amplitude0, center0, gamma0],
        bounds=(
            [0, 0, float(frequency_ghz.min()), 1e-6],
            [1.5, 2.0, float(frequency_ghz.max()), float(frequency_ghz.max() - frequency_ghz.min())],
        ),
        maxfev=20000,
    )

    offset, amplitude, center_ghz, gamma_ghz = fit_params
    return {
        "source_bundle": str(Path(bundle_path)),
        "qubit": data.qubit_name,
        "old_f10_hz": data.qubit_f01_hz,
        "old_f10_ghz": data.qubit_f01_hz / 1e9,
        "updated_f10_hz": float(center_ghz * 1e9),
        "updated_f10_ghz": float(center_ghz),
        "delta_hz": float(center_ghz * 1e9 - data.qubit_f01_hz),
        "delta_mhz": float(center_ghz * 1000 - data.qubit_f01_hz / 1e6),
        "fit_width_mhz": float(gamma_ghz * 1000),
        "fit_offset": float(offset),
        "fit_amplitude": float(amplitude),
        "slice_amp_prefactor": float(data.amp_prefactor[best_amp_index]),
        "slice_rabi_amp_mhz": float(data.rabi_amp_mhz[best_amp_index]),
    }


if __name__ == "__main__":
    project_dir = Path.cwd()
    bundle_path = project_dir / "data" / "echo_lorentzian_18-02-11-107098_data_bundle.npz"
    update = extract_f10_from_latest_echo(bundle_path)

    output_path = project_dir / "data" / "qubit_frequency_updates.json"
    output_path.write_text(json.dumps({"q1": update}, indent=2), encoding="utf-8")

    print(json.dumps(update, indent=2))
    print(f"Wrote {output_path}")
