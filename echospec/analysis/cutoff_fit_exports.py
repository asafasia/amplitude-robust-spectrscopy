"""Verify and load the common fitted metrics used by Figure 4 and Figure S8."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .robust_dip_fit import FIT_SETTINGS

ROOT = Path(__file__).resolve().parents[2]


def load_cutoff_fits(source, lengths, cutoffs, amplitudes, original_accepted):
    directory = Path(source) / "robust_fits"
    path = directory / "metrics.npz"
    provenance_path = directory / "provenance.json"
    metadata = json.loads(provenance_path.read_text())
    if not metadata["complete"] or metadata["settings"] != FIT_SETTINGS:
        raise ValueError("Fitted cutoff metrics are incomplete or use other settings")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata["arrays_sha256"]:
        raise ValueError("Fitted cutoff arrays changed")
    for source_name, expected in metadata["source_sha256"].items():
        if hashlib.sha256((ROOT / source_name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Fitted cutoff source changed: {source_name}")
    with np.load(path, allow_pickle=False) as saved:
        for key, expected in (("durations_us", lengths), ("cutoffs", cutoffs),
                              ("rabi_mhz", amplitudes)):
            np.testing.assert_array_equal(saved[key], expected)
        encoded = json.loads(str(saved["configuration"]))
        if encoded != {key: metadata[key] for key in ("source_sha256", "settings")}:
            raise ValueError("Fitted arrays and provenance describe different fits")
        keys = ("accepted", "fwhm_hz", "contrast", "center_hz", "relative_rms",
                "relative_width_error", "radius_mhz", "status")
        arrays = {key: saved[key].copy() for key in keys}
    shape = (len(lengths), len(cutoffs), len(amplitudes))
    if any(values.shape != shape for values in arrays.values()):
        raise ValueError("Unexpected fitted metric dimensions")
    accepted = arrays["accepted"]
    if np.any(accepted & ~original_accepted):
        raise ValueError("A fit reaccepted an originally rejected spectrum")
    if (not np.isfinite(arrays["fwhm_hz"][accepted]).all()
            or np.any(arrays["fwhm_hz"][accepted] <= 0)
            or not np.isfinite(arrays["contrast"][accepted]).all()
            or np.any((arrays["contrast"][accepted] <= 0)
                      | (arrays["contrast"][accepted] > 1))):
        raise ValueError("Invalid accepted fitted width or contrast")
    hashes = {
        str(path.relative_to(ROOT)): digest,
        str(provenance_path.relative_to(ROOT)):
            hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
    }
    return arrays, metadata, hashes
