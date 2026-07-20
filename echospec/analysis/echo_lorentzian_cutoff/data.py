"""Discovery and loading of raw OPX1000 echo-Lorentzian cutoff sweeps."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_DATA_DIR = Path("/Users/asafsolonnikov/Developer/data_opx1000")
SETTING_FIELDS = (
    "lorentzian_length_in_ns",
    "lorentzian_peak_amplitude",
    "frequency_span_in_mhz",
    "frequency_step_in_mhz",
    "num_shots",
    "min_amp_factor",
    "max_amp_factor",
    "amp_factor_step",
)


@dataclass(frozen=True)
class Campaign:
    """Matched raw runs with identical settings and different cutoffs."""

    label: str
    date: str
    cutoffs: tuple[float, ...]
    runs: Mapping[float, Path]
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class RawSweep:
    """One numerical 2D sweep plus its experimental calibration context."""

    run_dir: Path
    parameters: Mapping[str, object]
    qubit: str
    detuning_hz: np.ndarray
    amp_prefactor: np.ndarray
    rabi_mhz: np.ndarray
    state: np.ndarray
    t2_reference_name: str
    t2_us: float
    t2_limit_hz: float


def opx1000_data_dir() -> Path:
    """Return the configured read-only OPX1000 data root."""
    return Path(os.environ.get("OPX1000_DATA_DIR", DEFAULT_DATA_DIR)).expanduser()


def discover_campaigns(
    data_dir: Path | str | None = None,
    *,
    minimum_cutoffs: int = 2,
) -> list[Campaign]:
    """Discover date-local matched raw campaigns, newest first."""
    root = Path(data_dir).expanduser() if data_dir else opx1000_data_dir()
    raw_root = root / "calibrations"
    grouped: dict[tuple[object, ...], list[tuple[float, Path, dict]]] = {}

    for parameter_file in sorted(raw_root.glob("*/echo_lorentzian/*/parameters.json")):
        run_dir = parameter_file.parent
        if (
            not (run_dir / "sweep.npz").exists()
            or not (run_dir / "results.npz").exists()
        ):
            continue
        parameters = json.loads(parameter_file.read_text())
        if not parameters.get("echo", False) or parameters.get("cutoff") is None:
            continue
        date = run_dir.parents[1].name
        key = (date,) + tuple(parameters.get(field) for field in SETTING_FIELDS)
        grouped.setdefault(key, []).append(
            (float(parameters["cutoff"]), run_dir, parameters)
        )

    campaigns: list[Campaign] = []
    for key, entries in grouped.items():
        latest_by_cutoff: dict[float, tuple[Path, dict]] = {}
        for cutoff, run_dir, parameters in entries:
            latest_by_cutoff[cutoff] = (run_dir, parameters)
        if len(latest_by_cutoff) < minimum_cutoffs:
            continue

        parameters = next(iter(latest_by_cutoff.values()))[1]
        cutoffs = tuple(sorted(latest_by_cutoff))
        label = (
            f"{key[0]} | {len(cutoffs)} cutoffs | "
            f"{float(parameters['lorentzian_length_in_ns']) / 1000:g} us | "
            f"peak {float(parameters['lorentzian_peak_amplitude']):g} | "
            f"span {float(parameters['frequency_span_in_mhz']):g} MHz | "
            f"{int(parameters['num_shots'])} shots"
        )
        campaigns.append(
            Campaign(
                label=label,
                date=str(key[0]),
                cutoffs=cutoffs,
                runs={cutoff: latest_by_cutoff[cutoff][0] for cutoff in cutoffs},
                parameters=parameters,
            )
        )

    return sorted(
        campaigns,
        key=lambda campaign: (campaign.date, len(campaign.cutoffs)),
        reverse=True,
    )


def load_raw_sweep(run_dir: Path | str) -> RawSweep:
    """Load one raw 2D state array and convert amplitude to peak Rabi MHz."""
    run_dir = Path(run_dir)
    parameters = json.loads((run_dir / "parameters.json").read_text())
    with (
        np.load(run_dir / "sweep.npz") as sweep,
        np.load(run_dir / "results.npz") as results,
    ):
        qubit = str(np.ravel(sweep["qubit"])[0])
        detuning_hz = np.asarray(sweep["detuning"], dtype=float)
        amp_prefactor = np.asarray(sweep["amp_prefactor"], dtype=float)
        state = np.squeeze(np.asarray(results["state"], dtype=float))

    expected = (len(detuning_hz), len(amp_prefactor))
    if state.shape != expected:
        if state.T.shape == expected:
            state = state.T
        else:
            raise ValueError(
                f"Unexpected state shape {state.shape}; expected {expected}"
            )

    qubits = json.loads((run_dir / "profile" / "qubits.json").read_text())["qubits"]
    pulses = json.loads((run_dir / "profile" / "pulses.json").read_text())["pulses"]
    x180_name = qubits[qubit]["operations"]["x180"]
    x180 = pulses[qubit][x180_name]
    pi_rabi_mhz = 1000.0 / (2.0 * float(x180["length_ns"]))
    drive_scale = float(parameters["lorentzian_peak_amplitude"]) / float(
        x180["amplitude"]
    )
    rabi_mhz = amp_prefactor * drive_scale * pi_rabi_mhz

    transmon = qubits[qubit]["transmon"]
    t2_name = "t2_echo_ns" if transmon.get("t2_echo_ns") else "t2_ramsey_ns"
    t2_us = float(transmon[t2_name]) / 1000.0
    t2_limit_hz = 1e6 / (np.pi * t2_us)

    return RawSweep(
        run_dir=run_dir,
        parameters=parameters,
        qubit=qubit,
        detuning_hz=detuning_hz,
        amp_prefactor=amp_prefactor,
        rabi_mhz=rabi_mhz,
        state=state,
        t2_reference_name=t2_name,
        t2_us=t2_us,
        t2_limit_hz=t2_limit_hz,
    )
