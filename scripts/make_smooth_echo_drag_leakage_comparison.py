"""Compare leakage from abrupt and smooth AC-Stark-corrected echo pulses.

The smooth protocols replace the instantaneous midpoint sign change by a
normalized ``-tanh(t/tau)`` zero crossing.  Their DRAG quadrature is the
derivative of the complete signed waveform, including that zero crossing.

This is a three-level model study.  The constants below are intentionally easy
to edit; running the script writes figures, numerical arrays, and provenance to
``outputs/smooth_echo_drag_leakage``.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echospec.figures import FigureVariant, apply_figure_style  # noqa: E402
from echospec.simulation.qutrit import (  # noqa: E402
    _smooth_echo_peak,
    simulate_qutrit_map,
)

OUTPUT_DIR = ROOT / "outputs" / "smooth_echo_drag_leakage"

# Matched accumulated-phase study parameters.
DURATION_US = 20.0
CUTOFF = 0.001
ORDER = 0.5
ANHARMONICITY_MHZ = -216.0
T1_US = 51.24
T2_US = 7.31
T_PHI_US = 1.0 / (1.0 / T2_US - 1.0 / (2.0 * T1_US))
STEPS_PER_US = 1000

# The tanh scale is not the total transition width.  At tau=2 ns the sign is
# already 0.995 of its asymptotic value at |t| approximately 3 tau.
ECHO_TRANSITION_NS = 2.0

RABI_MHZ = np.linspace(10.0, 80.0, 15)
DETUNING_MHZ = np.linspace(-0.35, 0.35, 71)
WAVEFORM_RABI_MHZ = 80.0

# The DRAG value is in the simulation convention
# Omega_Q = -beta * d(Omega_I)/dt / alpha.  Kappa must be reoptimized when beta
# or the midpoint shape changes; the two values below come from focused scans
# for this exact model and should not be treated as hardware calibrations.
KAPPA_REFERENCE_MHZ_INV = 0.00225
DRAG_BETA = 1.0
KAPPA_DRAG_RETUNED_MHZ_INV = -0.0025


@dataclass(frozen=True)
class Protocol:
    key: str
    label: str
    echo_transition_us: float
    drag_beta: float
    stark_kappa_mhz_inv: float
    color: str


PROTOCOLS = (
    Protocol(
        key="abrupt_kappa_only",
        label="Abrupt midpoint, κ only",
        echo_transition_us=0.0,
        drag_beta=0.0,
        stark_kappa_mhz_inv=KAPPA_REFERENCE_MHZ_INV,
        color="#6c757d",
    ),
    Protocol(
        key="smooth_no_drag",
        label="Smooth midpoint, no DRAG",
        echo_transition_us=ECHO_TRANSITION_NS / 1000.0,
        drag_beta=0.0,
        stark_kappa_mhz_inv=KAPPA_REFERENCE_MHZ_INV,
        color="#0072b2",
    ),
    Protocol(
        key="smooth_drag_same_kappa",
        label="Smooth + DRAG, same κ",
        echo_transition_us=ECHO_TRANSITION_NS / 1000.0,
        drag_beta=DRAG_BETA,
        stark_kappa_mhz_inv=KAPPA_REFERENCE_MHZ_INV,
        color="#009e73",
    ),
    Protocol(
        key="smooth_drag_retuned",
        label="Smooth + DRAG, retuned κ",
        echo_transition_us=ECHO_TRANSITION_NS / 1000.0,
        drag_beta=DRAG_BETA,
        stark_kappa_mhz_inv=KAPPA_DRAG_RETUNED_MHZ_INV,
        color="#d55e00",
    ),
)


def fitted_central_minima(excited: np.ndarray) -> np.ndarray:
    """Fit each local population minimum with three neighboring points."""
    centers = []
    for row in excited:
        index = int(np.argmin(row))
        center = float(DETUNING_MHZ[index])
        if 0 < index < DETUNING_MHZ.size - 1:
            coefficients = np.polyfit(
                DETUNING_MHZ[index - 1 : index + 2],
                row[index - 1 : index + 2],
                2,
            )
            if coefficients[0] > 0.0:
                candidate = float(
                    -coefficients[1] / (2.0 * coefficients[0])
                )
                if (
                    DETUNING_MHZ[index - 1]
                    <= candidate
                    <= DETUNING_MHZ[index + 1]
                ):
                    center = candidate
        centers.append(center)
    return np.asarray(centers)


def simulate(protocol: Protocol):
    """Run one protocol over the common amplitude-detuning grid."""
    return simulate_qutrit_map(
        duration_us=DURATION_US,
        detuning_mhz=DETUNING_MHZ,
        rabi_mhz=RABI_MHZ,
        t1_us=T1_US,
        t_phi_us=T_PHI_US,
        anharmonicity_mhz=ANHARMONICITY_MHZ,
        num_steps_per_half=int(DURATION_US * STEPS_PER_US / 2.0),
        cutoff=CUTOFF,
        echo=True,
        echo_transition_us=protocol.echo_transition_us,
        order=ORDER,
        drag_beta=protocol.drag_beta,
        stark_kappa_mhz_inv=protocol.stark_kappa_mhz_inv,
        stark_correction_mode="accumulated_phase",
    )


def waveform(protocol: Protocol) -> tuple[np.ndarray, np.ndarray]:
    """Return time and the hardware-style complex envelope ``I + iQ``."""
    half_duration = DURATION_US / 2.0
    sigma_us = half_duration / np.sqrt(CUTOFF ** (-1.0 / ORDER) - 1.0)
    time_us = np.linspace(-half_duration, half_duration, 40_001)
    scaled_time = time_us / sigma_us
    base = (1.0 + scaled_time**2) ** (-ORDER)
    base_derivative = (
        -2.0
        * ORDER
        * time_us
        / sigma_us**2
        * (1.0 + scaled_time**2) ** (-ORDER - 1.0)
    )

    if protocol.echo_transition_us > 0.0:
        normalization = _smooth_echo_peak(
            half_duration_us=half_duration,
            sigma_us=sigma_us,
            order=ORDER,
            transition_us=protocol.echo_transition_us,
        )
        midpoint_sign = -np.tanh(time_us / protocol.echo_transition_us)
        midpoint_derivative = -(
            1.0 - midpoint_sign**2
        ) / protocol.echo_transition_us
        in_phase = midpoint_sign * base / normalization
        in_phase_derivative = (
            midpoint_derivative * base + midpoint_sign * base_derivative
        ) / normalization
    else:
        midpoint_sign = np.where(time_us < 0.0, 1.0, -1.0)
        in_phase = midpoint_sign * base
        # Match the simulator: the distribution-valued jump derivative is not
        # included for the abrupt protocol.
        in_phase_derivative = midpoint_sign * base_derivative

    alpha = 2.0 * np.pi * ANHARMONICITY_MHZ
    quadrature = -protocol.drag_beta * in_phase_derivative / alpha
    correction_mhz = (
        protocol.stark_kappa_mhz_inv
        * (WAVEFORM_RABI_MHZ * in_phase) ** 2
    )
    phase_steps = (
        np.pi
        * (correction_mhz[:-1] + correction_mhz[1:])
        * np.diff(time_us)
    )
    phase = np.concatenate(([0.0], np.cumsum(phase_steps)))
    # The qutrit solver evolves drive=Omega_I-i*Omega_Q in the rotating frame;
    # its hardware-style envelope is the complex conjugate shown here.
    played = (
        WAVEFORM_RABI_MHZ * (in_phase + 1j * quadrature) * np.exp(-1j * phase)
    )
    return time_us, played


def save_figure(figure: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "png"):
        figure.savefig(
            OUTPUT_DIR / f"{stem}.{suffix}",
            dpi=260 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_waveforms() -> None:
    figure, axes = plt.subplots(
        1,
        len(PROTOCOLS),
        figsize=(11.8, 2.65),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, protocol in zip(axes, PROTOCOLS, strict=True):
        time_us, played = waveform(protocol)
        mask = np.abs(time_us) <= 0.06
        axis.plot(1e3 * time_us[mask], played.real[mask], label="I")
        axis.plot(1e3 * time_us[mask], played.imag[mask], "--", label="Q")
        axis.plot(
            1e3 * time_us[mask],
            np.abs(played[mask]),
            color="0.4",
            lw=0.75,
            ls=":",
            label=r"$|I+iQ|$",
        )
        axis.axvline(0.0, color="0.55", lw=0.6, ls=":")
        axis.set_title(protocol.label)
        axis.set_xlabel("Time about midpoint (ns)")
    axes[0].set_ylabel(r"Envelope $\Omega/2\pi$ (MHz)")
    axes[-1].legend(fontsize=6.5, frameon=False)
    save_figure(figure, "00_waveforms")


def plot_population_maps(results: dict[str, object]) -> None:
    pe_max = max(float(result.excited.max()) for result in results.values())
    pf_max = max(
        float(result.second_excited.max()) for result in results.values()
    )
    figure, axes = plt.subplots(
        2,
        len(PROTOCOLS),
        figsize=(12.0, 5.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    pe_images = []
    pf_images = []
    for column, protocol in enumerate(PROTOCOLS):
        result = results[protocol.key]
        pe_images.append(
            axes[0, column].pcolormesh(
                DETUNING_MHZ,
                RABI_MHZ,
                result.excited,
                shading="auto",
                vmin=0.0,
                vmax=pe_max,
                rasterized=True,
            )
        )
        pf_images.append(
            axes[1, column].pcolormesh(
                DETUNING_MHZ,
                RABI_MHZ,
                result.second_excited,
                shading="auto",
                vmin=0.0,
                vmax=pf_max,
                cmap="magma",
                rasterized=True,
            )
        )
        axes[0, column].set_title(protocol.label)
        axes[1, column].set_xlabel(r"$\Delta/2\pi$ (MHz)")
        for row in range(2):
            axes[row, column].axvline(0.0, color="white", lw=0.55, ls=":")
    axes[0, 0].set_ylabel(r"$P_e$" + "\n" + r"$\Omega_0/2\pi$ (MHz)")
    axes[1, 0].set_ylabel(r"$P_f$" + "\n" + r"$\Omega_0/2\pi$ (MHz)")
    figure.colorbar(pe_images[-1], ax=axes[0], pad=0.012, label=r"$P_e$")
    figure.colorbar(pf_images[-1], ax=axes[1], pad=0.012, label=r"$P_f$")
    save_figure(figure, "01_population_maps")


def plot_metrics(metrics: dict[str, dict[str, np.ndarray]]) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(9.0, 2.7),
        constrained_layout=True,
    )
    for protocol in PROTOCOLS:
        values = metrics[protocol.key]
        axes[0].semilogy(
            RABI_MHZ,
            values["max_pf_by_rabi"],
            "o-",
            ms=3.0,
            color=protocol.color,
            label=protocol.label,
        )
        axes[1].plot(
            RABI_MHZ,
            1e3 * values["centers_mhz"],
            "o-",
            ms=3.0,
            color=protocol.color,
        )
        axes[2].plot(
            RABI_MHZ,
            values["pe_contrast"],
            "o-",
            ms=3.0,
            color=protocol.color,
        )
    axes[0].set(
        xlabel=r"$\Omega_0/2\pi$ (MHz)",
        ylabel=r"$\max_{\Delta} P_f$",
        title="Worst final leakage",
    )
    axes[1].axhline(0.0, color="0.5", lw=0.6, ls=":")
    axes[1].set(
        xlabel=r"$\Omega_0/2\pi$ (MHz)",
        ylabel="Fitted center (kHz)",
        title="Center stability",
    )
    axes[2].set(
        xlabel=r"$\Omega_0/2\pi$ (MHz)",
        ylabel=r"$\max P_e-\min P_e$",
        title="Central-window contrast",
    )
    axes[0].legend(fontsize=6.3, frameon=False)
    save_figure(figure, "02_metrics")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_figure_style(FigureVariant.PAPER)
    results = {}
    metrics = {}
    summaries = {}
    for protocol in PROTOCOLS:
        started = time.perf_counter()
        print(f"Simulating {protocol.label}", flush=True)
        result = simulate(protocol)
        elapsed_s = time.perf_counter() - started
        centers = fitted_central_minima(result.excited)
        max_pf_by_rabi = result.second_excited.max(axis=1)
        center_indices = np.argmin(result.excited, axis=1)
        pf_at_center = result.second_excited[
            np.arange(RABI_MHZ.size), center_indices
        ]
        pe_contrast = result.excited.max(axis=1) - result.excited.min(axis=1)
        results[protocol.key] = result
        metrics[protocol.key] = {
            "centers_mhz": centers,
            "max_pf_by_rabi": max_pf_by_rabi,
            "pf_at_center": pf_at_center,
            "pe_contrast": pe_contrast,
        }
        summaries[protocol.key] = {
            **asdict(protocol),
            "maximum_final_pf": float(result.second_excited.max()),
            "maximum_pf_at_fitted_center": float(pf_at_center.max()),
            "center_rms_khz": float(1e3 * np.sqrt(np.mean(centers**2))),
            "maximum_abs_center_khz": float(1e3 * np.max(np.abs(centers))),
            "minimum_pe_contrast": float(pe_contrast.min()),
            "elapsed_s": elapsed_s,
        }
        print(
            f"  max Pf={summaries[protocol.key]['maximum_final_pf']:.6g}, "
            f"center RMS={summaries[protocol.key]['center_rms_khz']:.3f} kHz, "
            f"elapsed={elapsed_s:.1f} s",
            flush=True,
        )

    arrays = {
        "rabi_mhz": RABI_MHZ,
        "detuning_mhz": DETUNING_MHZ,
    }
    for protocol in PROTOCOLS:
        result = results[protocol.key]
        arrays.update(
            {
                f"{protocol.key}_pg": result.ground,
                f"{protocol.key}_pe": result.excited,
                f"{protocol.key}_pf": result.second_excited,
                f"{protocol.key}_centers_mhz": metrics[protocol.key][
                    "centers_mhz"
                ],
                f"{protocol.key}_max_pf_by_rabi": metrics[protocol.key][
                    "max_pf_by_rabi"
                ],
                f"{protocol.key}_pf_at_center": metrics[protocol.key][
                    "pf_at_center"
                ],
                f"{protocol.key}_pe_contrast": metrics[protocol.key][
                    "pe_contrast"
                ],
            }
        )
    np.savez_compressed(OUTPUT_DIR / "comparison_data.npz", **arrays)

    provenance = {
        "description": (
            "Three-level leakage comparison for abrupt and smooth "
            "AC-Stark-corrected echo-root-Lorentzian pulses"
        ),
        "model": "three-level transmon Lindblad/RK4",
        "duration_us": DURATION_US,
        "cutoff": CUTOFF,
        "order": ORDER,
        "anharmonicity_mhz": ANHARMONICITY_MHZ,
        "t1_us": T1_US,
        "t2_us": T2_US,
        "t_phi_us": T_PHI_US,
        "steps_per_us": STEPS_PER_US,
        "echo_transition_ns": ECHO_TRANSITION_NS,
        "stark_correction_mode": "accumulated_phase",
        "rabi_mhz": RABI_MHZ.tolist(),
        "detuning_mhz": DETUNING_MHZ.tolist(),
        "protocols": summaries,
        "limitations": [
            "Final P_f is reported; maximum transient leakage is not retained.",
            "The Duffing model has three levels and uses the rotating-wave approximation.",
            "Kappa and beta are simulation values, not hardware calibrations.",
            "No measured microwave transfer function or waveform predistortion is applied.",
        ],
        "literature": [
            {
                "title": "Simple Pulses for Elimination of Leakage in Weakly Nonlinear Qubits",
                "doi": "10.1103/PhysRevLett.103.110501",
                "relevance": "Introduces the derivative quadrature used by DRAG.",
            },
            {
                "title": "Analytic control methods for high-fidelity unitary operations in a weakly nonlinear oscillator",
                "doi": "10.1103/PhysRevA.83.012308",
                "relevance": "Develops the adiabatic expansion and generalized DRAG corrections.",
            },
            {
                "title": "Reducing Leakage of Single-Qubit Gates for Superconducting Quantum Processors Using Analytical Control Pulse Envelopes",
                "doi": "10.1103/PRXQuantum.5.030353",
                "relevance": "Motivates smooth spectral shaping together with derivative corrections.",
            },
        ],
        "reproduction_command": (
            "PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python "
            "scripts/make_smooth_echo_drag_leakage_comparison.py"
        ),
    }
    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    plot_waveforms()
    plot_population_maps(results)
    plot_metrics(metrics)
    print(f"Saved comparison to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
