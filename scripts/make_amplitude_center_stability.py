"""Plot resonance-center stability for the amplitude-robustness data set.

The per-amplitude Gaussian estimator and preprocessing intentionally match
``notebooks/paper/32_main_amplitude_robustness.ipynb``.  The center-specific
quality mask omits the manuscript's ``abs(center) <= 50 kHz`` gate so that the
quantity being tested is not also used to select the reported points.
"""

from __future__ import annotations

# Backend and local-source setup must precede pyplot and echospec imports.
# ruff: noqa: E402, I001

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".codex_tmp" / "mpl"))
os.environ.setdefault("MPLBACKEND", "Agg")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

from echospec.figures import FigureVariant, apply_figure_style, save_figure


DATA_PATH = (
    PROJECT_ROOT
    / "ips_plots/data/lorentzian_echo/echo_lorentzian_12-33-44-667799_data_bundle.npz"
)
EDGE_POINTS = 40
SMOOTH_SIGMA_POINTS = 1.0
INITIAL_SIGMA_MHZ = 0.040
MIN_CONTRAST = 0.02
MAX_FWHM_MHZ = 0.12
# Independently selected in the matched simulation by Q >= 0.10; see Fig. S8.
OPERATING_RABI_MIN_MHZ = 2.475
OPERATING_RABI_MAX_MHZ = 8.973
# q1 OPX1000 EF spectroscopy: f12-f01 = 4.159106667-4.267106667 GHz.
ANHARMONICITY_MHZ = -216.0
SHAPED_DURATION_US = 10.0
SHAPED_CUTOFF = 0.002
SHAPED_ORDER = 0.5


def dip_gaussian(
    detuning_mhz: np.ndarray,
    offset: float,
    depth: float,
    center_mhz: float,
    sigma_mhz: float,
) -> np.ndarray:
    """Operational Gaussian dip estimator used in the Supplemental."""
    return offset - depth * np.exp(
        -0.5 * ((detuning_mhz - center_mhz) / sigma_mhz) ** 2
    )


def load_amplitude_sweep(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load detuning, calibrated peak Rabi frequency, and q1 populations."""
    with np.load(path, allow_pickle=True) as bundle:
        qubits = json.loads(bundle["profile__qubits.json"].item())["qubits"]
        pulses = json.loads(bundle["profile__pulses.json"].item())["pulses"]
        settings = json.loads(bundle["parameters.json"].item())
        qubit = str(bundle["data__qubit"][0])
        pi_pulse = pulses[qubit][qubits[qubit]["operations"]["x180"]]
        amp_prefactor = np.asarray(bundle["data__amp_prefactor"], dtype=float)
        pi_rabi_mhz = 1000 / (2 * pi_pulse["length_ns"])
        rabi_mhz = (
            amp_prefactor
            * float(settings["lorentzian_peak_amplitude"])
            / pi_pulse["amplitude"]
            * pi_rabi_mhz
        )
        return (
            np.asarray(bundle["data__detuning"], dtype=float) / 1e6,
            rabi_mhz,
            np.asarray(bundle["data__state"][0], dtype=float),
        )


def fit_slice(detuning_mhz: np.ndarray, trace: np.ndarray) -> dict[str, float]:
    """Fit one amplitude slice and retain covariance-derived center error."""
    x = np.asarray(detuning_mhz[EDGE_POINTS:-EDGE_POINTS], dtype=float)
    y = gaussian_filter1d(
        np.asarray(trace[EDGE_POINTS:-EDGE_POINTS], dtype=float),
        SMOOTH_SIGMA_POINTS,
    )
    step = float(np.median(np.diff(x)))
    span = float(x.max() - x.min())
    edge_count = max(5, x.size // 10)
    offset0 = float(np.median(np.r_[y[:edge_count], y[-edge_count:]]))
    minimum_index = int(np.argmin(y))
    depth0 = max(offset0 - float(y[minimum_index]), 1e-3)

    try:
        values, covariance = curve_fit(
            dip_gaussian,
            x,
            y,
            p0=[offset0, depth0, float(x[minimum_index]), INITIAL_SIGMA_MHZ],
            bounds=(
                [-0.2, 0.0, x.min(), abs(step) / 2],
                [1.2, 1.2, x.max(), span],
            ),
            maxfev=20_000,
        )
        errors = np.sqrt(np.diag(covariance))
    except (RuntimeError, ValueError):
        return {"finite": False}

    _, depth, center, sigma = values
    _, _, center_error, sigma_error = errors
    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    return {
        "finite": bool(np.all(np.isfinite(np.r_[values, errors]))),
        "contrast": float(depth),
        "center_mhz": float(center),
        "center_error_mhz": float(center_error),
        "fwhm_mhz": float(fwhm_factor * abs(sigma)),
        "fwhm_error_mhz": float(fwhm_factor * sigma_error),
    }


def constant_center_statistics(
    centers_mhz: np.ndarray, center_errors_mhz: np.ndarray
) -> dict[str, float]:
    """Return the weighted constant fit and unweighted residual summaries."""
    weights = 1 / center_errors_mhz**2
    mean_mhz = float(np.sum(weights * centers_mhz) / np.sum(weights))
    mean_error_mhz = float(1 / np.sqrt(np.sum(weights)))
    residuals_mhz = centers_mhz - mean_mhz
    chi_squared = float(np.sum((residuals_mhz / center_errors_mhz) ** 2))
    return {
        "mean_mhz": mean_mhz,
        "mean_error_mhz": mean_error_mhz,
        "rms_mhz": float(np.sqrt(np.mean(residuals_mhz**2))),
        "max_abs_mhz": float(np.max(np.abs(residuals_mhz))),
        "reduced_chi_squared": chi_squared / (len(centers_mhz) - 1),
    }


def main() -> None:
    apply_figure_style(FigureVariant.PAPER)
    detuning_mhz, rabi_mhz, state = load_amplitude_sweep(DATA_PATH)
    fits = [fit_slice(detuning_mhz, state[:, index]) for index in range(len(rabi_mhz))]

    finite = np.asarray([fit.get("finite", False) for fit in fits], dtype=bool)
    contrast = np.asarray([fit.get("contrast", np.nan) for fit in fits])
    fwhm_mhz = np.asarray([fit.get("fwhm_mhz", np.nan) for fit in fits])
    center_mhz = np.asarray([fit.get("center_mhz", np.nan) for fit in fits])
    center_error_mhz = np.asarray(
        [fit.get("center_error_mhz", np.nan) for fit in fits]
    )

    # Do not gate on center: that would circularly enforce the tested stability.
    center_quality = (
        finite
        & (contrast >= MIN_CONTRAST)
        & (fwhm_mhz <= MAX_FWHM_MHZ)
    )
    accepted = (
        center_quality
        & (rabi_mhz >= OPERATING_RABI_MIN_MHZ)
        & (rabi_mhz <= OPERATING_RABI_MAX_MHZ)
    )
    centers = center_mhz[accepted]
    center_errors = center_error_mhz[accepted]
    statistics = constant_center_statistics(centers, center_errors)
    residuals_khz = 1e3 * (centers - statistics["mean_mhz"])

    fig, ax = plt.subplots(figsize=(7.2, 3.1), constrained_layout=True)
    rms_khz = 1e3 * statistics["rms_mhz"]
    ax.axhspan(
        -rms_khz,
        rms_khz,
        color="0.90",
        zorder=0,
        label="Empirical RMS band",
    )
    ax.errorbar(
        rabi_mhz[accepted],
        residuals_khz,
        yerr=1e3 * center_errors,
        fmt="o",
        ms=3.2,
        color="#00838f",
        ecolor="0.58",
        elinewidth=0.75,
        capsize=1.5,
        label="Experiment",
        zorder=3,
    )
    ax.axhline(0, color="0.20", lw=0.9, ls="--", label="Weighted mean")
    ax.set_xlabel(r"$\Omega_0/2\pi$ (MHz)")
    ax.set_ylabel(r"$\delta_0-\overline{\delta_0}$ (kHz)")
    ax.set_xlim(OPERATING_RABI_MIN_MHZ - 0.15, OPERATING_RABI_MAX_MHZ + 0.15)
    ax.legend(loc="lower left", ncol=3, fontsize=7.5)
    ax.text(
        0.02,
        0.97,
        "\n".join(
            [
                rf"$\overline{{\delta_0}}={1e3 * statistics['mean_mhz']:.2f}"
                rf"\pm{1e3 * statistics['mean_error_mhz']:.2f}\,\mathrm{{kHz}}$ (fit)",
                rf"$\mathrm{{RMS}}={1e3 * statistics['rms_mhz']:.2f}\,\mathrm{{kHz}}$",
                rf"$\max|\delta_0-\overline{{\delta_0}}|="
                rf"{1e3 * statistics['max_abs_mhz']:.2f}\,\mathrm{{kHz}}$",
                rf"$\chi_\nu^2={statistics['reduced_chi_squared']:.2f}$",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
    )

    saved = save_figure(
        fig,
        "03_amplitude_center_stability",
        variant=FigureVariant.PAPER,
        formats=("pdf", "png", "svg"),
        dpi=300,
    )

    # Main-text comparison: measured center motion and the corresponding
    # constant-versus-shaped weak-drive AC-Stark scales over the same window.
    theory_rabi_mhz = np.linspace(
        OPERATING_RABI_MIN_MHZ, OPERATING_RABI_MAX_MHZ, 400
    )
    constant_stark_khz = 1e3 * (
        -theory_rabi_mhz**2 / (2 * ANHARMONICITY_MHZ)
    )
    sigma_us = (SHAPED_DURATION_US / 2) / np.sqrt(
        SHAPED_CUTOFF ** (-1 / SHAPED_ORDER) - 1
    )
    mean_square_envelope = (
        2
        * sigma_us
        / SHAPED_DURATION_US
        * np.arctan(SHAPED_DURATION_US / (2 * sigma_us))
    )
    shaped_stark_khz = mean_square_envelope * constant_stark_khz

    main_fig, (center_ax, stark_ax) = plt.subplots(
        1, 2, figsize=(7.2, 3.15), constrained_layout=True
    )
    center_ax.axhspan(-rms_khz, rms_khz, color="0.90", zorder=0)
    center_ax.errorbar(
        rabi_mhz[accepted],
        residuals_khz,
        yerr=1e3 * center_errors,
        fmt="o",
        ms=3.0,
        color="#00838f",
        ecolor="0.58",
        elinewidth=0.7,
        capsize=1.3,
        zorder=3,
    )
    center_ax.axhline(0, color="0.20", lw=0.9, ls="--")
    center_ax.set_xlim(
        OPERATING_RABI_MIN_MHZ - 0.15, OPERATING_RABI_MAX_MHZ + 0.15
    )
    center_ax.set_xlabel(r"$\Omega_0/2\pi$ (MHz)")
    center_ax.set_ylabel(r"$\delta_0-\overline{\delta_0}$ (kHz)")
    center_ax.set_title("(a) Measured resonance center", fontsize=8.5)
    center_ax.text(
        0.03,
        0.97,
        "\n".join(
            [
                rf"RMS $={rms_khz:.2f}\,\mathrm{{kHz}}$",
                rf"max. excursion $={1e3 * statistics['max_abs_mhz']:.2f}"
                r"\,\mathrm{kHz}$",
            ]
        ),
        transform=center_ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
    )

    stark_ax.plot(
        theory_rabi_mhz,
        constant_stark_khz,
        color="#d55e00",
        lw=1.6,
        label="Constant drive",
    )
    stark_ax.plot(
        theory_rabi_mhz,
        shaped_stark_khz,
        color="#0072b2",
        lw=1.6,
        label="Shaped-pulse average",
    )
    stark_ax.axhline(
        rms_khz,
        color="0.35",
        lw=0.9,
        ls="--",
        label="Measured RMS excursion",
    )
    stark_ax.axhline(
        1e3 * statistics["max_abs_mhz"],
        color="0.35",
        lw=0.9,
        ls=":",
        label="Measured max. excursion",
    )
    stark_ax.set_yscale("log")
    stark_ax.set_xlim(
        OPERATING_RABI_MIN_MHZ - 0.15, OPERATING_RABI_MAX_MHZ + 0.15
    )
    stark_ax.set_xlabel(r"$\Omega_0/2\pi$ (MHz)")
    stark_ax.set_ylabel("Shift or excursion (kHz)")
    stark_ax.set_title("(b) AC-Stark shift scale", fontsize=8.5)
    stark_ax.legend(loc="upper left", fontsize=6.8)
    stark_ax.grid(which="both", alpha=0.18)

    main_saved = save_figure(
        main_fig,
        "04_main_center_ac_stark_comparison",
        variant=FigureVariant.PAPER,
        formats=("pdf", "png", "svg"),
        dpi=300,
    )
    print(
        f"Operating window retains {accepted.sum()}/{len(accepted)} slices "
        f"from {OPERATING_RABI_MIN_MHZ:.3f} to "
        f"{OPERATING_RABI_MAX_MHZ:.3f} MHz."
    )
    print(
        f"constant center = {1e3 * statistics['mean_mhz']:.6f} +/- "
        f"{1e3 * statistics['mean_error_mhz']:.6f} kHz"
    )
    print(f"RMS residual = {1e3 * statistics['rms_mhz']:.6f} kHz")
    print(f"max |residual| = {1e3 * statistics['max_abs_mhz']:.6f} kHz")
    print(f"reduced chi^2 = {statistics['reduced_chi_squared']:.6f}")
    print("Saved:", ", ".join(str(path) for path in saved))
    print("Saved:", ", ".join(str(path) for path in main_saved))


if __name__ == "__main__":
    main()
