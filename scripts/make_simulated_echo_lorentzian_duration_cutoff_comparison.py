"""Generate duration-resolved simulated cutoff comparisons for the supplement."""

from __future__ import annotations

# Backend and local-source setup must precede pyplot and echospec imports.
# ruff: noqa: E402, I001

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.figures import FigureVariant, apply_figure_style, save_figure


DURATIONS_US = (10.0, 20.0, 30.0)
CUTOFFS = (0.010, 0.007, 0.002)
DETUNING_MHZ = np.linspace(-1.0, 1.0, 101)
RABI_MHZ = np.linspace(0.0, 50.0, 81)

# Measurement-linked q1 values from paper/coherence_parameters.tex.  T2 is a
# conservative effective value based on Ramsey T2*, not a Hahn-echo result.
T1_US = 51.24
T_PHI_US = 7.87
ORDER = 0.5
U_STEPS_PER_HALF = 1600


def _rhs(
    bloch: np.ndarray,
    *,
    detuning: np.ndarray,
    drive: np.ndarray,
    inv_t1: float,
    inv_t2: float,
) -> np.ndarray:
    """Bloch-equation derivative in physical time, with ground-state z=+1."""
    x, y, z = bloch
    return np.stack(
        (
            detuning * y - inv_t2 * x,
            -detuning * x - drive * z - inv_t2 * y,
            drive * y + inv_t1 * (1.0 - z),
        )
    )


def _integrate_half(
    bloch: np.ndarray,
    *,
    u_start: float,
    u_stop: float,
    sigma_us: float,
    detuning: np.ndarray,
    rabi: np.ndarray,
    drive_sign: float,
) -> np.ndarray:
    """Integrate one pulse half on t=sigma*sinh(u) using vectorized RK4."""
    du = (u_stop - u_start) / U_STEPS_PER_HALF
    inv_t1 = 1.0 / T1_US
    inv_t2 = 1.0 / (2.0 * T1_US) + 1.0 / T_PHI_US

    def derivative(state: np.ndarray, u: float) -> np.ndarray:
        envelope = 1.0 / np.cosh(u)
        dt_du = sigma_us * np.cosh(u)
        return dt_du * _rhs(
            state,
            detuning=detuning,
            drive=drive_sign * rabi * envelope,
            inv_t1=inv_t1,
            inv_t2=inv_t2,
        )

    u = u_start
    for _ in range(U_STEPS_PER_HALF):
        k1 = derivative(bloch, u)
        k2 = derivative(bloch + 0.5 * du * k1, u + 0.5 * du)
        k3 = derivative(bloch + 0.5 * du * k2, u + 0.5 * du)
        k4 = derivative(bloch + du * k3, u + du)
        bloch = bloch + (du / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        u += du
    return bloch


def simulate_map(duration_us: float, cutoff: float, *, echo: bool) -> np.ndarray:
    """Return final excited-state probability over detuning and peak Rabi rate."""
    sigma_us = (duration_us / 2.0) / np.sqrt(cutoff ** (-1.0 / ORDER) - 1.0)
    u_edge = float(np.arcsinh((duration_us / 2.0) / sigma_us))

    detuning, rabi = np.meshgrid(
        2.0 * np.pi * DETUNING_MHZ,
        2.0 * np.pi * RABI_MHZ,
    )
    bloch = np.zeros((3, *detuning.shape), dtype=float)
    bloch[2] = 1.0
    bloch = _integrate_half(
        bloch,
        u_start=-u_edge,
        u_stop=0.0,
        sigma_us=sigma_us,
        detuning=detuning,
        rabi=rabi,
        drive_sign=1.0,
    )
    bloch = _integrate_half(
        bloch,
        u_start=0.0,
        u_stop=u_edge,
        sigma_us=sigma_us,
        detuning=detuning,
        rabi=rabi,
        drive_sign=-1.0 if echo else 1.0,
    )
    population = (1.0 - bloch[2]) / 2.0
    if not np.all(np.isfinite(population)):
        raise RuntimeError("Simulation produced nonfinite populations")
    if population.min() < -1e-6 or population.max() > 1.0 + 1e-6:
        raise RuntimeError(
            "Simulation left the physical probability interval: "
            f"{population.min():.6g} to {population.max():.6g}"
        )
    return np.clip(population, 0.0, 1.0)


def build_figure(duration_us: float) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    apply_figure_style(FigureVariant.PAPER)
    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 5.15),
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    maps: dict[str, np.ndarray] = {}
    for echo in (True, False):
        protocol = "echo" if echo else "lorentzian"
        for cutoff in CUTOFFS:
            maps[f"{protocol}_cutoff_{cutoff:g}"] = simulate_map(
                duration_us,
                cutoff,
                echo=echo,
            )

    fig, axes = plt.subplots(
        2,
        3,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panel_labels = iter("abcdef")
    image = None
    for row, protocol in enumerate(("echo", "lorentzian")):
        for column, cutoff in enumerate(CUTOFFS):
            ax = axes[row, column]
            image = ax.pcolormesh(
                DETUNING_MHZ,
                RABI_MHZ,
                maps[f"{protocol}_cutoff_{cutoff:g}"],
                shading="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=0.5,
                rasterized=True,
            )
            ax.axvline(0.0, color="white", lw=0.45, ls="--", alpha=0.75)
            ax.text(
                0.025,
                0.95,
                f"({next(panel_labels)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="white",
                fontweight="bold",
            )
            if row == 0:
                ax.set_title(rf"$c={cutoff:.3f}$")
            if row == 1:
                ax.set_xlabel(r"$\Delta/2\pi$ (MHz)")
            if column == 0:
                ax.set_ylabel(r"$\Omega_0/2\pi$ (MHz)")

    if image is None:
        raise RuntimeError("No panels were generated")
    colorbar = fig.colorbar(image, ax=axes, pad=0.015, fraction=0.035)
    colorbar.set_label(r"$P_e$")
    colorbar.ax.tick_params(labelsize=6)
    return fig, maps


def main() -> None:
    output_dir = ROOT / "figures" / "paper"
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_t2_us = 1.0 / (1.0 / (2.0 * T1_US) + 1.0 / T_PHI_US)
    print(
        f"Simulation coherence: T1={T1_US:.2f} us, "
        f"Tphi={T_PHI_US:.2f} us, T2_eff={effective_t2_us:.3f} us"
    )
    for duration_us in DURATIONS_US:
        fig, maps = build_figure(duration_us)
        duration_label = int(duration_us)
        stem = f"09_simulated_echo_lorentzian_{duration_label}us"
        saved = save_figure(
            fig,
            stem,
            variant=FigureVariant.PAPER,
            formats=("pdf", "png", "svg"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
        plt.close(fig)
        data_path = output_dir / f"{stem}.npz"
        np.savez_compressed(
            data_path,
            duration_us=duration_us,
            cutoffs=np.asarray(CUTOFFS),
            detuning_mhz=DETUNING_MHZ,
            rabi_mhz=RABI_MHZ,
            t1_us=T1_US,
            t_phi_us=T_PHI_US,
            order=ORDER,
            u_steps_per_half=U_STEPS_PER_HALF,
            **maps,
        )
        for path in (*saved, data_path):
            print(path)


if __name__ == "__main__":
    main()
