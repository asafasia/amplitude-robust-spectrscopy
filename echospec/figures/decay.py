from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from echospec.figures.style import FigureVariant, normalize_variant


def make_t1_t2_decay_figure(
    *,
    t1_us: float = 30.0,
    t2_us: float = 14.0,
    t_stop_us: float = 80.0,
    num_points: int = 800,
    variant: FigureVariant | str = FigureVariant.PRESENTATION_IPS,
) -> Figure:
    """Create a slide-friendly T1/T2 decay figure."""
    variant = normalize_variant(variant)

    t_us = np.linspace(0, t_stop_us, num_points)
    p_excited = np.exp(-t_us / t1_us)
    coherence = np.exp(-t_us / t2_us)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.5, 11.5) if variant is FigureVariant.PRESENTATION_IPS else None,
        sharex=True,
    )
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.18, right=0.96, hspace=0.42)

    axes[0].plot(t_us, p_excited, color="#1f77b4", linewidth=4.5)
    axes[0].axvline(t1_us, color="#1f77b4", linestyle="--", linewidth=2.8, alpha=0.85)
    axes[0].text(
        t1_us + 2,
        np.exp(-1) + 0.08,
        rf"$T_1 = {t1_us:g}\,\mu s$",
        color="#1f77b4",
        fontsize=26,
        weight="bold",
    )
    axes[0].set_title(r"Energy relaxation ($T_1$)", pad=14)
    axes[0].set_ylabel("Excited-state\npopulation")
    axes[0].set_ylim(-0.03, 1.05)

    axes[1].plot(t_us, coherence, color="#d62728", linewidth=4.5)
    axes[1].axvline(t2_us, color="#d62728", linestyle="--", linewidth=2.8, alpha=0.85)
    axes[1].text(
        t2_us + 2,
        np.exp(-1) + 0.08,
        rf"$T_2 = {t2_us:g}\,\mu s$",
        color="#d62728",
        fontsize=26,
        weight="bold",
    )
    axes[1].set_title(r"Coherence decay ($T_2$)", pad=14)
    axes[1].set_xlabel(r"Time ($\mu s$)")
    axes[1].set_ylabel("Coherence\namplitude")
    axes[1].set_ylim(-0.03, 1.05)

    for ax in axes:
        ax.grid(True, alpha=0.28, linewidth=1.2)
        ax.tick_params(width=2.0, length=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Qubit Decay Times", y=0.975, weight="bold")
    return fig
