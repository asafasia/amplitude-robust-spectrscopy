from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from echospec.figures.paths import (
    PAPER_FIGURES_DIR,
    PRESENTATION_IPS_FIGURES_DIR,
    STYLES_DIR,
)


class FigureVariant(str, Enum):
    PAPER = "paper"
    PRESENTATION_IPS = "presentation_ips"


_STYLE_FILES = {
    FigureVariant.PAPER: STYLES_DIR / "paper.mplstyle",
    FigureVariant.PRESENTATION_IPS: STYLES_DIR / "presentation_ips.mplstyle",
}

_OUTPUT_DIRS = {
    FigureVariant.PAPER: PAPER_FIGURES_DIR,
    FigureVariant.PRESENTATION_IPS: PRESENTATION_IPS_FIGURES_DIR,
}


def normalize_variant(variant: FigureVariant | str) -> FigureVariant:
    return variant if isinstance(variant, FigureVariant) else FigureVariant(variant)


def apply_figure_style(variant: FigureVariant | str) -> None:
    """Apply the Matplotlib style for a paper or IPS presentation figure."""
    variant = normalize_variant(variant)
    plt.style.use(_STYLE_FILES[variant])


def output_dir_for(variant: FigureVariant | str) -> Path:
    variant = normalize_variant(variant)
    return _OUTPUT_DIRS[variant]


def save_figure(
    fig: Figure,
    stem: str,
    *,
    variant: FigureVariant | str,
    formats: Iterable[str] = ("png", "svg"),
) -> list[Path]:
    """Save a figure to the correct output folder for its visual target."""
    output_dir = output_dir_for(variant)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path)
        saved_paths.append(path)

    return saved_paths
