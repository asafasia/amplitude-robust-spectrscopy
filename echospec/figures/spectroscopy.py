from __future__ import annotations

from typing import Literal

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from echospec.figures.style import FigureVariant, normalize_variant

DetailLevel = Literal["full", "simple"]


def detail_level_for(variant: FigureVariant | str) -> DetailLevel:
    variant = normalize_variant(variant)
    return "simple" if variant is FigureVariant.PRESENTATION_IPS else "full"


def polish_axes_for_variant(
    ax: Axes,
    *,
    variant: FigureVariant | str,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Apply small figure-target-specific axis decisions after plotting."""
    variant = normalize_variant(variant)
    detail_level = detail_level_for(variant)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if detail_level == "simple":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(which="both", top=False, right=False)


def finish_figure(fig: Figure, *, variant: FigureVariant | str) -> Figure:
    """Finalize spacing according to the figure target."""
    normalize_variant(variant)
    fig.tight_layout()
    return fig
