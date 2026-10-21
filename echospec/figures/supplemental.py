"""Shared typography for the supplemental figures, based on the cutoff map."""

from pathlib import Path

import matplotlib.pyplot as plt

STYLE_PATH = Path(__file__).resolve().parents[2] / "styles/supplemental.mplstyle"
FIGURE_WIDTH = 7.2


def apply_supplemental_style() -> None:
    """Start from a known baseline so earlier paper/report styles cannot leak in."""
    plt.style.use(["default", str(STYLE_PATH)])
