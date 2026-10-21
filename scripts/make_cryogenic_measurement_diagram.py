"""Generate the paper's dilution-refrigerator wiring schematic.

The diagram is built entirely from explicit Matplotlib vector geometry.  Its
vertical signal lines and horizontal temperature boundaries deliberately echo
the visual language of a conventional cryogenic wiring diagram while keeping
the actual measurement chain used in this work.
"""

from __future__ import annotations

from math import cos, radians, sin
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Arc,
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Polygon,
    Rectangle,
)
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "paper" / "figures"
OUTPUT_STEM = OUTPUT_DIR / "cryogenic_measurement_chain"

INK = "#202020"
GRAY = "#686868"
LIGHT_GRAY = "#B8B8B8"
BLUE = "#254BDB"
GREEN = "#16824B"
RED = "#EF3B38"
HEMT_FILL = "#F1F1F1"
SHIELD_FILL = "#FAFAFA"


def line(
    ax: plt.Axes,
    x: list[float],
    y: list[float],
    *,
    color: str = INK,
    linewidth: float = 1.25,
    linestyle: str = "-",
    zorder: int = 3,
) -> None:
    ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=zorder,
    )


def connector(ax: plt.Axes, x: float, y: float) -> None:
    """Draw a compact feedthrough/connector symbol centered on ``(x, y)``."""
    ax.add_patch(
        Rectangle(
            (x - 0.065, y - 0.16),
            0.13,
            0.32,
            facecolor="white",
            edgecolor=GRAY,
            linewidth=0.8,
            zorder=8,
        )
    )
    line(ax, [x - 0.13, x + 0.13], [y, y], color=GRAY, linewidth=0.7, zorder=9)


def attenuator(ax: plt.Axes, x: float, y: float, value: str) -> None:
    """Draw a narrow inline attenuator and put its value beside the line."""
    ax.add_patch(
        Rectangle(
            (x - 0.105, y - 0.20),
            0.21,
            0.40,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=1.0,
            zorder=8,
        )
    )
    ax.text(
        x + 0.16,
        y,
        value,
        ha="left",
        va="center",
        fontsize=7.4,
        color=BLUE,
        zorder=9,
    )


def low_pass_filter(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    label: str | None = "300 MHz",
) -> None:
    """Draw the square-corner filter body and reference-style bent-line glyph."""
    ax.add_patch(
        Rectangle(
            (x - 0.135, y - 0.20),
            0.27,
            0.40,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=0.95,
            zorder=8,
        )
    )
    line(
        ax,
        [x - 0.045, x - 0.045, x + 0.045],
        [y - 0.14, y + 0.05, y + 0.14],
        color=BLUE,
        linewidth=0.95,
        zorder=9,
    )
    if label is None:
        return
    ax.text(
        x + 0.30,
        y,
        label,
        ha="left",
        va="center",
        fontsize=6.7,
        color=GRAY,
        zorder=9,
    )


def circulator(ax: plt.Axes, x: float, y: float) -> None:
    size = 0.50
    ax.add_patch(
        Rectangle(
            (x - size / 2, y - size / 2),
            size,
            size,
            facecolor="white",
            edgecolor=LIGHT_GRAY,
            linewidth=1.0,
            zorder=7,
        )
    )
    ax.add_patch(
        Circle(
            (x, y),
            0.17,
            facecolor="white",
            edgecolor=GRAY,
            linewidth=0.85,
            zorder=8,
        )
    )
    # Use explicit geometry so the clockwise arrow scales with the component.
    radius = 0.105
    ax.add_patch(
        Arc(
            (x, y),
            2 * radius,
            2 * radius,
            theta1=-135,
            theta2=135,
            color=INK,
            linewidth=0.95,
            capstyle="round",
            zorder=9,
        )
    )
    angle = radians(-135)
    radial = (cos(angle), sin(angle))
    tangent = (sin(angle), -cos(angle))
    tip = (x + radius * radial[0], y + radius * radial[1])
    base = (tip[0] - 0.065 * tangent[0], tip[1] - 0.065 * tangent[1])
    ax.add_patch(
        Polygon(
            [
                tip,
                (base[0] + 0.032 * radial[0], base[1] + 0.032 * radial[1]),
                (base[0] - 0.032 * radial[0], base[1] - 0.032 * radial[1]),
            ],
            closed=True,
            facecolor=INK,
            edgecolor=INK,
            linewidth=0.3,
            zorder=10,
        )
    )


def bias_coil(ax: plt.Axes, x: float, y: float) -> None:
    """Draw smooth flattened windings and a rounded return lead."""
    vertices = [(0, 0.25), (-0.035, 0.25), (-0.035, 0.18)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO]
    curves = [
        [(-0.035, 0.155), (-0.44, 0.18), (-0.44, 0.09)],
        [(-0.44, -0.005), (0.44, -0.005), (0.44, 0.04)],
        [(0.44, 0.085), (-0.44, 0.085), (-0.44, -0.01)],
        [(-0.44, -0.105), (0.44, -0.105), (0.44, -0.06)],
        [(0.44, -0.015), (-0.44, -0.015), (-0.44, -0.11)],
        [(-0.44, -0.19), (-0.10, -0.18), (-0.015, -0.18)],
        [(0.04, -0.18), (0.04, -0.14), (0.04, -0.10)],
    ]
    for curve in curves:
        vertices.extend(curve)
        codes.extend([MplPath.CURVE4] * 3)
    vertices.append((0.04, 0.25))
    codes.append(MplPath.LINETO)
    ax.add_patch(
        PathPatch(
            MplPath([(x + dx, y + dy) for dx, dy in vertices], codes),
            facecolor="none",
            edgecolor=GREEN,
            linewidth=1.0,
            capstyle="round",
            joinstyle="round",
            zorder=9,
        )
    )


def hemt(ax: plt.Axes, x: float, y: float) -> None:
    ax.add_patch(
        Polygon(
            [(x - 0.25, y - 0.29), (x, y + 0.29), (x + 0.25, y - 0.29)],
            closed=True,
            facecolor=HEMT_FILL,
            edgecolor=INK,
            linewidth=1.1,
            zorder=8,
        )
    )


def direction_marker(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    direction: str,
    color: str,
) -> None:
    dy = 0.12 if direction == "up" else -0.12
    ax.add_patch(
        FancyArrowPatch(
            (x, y - dy),
            (x, y + dy),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.9,
            color=color,
            zorder=10,
        )
    )


def build_figure() -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0,
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 7.4), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 10.7)
    ax.set_aspect("equal")
    ax.axis("off")

    # Temperature boundaries: sparse dashed rules are the main organizing
    # structure, matching standard cryogenic wiring schematics.
    stages = [
        (9.25, "300 K"),
        (8.10, "50 K"),
        (7.20, "10 K"),
        (6.20, "4 K"),
        (5.20, "Still"),
        (4.20, "Cold plate"),
        (3.20, "MXC"),
    ]
    for y, label in stages:
        line(
            ax,
            [1.55, 9.92],
            [y, y],
            color=INK,
            linewidth=0.85,
            linestyle=(0, (3.0, 2.8)),
            zorder=1,
        )
        ax.text(
            1.42,
            y,
            label,
            ha="right",
            va="center",
            fontsize=7.6,
            color=INK,
        )

    x_qubit = 2.55
    x_readout = 4.10
    x_bias = 5.65
    x_hemt_power = 7.15
    x_return = 8.75

    # Room-temperature equipment and feedthroughs.
    ax.text(3.33, 10.43, "Input", ha="center", va="bottom", fontsize=8.8, color=INK)
    ax.text(x_return, 10.43, "Output", ha="center", va="bottom", fontsize=8.8, color=INK)
    ax.text(
        6.05,
        10.67,
        "Control PC  ↔  OPX1000 / MW-FEM",
        ha="center",
        va="top",
        fontsize=8.6,
        color=INK,
    )
    top_labels = [
        (x_qubit, "Qubit drive\nRF OUT"),
        (x_readout, "Readout drive\nRF OUT"),
        (x_bias, "DC bias"),
        (x_hemt_power, "HEMT\nDC power"),
        (x_return, "Readout acquire\nRF IN"),
    ]
    for x, label in top_labels:
        ax.text(x, 9.68, label, ha="center", va="bottom", fontsize=6.7, color=INK)
        connector(ax, x, 9.25)

    # The two driven microwave lines and the optional DC line descend through
    # the refrigerator.  Values match the measurement setup in the manuscript.
    line(ax, [x_qubit, x_qubit], [9.25, 1.05], linewidth=1.25)
    line(ax, [x_readout, x_readout], [9.25, 1.36], linewidth=1.25)
    line(ax, [x_bias, x_bias], [9.25, 2.45], color=GREEN, linewidth=1.2)
    direction_marker(ax, x_qubit, 8.72, direction="down", color=BLUE)
    direction_marker(ax, x_readout, 8.72, direction="down", color=BLUE)
    direction_marker(ax, x_bias, 8.72, direction="down", color=GREEN)

    attenuation_y = [7.83, 5.93, 4.93, 3.93, 2.93]
    for y, value in zip(attenuation_y, ["10 dB", "10 dB", "10 dB", "20 dB", "20 dB"], strict=True):
        attenuator(ax, x_qubit, y, value)
    for y, value in zip(attenuation_y, ["6 dB", "10 dB", "6 dB", "20 dB", "20 dB"], strict=True):
        attenuator(ax, x_readout, y, value)

    low_pass_filter(ax, x_bias, 2.93)

    # The acquisition path rises from the device through three circulators and
    # the 4 K HEMT.  Its DC supply is kept visually distinct from the RF path.
    line(ax, [x_return, x_return], [2.72, 9.25], linewidth=1.3)
    direction_marker(ax, x_return, 8.72, direction="up", color=RED)
    hemt(ax, x_return, 6.45)
    line(ax, [x_hemt_power, x_hemt_power, x_return - 0.25], [9.25, 6.45, 6.45], color=GRAY, linewidth=0.9)
    ax.add_patch(Rectangle((x_hemt_power - 0.08, 6.34), 0.16, 0.22, facecolor=RED, edgecolor=RED, zorder=9))
    ax.text(
        x_return - 0.38,
        6.45,
        "HEMT",
        ha="right",
        va="center",
        fontsize=7.0,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
        zorder=10,
    )

    circulator_y = 2.72
    line(ax, [6.20, x_return], [circulator_y, circulator_y], linewidth=1.25)
    for x in (7.15, 7.72, 8.29):
        circulator(ax, x, circulator_y)

    # Mixing-chamber package.  The two generated tones enter the QPU and the
    # returning readout signal exits through the isolated chain.
    shield = Rectangle(
        (1.95, 0.10),
        7.25,
        2.24,
        facecolor=SHIELD_FILL,
        edgecolor=INK,
        linewidth=1.15,
        zorder=2,
    )
    ax.add_patch(shield)
    ax.text(8.98, 0.33, "Magnetic shield", ha="right", va="center", fontsize=7.2, color=INK, zorder=7)

    qpu = FancyBboxPatch(
        (4.62, 0.60),
        1.36,
        0.96,
        boxstyle="round,pad=0.015,rounding_size=0.035",
        facecolor="white",
        edgecolor=INK,
        linewidth=1.0,
        zorder=7,
    )
    ax.add_patch(qpu)
    ax.add_patch(Rectangle((4.83, 0.86), 0.32, 0.42, facecolor=RED, edgecolor=RED, zorder=8))
    qubit_points = [(5.35 + 0.18 * col, 1.37 - 0.19 * row) for row in range(3) for col in range(3)]
    qubit_points += [(5.35 + 0.18 * col, 0.78) for col in range(2)]
    for qx, qy in qubit_points:
        ax.add_patch(Circle((qx, qy), 0.035, facecolor="white", edgecolor=GRAY, linewidth=0.7, zorder=8))
    ax.text(5.30, 0.33, "QPU · 11 qubits", ha="center", va="center", fontsize=7.4, color=INK)

    line(ax, [x_qubit, 4.62], [1.05, 1.05], linewidth=1.25)
    line(ax, [x_readout, 4.62], [1.36, 1.36], linewidth=1.25)
    line(ax, [5.98, 6.20, 6.20], [1.18, 1.18, circulator_y], linewidth=1.25)
    line(
        ax,
        [x_bias, x_bias],
        [2.45, 2.15],
        color=GREEN,
        linewidth=1.1,
    )
    bias_coil(ax, x_bias, 1.90)
    ax.text(
        6.48,
        1.90,
        "Global\nbias coil",
        ha="left",
        va="center",
        fontsize=7.4,
        color=INK,
        zorder=9,
    )

    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    for suffix in ("svg", "pdf", "png"):
        path = OUTPUT_STEM.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
