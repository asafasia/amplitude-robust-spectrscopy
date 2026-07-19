"""Generate the dilution-refrigerator measurement-chain schematic.

The figure is intentionally drawn from explicit geometry rather than an
automatic graph layout.  This keeps temperature stages, signal directions,
and microwave components aligned and makes the SVG/PDF output reproducible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "paper" / "figures"
OUTPUT_STEM = OUTPUT_DIR / "cryogenic_measurement_chain"

# Restrained, publication-oriented palette.  Signal families are distinguished
# primarily by labels and direction rather than decorative color.
INK = "#20262D"
MUTED = "#5F6871"
STRUCTURE = "#A7B3BD"
STAGE_FILL = "#E8EEF2"
DRIVE = "#316AA3"
READ_IN = "#387B79"
READ_OUT = "#4D89AC"
BIAS = "#6D8A50"
HEMT_POWER = "#957144"
ATTEN_FILL = "#F6E8C8"
ATTEN_EDGE = "#B6914D"
FILTER_FILL = "#DFEBF3"
FILTER_EDGE = "#6E96B1"
ROOM_FILL = "#F3F7FA"
FRIDGE_FILL = "#FCFCFB"
DEVICE_FILL = "#F1ECF4"
DEVICE_EDGE = "#8D7C94"
HEMT = "#586F9B"
FONT_SCALE = 1.15


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 8.0,
    linewidth: float = 0.9,
    radius: float = 0.025,
    zorder: int = 5,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize * FONT_SCALE,
        zorder=zorder + 1,
    )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    linewidth: float = 1.15,
    mutation_scale: float = 8.0,
    zorder: int = 3,
    connectionstyle: str = "arc3",
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        shrinkA=0,
        shrinkB=0,
        connectionstyle=connectionstyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def line(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    *,
    color: str = INK,
    linewidth: float = 1.0,
    zorder: int = 2,
    linestyle: str = "-",
) -> None:
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder,
    )


def attenuator(ax: plt.Axes, x: float, y: float, value: str) -> None:
    rounded_box(
        ax,
        x - 0.25,
        y - 0.15,
        0.50,
        0.30,
        value,
        facecolor=ATTEN_FILL,
        edgecolor=ATTEN_EDGE,
        fontsize=6.6,
        linewidth=0.75,
        radius=0.012,
        zorder=8,
    )


def low_pass_filter(ax: plt.Axes, x: float, y: float) -> None:
    rounded_box(
        ax,
        x - 0.31,
        y - 0.19,
        0.62,
        0.38,
        "LPF\n7.5 GHz",
        facecolor=FILTER_FILL,
        edgecolor=FILTER_EDGE,
        fontsize=6.0,
        linewidth=0.75,
        radius=0.015,
        zorder=8,
    )


def circulator(ax: plt.Axes, x: float, y: float) -> None:
    radius = 0.17
    ax.add_patch(
        Circle(
            (x, y),
            radius,
            facecolor=ATTEN_FILL,
            edgecolor=ATTEN_EDGE,
            linewidth=0.85,
            zorder=8,
        )
    )
    # Use a math glyph so the symbol remains available with publication fonts
    # that do not include the Unicode clockwise-arrow character.
    ax.text(x, y - 0.005, r"$\circlearrowright$", ha="center", va="center", fontsize=8.5 * FONT_SCALE, color=INK, zorder=9)


def hemt(ax: plt.Axes, x: float, y: float, *, label: bool = True) -> None:
    width, height = 0.47, 0.48
    triangle = Polygon(
        [(x - width / 2, y), (x + width / 2, y + height / 2), (x + width / 2, y - height / 2)],
        closed=True,
        facecolor=HEMT,
        edgecolor="#354452",
        linewidth=0.8,
        zorder=8,
    )
    ax.add_patch(triangle)
    if label:
        ax.text(x, y - 0.36, "HEMT", ha="center", va="top", fontsize=6.4 * FONT_SCALE, color=MUTED)


def build_figure() -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "Liberation Sans",
            ],
            "font.size": 8 * FONT_SCALE,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0,
        }
    )
    fig, ax = plt.subplots(figsize=(15.6, 8.2), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 15.8)
    ax.set_ylim(0, 8.35)
    ax.set_aspect("equal")
    ax.axis("off")

    room = FancyBboxPatch(
        (0.25, 0.95),
        2.45,
        7.15,
        boxstyle="round,pad=0.02,rounding_size=0.055",
        facecolor=ROOM_FILL,
        edgecolor="#9FA8AF",
        linewidth=0.9,
        zorder=0,
    )
    fridge = FancyBboxPatch(
        (2.95, 0.95),
        12.55,
        7.15,
        boxstyle="round,pad=0.02,rounding_size=0.065",
        facecolor=FRIDGE_FILL,
        edgecolor=STRUCTURE,
        linewidth=0.95,
        zorder=0,
    )
    ax.add_patch(room)
    ax.add_patch(fridge)
    ax.text(
        1.475,
        7.82,
        "ROOM-TEMPERATURE ELECTRONICS",
        ha="center",
        va="center",
        fontsize=6.3 * FONT_SCALE,
        fontweight="bold",
        color=MUTED,
    )

    stage_x = [3.55, 5.10, 6.65, 8.20, 9.75, 11.30, 12.85]
    stage_labels = ["300 K", "50 K", "10 K", "4 K", "2.7 K\nStill", "900 mK\nCold Plate", "9 mK\nMixing Chamber"]
    for x, label in zip(stage_x, stage_labels, strict=True):
        ax.add_patch(Rectangle((x - 0.055, 1.05), 0.11, 6.48, facecolor=STAGE_FILL, edgecolor="none", zorder=1))
        line(ax, [x, x], [1.00, 7.58], color=STRUCTURE, linewidth=0.65, linestyle=(0, (2.4, 3.4)), zorder=2)
        ax.text(x, 7.70, label, ha="center", va="bottom", fontsize=7.2 * FONT_SCALE, fontweight="bold", color=INK)

    y_drive, y_readout_tx, y_readout_rx = 6.20, 4.80, 3.40
    y_hemt_power, y_bias = 2.20, 1.25
    x_line_start, x_device = 2.58, 14.05

    row_specs = [
        ("QUBIT DRIVE · RF OUT", y_drive, DRIVE),
        ("READOUT DRIVE · RF OUT", y_readout_tx, READ_IN),
        ("READOUT ACQUIRE · RF IN", y_readout_rx, READ_OUT),
        ("HEMT DC POWER", y_hemt_power, HEMT_POWER),
        ("FLUX / DC BIAS · NOT USED", y_bias, BIAS),
    ]
    for label, y, color in row_specs:
        ax.text(3.05, y + 0.22, label, ha="left", va="bottom", fontsize=6.6 * FONT_SCALE, fontweight="bold", color=color)

    arrow(ax, (x_line_start, y_drive), (x_device, y_drive), color=DRIVE, linewidth=1.55, mutation_scale=9)
    arrow(ax, (x_line_start, y_readout_tx), (x_device, y_readout_tx), color=READ_IN, linewidth=1.55, mutation_scale=9)
    arrow(ax, (x_device, y_readout_rx), (x_line_start, y_readout_rx), color=READ_OUT, linewidth=1.55, mutation_scale=9)
    arrow(ax, (1.965, y_bias), (x_device, y_bias), color=BIAS, linewidth=1.45, mutation_scale=9)

    opx = FancyBboxPatch(
        (0.48, 3.02),
        2.04,
        3.92,
        boxstyle="round,pad=0.025,rounding_size=0.045",
        facecolor="#E7EFF7",
        edgecolor="#5F7C96",
        linewidth=1.1,
        zorder=5,
    )
    ax.add_patch(opx)
    ax.text(1.50, 6.72, "OPX1000 / MW-FEM", ha="center", va="center", fontsize=8.8 * FONT_SCALE, fontweight="bold", color=INK, zorder=7)
    opx_ports = [
        (y_drive, "Qubit drive\nRF OUT", DRIVE, "#EAF2FA"),
        (y_readout_tx, "Readout drive\nRF OUT", READ_IN, "#E9F4F2"),
        (y_readout_rx, "Readout acquire\nRF IN", READ_OUT, "#EAF3F8"),
    ]
    for y, label, color, fill in opx_ports:
        rounded_box(
            ax,
            0.72,
            y - 0.22,
            1.45,
            0.44,
            label,
            facecolor=fill,
            edgecolor=color,
            fontsize=6.5,
            linewidth=0.8,
            radius=0.018,
            zorder=7,
        )
        ax.add_patch(Circle((2.52, y), 0.064, facecolor=color, edgecolor="white", linewidth=0.7, zorder=10))

    # The control computer is deliberately separate from the signal-chain
    # blocks: it submits QUA programs and receives results over the control
    # network, while the MW-FEM performs the real-time RF processing.
    pc = FancyBboxPatch(
        (0.72, 7.23),
        1.56,
        0.45,
        boxstyle="round,pad=0.02,rounding_size=0.018",
        facecolor="#EEF4F8",
        edgecolor="#6C8496",
        linewidth=0.85,
        zorder=6,
    )
    ax.add_patch(pc)
    ax.text(1.50, 7.51, "CONTROL PC", ha="center", va="center", fontsize=6.8 * FONT_SCALE, fontweight="bold", color=INK, zorder=7)
    ax.text(1.50, 7.36, "QUA program / data", ha="center", va="center", fontsize=5.5 * FONT_SCALE, color=MUTED, zorder=7)
    control_link = FancyArrowPatch(
        (1.50, 7.23),
        (1.50, 6.94),
        arrowstyle="<->",
        mutation_scale=6.5,
        linewidth=0.75,
        color=MUTED,
        shrinkA=0,
        shrinkB=0,
        zorder=7,
    )
    ax.add_patch(control_link)
    ax.text(1.62, 7.08, "Ethernet control", ha="left", va="center", fontsize=5.2 * FONT_SCALE, color=MUTED, zorder=7)

    rounded_box(ax, 0.985, y_bias - 0.23, 0.98, 0.46, "DC bias supply", facecolor="#EDF3EA", edgecolor=BIAS, fontsize=6.8)

    drive_attenuators = [(5.10, "10 dB"), (8.20, "10 dB"), (9.75, "10 dB"), (11.30, "20 dB"), (12.85, "20 dB")]
    read_attenuators = [(5.10, "6 dB"), (8.20, "10 dB"), (9.75, "6 dB"), (11.30, "20 dB"), (12.85, "20 dB")]
    for x, value in drive_attenuators:
        attenuator(ax, x, y_drive, value)
    for x, value in read_attenuators:
        attenuator(ax, x, y_readout_tx, value)
    low_pass_filter(ax, 8.20, y_bias)
    low_pass_filter(ax, 12.85, y_bias)

    hemt(ax, 8.20, y_readout_rx, label=False)
    ax.text(8.20, y_readout_rx + 0.38, "HEMT · 4 K", ha="center", va="bottom", fontsize=6.4 * FONT_SCALE, color=MUTED)
    for x in (12.10, 12.48, 12.86):
        circulator(ax, x, y_readout_rx)

    rounded_box(
        ax,
        0.90,
        y_hemt_power - 0.21,
        1.15,
        0.42,
        "HEMT DC power",
        facecolor="#F3EDE3",
        edgecolor=HEMT_POWER,
        fontsize=6.7,
    )
    line(ax, [2.05, 8.20], [y_hemt_power, y_hemt_power], color=HEMT_POWER, linewidth=0.9, zorder=2)
    arrow(
        ax,
        (8.20, y_hemt_power),
        (8.20, y_readout_rx - 0.24),
        color=HEMT_POWER,
        linewidth=0.9,
        mutation_scale=6,
        zorder=7,
    )

    shield = FancyBboxPatch(
        (14.05, 1.05),
        1.18,
        6.20,
        boxstyle="round,pad=0.025,rounding_size=0.045",
        facecolor=DEVICE_FILL,
        edgecolor=DEVICE_EDGE,
        linewidth=1.0,
        zorder=6,
    )
    ax.add_patch(shield)
    ax.text(14.64, 7.02, "Magnetic shield", ha="center", va="top", fontsize=7.2 * FONT_SCALE, fontweight="bold", color=INK, zorder=7)
    sample = FancyBboxPatch(
        (14.28, 3.02),
        0.72,
        2.25,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor="#FAFAF9",
        edgecolor="#929A9F",
        linewidth=0.8,
        zorder=7,
    )
    ax.add_patch(sample)
    ax.text(14.64, 4.92, "QPU", ha="center", va="center", fontsize=8.0 * FONT_SCALE, fontweight="bold", color=INK, zorder=8)
    qubit_points = [(14.45 + 0.19 * col, 4.55 - 0.26 * row) for row in range(4) for col in range(3)]
    for qx, qy in qubit_points[:11]:
        ax.add_patch(Circle((qx, qy), 0.035, facecolor="white", edgecolor=DEVICE_EDGE, linewidth=0.65, zorder=8))
    ax.text(14.64, 3.34, "11 qubits", ha="center", va="center", fontsize=6.0 * FONT_SCALE, color=MUTED, zorder=8)

    for y, color in ((y_drive, DRIVE), (y_readout_tx, READ_IN), (y_readout_rx, READ_OUT), (y_bias, BIAS)):
        ax.add_patch(Circle((14.05, y), 0.060, facecolor=color, edgecolor="white", linewidth=0.7, zorder=10))

    ax.text(3.10, 0.56, "SYMBOLS", ha="left", va="center", fontsize=6.8 * FONT_SCALE, fontweight="bold", color=MUTED)
    attenuator(ax, 4.18, 0.56, "10 dB")
    ax.text(4.52, 0.56, "attenuator", ha="left", va="center", fontsize=6.4 * FONT_SCALE, color=INK)
    low_pass_filter(ax, 6.10, 0.56)
    ax.text(6.50, 0.56, "low-pass filter", ha="left", va="center", fontsize=6.4 * FONT_SCALE, color=INK)
    circulator(ax, 8.10, 0.56)
    ax.text(8.38, 0.56, "circulator", ha="left", va="center", fontsize=6.4 * FONT_SCALE, color=INK)
    hemt(ax, 9.85, 0.56, label=False)
    ax.text(10.18, 0.56, "HEMT amplifier", ha="left", va="center", fontsize=6.4 * FONT_SCALE, color=INK)
    arrow(ax, (12.10, 0.56), (12.85, 0.56), color=INK, linewidth=0.9, mutation_scale=6)
    ax.text(12.98, 0.56, "signal direction", ha="left", va="center", fontsize=6.4 * FONT_SCALE, color=INK)

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
