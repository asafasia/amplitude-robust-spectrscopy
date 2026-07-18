import matplotlib.pyplot as plt


PRESENTATION_RC = {
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#222222",
    "axes.labelcolor": "#111111",
    "axes.labelsize": 14,
    "axes.linewidth": 1.0,
    "axes.titlesize": 0,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "legend.frameon": False,
    "image.cmap": "viridis",
}


def use_presentation_style():
    plt.style.use("default")
    plt.rcParams.update(PRESENTATION_RC)


def presentation_figure(width=9.0, height=5.6):
    return plt.subplots(figsize=(width, height), constrained_layout=True)


def polish_axes(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=1)
