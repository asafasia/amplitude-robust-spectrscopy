"""Render the pulse comparison in paper/notes/POLYNOMIAL_ECHO_PULSE.md."""

from __future__ import annotations

import hashlib
import json
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
from scripts import optimize_echo_tail as model  # noqa: E402

OUT = ROOT / "figures/paper/polynomial_echo_note"
COLORS = ["#586579", "#176eaa", "#d47823"]
LABELS = ["Hard cutoff", "Polynomial p = 5 + cutoff subtraction", "C² subtraction"]
STYLES = ["-", "--", "-."]


def curves(t):
    """Use the existing comparison implementation for both corrected envelopes."""
    hard = np.where(np.abs(t) < 15, (1 + (t / model.SIGMA)**2)**-.5, 0.)
    return np.array([
        hard, model.envelope(t, family="polynomial5")[0],
        model.envelope(t, family="c2")[0],
    ])


def sign(t):
    return np.where(t < -.008, 1., np.where(
        t > .008, -1., -np.sin(np.pi * t / .016)))


def plot_curves(ax, x, values):
    for y, color, label, style in zip(values, COLORS, LABELS, STYLES, strict=True):
        ax.plot(x, y, color=color, ls=style, lw=2, label=label)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": .16,
        "savefig.facecolor": "white", "pdf.fonttype": 42,
    })
    full = np.unique(np.r_[np.linspace(-16, 16, 16001),
                           np.linspace(-.08, .08, 1601)])
    left = np.linspace(-15.5, -10, 2201)
    right = -left[::-1]
    center = np.linspace(-.04, .04, 3201)
    data = {"full_time_us": full, "full_envelopes": curves(full),
            "left_time_us": left, "left_envelopes": curves(left),
            "right_time_us": right, "right_envelopes": curves(right),
            "center_time_us": center, "center_envelopes": curves(center),
            "echo_sign": sign(center)}
    data["center_i_over_a"] = data["center_envelopes"] * data["echo_sign"]
    np.savez_compressed(OUT / "waveforms.npz", **data)

    fig = plt.figure(figsize=(12, 10), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1])
    overview = fig.add_subplot(gs[0, :])
    edge_left = fig.add_subplot(gs[1, 0])
    edge_right = fig.add_subplot(gs[1, 1])
    pulse = fig.add_subplot(gs[2, 0])
    factor = fig.add_subplot(gs[2, 1])
    fig.suptitle("Polynomial edge correction and 16 ns echo smoothing", fontsize=17)
    plot_curves(overview, full, data["full_envelopes"])
    overview.set(yscale="symlog", ylim=(-1e-9, 1.5), xlim=(-16, 16),
                 xlabel="Time from midpoint (µs)", ylabel="Unsigned envelope g(t)",
                 title="a   Full 30 µs envelope — logarithmic above 10⁻⁸")
    overview.set_yscale("symlog", linthresh=1e-8)
    overview.set_yticks([0, 1e-6, 1e-4, 1e-2, 1])
    overview.set_yticklabels(["0", "10⁻⁶", "10⁻⁴", "10⁻²", "1"])
    overview.legend(loc="upper right", fontsize=10)
    for ax, t, key, endpoint, title in [
        (edge_left, left, "left_envelopes", -15, "b   Turn-on edge"),
        (edge_right, right, "right_envelopes", 15, "c   Turn-off edge"),
    ]:
        plot_curves(ax, t, data[key])
        ax.axvline(endpoint, color="#9c3549", lw=1, ls=":")
        ax.set_yscale("symlog", linthresh=1e-9)
        ax.set(xlim=(t[0], t[-1]), ylim=(-2e-10, .002), title=title,
               xlabel="Time from midpoint (µs)", ylabel="Unsigned envelope g(t)")
        ax.set_yticks([0, 1e-8, 1e-6, 1e-4, 1e-3])
        ax.set_yticklabels(["0", "10⁻⁸", "10⁻⁶", "10⁻⁴", "10⁻³"])
    edge_left.text(.04, .94, "Hard cutoff jumps by c = 0.001", va="top",
                   transform=edge_left.transAxes, fontsize=10)
    edge_right.text(.04, .94, "Both corrections reach zero smoothly", va="top",
                    transform=edge_right.transAxes, fontsize=10)

    plot_curves(pulse, center * 1000, data["center_i_over_a"])
    abrupt = np.where(center < 0, 1., -1.) * data["center_envelopes"][0]
    pulse.plot(center * 1000, abrupt, color="black", ls=":", lw=1.4,
               label="Abrupt echo reference")
    pulse.set(title="d   Signed I envelope near the midpoint",
              xlabel="Time from midpoint (ns)", ylabel="I(t) / A",
              xlim=(-40, 40), ylim=(-1.12, 1.12))
    pulse.text(.03, .06, "The three smoothed curves nearly overlap here.",
               transform=pulse.transAxes, fontsize=9)
    pulse.legend(handles=[pulse.lines[-1]], loc="upper right", fontsize=9)
    factor.plot(center * 1000, data["echo_sign"], color="#70439b", lw=2.5,
                label="16 ns sine transition")
    factor.plot([-40, 0, 0, 40], [1, 1, -1, -1], color="black", ls=":",
                lw=1.4, label="Abrupt sign flip")
    factor.set(title="e   Common echo sign factor",
               xlabel="Time from midpoint (ns)", ylabel="s(t)",
               xlim=(-20, 20), ylim=(-1.3, 1.5))
    factor.annotate("", xy=(-8, 1.22), xytext=(8, 1.22),
                    arrowprops={"arrowstyle": "<->", "color": "#70439b"})
    factor.text(0, 1.29, "16 ns", ha="center", fontsize=10, color="#70439b")
    factor.legend(loc="lower left", fontsize=9)
    for ax in [pulse, factor]:
        ax.axvspan(-8, 8, color="#70439b", alpha=.08)
        ax.axhline(0, color="gray", lw=.7)
        for x in [-8, 8]:
            ax.axvline(x, color="#70439b", lw=.8, ls=":")
    fig.savefig(OUT / "comparison.png", dpi=180)
    fig.savefig(OUT / "comparison.pdf")
    plt.close(fig)
    sources = [Path(__file__).resolve(), ROOT / "scripts/optimize_echo_tail.py"]
    provenance = {
        "duration_us": 30, "cutoff": .001, "sigma_us": float(model.SIGMA),
        "polynomial_power": 5, "echo_transition_ns": 16,
        "families": LABELS, "normalization": "Unsigned g(0)=1; no area rescaling",
        "edge_axis": "symlog, linear below 1e-9; zero is shown explicitly",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(
            p.read_bytes()).hexdigest() for p in sources},
        "reproduce": "python scripts/plot_polynomial_echo_note.py",
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(OUT / "comparison.png")


if __name__ == "__main__":
    main()
