"""Plot completed p=2 survey checkpoints without modifying the running scan."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.analysis.cutoff_metrics import cutoff_metrics  # noqa: E402
from echospec.figures.coherence import REFERENCE_FWHM_HZ  # noqa: E402

SOURCE = ROOT / "outputs/polynomial_duration_cutoff_poly2_log/n100"


def log_edges(values):
    centers = np.log(values)
    return np.exp(np.r_[centers[0] - (centers[1] - centers[0]) / 2,
                        (centers[1:] + centers[:-1]) / 2,
                        centers[-1] + (centers[-1] - centers[-2]) / 2])


def main():
    config = json.loads((SOURCE / "simulations/configuration.json").read_text())
    durations = config["durations_us"]
    cutoffs, amps = np.array(config["cutoffs"]), np.array(config["rabi_mhz"])
    # Freeze the file list: newly completed maps belong to the next preview.
    paths = sorted((SOURCE / "simulations").glob("L*us/cutoff_*.npz"))
    shape = (len(durations), len(cutoffs), len(amps))
    width, contrast = np.full(shape, np.nan), np.full(shape, np.nan)
    accepted, completed = np.zeros(shape, bool), np.zeros(shape[:2], bool)
    hashes = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            specification = json.loads(str(data["configuration"]))
            if specification["config"] != config:
                raise ValueError(f"Checkpoint configuration mismatch: {path}")
            pulse = specification["pulse"]
            row = durations.index(pulse["duration_us"])
            ci = int(path.stem.split("_")[-1])
            if pulse["cutoff"] != cutoffs[ci]:
                raise ValueError(f"Checkpoint cutoff mismatch: {path}")
            for ai, spectrum in enumerate(data["populations"][1]):
                metrics = cutoff_metrics(data["detuning_mhz"], spectrum,
                                         shots=config["shots"])
                width[row, ci, ai] = metrics["fwhm_hz"]
                contrast[row, ci, ai] = metrics["contrast"]
                accepted[row, ci, ai] = metrics["accepted"]
            completed[row, ci] = True
        hashes[str(path.relative_to(SOURCE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    visible = np.flatnonzero(completed.any(axis=1))
    if not visible.size:
        raise ValueError("No completed checkpoints yet")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "hatch.linewidth": .4})
    fig, axes = plt.subplots(len(visible), 2, figsize=(9, 2.25 * len(visible) + 1.6),
                             squeeze=False, sharex=True, sharey=True,
                             layout="constrained")
    for col, (values, vmax, cmap, label) in enumerate([
        (width / REFERENCE_FWHM_HZ, 10, "magma", r"FWHM / $\Gamma_{T_2^*}$"),
        (contrast, 1, "viridis", "Contrast"),
    ]):
        color = plt.get_cmap(cmap).copy()
        color.set_bad((0, 0, 0, 0))
        background = ListedColormap(["#dce1e4"])
        background.set_bad((0, 0, 0, 0))
        for i, row in enumerate(visible):
            ax = axes[i, col]
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   facecolor="#fff9e9", edgecolor="#c3bba6",
                                   hatch="///", linewidth=0, zorder=-1))
            saved = np.broadcast_to(completed[row, :, None], (len(cutoffs), len(amps)))
            ax.pcolormesh(log_edges(cutoffs), log_edges(amps),
                          np.ma.masked_where(~saved, np.zeros(saved.shape)).T,
                          cmap=background, rasterized=True)
            artist = ax.pcolormesh(
                log_edges(cutoffs), log_edges(amps),
                np.ma.masked_where(~accepted[row], values[row]).T,
                cmap=color, vmin=0, vmax=vmax, rasterized=True,
            )
            ax.contour(cutoffs, amps,
                       np.ma.masked_where(~accepted[row], contrast[row]).T,
                       levels=[.2], colors="white", linewidths=.6)
            ax.set(xscale="log", yscale="log", xlim=(1e-5, 1), ylim=(1, 100),
                   xticks=[1e-5, 1e-4, 1e-3, 1e-2, .1, 1], yticks=[1, 10, 100])
            ax.set_title(f"{durations[row]:g} µs — {completed[row].sum()}/100 cutoffs saved",
                         loc="left")
            if col == 0:
                ax.set_ylabel("Nominal drive (MHz)")
            if i == len(visible) - 1:
                ax.set_xlabel("Cutoff")
        bar = fig.colorbar(artist, ax=list(axes[:, col]), location="top", shrink=.9,
                           aspect=30, pad=.03)
        bar.set_label(label)
    absent = [f"{durations[row]:g}" for row in range(len(durations)) if row not in visible]
    suffix = f"; {', '.join(absent)} µs not yet saved" if absent else ""
    fig.suptitle(f"Partial simulation results: p = 2 | {len(paths)}/500 maps\n"
                 f"100 log-spaced drives, 1–100 MHz; 16 ns inversion{suffix}", fontsize=12)
    fig.legend(handles=[Patch(facecolor="#fff9e9", hatch="///", edgecolor="#c3bba6",
                              label="Not yet simulated"),
                        Patch(facecolor="#dce1e4", label="Saved, dip rejected")],
               loc="outside lower center", ncol=2, frameon=False)
    destination = SOURCE / "preview"
    destination.mkdir(exist_ok=True)
    fig.savefig(destination / "partial_results.png", dpi=180)
    plt.close(fig)
    np.savez_compressed(destination / "partial_results.npz", durations_us=durations,
                        cutoffs=cutoffs, rabi_mhz=amps, fwhm_hz=width,
                        contrast=contrast, accepted=accepted, completed=completed)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    completed_maps=len(paths), expected_maps=500,
                    completed_per_duration=completed.sum(axis=1).tolist(),
                    configuration=config, checkpoint_sha256=hashes,
                    reference_linewidth_hz=REFERENCE_FWHM_HZ,
                    status="Partial preview; finer-grid convergence checks pending",
                    white_contour="Contrast = 0.2; interpolated between accepted samples")
    (destination / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(destination / "partial_results.png")
    print("Completed per duration:", metadata["completed_per_duration"])


if __name__ == "__main__":
    main()
