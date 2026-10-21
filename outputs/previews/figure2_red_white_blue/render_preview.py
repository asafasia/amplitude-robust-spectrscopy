"""Preview Figure 2 with a blue-white-red map, without changing paper assets."""
from pathlib import Path
import hashlib
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import make_main_central_spectroscopy_experiment as source

OUT = Path(__file__).resolve().parent
CMAP = LinearSegmentedColormap.from_list(
    "blue_white_red", ["#2166ac", "#ffffff", "#b2182b"], N=257
)


def save_preview(fig, *args, **kwargs):
    for ax in fig.axes[:6]:
        for collection in ax.collections:
            collection.set_cmap(CMAP)
            collection.set_clim(0.0, 1.0)
    paths = []
    for suffix in ("png", "pdf"):
        path = OUT / f"figure2_blue_white_red.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
        paths.append(path)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    protected = list((ROOT / "figures/paper").glob("02_central_spectroscopy.*"))
    protected += [ROOT / "paper/main.pdf", ROOT / "paper/sections/02_pulse_protocol.tex"]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    source.save_figure = save_preview
    source.main()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in before.items())
    print("Verified: original Figure 2 and paper files unchanged.")
