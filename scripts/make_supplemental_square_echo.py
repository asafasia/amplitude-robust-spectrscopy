"""Export and plot the saved single-square/root echo comparison, without solving.

The default rebuild uses the portable paper dataset. --refresh-source reimports
the existing simulation archives, verifies their recorded response hash, and
selects original samples without interpolation, normalization, or fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from echospec.figures.supplemental import FIGURE_WIDTH, apply_supplemental_style
from echospec.paper_data import save_paper_dataset

ROOT = Path(__file__).resolve().parents[1]
STEM = "09_square_echo_comparison"
SOURCE = ROOT / "outputs/square_root_echo_comparison_20us"
DATA = ROOT / "paper/data/numerical"
FIGURES = ROOT / "figures/paper/supplemental"
COLORS = {"square": "#c66a20", "root": "#126aa2"}
LABELS = {"square": "Square echo", "root": "Root-Lorentzian echo"}
SLICE_DRIVES = (3.05, 10.05)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_index(values: np.ndarray, target: float) -> int:
    """Require the requested saved sample; never silently select a nearby drive."""
    indices = np.flatnonzero(np.isclose(values, target, rtol=0, atol=1e-10))
    if len(indices) != 1:
        raise ValueError(f"Expected one saved sample at {target} MHz")
    return int(indices[0])


def check_populations(population: np.ndarray, shape: tuple[int, ...]) -> None:
    if population.shape != shape or not np.all(np.isfinite(population)):
        raise ValueError("Invalid population shape or nonfinite population")
    if np.min(population) < -1e-8 or np.max(population) > 1 + 1e-8:
        raise ValueError("Population outside [0, 1]")
    if not np.allclose(population.sum(axis=-1), 1, rtol=0, atol=1e-7):
        raise ValueError("Populations are not normalized")


def prepare_arrays(responses, controls) -> dict[str, np.ndarray]:
    """Select the reported drive range and retain original populations verbatim."""
    drive = responses["amplitude_mhz"]
    detuning = responses["detuning_mhz"]
    wide_detuning = controls["wide_detuning_mhz"]
    for axis in (drive, detuning, wide_detuning):
        if axis.ndim != 1 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
            raise ValueError("Source coordinates must be finite and strictly increasing")
    selected = np.flatnonzero((drive >= 0.1) & (drive <= 20.05))
    slices = np.array([exact_index(drive[selected], x) for x in SLICE_DRIVES])
    wide_index = exact_index(controls["amplitude_mhz"], SLICE_DRIVES[-1])
    arrays = {
        "amplitude_mhz": drive[selected],
        "detuning_mhz": detuning.copy(),
        "source_amplitude_indices": selected,
        "slice_indices": slices,
        "wide_amplitude_mhz": np.asarray(SLICE_DRIVES[-1]),
        "wide_detuning_mhz": wide_detuning.copy(),
    }
    for protocol in COLORS:
        population = responses[f"dissipative3_{protocol}"][selected].copy()
        wide = controls[f"wide_{protocol}"][wide_index].copy()
        check_populations(population, (len(selected), len(detuning), 3))
        check_populations(wide, (len(wide_detuning), 3))
        arrays[f"{protocol}_population"] = population
        arrays[f"{protocol}_variation"] = np.ptp(population[:, :, 1], axis=1)
        arrays[f"wide_{protocol}_population"] = wide
    return arrays


def import_saved(source: Path):
    records = {
        name: json.loads((source / f"{name}.json").read_text())
        for name in (
            "configuration", "postprocessing_provenance", "validation",
            "additional_validation",
        )
    }
    if sha256(source / "responses.npz") != records["postprocessing_provenance"]["response_sha256"]:
        raise ValueError("Saved responses do not match their recorded SHA-256")
    config = records["configuration"]["configuration"]
    expected = {
        "levels": 3, "duration_us": 20.0, "cutoff": 0.005, "order": 0.5,
        "echo": True, "zeroed_pulse": False, "drag_beta": 0.0,
        "stark_kappa_mhz_inv": 0.0, "anharmonicity_mhz": -216.0,
        "t1_us": 43.8, "t2_us": 7.6,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved configuration differs from the manuscript comparison")
    with np.load(source / "responses.npz", allow_pickle=False) as response:
        with np.load(source / "additional_controls.npz", allow_pickle=False) as controls:
            for name in ("amplitude_mhz", "detuning_mhz"):
                np.testing.assert_array_equal(response[name], config[name])
            np.testing.assert_allclose(
                response["detuning_mhz"], np.linspace(-0.5, 0.5, 401),
                rtol=0, atol=1e-12,
            )
            arrays = prepare_arrays(response, controls)
    input_names = [f"{name}.json" for name in records]
    input_names += ["responses.npz", "additional_controls.npz"]
    metrics = [
        {
            "amplitude_mhz": float(arrays["amplitude_mhz"][i]),
            **{f"{name}_variation": float(arrays[f"{name}_variation"][i])
               for name in COLORS},
        }
        for i in arrays["slice_indices"]
    ]
    provenance = {
        "scope": "Supplemental Fig. S9; single inversion, fixed 20 us duration",
        "source_directory": source.name,
        "source_sha256": {name: sha256(source / name) for name in input_names},
        "source_records": records,
        "population_order": ["ground", "excited", "second_excited"],
        "selection": "Original rows in [0.1, 20.05] MHz; exact 3.05 and 10.05 MHz slices",
        "processing": "No simulation, interpolation, fit, smoothing, or normalization",
        "variation_definition": "max(Pe)-min(Pe) over the saved +/-0.5 MHz window",
        "slice_metrics": metrics,
        "matching": "Equal duration and peak drive, not equal pulse area or energy",
        "validation_note": (
            "Square segments use exact Lindblad exponentials. The refined four-level "
            "root checks in additional_validation supersede the coarse root checks "
            "in validation. Checks at selected detunings do not bound every map pixel."
        ),
        "reproduce": "PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python scripts/make_supplemental_square_echo.py",
    }
    return arrays, provenance


def render(arrays, output: Path) -> None:
    apply_supplemental_style()
    fig, axes = plt.subplots(3, 2, figsize=(FIGURE_WIDTH, 6.5))
    fig.subplots_adjust(left=0.095, right=0.89, bottom=0.075, top=0.96,
                        wspace=0.33, hspace=0.55)
    drive = arrays["amplitude_mhz"]
    detuning_khz = arrays["detuning_mhz"] * 1000
    for ax, name, letter in zip(axes[0], COLORS, "ab", strict=True):
        mesh = ax.pcolormesh(
            detuning_khz, drive, arrays[f"{name}_population"][:, :, 1],
            shading="nearest", cmap="viridis", vmin=0, vmax=0.8,
            rasterized=True,
        )
        ax.set(title=f"({letter}) {LABELS[name]}", yscale="log", ylim=(0.1, 20.05),
               xlim=(-500, 500), xlabel=r"Detuning $\Delta/(2\pi)$ (kHz)",
               ylabel=r"Peak drive $\Omega_0/(2\pi)$ (MHz)")
        ax.set_yticks([0.1, 1, 10], labels=["0.1", "1", "10"])
    box = axes[0, 1].get_position()
    cax = fig.add_axes([0.91, box.y0, 0.016, box.height])
    fig.colorbar(mesh, cax=cax, label=r"$P_e$", ticks=[0, 0.4, 0.8])
    for ax, index, letter in zip(axes[1], arrays["slice_indices"], "cd", strict=True):
        for name in COLORS:
            ax.plot(detuning_khz, arrays[f"{name}_population"][index, :, 1],
                    color=COLORS[name], label=LABELS[name])
        ax.set(title=rf"({letter}) $\Omega_0/(2\pi)={drive[index]:.2f}$ MHz",
               xlim=(-500, 500), ylim=(0, 0.8), ylabel=r"$P_e$",
               xlabel=r"Detuning $\Delta/(2\pi)$ (kHz)")
    axes[1, 0].legend(loc="lower left", fontsize=7)
    ax = axes[2, 0]
    for name in COLORS:
        ax.loglog(drive, arrays[f"{name}_variation"], color=COLORS[name])
    ax.set(title="(e) Central-window population variation", xlim=(0.1, 20.05),
           ylim=(1e-4, 1), ylabel=r"$V = \max P_e - \min P_e$",
           xlabel=r"Peak drive $\Omega_0/(2\pi)$ (MHz)")
    ax.set_xticks([0.1, 1, 10], labels=["0.1", "1", "10"])
    ax = axes[2, 1]
    ax.axvspan(-0.5, 0.5, color="0.85", zorder=0)
    for name in COLORS:
        ax.plot(arrays["wide_detuning_mhz"], arrays[f"wide_{name}_population"][:, 1],
                color=COLORS[name], linewidth=0.8)
    ax.set(title=r"(f) Wider window, $\Omega_0/(2\pi)=10.05$ MHz",
           xlim=(-10, 10), ylim=(0, 0.8), ylabel=r"$P_e$",
           xlabel=r"Detuning $\Delta/(2\pi)$ (MHz)")
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(output / f"square_echo_comparison.{extension}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--refresh-source", action="store_true")
    args = parser.parse_args()
    data_file = DATA / f"{STEM}.npz"
    metadata_file = DATA / f"{STEM}.json"
    if args.refresh_source or not data_file.exists():
        arrays, provenance = import_saved(args.source_dir)
    else:
        with np.load(data_file, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        provenance = json.loads(metadata_file.read_text())["provenance"]
    paths = save_paper_dataset(STEM, category="numerical", arrays=arrays,
                               provenance=provenance)
    render(arrays, FIGURES)
    figure_provenance = {
        "dataset_sha256": {path.name: sha256(path) for path in paths},
        "generator_sha256": sha256(Path(__file__)),
        "color_limits": [0, 0.8],
        "map_sampling": "Original amplitude/detuning coordinates; nearest cells",
        "slice_metrics": provenance["slice_metrics"],
    }
    (FIGURES / "square_echo_comparison_provenance.json").write_text(
        json.dumps(figure_provenance, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(figure_provenance["slice_metrics"], indent=2))
    print(FIGURES / "square_echo_comparison.pdf")


if __name__ == "__main__":
    main()
