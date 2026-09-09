"""Render measured Figure 4, retaining the historical asset filename.

Render from the experimental paper cache by default. To reimport measurements,
pass --source-data-dir /path/to/opx1000-codes/data or set OPX1000_DATA_DIR.
"""

from __future__ import annotations

# ruff: noqa: E402, I001
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echospec.figures import FigureVariant, apply_figure_style, save_figure
from echospec.paper_data import save_paper_dataset

OUTPUT_STEM = "04_main_ac_stark_correction_maps"
CACHE_PATH = ROOT / "paper/data/experimental" / f"{OUTPUT_STEM}.npz"
DISPLAY_RABI_LIMITS_MHZ = (10.0, 60.0)
DISPLAY_DETUNING_HALF_WIDTH_MHZ = 0.15
CAMPAIGN = Path("drag_beta_kappa_calibration/2026-09-05_20-37-12")
RUNS = (
    ("plain", "beta_0", "calibrations/2026-09-05/drag_kappa_joint_01/22-25-01-799230"),
    (
        "drag",
        "drag_beta_-0.22",
        "calibrations/2026-09-06/drag_kappa_joint_02/00-12-36-749691",
    ),
)


def load_source(data_dir: Path) -> tuple[dict, dict]:
    """Read selected populations and saved fits, recording source checksums."""
    source_hashes = {}

    def source_path(relative: Path) -> Path:
        path = data_dir / relative
        source_hashes[relative.as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        return path

    plan = json.loads(source_path(CAMPAIGN / "approved_plan.json").read_text())
    records = {
        row["label"]: row
        for row in json.loads(source_path(CAMPAIGN / "records.json").read_text())
    }
    arrays = {}
    pulses = {}
    for prefix, label, relative in RUNS:
        run = Path(relative)
        metadata = json.loads(source_path(run / "metadata.json").read_text())
        parameters = json.loads(source_path(run / "parameters.json").read_text())
        pulse = metadata["pulse"]
        record = records[label]
        expected_beta = 0.0 if prefix == "plain" else -0.22
        expected_transition = 0.0 if prefix == "plain" else 16.0
        expected = {
            "pulse_shape": "root_lorentzian",
            "echo": True,
            "lorentzian_length_in_ns": 50000,
            "cutoff": 0.00075,
            "drag_beta": expected_beta,
            "stark_kappa_mhz_inv": 0.0,
            "ac_stark_correction": False,
            "applied_echo_transition_time_ns": expected_transition,
            "three_state_discrimination_available": False,
        }
        for key, value in expected.items():
            if pulse[key] != value:
                raise ValueError(f"{label}: unexpected {key}: {pulse[key]}")
        if parameters["num_shots"] != 2000 or record["drag_beta"] != expected_beta:
            raise ValueError(f"{label}: shot count or record beta mismatch")
        with np.load(source_path(run / "sweep.npz"), allow_pickle=False) as sweep:
            if sweep["qubit"].tolist() != ["q6"]:
                raise ValueError("Figure 4 requires the q6 measurement")
            detuning = np.asarray(sweep["detuning"], dtype=float) / 1e6
            amplitude = np.asarray(sweep["amp_prefactor"], dtype=float)
        with np.load(source_path(run / "results.npz"), allow_pickle=False) as result:
            # Source plotter specifies qubit, detuning, amplitude order.
            state = np.asarray(result["state"], dtype=float)
            if state.shape != (1, detuning.size, amplitude.size):
                raise ValueError(f"Unexpected state shape: {state.shape}")
            arrays[f"{prefix}_pe"] = state[0].T
        rabi = np.asarray(record["rabi_frequency_mhz"], dtype=float)
        if prefix == "plain":
            arrays.update(
                detuning_mhz=detuning,
                rabi_mhz=rabi,
                amplitude_v=amplitude * pulse["lorentzian_peak_amplitude"],
            )
        else:
            np.testing.assert_array_equal(detuning, arrays["detuning_mhz"])
            np.testing.assert_array_equal(rabi, arrays["rabi_mhz"])
            np.testing.assert_array_equal(
                amplitude * pulse["lorentzian_peak_amplitude"], arrays["amplitude_v"]
            )
        arrays[f"{prefix}_centers_mhz"] = (
            np.asarray(record["center_hz_vs_amplitude"]) / 1e6
        )
        arrays[f"{prefix}_fit_accepted"] = np.asarray(
            record["center_fit_accepted"], dtype=bool
        )
        for key in ("center_rms_hz", "weighted_center_hz", "spectroscopy_contrast"):
            arrays[f"{prefix}_{key}"] = np.asarray(record[key])
        pulses[prefix] = pulse
    arrays.update(
        duration_us=np.asarray(50.0),
        cutoff=np.asarray(0.00075),
        drag_beta=np.asarray([0.0, -0.22]),
        kappa_mhz_inv=np.zeros(2),
        applied_echo_transition_ns=np.asarray([0.0, 16.0]),
        num_shots=np.asarray(2000),
    )
    provenance = {
        "figure_asset": f"figures/paper/{OUTPUT_STEM}.pdf",
        "manuscript_scope": "letter",
        "figure_number": 4,
        "generator": "scripts/make_main_ac_stark_correction_maps.py",
        "reproduction_command": "python scripts/make_main_ac_stark_correction_maps.py",
        "source_import_command": "python scripts/make_main_ac_stark_correction_maps.py --source-data-dir /path/to/opx1000-codes/data",
        "source_root": "OPX1000_DATA_DIR (default sibling opx1000-codes/data)",
        "source_campaign": CAMPAIGN.as_posix(),
        "source_sha256": source_hashes,
        "approved_plan": plan,
        "applied_pulse_metadata": pulses,
        "population_definition": "Saved two-state discriminated state average; no readout rescaling",
        "detuning_convention": "Unshifted acquisition detuning relative to the configured q6 reference",
        "rabi_axis": "Saved calibrated nominal Rabi frequency; no waveform-peak renormalization",
        "fit_source": "Campaign records.json; no refitting or extra amplitude selection",
        "fit_acceptance": "Finite center and positive finite error and contrast; R squared >= 0.1",
        "center_rms_definition": "Unweighted RMS of accepted centers about their inverse-variance-weighted mean; not RMS about zero",
        "limitations": "Sequential runs; midpoint smoothing and beta both change; no resolved P_f measurement",
        "display": {"population_limits": [0.0, 0.6], "frequency_alignment_mhz": 0.0},
        "array_dimensions": {
            "plain_pe": ["rabi_mhz", "detuning_mhz"],
            "drag_pe": ["rabi_mhz", "detuning_mhz"],
        },
    }
    return arrays, provenance


def validate_arrays(data: dict) -> None:
    """Reject mismatched grids, invalid populations, or fit array lengths."""
    for key in ("detuning_mhz", "rabi_mhz"):
        if data[key].shape != (200,) or not np.all(np.diff(data[key]) > 0):
            raise ValueError(f"Invalid {key} grid")
    for prefix, _, _ in RUNS:
        pe = data[f"{prefix}_pe"]
        if pe.shape != (200, 200) or not np.all(np.isfinite(pe)):
            raise ValueError(f"Invalid {prefix} population map")
        if np.any((pe < 0) | (pe > 1)):
            raise ValueError("Measured state average lies outside [0, 1]")
        for suffix in ("centers_mhz", "fit_accepted"):
            if data[f"{prefix}_{suffix}"].shape != (200,):
                raise ValueError(f"Invalid {prefix} fit array")


def center_display_data(data: dict, provenance: dict) -> None:
    """Subtract each run's arithmetic mean accepted center over the shown range.

    Always derive display coordinates from acquisition coordinates, so repeated
    cache renders cannot accumulate shifts. Retain the original arrays intact.
    """
    rabi = data["rabi_mhz"]
    selected = (rabi >= DISPLAY_RABI_LIMITS_MHZ[0]) & (
        rabi <= DISPLAY_RABI_LIMITS_MHZ[1]
    )
    data["display_rabi_mask"] = selected
    offsets = {}
    for prefix, _, _ in RUNS:
        centers = data[f"{prefix}_centers_mhz"]
        accepted = selected & data[f"{prefix}_fit_accepted"] & np.isfinite(centers)
        if not np.any(accepted):
            raise ValueError(f"No accepted {prefix} centers in the display range")
        mean = float(np.mean(centers[accepted]))
        offsets[prefix] = mean
        data[f"{prefix}_display_center_mask"] = accepted
        data[f"{prefix}_display_mean_center_mhz"] = np.asarray(mean)
        data[f"{prefix}_display_detuning_mhz"] = data["detuning_mhz"] - mean
        data[f"{prefix}_display_centers_mhz"] = centers - mean
        data[f"{prefix}_display_center_rms_hz"] = np.asarray(
            1e6 * np.sqrt(np.mean((centers[accepted] - mean) ** 2))
        )
    provenance["display"] = {
        "population_limits": [0.0, 0.6],
        "rabi_limits_mhz": list(DISPLAY_RABI_LIMITS_MHZ),
        "detuning_limits_mhz": [
            -DISPLAY_DETUNING_HALF_WIDTH_MHZ,
            DISPLAY_DETUNING_HALF_WIDTH_MHZ,
        ],
        "subtracted_mean_center_mhz": offsets,
        "alignment": "One constant per run: arithmetic mean of finite accepted fitted centers over 10 <= calibrated Rabi frequency <= 60 MHz",
        "population_processing": "No interpolation or rescaling; original populations retained",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path)
    args = parser.parse_args()
    source = args.source_data_dir or os.environ.get("OPX1000_DATA_DIR")
    if source or not CACHE_PATH.exists():
        data_dir = Path(source) if source else ROOT.parent / "opx1000-codes/data"
        data, provenance = load_source(data_dir)
    else:
        with np.load(CACHE_PATH, allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        provenance = json.loads(CACHE_PATH.with_suffix(".json").read_text())[
            "provenance"
        ]
    validate_arrays(data)
    center_display_data(data, provenance)
    displayed_populations = [
        data[f"{prefix}_pe"][
            np.ix_(
                data["display_rabi_mask"],
                np.abs(data[f"{prefix}_display_detuning_mhz"])
                <= DISPLAY_DETUNING_HALF_WIDTH_MHZ,
            )
        ]
        for prefix, _, _ in RUNS
    ]
    population_min = min(float(values.min()) for values in displayed_populations)
    population_max = max(float(values.max()) for values in displayed_populations)
    vmin = float(np.floor(population_min * 10) / 10)
    vmax = float(np.ceil(population_max * 10) / 10)
    if vmax <= vmin:
        vmax = vmin + 0.1
    data["display_population_limits"] = np.asarray([vmin, vmax])
    provenance["display"].update(
        population_limits=[vmin, vmax],
        measured_population_extrema=[population_min, population_max],
        population_limit_rule="Shared extrema of both maps within the displayed Rabi and centered-detuning ranges; round minimum down and maximum up to multiples of 0.1",
    )
    paper_paths = save_paper_dataset(
        OUTPUT_STEM, category="experimental", arrays=data, provenance=provenance
    )
    apply_figure_style(FigureVariant.PAPER)
    fig = plt.figure(figsize=(3.35, 1.85), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.045), wspace=0.08)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    for axis, prefix, title in zip(
        axes, ("plain", "drag"), (r"(a) $\beta=0$", r"(b) $\beta=-0.22$"), strict=True
    ):
        plot = axis.pcolormesh(
            data[f"{prefix}_display_detuning_mhz"],
            data["rabi_mhz"],
            data[f"{prefix}_pe"],
            shading="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        centers = np.where(
            data[f"{prefix}_display_center_mask"],
            data[f"{prefix}_display_centers_mhz"],
            np.nan,
        )
        (line,) = axis.plot(centers, data["rabi_mhz"], color="white", lw=0.65)
        line.set_path_effects(
            [
                path_effects.Stroke(linewidth=1.1, foreground="black"),
                path_effects.Normal(),
            ]
        )
        axis.axvline(0, color="white", ls="--", lw=0.55)
        axis.set(
            title=title,
            xlabel="Centered detuning (MHz)",
            xlim=(-DISPLAY_DETUNING_HALF_WIDTH_MHZ, DISPLAY_DETUNING_HALF_WIDTH_MHZ),
            ylim=DISPLAY_RABI_LIMITS_MHZ,
            xticks=[-0.1, 0, 0.1],
            yticks=[10, 20, 30, 40, 50, 60],
        )
    axes[0].set_ylabel(r"$\Omega_{\mathrm{cal}}/2\pi$ (MHz)")
    axes[1].tick_params(labelleft=False)
    fig.colorbar(plot, cax=fig.add_subplot(grid[0, 2]), label=r"$P_e$")
    paths = save_figure(
        fig,
        OUTPUT_STEM,
        variant=FigureVariant.PAPER,
        formats=("pdf", "png", "svg"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    for path in paths:
        if path.suffix == ".svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    for path in (*paths, *paper_paths):
        print(path)


if __name__ == "__main__":
    main()
