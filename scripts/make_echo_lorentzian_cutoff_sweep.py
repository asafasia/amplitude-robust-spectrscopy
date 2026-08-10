from __future__ import annotations

# Backend and local-source setup must precede pyplot and echospec imports.
# ruff: noqa: E402, I001

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ars-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echospec.figures import FigureVariant, apply_figure_style, save_figure
from echospec.paper_data import save_paper_dataset


DATA_ROOT = Path(
    os.environ.get("OPX1000_DATA_DIR", ROOT.parent / "data_opx1000")
).expanduser()
OUTPUT_STEM = "08_echo_lorentzian_cutoff_sweep"

# A retained linewidth must describe a resolved, bounded probability feature.
MIN_FIT_R_SQUARED = 0.60
MIN_FIT_AMPLITUDE = 0.05
MAX_FIT_AMPLITUDE = 1.00
MIN_RESOLUTION_BINS = 2.0
MAX_FWHM_SWEEP_FRACTION = 0.50


@dataclass(frozen=True)
class Campaign:
    key: str
    title: str
    relative_directory: Path
    results_filename: str
    expected_cutoffs: int
    frequency_span_mhz: float
    frequency_step_mhz: float
    expected_shots: int

    @property
    def directory(self) -> Path:
        return DATA_ROOT / self.relative_directory


CAMPAIGNS = (
    Campaign(
        key="broad_survey",
        title="Broad survey",
        relative_directory=Path(
            "cutoff_amp_fwhm_map/20260712_171809_single_cutoff_full"
        ),
        results_filename="cutoff_amp_fwhm_map_fit_results.csv",
        expected_cutoffs=20,
        frequency_span_mhz=5.0,
        frequency_step_mhz=0.005,
        expected_shots=60,
    ),
    Campaign(
        key="focused_sweep",
        title="Focused sweep",
        relative_directory=Path(
            "echo_lorentzian_cutoff_sweep/medium_cutoff/"
            "20260711_mid_001_to_01_10cutoffs_10mhz_400freq/domain_10mhz"
        ),
        results_filename="cutoff_sweep_fit_results.csv",
        expected_cutoffs=10,
        frequency_span_mhz=10.0,
        frequency_step_mhz=0.02506265664160401,
        expected_shots=100,
    ),
)


def load_campaign(campaign: Campaign) -> tuple[pd.DataFrame, dict[str, object]]:
    directory = campaign.directory
    manifest_path = directory / "manifest.json"
    results_path = directory / campaign.results_filename
    missing = [path for path in (manifest_path, results_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing campaign inputs: {missing}")

    manifest = json.loads(manifest_path.read_text())
    parameters = manifest["base_parameters"]
    checks = {
        "echo": parameters["echo"] is True,
        "experimental": parameters["simulate"] is False,
        "pulse_length": int(parameters["lorentzian_length_in_ns"]) == 20_000,
        "pulse_shape": parameters["pulse_shape"] == "root_lorentzian",
        "peak_amplitude": np.isclose(
            float(parameters["lorentzian_peak_amplitude"]), 0.2
        ),
        "shots": int(parameters["num_shots"]) == campaign.expected_shots,
        "frequency_span": np.isclose(
            float(parameters["frequency_span_in_mhz"]),
            campaign.frequency_span_mhz,
        ),
        "frequency_step": np.isclose(
            float(parameters["frequency_step_in_mhz"]),
            campaign.frequency_step_mhz,
        ),
        "completed_runs": int(manifest["completed_runs"])
        == campaign.expected_cutoffs,
    }
    if manifest.get("complete") is False or manifest.get("interrupted") is True:
        checks["complete"] = False
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{campaign.key}: failed manifest checks: {failed}")

    data = pd.read_csv(results_path)
    required_columns = {
        "cutoff",
        "qubit",
        "rabi_frequency_mhz",
        "gaussian_center_hz",
        "fwhm_hz",
        "fwhm_t2_units",
        "fit_abs_amplitude",
        "fit_r_squared",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"{campaign.key}: missing result columns {sorted(missing_columns)}"
        )
    if set(data["qubit"].dropna().unique()) != {"q1"}:
        raise ValueError(f"{campaign.key}: expected only q1 data")
    if data["cutoff"].nunique() != campaign.expected_cutoffs:
        raise ValueError(
            f"{campaign.key}: expected {campaign.expected_cutoffs} cutoffs, "
            f"found {data['cutoff'].nunique()}"
        )
    if data.duplicated(["cutoff", "rabi_frequency_mhz"]).any():
        raise ValueError(f"{campaign.key}: duplicate cutoff/amplitude rows")
    return data, manifest


def quality_mask(data: pd.DataFrame, campaign: Campaign) -> pd.Series:
    min_fwhm_hz = MIN_RESOLUTION_BINS * campaign.frequency_step_mhz * 1e6
    max_fwhm_hz = (
        MAX_FWHM_SWEEP_FRACTION * campaign.frequency_span_mhz * 1e6
    )
    half_span_hz = 0.5 * campaign.frequency_span_mhz * 1e6
    finite = np.isfinite(
        data[
            [
                "gaussian_center_hz",
                "fwhm_hz",
                "fwhm_t2_units",
                "fit_abs_amplitude",
                "fit_r_squared",
            ]
        ]
    ).all(axis=1)
    return (
        finite
        & data["fit_abs_amplitude"].between(
            MIN_FIT_AMPLITUDE, MAX_FIT_AMPLITUDE
        )
        & (data["fit_r_squared"] >= MIN_FIT_R_SQUARED)
        & data["fwhm_hz"].between(min_fwhm_hz, max_fwhm_hz)
        & (data["gaussian_center_hz"].abs() <= half_span_hz)
    )


def centers_to_edges(values: np.ndarray, *, logarithmic: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(np.diff(values) <= 0):
        raise ValueError("Grid centers must be a strictly increasing 1D array")
    if logarithmic:
        if np.any(values <= 0):
            raise ValueError("Logarithmic grid centers must be positive")
        transformed = np.log(values)
        midpoints = 0.5 * (transformed[:-1] + transformed[1:])
        edges = np.r_[
            transformed[0] - (midpoints[0] - transformed[0]),
            midpoints,
            transformed[-1] + (transformed[-1] - midpoints[-1]),
        ]
        return np.exp(edges)
    midpoints = 0.5 * (values[:-1] + values[1:])
    return np.r_[
        values[0] - (midpoints[0] - values[0]),
        midpoints,
        values[-1] + (values[-1] - midpoints[-1]),
    ]


def map_matrix(
    data: pd.DataFrame, accepted: pd.Series, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    screened = data.copy()
    resolution = 1.0 / screened["fwhm_t2_units"]
    if metric == "resolution":
        screened["metric"] = resolution
    elif metric == "signal_weighted_resolution":
        screened["metric"] = screened["fit_abs_amplitude"] * resolution
    else:
        raise ValueError(f"Unknown map metric: {metric}")
    screened.loc[~accepted, "metric"] = np.nan
    matrix = screened.pivot(
        index="rabi_frequency_mhz",
        columns="cutoff",
        values="metric",
    ).sort_index(axis=0).sort_index(axis=1)
    return (
        matrix.columns.to_numpy(dtype=float),
        matrix.index.to_numpy(dtype=float),
        matrix.to_numpy(dtype=float),
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.97,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        color="black",
    )


def main() -> None:
    apply_figure_style(FigureVariant.PAPER)
    loaded = [load_campaign(campaign) for campaign in CAMPAIGNS]
    datasets = []
    for campaign, (data, manifest) in zip(CAMPAIGNS, loaded, strict=True):
        accepted = quality_mask(data, campaign)
        datasets.append((campaign, data, accepted, manifest))

    t2_values = np.concatenate(
        [data["t2_s"].dropna().unique() for _, data, _, _ in datasets]
    )
    if not np.allclose(t2_values, t2_values[0]):
        raise ValueError(f"Campaigns use different T2 references: {t2_values}")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.0, 4.8),
        sharex="col",
        sharey=True,
        constrained_layout=True,
    )

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e7e7e7")
    metrics = (
        (
            "resolution",
            LogNorm(vmin=0.1, vmax=1.0),
            r"Resolution $\mathcal{R}=\frac{1/(\pi T_2)}{\mathrm{FWHM}}$",
        ),
        (
            "signal_weighted_resolution",
            LogNorm(vmin=0.1, vmax=1.0),
            r"Signal-weighted resolution $|A_{\rm fit}|\mathcal{R}$",
        ),
    )
    row_meshes = []
    panel = 0
    for row, (metric, norm, colorbar_label) in enumerate(metrics):
        row_mesh = None
        for column, (campaign, data, accepted, _) in enumerate(datasets):
            ax = axes[row, column]
            cutoffs, rabi_mhz, values = map_matrix(data, accepted, metric)
            row_mesh = ax.pcolormesh(
                centers_to_edges(cutoffs, logarithmic=True),
                centers_to_edges(rabi_mhz, logarithmic=False),
                np.ma.masked_invalid(values),
                shading="flat",
                cmap=cmap,
                norm=norm,
                rasterized=True,
            )
            ax.set_xscale("log")
            ax.set_xlim(cutoffs.min(), cutoffs.max())
            ax.set_ylim(0.0, 16.6)
            if row == 0:
                ax.set_title(campaign.title)
            if row == 1:
                ax.set_xlabel(r"Cutoff $c$")
            if column == 0:
                ax.set_ylabel(
                    r"$\Omega_0/2\pi$ (MHz)"
                )
            add_panel_label(ax, f"({chr(ord('a') + panel)})")
            panel += 1
        if row_mesh is None:
            raise RuntimeError(f"No map was plotted for {metric}")
        row_meshes.append((row_mesh, colorbar_label))

    for row, (row_mesh, colorbar_label) in enumerate(row_meshes):
        colorbar = fig.colorbar(
            row_mesh,
            ax=axes[row, :],
            pad=0.015,
            aspect=24,
        )
        colorbar.set_label(colorbar_label)

    saved = save_figure(
        fig,
        OUTPUT_STEM,
        variant=FigureVariant.PAPER,
        formats=("pdf", "png", "svg"),
        dpi=300,
    )
    plt.close(fig)

    provenance = {
        "output_stem": OUTPUT_STEM,
        "t2_s": float(t2_values[0]),
        "quality_screen": {
            "minimum_fit_r_squared": MIN_FIT_R_SQUARED,
            "fit_abs_amplitude_range": [
                MIN_FIT_AMPLITUDE,
                MAX_FIT_AMPLITUDE,
            ],
            "minimum_fwhm_frequency_bins": MIN_RESOLUTION_BINS,
            "maximum_fwhm_fraction_of_sweep_span": MAX_FWHM_SWEEP_FRACTION,
            "fit_center_required_inside_sweep": True,
        },
        "map_metrics": {
            "resolution": "[1/(pi*T2)]/FWHM",
            "signal_weighted_resolution":
                "fit_abs_amplitude * [1/(pi*T2)]/FWHM",
            "color_normalization": "logarithmic",
            "color_limits": [0.1, 1.0],
        },
        "campaigns": [],
    }
    for campaign, data, accepted, manifest in datasets:
        provenance["campaigns"].append(
            {
                "key": campaign.key,
                "source_directory": str(campaign.directory),
                "source_results": str(
                    campaign.directory / campaign.results_filename
                ),
                "run_started_at": manifest.get("run_started_at"),
                "run_finished_at": manifest.get("run_finished_at"),
                "shots_per_point": campaign.expected_shots,
                "frequency_span_mhz": campaign.frequency_span_mhz,
                "frequency_step_mhz": campaign.frequency_step_mhz,
                "cutoff_count": int(data["cutoff"].nunique()),
                "amplitude_count": int(data["rabi_frequency_mhz"].nunique()),
                "fit_count": int(len(data)),
                "accepted_fit_count": int(accepted.sum()),
                "accepted_fit_fraction": float(accepted.mean()),
            }
        )
    provenance_path = ROOT / "figures" / "paper" / f"{OUTPUT_STEM}_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    paper_arrays: dict[str, np.ndarray] = {
        "t2_s": np.asarray(float(t2_values[0])),
    }
    portable_campaigns = []
    exported_columns = (
        "cutoff",
        "rabi_frequency_mhz",
        "gaussian_center_hz",
        "fwhm_hz",
        "fwhm_t2_units",
        "fit_abs_amplitude",
        "fit_r_squared",
    )
    for campaign, data, accepted, manifest in datasets:
        for column in exported_columns:
            paper_arrays[f"{campaign.key}_{column}"] = data[column].to_numpy(
                dtype=float
            )
        paper_arrays[f"{campaign.key}_accepted"] = accepted.to_numpy(
            dtype=bool
        )
        for metric in ("resolution", "signal_weighted_resolution"):
            cutoffs, rabi_mhz, values = map_matrix(data, accepted, metric)
            paper_arrays[f"{campaign.key}_{metric}_cutoff"] = cutoffs
            paper_arrays[f"{campaign.key}_{metric}_rabi_mhz"] = rabi_mhz
            paper_arrays[f"{campaign.key}_{metric}"] = values
        portable_campaigns.append(
            {
                "key": campaign.key,
                "source_directory": str(campaign.relative_directory),
                "source_results": campaign.results_filename,
                "run_started_at": manifest.get("run_started_at"),
                "run_finished_at": manifest.get("run_finished_at"),
                "shots_per_point": campaign.expected_shots,
                "frequency_span_mhz": campaign.frequency_span_mhz,
                "frequency_step_mhz": campaign.frequency_step_mhz,
                "cutoff_count": int(data["cutoff"].nunique()),
                "amplitude_count": int(data["rabi_frequency_mhz"].nunique()),
                "fit_count": int(len(data)),
                "accepted_fit_count": int(accepted.sum()),
            }
        )
    paper_data_paths = save_paper_dataset(
        OUTPUT_STEM,
        category="experimental",
        arrays=paper_arrays,
        provenance={
            "figure_asset": f"figures/paper/{OUTPUT_STEM}.pdf",
            "manuscript_scope": "supplemental",
            "generator": "scripts/make_echo_lorentzian_cutoff_sweep.py",
            "reproduction_command": (
                "PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python "
                "scripts/make_echo_lorentzian_cutoff_sweep.py"
            ),
            "source_root_environment": "OPX1000_DATA_DIR",
            "measurement": {
                "qubit": "q1",
                "pulse_shape": "echo-root-Lorentzian",
                "duration_us": 20.0,
                "peak_waveform_amplitude": 0.2,
            },
            "quality_screen": provenance["quality_screen"],
            "map_metrics": provenance["map_metrics"],
            "campaigns": portable_campaigns,
            "dimension_conventions": {
                "raw_fit_columns": (
                    "campaign-prefixed fit columns and accepted masks share "
                    "the fit_row axis"
                ),
                "map_values": (
                    "campaign-prefixed resolution maps have axes "
                    "[rabi_mhz, cutoff]"
                ),
                "map_coordinates": (
                    "the matching *_rabi_mhz and *_cutoff arrays are the map "
                    "axis coordinates"
                ),
            },
        },
    )

    print(f"T2 reference: {t2_values[0] * 1e6:.6f} us")
    for campaign, data, accepted, _ in datasets:
        print(
            f"{campaign.title}: retained {accepted.sum()}/{len(data)} fits "
            f"({accepted.mean():.1%})"
        )
    for path in (*saved, provenance_path, *paper_data_paths):
        print(path)


if __name__ == "__main__":
    main()
