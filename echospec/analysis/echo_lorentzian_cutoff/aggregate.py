"""Load the retained July 12 aggregate cutoff campaign."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import opx1000_data_dir

DEFAULT_AGGREGATE_RELATIVE_PATH = Path(
    "cutoff_amp_fwhm_map/20260712_171809_single_cutoff_full"
)
MIN_FIT_R_SQUARED = 0.60
MIN_FIT_AMPLITUDE = 0.05
MAX_FIT_AMPLITUDE = 1.00
MIN_RESOLUTION_BINS = 2.0
MAX_FWHM_SWEEP_FRACTION = 0.50


@dataclass(frozen=True)
class AggregateCampaign:
    """A completed aggregate campaign with stored OPX fits and rendered maps."""

    directory: Path
    manifest: Mapping[str, object]
    results: pd.DataFrame
    cutoffs: tuple[float, ...]
    figures: Mapping[float, Path]

    @property
    def parameters(self) -> Mapping[str, object]:
        return self.manifest["base_parameters"]


def load_aggregate_campaign(
    data_dir: Path | str | None = None,
    *,
    relative_path: Path | str = DEFAULT_AGGREGATE_RELATIVE_PATH,
) -> AggregateCampaign:
    """Load the exact retained campaign requested by the paper analysis."""
    root = Path(data_dir).expanduser() if data_dir else opx1000_data_dir()
    directory = root / Path(relative_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    results = pd.read_csv(directory / "cutoff_amp_fwhm_map_fit_results.csv")

    if not manifest.get("complete") or manifest.get("failures"):
        raise ValueError(f"Campaign is not complete and failure-free: {directory}")
    cutoffs = tuple(float(run["cutoff"]) for run in manifest["runs"])
    if int(manifest["completed_runs"]) != len(cutoffs):
        raise ValueError("Manifest completed-run count does not match its run list.")
    if results["cutoff"].nunique() != len(cutoffs):
        raise ValueError("Fit table cutoff count does not match the manifest.")

    figures: dict[float, Path] = {}
    for run in manifest["runs"]:
        cutoff = float(run["cutoff"])
        attempt_name = str(run["output_dir"]).replace("\\", "/").split("/")[-1]
        candidates = sorted(
            (directory / attempt_name / "individual_figures").glob("*.png")
        )
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one individual figure for cutoff {cutoff}, found {candidates}"
            )
        figures[cutoff] = candidates[0]

    return AggregateCampaign(
        directory=directory,
        manifest=manifest,
        results=results,
        cutoffs=cutoffs,
        figures=figures,
    )


def aggregate_quality_mask(campaign: AggregateCampaign) -> pd.Series:
    """Apply the paper-quality gate to stored OPX1000 fit records."""
    data = campaign.results
    parameters = campaign.parameters
    step_hz = float(parameters["frequency_step_in_mhz"]) * 1e6
    span_hz = float(parameters["frequency_span_in_mhz"]) * 1e6
    required = [
        "gaussian_center_hz",
        "fwhm_hz",
        "fwhm_t2_units",
        "fit_abs_amplitude",
        "fit_r_squared",
    ]
    finite = np.isfinite(data[required]).all(axis=1)
    return (
        finite
        & data["fit_abs_amplitude"].between(MIN_FIT_AMPLITUDE, MAX_FIT_AMPLITUDE)
        & (data["fit_r_squared"] >= MIN_FIT_R_SQUARED)
        & data["fwhm_hz"].between(
            MIN_RESOLUTION_BINS * step_hz,
            MAX_FWHM_SWEEP_FRACTION * span_hz,
        )
        & (data["gaussian_center_hz"].abs() <= 0.5 * span_hz)
    )


def selected_cutoff_rows(
    campaign: AggregateCampaign,
    cutoff: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return one cutoff's rows and aligned acceptance mask."""
    selected = campaign.results[np.isclose(campaign.results["cutoff"], cutoff)].copy()
    accepted = aggregate_quality_mask(campaign).loc[selected.index]
    return selected.reset_index(drop=True), accepted.reset_index(drop=True)
