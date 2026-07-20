"""Apply the forked OPX1000 FWHM pipeline to raw cutoff sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr

from .data import RawSweep, load_raw_sweep
from .opx1000_fwhm import add_gaussian_fwhm_analysis

MIN_FIT_R_SQUARED = 0.10
MIN_FIT_AMPLITUDE = 0.05
MAX_FIT_AMPLITUDE = 1.00
MIN_RESOLUTION_BINS = 2.0
MAX_FWHM_SWEEP_FRACTION = 0.50


@dataclass(frozen=True)
class CutoffAnalysis:
    """Raw sweep and OPX1000-derived linewidth metrics."""

    raw: RawSweep
    dataset: xr.Dataset

    @property
    def fwhm_hz(self) -> np.ndarray:
        return np.asarray(self.dataset.gaussian_fwhm_hz.sel(qubit=self.raw.qubit))

    @property
    def center_hz(self) -> np.ndarray:
        return np.asarray(self.dataset.gaussian_center_hz.sel(qubit=self.raw.qubit))

    @property
    def fit_amplitude(self) -> np.ndarray:
        return np.asarray(
            self.dataset.gaussian_fit_abs_amplitude.sel(qubit=self.raw.qubit)
        )

    @property
    def fit_r_squared(self) -> np.ndarray:
        return np.asarray(self.dataset.gaussian_fit_r_squared.sel(qubit=self.raw.qubit))

    @property
    def valid(self) -> np.ndarray:
        frequency_bin_hz = float(np.median(np.diff(self.raw.detuning_hz)))
        span_hz = float(np.ptp(self.raw.detuning_hz))
        finite = (
            np.isfinite(self.fwhm_hz)
            & np.isfinite(self.center_hz)
            & np.isfinite(self.fit_amplitude)
            & np.isfinite(self.fit_r_squared)
        )
        return (
            finite
            & (self.fit_amplitude >= MIN_FIT_AMPLITUDE)
            & (self.fit_amplitude <= MAX_FIT_AMPLITUDE)
            & (self.fit_r_squared >= MIN_FIT_R_SQUARED)
            & (self.fwhm_hz >= MIN_RESOLUTION_BINS * frequency_bin_hz)
            & (self.fwhm_hz <= MAX_FWHM_SWEEP_FRACTION * span_hz)
            & (np.abs(self.center_hz) <= 0.5 * span_hz)
        )

    @property
    def resolution(self) -> np.ndarray:
        return np.divide(
            self.raw.t2_limit_hz,
            self.fwhm_hz,
            out=np.full_like(self.fwhm_hz, np.nan),
            where=self.fwhm_hz > 0,
        )

    @property
    def signal_resolution(self) -> np.ndarray:
        return self.fit_amplitude * self.resolution


def analyze_sweep(raw: RawSweep) -> CutoffAnalysis:
    """Recalculate all amplitude FWHMs with the OPX1000 fitting code."""
    dataset = xr.Dataset(
        {
            "state": (
                ("qubit", "detuning", "amp_prefactor"),
                raw.state[np.newaxis, :, :],
            )
        },
        coords={
            "qubit": [raw.qubit],
            "detuning": raw.detuning_hz,
            "amp_prefactor": raw.amp_prefactor,
            "t2_star_fwhm_limit_hz": raw.t2_limit_hz,
        },
        attrs={"raw_source": str(raw.run_dir)},
    )
    fitted = add_gaussian_fwhm_analysis(
        dataset,
        use_state_discrimination=True,
    )
    return CutoffAnalysis(raw=raw, dataset=fitted)


@lru_cache(maxsize=32)
def analyze_run(run_dir: str | Path) -> CutoffAnalysis:
    """Load and analyze one run, caching it for responsive cutoff switching."""
    return analyze_sweep(load_raw_sweep(Path(run_dir)))


def best_amplitude_index(analysis: CutoffAnalysis) -> int:
    """Choose the accepted point with maximum signal-times-resolution."""
    score = np.where(analysis.valid, analysis.signal_resolution, -np.inf)
    if np.isfinite(score).any():
        return int(np.argmax(score))
    return len(analysis.raw.amp_prefactor) // 2
