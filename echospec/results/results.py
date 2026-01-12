import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from numpy.typing import NDArray
from typing import Sequence
from matplotlib.figure import Figure

from echospec.analysis.fwhm import fwhm_gaussian_fit
from echospec.utils.parameters import Parameters
from echospec.plotting.spectroscopy import plot_spectroscopy_2d


@dataclass
class ResultsBase:
    data: NDArray[np.floating]

    OBS_INDEX = {"x": 0, "y": 1, "z": 2}

    def x(self) -> NDArray[np.floating]:
        return self.data[self.OBS_INDEX["x"]]

    def y(self) -> NDArray[np.floating]:
        return self.data[self.OBS_INDEX["y"]]

    def z(self) -> NDArray[np.floating]:
        return self.data[self.OBS_INDEX["z"]]

    @property
    def final_z(self) -> float:
        return float(self.z()[-1])

    def __post_init__(self):
        if self.data.shape[0] < 3:
            raise ValueError("First axis must be observables (x,y,z)")


@dataclass
class ResultsSingleRun(ResultsBase):
    data: NDArray[np.floating]   # (3, n_time)
    time: NDArray[np.floating]   # (n_time,)

    @property
    def final_z(self) -> float:
        return float(self.z()[-1])


@dataclass
class ResultsSpectroscopy1D(ResultsBase):
    data: NDArray[np.floating]        # (3, n_detuning, n_time)
    time: NDArray[np.floating]        # (n_time,)
    detunings: NDArray[np.floating]   # (n_detuning,)
    params: Parameters

    _fit_cache: Optional[Tuple[float, float]] = field(
        default=None, init=False, repr=False
    )

    # ---------- constructors ----------

    @classmethod
    def from_single_runs(
        cls,
        runs: list[ResultsSingleRun],
        detunings: NDArray[np.floating],
        # rabi_frequency: float,
        params: Parameters,
    ) -> "ResultsSpectroscopy1D":
        """
        Build spectroscopy result from a list of single-run results.
        """
        if len(runs) == 0:
            raise ValueError("No single-run results provided.")

        time = runs[0].time
        for r in runs:
            if not np.allclose(r.time, time):
                raise ValueError("All runs must share the same time axis.")

        data = np.stack([r.data for r in runs], axis=1)

        return cls(
            data=data,
            time=time,
            detunings=detunings,
            params=params,
        )

    @property
    def rabi_frequency(self) -> float:
        return self.params.rabi_frequency
    # ---------- analysis ----------

    def final_z_vs_detuning(self) -> NDArray[np.floating]:
        return self.z()[:, -1]

    def z_trace(self, i_detuning: int) -> NDArray[np.floating]:
        return self.z()[i_detuning]

    def _compute_gaussian_fit(self) -> Tuple[float, float]:
        """Run Gaussian fit once and cache (fwhm, snr)."""
        if self._fit_cache is None:
            z_final = self.final_z_vs_detuning()
            _, fwhm, snr, _ = fwhm_gaussian_fit(
                x=self.detunings,
                y=z_final,
            )
            self._fit_cache = (fwhm, snr)
        return self._fit_cache

    @property
    def fwhm(self) -> Optional[float]:
        return self._compute_gaussian_fit()[0]

    @property
    def snr(self) -> Optional[float]:
        return self._compute_gaussian_fit()[1]

    @property
    def populations(self) -> NDArray[np.floating]:
        return (1 - self.final_z_vs_detuning()) / 2

    # ---------- plotting ----------

    def plot(self):
        from echospec.plotting.spectroscopy import plot_spectroscopy

        fig = plot_spectroscopy(
            populations=self.populations,
            detunings=self.detunings,
            params=self.params,
            fwhm=self.fwhm,
            with_fwhm=True,
        )
        return fig


@dataclass
class ResultsSpectroscopy2D:
    spectroscopies: Sequence[ResultsSpectroscopy1D]
    amplitudes: NDArray[np.floating]   # (n_amp,)
    detunings: NDArray[np.floating]    # (n_detuning,)

    def __post_init__(self):
        if len(self.spectroscopies) != len(self.amplitudes):
            raise ValueError(
                "Each amplitude must correspond to one 1D spectroscopy."
            )

        # sanity check: all 1D results must match detunings
        for spec in self.spectroscopies:
            if not np.allclose(spec.detunings, self.detunings):
                raise ValueError(
                    "All 1D spectroscopies must share the same detunings."
                )

    # ---------- constructors ----------

    @classmethod
    def from_spectroscopy_1d(
        cls,
        spectroscopies: Sequence[ResultsSpectroscopy1D],
        amplitudes: NDArray[np.floating],

    ) -> "ResultsSpectroscopy2D":
        """
        Construct 2D spectroscopy from a list of 1D spectroscopy results.
        """
        if len(spectroscopies) == 0:
            raise ValueError("No 1D spectroscopy results provided.")

        detunings = spectroscopies[0].detunings

        return cls(
            spectroscopies=spectroscopies,
            amplitudes=amplitudes,
            detunings=detunings,
        )

    # ---------- accessors ----------

    def spectroscopy_1d(self, i: int) -> ResultsSpectroscopy1D:
        return self.spectroscopies[i]

    # ---------- analysis ----------

    def final_z_map(self) -> NDArray[np.floating]:
        """
        Final Z expectation value.

        Shape:
            (n_amp, n_detuning)
        """
        return np.stack(
            [spec.final_z_vs_detuning() for spec in self.spectroscopies],
            axis=0,
        )

    def plot(self) -> Figure:

        fig = plot_spectroscopy_2d(
            populations=self.populations,
            amplitudes=self.amplitudes,
            detunings=self.detunings,
            params=self.spectroscopies[0].params,
            fwhm=[spec.fwhm for spec in self.spectroscopies],
            with_fwhm=True,

        )
        return fig

    @property
    def populations(self) -> NDArray[np.floating]:
        """
        Final excited-state population.

        Shape:
            (n_amp, n_detuning)
        """
        return (1 - self.final_z_map()) / 2

    @property
    def fwhm_map(self) -> NDArray[np.floating]:
        """
        FWHM values for each amplitude.

        Shape:
            (n_amp,)
        """
        return np.array(
            [spec.fwhm if spec.fwhm is not None else np.nan for spec in self.spectroscopies]
        )

    @property
    def snr_map(self) -> NDArray[np.floating]:
        """
        SNR values for each amplitude.

        Shape:
            (n_amp,)
        """
        return np.array(
            [spec.snr if spec.snr is not None else np.nan for spec in self.spectroscopies]
        )


if __name__ == "__main__":
    detunings = np.linspace(-10, 10, 5)
    times = np.linspace(0, 1, 100)
    data = np.random.rand(3, len(detunings), len(times))
