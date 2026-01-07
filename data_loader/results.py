from dataclasses import dataclass
from typing import Any
import numpy as np
from echospec.analysis.fwhm import fwhm_gaussian_fit


@dataclass
class SpectroscopyResult:
    meta: Any
    raw: dict

    def detuning(self) -> np.ndarray:
        return self.raw["detuning"]

    def signal(self) -> np.ndarray:
        return self.raw["signal"]

    def slice_at_amplitude(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (detuning, signal_slice)
        """
        return self.detuning(), self.signal()[idx]

    def fwhm(self, idx: int) -> float:  # type: ignore
        det, sig = self.slice_at_amplitude(idx)
        # return fwhm_gaussian_fit(det, sig)
        pass
