"""Experiments module for amplitude-robust spectroscopy."""

from echospec.experiments.spectroscopy import Spectroscopy, OptionsSpectroscopy, ResultsSpectroscopy1D
from echospec.experiments.spectroscopy_vs_amplitude import AmplitudeSweepSpectroscopy, OptionsSpectroscopy2d

__all__ = [
    "Spectroscopy",
    "OptionsSpectroscopy",
    "ResultsSpectroscopy1D",
    "AmplitudeSweepSpectroscopy",
    "OptionsSpectroscopy2d",
]
