"""Experiments module for amplitude-robust spectroscopy."""

from echospec.experiments.spectroscopy import Spectroscopy, OptionsSpectroscopy, ResultsSpectroscopy
from echospec.experiments.spectroscopy_vs_amplitude import AmplitudeSweepSpectroscopy, OptionsSpectroscopy2d

__all__ = [
    "Spectroscopy",
    "OptionsSpectroscopy",
    "ResultsSpectroscopy",
    "AmplitudeSweepSpectroscopy",
    "OptionsSpectroscopy2d",
]
