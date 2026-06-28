"""Experiments module for amplitude-robust spectroscopy."""

from echospec.experiments.spectroscopy import Spectroscopy, OptionsSpectroscopy, ResultsSpectroscopy1D
from echospec.experiments.spectroscopy_vs_amplitude import AmplitudeSweepSpectroscopy, OptionsSpectroscopy2d
from echospec.experiments.lorentzian_rabi_amplitude import (
    LorentzianRabiAmplitudeExperiment,
    LorentzianRabiAmplitudeSweepExperiment,
    OptionsLorentzianRabiAmplitude,
    ResultsLorentzianRabiAmplitude,
    ResultsLorentzianRabiAmplitudeSweep,
    fit_final_z_to_cosine,
    lorentzian_area_scale,
)
from echospec.experiments.lorentzian_rabi_scan import (
    LorentzianRabiLengthCutoffScanExperiment,
    OptionsLorentzianRabiLengthCutoffScan,
    ResultsLorentzianRabiLengthCutoffScan,
)

__all__ = [
    "Spectroscopy",
    "OptionsSpectroscopy",
    "ResultsSpectroscopy1D",
    "AmplitudeSweepSpectroscopy",
    "OptionsSpectroscopy2d",
    "LorentzianRabiAmplitudeExperiment",
    "LorentzianRabiAmplitudeSweepExperiment",
    "OptionsLorentzianRabiAmplitude",
    "ResultsLorentzianRabiAmplitude",
    "ResultsLorentzianRabiAmplitudeSweep",
    "fit_final_z_to_cosine",
    "lorentzian_area_scale",
    "LorentzianRabiLengthCutoffScanExperiment",
    "OptionsLorentzianRabiLengthCutoffScan",
    "ResultsLorentzianRabiLengthCutoffScan",
]
