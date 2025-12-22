"""Base classes for simulation experiments."""

from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import TypeVar, Generic, Optional

from echospec.utils.parameters import Parameters
from echospec.utils.config import DATA_PATH
from echospec.simulation.run import Options


ResultsT = TypeVar("ResultsT")


class BaseExperiment(ABC, Generic[ResultsT]):
    """
    Abstract base class for all simulation experiments.

    Provides a common interface for running simulations and accessing results.
    """

    def __init__(
        self,
        params: Parameters,
        options: Optional[Options] = None,
    ) -> None:
        self.params = params
        self.options = options or Options()
        self.results: Optional[ResultsT] = None
        self.current_figure: Optional[plt.Figure] = None

    @abstractmethod
    def run(self) -> ResultsT:
        """
        Execute the experiment.

        Returns
        -------
        ResultsT
            The results of the experiment.
        """
        pass

    @abstractmethod
    def plot(self) -> None:
        """
        Plot the results of the experiment.

        Raises
        ------
        RuntimeError
            If run() has not been called yet.
        """
        pass

    def _check_results(self) -> None:
        """
        Check that results are available.

        Raises
        ------
        RuntimeError
            If run() has not been called yet.
        """
        if self.results is None:
            raise RuntimeError("No results available. Call run() first.")

    @abstractmethod
    def _get_experiment_name(self) -> str:
        """
        Get the name of the experiment for saving purposes.

        Returns
        -------
        str
            The experiment name (e.g., 'spectroscopy', 'amplitude_sweep')
        """
        pass

    @abstractmethod
    def _save_results(self, save_dir: Path) -> None:
        """
        Save experiment-specific results to the given directory.

        Parameters
        ----------
        save_dir : Path
            Directory where results should be saved
        """
        pass

    def save(self, base_path: Optional[Path] = None) -> Path:
        """
        Save experiment data, parameters, options, and figure.

        Creates directory structure: base_path / date / experiment_name_timestamp /

        Parameters
        ----------
        base_path : Optional[Path]
            Base directory for saving. If None, uses DATA_PATH from config

        Returns
        -------
        Path
            The directory where files were saved

        Raises
        ------
        RuntimeError
            If run() has not been called yet
        """
        self._check_results()

        # Set default base path from config
        if base_path is None:
            base_path = DATA_PATH
        else:
            base_path = Path(base_path)

        # Create date folder (e.g., "22-12-25")
        date_str = datetime.now().strftime("%d-%m-%y")
        date_dir = base_path / date_str
        date_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment folder with timestamp (e.g., "spectroscopy_14-30-45")
        time_str = datetime.now().strftime("%H-%M-%S")
        experiment_name = self._get_experiment_name()
        save_dir = date_dir / f"{experiment_name}_{time_str}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save parameters as JSON
        self._save_params_json(save_dir)

        # Save options as JSON
        self._save_options_json(save_dir)

        # Save experiment-specific results
        self._save_results(save_dir)

        # Always save figure
        self._save_figure(save_dir)

        print(f"Saved experiment data to: {save_dir}")
        return save_dir

    def _save_params_json(self, save_dir: Path) -> None:
        """Save parameters to JSON file."""
        params_dict = asdict(self.params)

        # Convert numpy types and enums to native Python types
        params_dict = self._convert_to_serializable(params_dict)

        with open(save_dir / "params.json", "w") as f:
            json.dump(params_dict, f, indent=2)

    def _save_options_json(self, save_dir: Path) -> None:
        """Save options to JSON file."""
        options_dict = asdict(self.options)

        # Convert numpy types to native Python types
        options_dict = self._convert_to_serializable(options_dict)

        with open(save_dir / "options.json", "w") as f:
            json.dump(options_dict, f, indent=2)

    def _save_figure(self, save_dir: Path) -> None:
        """Save the current matplotlib figure."""
        fig = self.current_figure if self.current_figure is not None else plt.gcf()
        if fig is not None:
            fig.savefig(
                save_dir / "plot.png",
                dpi=300,
                bbox_inches="tight",
            )

    @staticmethod
    def _convert_to_serializable(obj):
        """Convert numpy types and other non-serializable objects to JSON-serializable types."""
        if isinstance(obj, dict):
            return {key: BaseExperiment._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [BaseExperiment._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif hasattr(obj, '__str__') and not isinstance(obj, (str, int, float, bool, type(None))):
            # Convert enum and other objects to string
            return str(obj)
        else:
            return obj
