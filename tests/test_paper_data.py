"""Tests for the reviewer-facing paper-data export contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from echospec.paper_data import save_paper_dataset


class PaperDataExportTests(unittest.TestCase):
    def test_writes_safe_npz_and_descriptive_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path, metadata_path = save_paper_dataset(
                "fig_s1_example",
                category="numerical",
                arrays={
                    "detuning_mhz": np.asarray([-1.0, 0.0, 1.0]),
                    "population": np.asarray([0.1, 0.2, 0.1]),
                },
                provenance={
                    "manuscript_scope": "supplemental",
                    "generator": "scripts/example.py",
                },
                data_root=Path(directory),
            )

            self.assertEqual(data_path.parent.name, "numerical")
            with np.load(data_path, allow_pickle=False) as data:
                np.testing.assert_array_equal(
                    data["detuning_mhz"],
                    [-1.0, 0.0, 1.0],
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], 1)
            self.assertEqual(metadata["category"], "numerical")
            self.assertEqual(
                metadata["arrays"]["population"]["shape"],
                [3],
            )

    def test_rejects_object_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TypeError):
                save_paper_dataset(
                    "bad_object_data",
                    category="experimental",
                    arrays={"labels": np.asarray([{"unsafe": True}], dtype=object)},
                    provenance={"generator": "test"},
                    data_root=Path(directory),
                )


if __name__ == "__main__":
    unittest.main()
