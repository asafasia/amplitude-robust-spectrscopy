"""Validated exports for the data used in the Letter and Supplemental Material."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np

PaperDataCategory = Literal["numerical", "experimental"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_DATA_ROOT = PROJECT_ROOT / "paper" / "data"
_VALID_CATEGORIES = {"numerical", "experimental"}
_VALID_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def save_paper_dataset(
    stem: str,
    *,
    category: PaperDataCategory,
    arrays: Mapping[str, Any],
    provenance: Mapping[str, Any],
    data_root: Path | None = None,
) -> tuple[Path, Path]:
    """Write an NPZ dataset and a human-readable JSON provenance sidecar.

    Array keys should include units where applicable (for example,
    ``detuning_mhz`` or ``duration_us``). Object arrays are rejected so every
    archive can be loaded safely with ``allow_pickle=False``.
    """
    if category not in _VALID_CATEGORIES:
        raise ValueError(
            f"category must be one of {sorted(_VALID_CATEGORIES)}, got {category!r}"
        )
    if not _VALID_NAME.fullmatch(stem):
        raise ValueError(
            "stem must contain only lowercase letters, numbers, '.', '_', or '-'"
        )
    if not arrays:
        raise ValueError("arrays must not be empty")

    prepared: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not _VALID_NAME.fullmatch(name):
            raise ValueError(f"Invalid array name: {name!r}")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(
                f"{name!r} has object dtype; paper datasets must not require pickle"
            )
        prepared[name] = array

    output_dir = (data_root or DEFAULT_PAPER_DATA_ROOT) / category
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"{stem}.npz"
    metadata_path = output_dir / f"{stem}.json"
    np.savez_compressed(data_path, **prepared)

    metadata = {
        "schema_version": 1,
        "dataset": stem,
        "category": category,
        "data_file": data_path.name,
        "arrays": {
            name: {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
            for name, array in sorted(prepared.items())
        },
        "provenance": dict(provenance),
    }
    try:
        encoded = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "provenance must contain finite JSON-compatible values"
        ) from error
    metadata_path.write_text(encoded + "\n", encoding="utf-8")
    return data_path, metadata_path
