"""Matched reuse must preserve the original simulation configuration and data."""

import hashlib

import pytest

from scripts.extend_cutoff_duration_40_50 import copy_verified, new_scan_arguments
from scripts.repeat_extended_cutoff_poly2 import scan_arguments
from scripts.scan_polynomial_cutoffs import parser, settings_from_args


def test_new_run_changes_only_the_requested_durations(tmp_path):
    original = settings_from_args(parser().parse_args(scan_arguments(tmp_path, 6)))
    new = settings_from_args(parser().parse_args(new_scan_arguments(tmp_path, 6)))
    assert new.pop("durations_us") == [40, 50]
    assert original.pop("durations_us") == [10, 20, 30, 45, 60]
    assert new == original


def test_verified_copy_rejects_modified_source_or_destination(tmp_path):
    source = tmp_path / "original.npz"
    source.write_bytes(b"original simulation")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "copy" / "simulation.npz"
    assert copy_verified(source, destination, digest) == digest
    assert destination.read_bytes() == source.read_bytes()
    destination.write_bytes(b"unrelated simulation")
    with pytest.raises(ValueError, match="copied data differ"):
        copy_verified(source, destination, digest)
    source.write_bytes(b"changed baseline")
    with pytest.raises(ValueError, match="Baseline checksum mismatch"):
        copy_verified(source, tmp_path / "second.npz", digest)
    assert not (tmp_path / "second.npz").exists()
