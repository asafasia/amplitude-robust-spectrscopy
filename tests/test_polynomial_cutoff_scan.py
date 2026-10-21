"""CLI grid, cached data identity, and multi-duration configuration."""

from dataclasses import asdict

import numpy as np
import pytest

from echospec.simulation.polynomial_echo import PolynomialEchoPulse
from scripts.scan_polynomial_cutoffs import parser, run_case, settings_from_args


@pytest.mark.parametrize("count", [10, 50])
def test_logarithmic_grid_includes_both_requested_endpoints(count):
    config = settings_from_args(parser().parse_args(["--cutoff-count", str(count)]))
    np.testing.assert_allclose(config["cutoffs"], np.logspace(-5, 0, count))
    assert config["cutoffs"][0] == 1e-5 and config["cutoffs"][-1] == 1
    assert config["durations_us"] == [30]
    assert config["drag_beta"] == -.5 and config["transition_us"] == .016


def test_duration_list_keeps_same_cutoff_and_drag():
    config = settings_from_args(parser().parse_args(["--durations-us", "30", "60"]))
    assert config["durations_us"] == [30, 60]
    assert len(config["cutoffs"]) == 10


def test_cache_refuses_mismatched_physics_and_incomplete_metrics_run(tmp_path):
    path = tmp_path / "case.npz"
    config = settings_from_args(parser().parse_args([]))
    pulse = asdict(PolynomialEchoPulse())
    with pytest.raises(FileNotFoundError):
        run_case(path, config, pulse, metrics_only=True)
    np.savez(path, configuration="stale")
    with pytest.raises(ValueError, match="Cache differs"):
        run_case(path, config, pulse)


@pytest.mark.parametrize("arguments", [
    ["--cutoff-min", "0"], ["--cutoff-count", "1"],
    ["--detuning-points", "100"], ["--detuning-span-mhz", ".1"],
    ["--durations-us", "30", "30"], ["--durations-us", "nan"],
    ["--t2-us", "200"], ["--dt-ns", "1"], ["--shots", "1001"],
])
def test_invalid_scan_configuration_fails_before_writes(arguments):
    with pytest.raises(ValueError):
        settings_from_args(parser().parse_args(arguments))
