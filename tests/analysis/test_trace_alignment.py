import numpy as np
import pytest

from echospec.analysis.trace_alignment import fit_population_offset


def test_recovers_offset_on_distinct_sampling_grids_without_changing_inputs():
    sim_x = np.linspace(-1, 1, 101)
    sim_y = 0.3 + 0.2 * sim_x**2
    exp_x = np.linspace(-0.95, 0.95, 20)
    noise = np.tile([-0.01, 0.01], 10)
    exp_y = np.interp(exp_x, sim_x, sim_y) - 0.04 + noise
    original = sim_y.copy()
    fit = fit_population_offset(exp_x, exp_y, sim_x, sim_y)
    assert fit["offset"] == pytest.approx(-0.04)
    assert fit["rmse_after"] == pytest.approx(0.01)
    assert fit["rmse_after"] < fit["rmse_before"]
    np.testing.assert_array_equal(sim_y, original)


def test_rejects_extrapolation():
    with pytest.raises(ValueError, match="cover"):
        fit_population_offset([-2, 0, 2], [0, 0, 0], [-1, 1], [0, 0])
