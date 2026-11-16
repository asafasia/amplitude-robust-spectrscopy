"""The quoted 50 MHz leakage must not use a maximum at another drive."""

import numpy as np

from scripts.make_midpoint_leakage_50mhz import leakage_metrics


def test_highest_drive_leakage_is_separate_from_whole_grid_maximum():
    populations = np.zeros((2, 3, 3, 3))
    populations[0, 2] = [[0, 0, 0], [0.04, 0.03, 0.02], [0.01, 0.02, 0.03]]
    populations[1, 2] = [[0, 0, 0], [0.002, 0.001, 0.003], [0.001, 0.0005, 0.0002]]
    metrics = leakage_metrics(populations, (0, 16), np.array([-0.5, 0, 0.5]))
    np.testing.assert_allclose([m["max_final_pf"] for m in metrics], [0.04, 0.003])
    np.testing.assert_allclose(
        [m["max_final_pf_at_highest_drive"] for m in metrics], [0.03, 0.001]
    )
    assert [m["highest_drive_maximum_detuning_mhz"] for m in metrics] == [0.5, -0.5]
