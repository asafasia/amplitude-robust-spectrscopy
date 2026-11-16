"""Display smoothing must not change measurements or fill rejected cells."""

import numpy as np

from echospec.figures.cutoff_contours import smooth_contrast_for_contour


def test_masked_cells_do_not_bias_boundary_field_or_become_valid():
    values = np.full((30, 30), .4)
    valid = np.ones(values.shape, dtype=bool)
    valid[10:20, 10:20] = False
    values[~valid] = 1e6
    original = values.copy()
    result = smooth_contrast_for_contour(values, valid)
    np.testing.assert_allclose(result.compressed(), .4, rtol=1e-14)
    np.testing.assert_array_equal(result.mask, ~valid)
    np.testing.assert_array_equal(values, original)


def test_small_ripples_are_suppressed_without_changing_broad_level():
    x, y = np.mgrid[:100, :100]
    background = .3 + .001 * x
    values = background + .02 * np.sin(2 * np.pi * y / 4)
    result = smooth_contrast_for_contour(values, np.ones(values.shape, dtype=bool))
    interior = np.s_[15:-15, 15:-15]
    assert np.max(abs(result[interior] - background[interior])) < 1e-4
