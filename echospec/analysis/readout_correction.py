"""Two-state assignment inversion, conditional on a fixed calibration matrix."""

import numpy as np


def correct_binary_readout(measured, matrix, shots):
    """Return unclipped excited-population estimates and shot-noise errors.

    Matrix rows are prepared g/e, columns are assigned g/e. This assumes the
    calibration describes the actual acquisition classifier, has not drifted,
    and population outside g/e is negligible. Calibration uncertainty and
    state-preparation errors are not included in the returned standard errors.
    """
    measured = np.asarray(measured, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 2 x 2 assignment matrix")
    if np.any((matrix < 0) | (matrix > 1)) or not np.allclose(
        matrix.sum(axis=1), 1, rtol=0, atol=1e-10
    ):
        raise ValueError("Assignment matrix must be row-stochastic")
    if not np.isfinite(measured).all() or np.any((measured < 0) | (measured > 1)):
        raise ValueError("Measured assignment fractions must lie in [0, 1]")
    if not np.isscalar(shots) or not np.isfinite(shots) or shots <= 0:
        raise ValueError("shots must be positive")
    contrast = matrix[1, 1] - matrix[0, 1]
    if abs(contrast) < 1e-10:
        raise ValueError("Assignment matrix cannot distinguish the two states")
    corrected = (measured - matrix[0, 1]) / contrast
    stderr = np.sqrt(measured * (1 - measured) / shots) / abs(contrast)
    return corrected, stderr
