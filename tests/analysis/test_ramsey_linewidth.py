"""Independent quadrature and limiting cases for Ramsey spectral widths."""

import numpy as np
import pytest
from scipy.integrate import quad

from echospec.analysis.ramsey_linewidth import (
    coherence_transform,
    cosine_transform,
    linewidth_mhz,
)


def test_finite_transform_matches_direct_time_integral():
    for offset in (0.0, 0.021, 0.13):
        def signal(t, offset=offset):
            return np.exp(-t / 7.6 - 2j * np.pi * offset * t)

        integral = quad(lambda t: signal(t).real, 0, 20)[0]
        integral += 1j * quad(lambda t: signal(t).imag, 0, 20)[0]
        np.testing.assert_allclose(coherence_transform(offset, 20, 7.6), integral)


def test_real_cosine_transform_retains_both_frequency_components_and_phase():
    frequency, fringe, phase = 0.98, 1.0, 0.4

    def integrand(t):
        return (np.exp(-t / 7.6) * np.cos(2 * np.pi * fringe * t + phase)
                * np.exp(-2j * np.pi * frequency * t))

    integral = quad(lambda t: integrand(t).real, 0, 20, limit=200)[0]
    integral += 1j * quad(lambda t: integrand(t).imag, 0, 20, limit=200)[0]
    np.testing.assert_allclose(
        cosine_transform(frequency, 20, 7.6, fringe, phase), integral,
        rtol=1e-10, atol=1e-10,
    )


@pytest.mark.parametrize("kind,factor", [
    ("absorption", 1), ("power", 1), ("magnitude", np.sqrt(3)),
])
def test_long_record_reaches_the_correct_convention_dependent_limit(kind, factor):
    expected = factor / (np.pi * 7.6)
    assert linewidth_mhz(np.inf, 7.6, kind) == pytest.approx(expected)
    assert linewidth_mhz(200, 7.6, kind) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("kind,coefficient", [
    ("absorption", 0.6033545644), ("power", 0.8858929414),
    ("magnitude", 1.2067091288),
])
def test_short_record_recovers_rectangular_window_width(kind, coefficient):
    # Negligible decay: one-sided cosine integral, sinc squared, and sinc modulus.
    assert linewidth_mhz(10, 1e10, kind) == pytest.approx(coefficient / 10, rel=1e-8)
