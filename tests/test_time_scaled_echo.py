"""Check physical time dilation, including phase and derivative correction."""

import numpy as np
import pytest

from echospec.simulation.flexible_echo import FlexibleEchoPulse
from echospec.simulation.time_scaled_echo import TimeScaledEchoPulse


def shaped_pulse():
    return FlexibleEchoPulse(
        duration_us=16, core_us=.016, transition_us=.037,
        phase_rad=.4, phase_scale_us=.3, rotation_mix=.1,
        amp_coefficients=(.1, -.2, .05, .1),
        phase_coefficients=(.4, -.2, .2, .1, -.1),
        odd_phase_coefficients=(.1, -.05, .02))


@pytest.mark.parametrize("factor", [0, -1, np.nan, np.inf])
def test_invalid_stretch_factor(factor):
    with pytest.raises(ValueError):
        TimeScaledEchoPulse(shaped_pulse(), factor)


def test_unit_stretch_preserves_completed_waveform():
    base = shaped_pulse()
    pulse = TimeScaledEchoPulse(base, 1)
    time = np.r_[np.linspace(-8, 8, 101), np.linspace(-.1, .1, 101)]
    np.testing.assert_array_equal(pulse.drive(time), base.drive(time))


def test_all_raw_time_scales_and_derivatives_stretch_together():
    base = shaped_pulse()
    pulse = TimeScaledEchoPulse(base, 1.875)
    time = np.array([-14.8, -.6, -.021, -.005, 0, .011, .4, 14.8])
    for method in ("envelope", "phase", "raw_drive_and_derivative"):
        value, derivative = getattr(pulse, method)(time)
        reference, dref = getattr(base, method)(time / pulse.factor)
        np.testing.assert_allclose(value, reference)
        np.testing.assert_allclose(derivative, dref / pulse.factor)
        eps = 1e-7
        finite_difference = (getattr(pulse, method)(time + eps)[0]
                             - getattr(pulse, method)(time - eps)[0]) / (2 * eps)
        np.testing.assert_allclose(derivative, finite_difference, atol=2e-7,
                                   rtol=2e-6)
    assert pulse.duration_us == 30
    assert pulse.sigma_us == base.sigma_us * pulse.factor
    assert pulse.transition_us == base.transition_us * pulse.factor


def test_drag_uses_stretched_derivative_and_physical_anharmonicity():
    base = shaped_pulse()
    pulse = TimeScaledEchoPulse(base, 2)
    time = np.linspace(-.1, .1, 101)
    raw, derivative = base.raw_drive_and_derivative(time / 2)
    drag = 1j * base.drag_beta * derivative / (2 * np.pi * abs(base.anharmonicity_mhz))
    np.testing.assert_allclose(pulse.drive(time), raw + drag / 2)
    assert np.max(abs(pulse.drive(time) - base.drive(time / 2))) > .01
    np.testing.assert_array_equal(pulse.drive(np.array([-17., -16., 16., 17.])), 0)
