import numpy as np

from echospec.analysis.rotary_echo_fit import (
    fit_frequency_shift,
    fit_fringe_spacing,
)


def test_fringe_fit_recovers_period_and_returns_phase_aliases():
    x = np.linspace(-.5, .5, 401)
    q, shift = 12.73, .123
    y = .43+.02*x-.08*np.cos(2*np.pi*q*(x-shift))
    result = fit_fringe_spacing(x, y)
    np.testing.assert_allclose(result["effective_time_us"], q, atol=1e-7)
    assert result["rms_probability"] < 1e-8
    assert np.min(abs(result["candidate_troughs_mhz"]-shift)) < 1e-8
    assert len(result["candidate_troughs_mhz"]) > 10


def test_full_pattern_recovers_shift_with_unknown_readout_gain_and_offset():
    x = np.linspace(-.5, .5, 301)

    def response(z):
        return .45-.10*np.exp(-z*z/.12)*np.cos(80*z+2*z*z)

    actual_shift = .1234
    y = .03+.83*response(x-actual_shift)
    result = fit_frequency_shift(x, y, response)
    np.testing.assert_allclose(result["resonance_shift_mhz"], actual_shift, atol=1e-8)
    np.testing.assert_allclose([result["gain"], result["offset"]], [.83, .03], atol=1e-8)
    assert np.isnan(result["local_standard_error_mhz"])
    assert result["rms_probability"] < 1e-8


def test_periodic_model_reports_multiple_equivalent_frequency_solutions():
    x = np.linspace(-.5, .5, 301)

    def response(z):
        return .45-.1*np.cos(2*np.pi*z/.08)

    result = fit_frequency_shift(x, response(x-.013), response)
    aliases = result["minima"][result["minima"][:, 1] < 1e-10, 0]
    assert len(aliases) >= 8
    np.testing.assert_allclose(np.diff(np.sort(aliases)), .08, atol=1e-7)


def test_noisy_fit_has_finite_uncertainty_and_recovers_shift():
    x = np.linspace(-.4, .4, 301)

    def response(z):
        return .45-.12*np.exp(-z*z/.08)*np.cos(80*z+2*z*z)

    random = np.random.default_rng(791)
    noise = .003
    y = response(x-.1234)+random.normal(0., noise, len(x))
    result = fit_frequency_shift(x, y, response, variance=np.full_like(x, noise**2))
    assert 0 < result["local_standard_error_mhz"] < .001
    assert abs(result["resonance_shift_mhz"]-.1234) < 4*result["local_standard_error_mhz"]
    assert .6 < result["reduced_chi_squared"] < 1.5
