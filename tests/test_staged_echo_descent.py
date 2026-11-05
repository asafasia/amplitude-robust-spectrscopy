"""Guardrails must cover unoptimized and earlier amplitude bands alike."""

from copy import deepcopy

import numpy as np
import pytest

from echospec.simulation.polynomial_echo import PolynomialEchoPulse
from scripts.staged_echo_descent import Engine, encode, pulse_from


@pytest.fixture
def example():
    engine = Engine.__new__(Engine)
    engine.amps = np.array([5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    engine.bw = np.full(7, 150.0)
    engine.bc = np.array([0.25, 0.45, 0.45, 0.43, 0.4, 0.38, 0.36])
    engine.i50 = 5
    record = dict(
        valid=True,
        rows=[
            dict(
                fwhm_hz=140000.0,
                contrast=c + 0.03,
                satellite_prominence=0.0,
                center_hz=0.0,
            )
            for c in engine.bc
        ],
    )
    return engine, record


@pytest.mark.parametrize(
    "index, metric, bad_value",
    [(0, "contrast", 0.24), (2, "fwhm_hz", 151000.0), (6, "contrast", 0.35)],
)
def test_original_guardrails_apply_outside_active_band(
    example, index, metric, bad_value
):
    engine, record = example
    assert np.min(engine.constraints(record, seen_hi=20)) > 0
    changed = deepcopy(record)
    changed["rows"][index][metric] = bad_value
    assert np.min(engine.constraints(changed, seen_hi=20)) < 0
    assert not engine.quality(changed)["accepted"]


def test_nominal_contrast_floor_is_separate_from_reference_guard(example):
    engine, record = example
    record["rows"][engine.i50]["contrast"] = 0.399
    assert record["rows"][engine.i50]["contrast"] > engine.bc[engine.i50]
    assert np.min(engine.constraints(record)) < 0
    assert not engine.quality(record)["accepted"]


def test_log_parameter_transform_round_trips():
    expected = PolynomialEchoPulse(
        duration_us=22.0, cutoff=0.00013, power=2.3, transition_us=0.047, drag_beta=0.0
    )
    actual = pulse_from(encode(expected))
    np.testing.assert_allclose(
        list(vars(actual).values()), list(vars(expected).values())
    )


def test_optional_band_floor_expands_with_the_curriculum(example):
    engine, record = example
    # The 60 MHz point has C=0.39, better than its original C=0.36, but below
    # an optional C=0.4 floor that becomes active when that band is reached.
    assert engine.quality(record, band_floor=0.4, seen_hi=20)["accepted"]
    assert not engine.quality(record, band_floor=0.4, seen_hi=60)["accepted"]
