"""Operating-range selection must not reward invisible or invalid features."""

import numpy as np
import pytest

from scripts.make_outer_tail_ramp_scan import operating_summary, pareto_flags


def row(amp, width=100000., snr=10., accepted=True):
    return dict(amplitude_mhz=amp, fwhm_hz=width, contrast_snr=snr,
                center_hz=0., accepted=accepted)


def test_score_uses_worst_sample_and_excludes_zero_drive():
    result = operating_summary([row(0, np.nan, np.nan, False),
                                row(50, 100000, 10), row(100, 200000, 6)])
    assert result["sampled_points"] == 2
    assert result["minimum_snr_per_mhz"] == 30
    assert result["worst_fwhm_khz"] == 200
    assert result["minimum_snr"] == 6
    assert result["points_passing_goal"] == 1


@pytest.mark.parametrize("bad", [row(100, accepted=False), row(100, np.nan),
                                 row(100, snr=np.nan), row(100, snr=-1)])
def test_invalid_high_drive_point_disqualifies_candidate(bad):
    result = operating_summary([row(50), bad])
    assert result["valid_points"] == 1
    assert result["minimum_snr_per_mhz"] == 0
    assert result["worst_fwhm_khz"] is None


def test_narrow_invisible_dip_does_not_win():
    narrow = operating_summary([row(100, 1000, .001)])
    useful = operating_summary([row(100, 100000, 10)])
    assert useful["minimum_snr_per_mhz"] > narrow["minimum_snr_per_mhz"]


def test_pareto_keeps_tradeoffs_and_ties():
    candidates = [dict(worst_fwhm_khz=w, minimum_snr=s)
                  for w, s in [(100, 5), (200, 10), (200, 5), (100, 5)]]
    candidates.append(dict(worst_fwhm_khz=None, minimum_snr=None))
    assert pareto_flags(candidates) == [True, True, False, True, False]
