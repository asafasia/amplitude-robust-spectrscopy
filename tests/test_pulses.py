import unittest

import numpy as np

from echospec.simulation.pulses import (
    PulseArgs,
    PulseType,
    choose_pulse,
    gaussian_envelope,
    lorentzian_envelope,
)


class TestPulses(unittest.TestCase):
    def test_lorentzian_envelope_reaches_cutoff_at_edges(self):
        args = PulseArgs(pulse_length=10.0, cutoff=1e-2, order=0.5)
        t = np.array([-5.0, 0.0, 5.0])

        envelope = lorentzian_envelope(t, args)

        np.testing.assert_allclose(envelope[1], 1.0)
        np.testing.assert_allclose(envelope[0], args.cutoff)
        np.testing.assert_allclose(envelope[2], args.cutoff)

    def test_gaussian_envelope_reaches_cutoff_at_edges(self):
        args = PulseArgs(pulse_length=10.0, cutoff=1e-2)
        t = np.array([-5.0, 0.0, 5.0])

        envelope = gaussian_envelope(t, args)

        np.testing.assert_allclose(envelope[1], 1.0)
        np.testing.assert_allclose(envelope[0], args.cutoff)
        np.testing.assert_allclose(envelope[2], args.cutoff)

    def test_choose_pulse_returns_static_drive_for_square_pulse(self):
        self.assertIsNone(choose_pulse(PulseType.SQUARE, eco_pulse=False))


if __name__ == "__main__":
    unittest.main()
