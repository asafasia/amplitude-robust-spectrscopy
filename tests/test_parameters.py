import math
import unittest

from echospec.simulation.pulses import PulseType
from echospec.utils.parameters import Parameters


class TestParameters(unittest.TestCase):
    def test_default_pulse_type(self):
        params = Parameters()

        self.assertEqual(params.pulse_type, PulseType.LORENTZIAN)

    def test_t2_combines_relaxation_and_dephasing(self):
        params = Parameters(T1=30.0, T_dephasing=20.0)

        self.assertTrue(math.isclose(params.T2, 15.0))
        self.assertTrue(math.isclose(params.gamma_relaxation, 1 / 30.0))
        self.assertTrue(math.isclose(params.gamma_dephasing, 1 / 20.0))


if __name__ == "__main__":
    unittest.main()
