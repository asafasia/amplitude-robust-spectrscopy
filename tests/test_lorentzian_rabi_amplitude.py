import math
import unittest

import numpy as np

from echospec.experiments.lorentzian_rabi_amplitude import (
    LorentzianRabiAmplitudeExperiment,
    LorentzianRabiAmplitudeSweepExperiment,
    OptionsLorentzianRabiAmplitude,
    fit_final_z_to_cosine,
    lorentzian_area_scale,
)
from echospec.experiments.lorentzian_rabi_scan import (
    ResultsLorentzianRabiLengthCutoffScan,
)
from echospec.simulation.pulses import PulseType
from echospec.utils.parameters import Parameters
from echospec.utils.units import Units as u


class TestLorentzianRabiAmplitude(unittest.TestCase):
    def test_fit_final_z_to_cosine_recovers_rabi_scale(self):
        amplitudes = np.linspace(0, 8, 80)
        rabi_scale = 0.75
        final_z = 0.93 * np.cos(rabi_scale * amplitudes + 0.1) + 0.03

        fit = fit_final_z_to_cosine(
            amplitudes,
            final_z,
            rabi_scale_guess=rabi_scale,
        )

        self.assertTrue(math.isclose(fit.rabi_scale, rabi_scale, rel_tol=1e-6))
        self.assertTrue(
            math.isclose(fit.pi_pulse_amplitude, np.pi / rabi_scale, rel_tol=1e-6)
        )

    def test_lorentzian_area_scale_is_positive(self):
        area = lorentzian_area_scale(
            pulse_length=10 * u.us,
            cutoff=1e-3,
            num_time_points=101,
        )

        self.assertGreater(area, 0)
        self.assertLess(area, 10 * u.us)

    def test_experiment_runs_without_decay_c_ops(self):
        params = Parameters(
            pulse_type=PulseType.LORENTZIAN,
            eco_pulse=False,
            anharmonicity=0.0,
        )
        options = OptionsLorentzianRabiAmplitude(
            plot=False,
            save=False,
            num_time_points=101,
        )
        pulse_lengths = np.array([1.0]) * u.us
        cutoffs = np.array([1e-2])
        amplitudes = np.linspace(0, 2, 7) * u.pi2 * u.MHz

        result = LorentzianRabiAmplitudeExperiment(
            pulse_lengths=pulse_lengths,
            cutoffs=cutoffs,
            amplitudes=amplitudes,
            params=params,
            options=options,
        ).run()

        self.assertEqual(result.final_z.shape, (1, 1, len(amplitudes)))
        self.assertEqual(result.rabi_scales.shape, (1, 1))
        self.assertGreater(result.rabi_scales[0, 0], 0)

    def test_single_shape_sweep_exposes_state_data_for_each_amplitude(self):
        params = Parameters(
            pulse_type=PulseType.LORENTZIAN,
            eco_pulse=False,
            anharmonicity=0.0,
        )
        options = OptionsLorentzianRabiAmplitude(
            plot=False,
            save=False,
            num_time_points=101,
        )
        amplitudes = np.linspace(0, 2, 7) * u.pi2 * u.MHz

        result = LorentzianRabiAmplitudeSweepExperiment(
            pulse_length=1.0 * u.us,
            cutoff=1e-2,
            amplitudes=amplitudes,
            params=params,
            options=options,
        ).run()

        self.assertEqual(result.data.shape, (len(amplitudes), 3, 101))
        self.assertEqual(result.final_state.shape, (len(amplitudes), 3))
        self.assertEqual(result.final_z.shape, (len(amplitudes),))
        self.assertGreater(result.pi_pulse_amplitude, 0)

    def test_length_cutoff_scan_result_plots_calculated_rabi(self):
        pulse_lengths = np.array([1.0, 2.0]) * u.us
        cutoffs = np.array([1e-3, 1e-2])
        amplitudes = np.linspace(0, 2, 7) * u.pi2 * u.MHz
        pi_pulse_amplitudes = np.ones((2, 2)) * u.pi2 * u.MHz

        result = ResultsLorentzianRabiLengthCutoffScan(
            pulse_lengths=pulse_lengths,
            cutoffs=cutoffs,
            amplitudes=amplitudes,
            rabi_scales=np.ones((2, 2)) * u.us,
            pi_pulse_amplitudes=pi_pulse_amplitudes,
            area_scales=np.ones((2, 2)) * u.us,
            sweep_results=[[], []],
        )

        self.assertEqual(result.pi_pulse_amplitudes.shape, (2, 2))
        np.testing.assert_allclose(
            result.pi_pulse_amplitudes / u.pi2 / u.MHz,
            np.ones((2, 2)),
        )


if __name__ == "__main__":
    unittest.main()
