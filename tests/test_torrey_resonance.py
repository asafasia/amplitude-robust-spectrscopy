import math
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from echospec.experiments.torrey_resonance import (
    OptionsTorreyResonance,
    TorreyFwhmVsRabiExperiment,
    TorreyResonanceExperiment,
    torrey_fwhm,
    torrey_snr,
    torrey_sigma_z,
)
from echospec.utils.parameters import Parameters
from echospec.utils.units import Units as u


class TestTorreyResonance(unittest.TestCase):
    def test_torrey_sigma_z_matches_center_and_far_limits(self):
        T1 = 30 * u.us
        T2 = 12 * u.us
        rabi_frequency = 0.1 * u.pi2 * u.MHz
        detunings = np.array([0.0, 1e9])

        sigma_z = torrey_sigma_z(detunings, rabi_frequency, T1, T2)

        expected_center = 1 / (1 + rabi_frequency**2 * T1 * T2)
        self.assertTrue(math.isclose(sigma_z[0], expected_center, rel_tol=1e-12))
        self.assertTrue(math.isclose(sigma_z[1], 1.0, rel_tol=1e-6))

    def test_torrey_fwhm_matches_half_depth_crossing(self):
        T1 = 30 * u.us
        T2 = 12 * u.us
        rabi_frequency = 0.1 * u.pi2 * u.MHz
        fwhm = torrey_fwhm(rabi_frequency, T1, T2)
        half_detuning = fwhm / 2

        center = torrey_sigma_z(np.array([0.0]), rabi_frequency, T1, T2)[0]
        half_depth = 0.5 * (1.0 + center)
        sigma_z_at_half_width = torrey_sigma_z(
            np.array([half_detuning]),
            rabi_frequency,
            T1,
            T2,
        )[0]

        self.assertTrue(
            math.isclose(sigma_z_at_half_width, half_depth, rel_tol=1e-12)
        )

    def test_torrey_snr_matches_peak_population_signal(self):
        T1 = 30 * u.us
        T2 = 12 * u.us
        rabi_frequency = 0.1 * u.pi2 * u.MHz

        center_sigma_z = torrey_sigma_z(
            np.array([0.0]),
            rabi_frequency,
            T1,
            T2,
        )[0]
        expected_snr = (1.0 - center_sigma_z) / 2.0

        self.assertTrue(
            math.isclose(
                torrey_snr(rabi_frequency, T1, T2),
                expected_snr,
                rel_tol=1e-12,
            )
        )

    def test_resonance_experiment_returns_fwhm_for_parameters(self):
        params = Parameters(
            T1=30 * u.us,
            T_dephasing=20 * u.us,
            rabi_frequency=0.1 * u.pi2 * u.MHz,
        )
        detunings = np.linspace(-1, 1, 101) * u.pi2 * u.MHz
        options = OptionsTorreyResonance(plot=False, save=False)

        result = TorreyResonanceExperiment(detunings, params, options).run()

        self.assertEqual(result.sigma_z.shape, detunings.shape)
        self.assertTrue(
            math.isclose(
                result.fwhm,
                torrey_fwhm(params.rabi_frequency, params.T1, params.T2),
                rel_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                result.snr,
                torrey_snr(params.rabi_frequency, params.T1, params.T2),
                rel_tol=1e-12,
            )
        )

    def test_resonance_experiment_saves_figure_to_fig_dir(self):
        params = Parameters(
            T1=30 * u.us,
            T_dephasing=20 * u.us,
            rabi_frequency=0.1 * u.pi2 * u.MHz,
        )
        detunings = np.linspace(-1, 1, 101) * u.pi2 * u.MHz
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment = TorreyResonanceExperiment(
                detunings,
                params,
                OptionsTorreyResonance(
                    plot=False,
                    save=False,
                    save_fig=True,
                    fig_dir=Path(tmpdir),
                ),
            )

            experiment.run()

            self.assertTrue((Path(tmpdir) / "torrey_resonance.png").exists())
            plt.close(experiment.current_figure)

    def test_resonance_plot_has_no_title_for_presentation(self):
        params = Parameters(
            T1=30 * u.us,
            T_dephasing=20 * u.us,
            rabi_frequency=0.1 * u.pi2 * u.MHz,
        )
        detunings = np.linspace(-1, 1, 101) * u.pi2 * u.MHz
        result = TorreyResonanceExperiment(
            detunings,
            params,
            OptionsTorreyResonance(plot=False, save=False),
        ).run()
        fig = result.plot()

        self.assertEqual(fig.axes[0].get_title(), "")
        plt.close(fig)

    def test_fwhm_vs_rabi_experiment_is_monotonic(self):
        params = Parameters(T1=30 * u.us, T_dephasing=20 * u.us)
        rabi_frequencies = np.linspace(0.0, 0.5, 9) * u.pi2 * u.MHz
        options = OptionsTorreyResonance(plot=False, save=False)

        result = TorreyFwhmVsRabiExperiment(
            rabi_frequencies,
            params,
            options,
        ).run()

        self.assertEqual(result.fwhm.shape, rabi_frequencies.shape)
        self.assertEqual(result.snr.shape, rabi_frequencies.shape)
        np.testing.assert_allclose(
            result.inverse_fwhm_mhz,
            1.0 / (result.fwhm / u.pi2 / u.MHz),
        )
        self.assertTrue(
            math.isclose(
                result.inverse_t2_limit_mhz,
                result.inverse_fwhm_mhz[0],
                rel_tol=1e-12,
            )
        )
        np.testing.assert_allclose(
            result.inverse_fwhm_t2_units,
            result.inverse_fwhm_mhz / result.inverse_t2_limit_mhz,
        )
        np.testing.assert_allclose(
            result.inverse_fwhm_snr_product,
            result.inverse_fwhm_t2_units * result.snr,
        )
        self.assertTrue(math.isclose(result.inverse_fwhm_t2_units[0], 1.0))
        self.assertTrue(np.all(np.diff(result.fwhm) >= 0))
        self.assertTrue(np.all(np.diff(result.snr) >= 0))

    def test_fwhm_vs_rabi_threshold_positions_are_interpolated(self):
        params = Parameters(T1=30 * u.us, T_dephasing=20 * u.us)
        rabi_frequencies = np.linspace(0.0, 2.0, 401) * u.pi2 * u.MHz
        options = OptionsTorreyResonance(
            plot=False,
            save=False,
            inverse_fwhm_threshold=1e-1,
            snr_threshold=0.2,
        )

        result = TorreyFwhmVsRabiExperiment(
            rabi_frequencies,
            params,
            options,
        ).run()

        self.assertIsNotNone(result.inverse_fwhm_threshold_rabi_mhz)
        self.assertIsNotNone(result.snr_threshold_rabi_mhz)
        self.assertGreater(result.inverse_fwhm_threshold_rabi_mhz, 0)
        self.assertGreater(result.snr_threshold_rabi_mhz, 0)

    def test_fwhm_vs_rabi_plot_can_include_product_line(self):
        params = Parameters(T1=30 * u.us, T_dephasing=20 * u.us)
        rabi_frequencies = np.linspace(0.0, 0.5, 9) * u.pi2 * u.MHz
        options = OptionsTorreyResonance(
            plot=False,
            save=False,
            plot_product=True,
        )

        result = TorreyFwhmVsRabiExperiment(
            rabi_frequencies,
            params,
            options,
        ).run()
        fig = result.plot()
        labels = [line.get_label() for line in fig.axes[0].lines]

        self.assertTrue(result.plot_product)
        self.assertIn("(1/FWHM) x SNR", labels)
        self.assertIsNone(fig.axes[0].get_legend())
        plt.close(fig)

    def test_fwhm_vs_rabi_plot_can_be_rendered_in_stages(self):
        params = Parameters(T1=30 * u.us, T_dephasing=20 * u.us)
        rabi_frequencies = np.linspace(0.0, 0.5, 9) * u.pi2 * u.MHz
        result = TorreyFwhmVsRabiExperiment(
            rabi_frequencies,
            params,
            OptionsTorreyResonance(plot=False, save=False, plot_product=True),
        ).run()

        fwhm_fig = result.plot(include_snr=False, include_product=False)
        self.assertEqual(len(fwhm_fig.axes), 1)
        self.assertNotIn(
            "SNR",
            [line.get_label() for line in fwhm_fig.axes[0].lines],
        )
        fwhm_ylim = fwhm_fig.axes[0].get_ylim()
        plt.close(fwhm_fig)

        snr_fig = result.plot(include_snr=True, include_product=False)
        self.assertEqual(len(snr_fig.axes), 2)
        self.assertNotIn(
            "(1/FWHM) x SNR",
            [line.get_label() for line in snr_fig.axes[0].lines],
        )
        snr_ylim = snr_fig.axes[0].get_ylim()
        snr_right_ylim = snr_fig.axes[1].get_ylim()
        plt.close(snr_fig)

        product_fig = result.plot(include_snr=True, include_product=True)
        self.assertEqual(len(product_fig.axes), 2)
        self.assertIn(
            "(1/FWHM) x SNR",
            [line.get_label() for line in product_fig.axes[0].lines],
        )
        self.assertEqual(fwhm_ylim, snr_ylim)
        self.assertEqual(fwhm_ylim, product_fig.axes[0].get_ylim())
        self.assertEqual(snr_right_ylim, product_fig.axes[1].get_ylim())
        plt.close(product_fig)

    def test_fwhm_vs_rabi_plot_shades_product_threshold_region(self):
        params = Parameters(T1=30 * u.us, T_dephasing=20 * u.us)
        rabi_frequencies = np.linspace(0.0, 2.0, 401) * u.pi2 * u.MHz
        options = OptionsTorreyResonance(
            plot=False,
            save=False,
            plot_product=True,
        )

        result = TorreyFwhmVsRabiExperiment(
            rabi_frequencies,
            params,
            options,
        ).run()
        fig = result.plot()
        labels = [line.get_label() for line in fig.axes[0].lines]

        self.assertNotIn("1/FWHM > 0.1", labels)
        self.assertNotIn("SNR > 0.2", labels)
        self.assertEqual(len(result.product_threshold_rabi_ranges_mhz), 1)
        left, right = result.product_threshold_rabi_ranges_mhz[0]
        midpoint = 0.5 * (left + right)
        midpoint_product = np.interp(
            midpoint,
            result.rabi_frequencies_mhz,
            result.inverse_fwhm_snr_product,
        )
        self.assertGreater(midpoint_product, result.product_threshold)
        self.assertGreaterEqual(len(fig.axes[0].patches), 1)
        plt.close(fig)

    def test_fwhm_vs_rabi_plot_has_no_title_for_presentation(self):
        params = Parameters(T1=30 * u.us, T_dephasing=20 * u.us)
        rabi_frequencies = np.linspace(0.0, 0.5, 9) * u.pi2 * u.MHz
        result = TorreyFwhmVsRabiExperiment(
            rabi_frequencies,
            params,
            OptionsTorreyResonance(plot=False, save=False),
        ).run()
        fig = result.plot()

        self.assertEqual(fig.axes[0].get_title(), "")
        plt.close(fig)

    def test_fwhm_vs_rabi_experiment_saves_figure_to_fig_dir(self):
        params = Parameters(T1=30 * u.us, T_dephasing=20 * u.us)
        rabi_frequencies = np.linspace(0.0, 0.5, 9) * u.pi2 * u.MHz
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment = TorreyFwhmVsRabiExperiment(
                rabi_frequencies,
                params,
                OptionsTorreyResonance(
                    plot=False,
                    save=False,
                    save_fig=True,
                    fig_dir=Path(tmpdir),
                ),
            )

            experiment.run()

            self.assertTrue((Path(tmpdir) / "torrey_fwhm_vs_rabi.png").exists())
            plt.close(experiment.current_figure)

    def test_fwhm_vs_rabi_plot_can_use_log_scale(self):
        params = Parameters(T1=30 * u.us, T_dephasing=20 * u.us)
        rabi_frequencies = np.logspace(-3, -1, 5) * u.pi2 * u.MHz
        options = OptionsTorreyResonance(
            plot=False,
            save=False,
            log_scale=True,
        )

        result = TorreyFwhmVsRabiExperiment(
            rabi_frequencies,
            params,
            options,
        ).run()
        fig = result.plot()

        self.assertTrue(result.log_scale)
        self.assertEqual(fig.axes[0].get_xscale(), "log")
        self.assertEqual(fig.axes[0].get_yscale(), "log")
        self.assertEqual(
            [tick.get_text() for tick in fig.axes[0].get_yticklabels()],
            ["1", "0.1", "0.01", "0.001"],
        )
        self.assertTrue(
            any(
                len(line.get_ydata()) == 2
                and np.allclose(line.get_ydata(), [1.0, 1.0])
                for line in fig.axes[0].lines
            )
        )
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
