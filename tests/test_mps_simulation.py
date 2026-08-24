import unittest
from unittest import mock

import numpy as np
import torch

from echospec.simulation.backends import run_with_solver
from echospec.simulation.config import SimulationConfig
from echospec.simulation.mps.runner import run_simulation
from echospec.simulation.mps.solver import (
    lindblad_rhs,
    linear_lindblad_rhs,
    make_superoperators,
    mps_is_available,
    resolve_device,
)
from echospec.simulation.qutip import LegacySolver
from echospec.simulation.qutip.runner import run_simulation as run_qutip_reference


def small_config() -> SimulationConfig:
    return SimulationConfig(
        levels=3,
        amplitude_mhz=(0.0, 2.0),
        detuning_mhz=(-0.1, 0.0, 0.1),
        duration_us=0.02,
        num_steps_per_half=80,
        cutoff=0.005,
        echo=True,
        anharmonicity_mhz=-217.0,
        t1_us=27.0,
        t2_us=6.5,
    )


class TestDeviceSelection(unittest.TestCase):
    def test_original_qutip_runner_is_exposed_in_backend_folder(self) -> None:
        self.assertEqual(LegacySolver.__module__, "echospec.simulation.run")

    def test_cpu_is_always_available(self) -> None:
        self.assertEqual(resolve_device("cpu").type, "cpu")

    def test_unavailable_mps_is_an_explicit_error(self) -> None:
        with mock.patch(
            "echospec.simulation.mps.solver.mps_is_available", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "MPS was requested"):
                resolve_device("mps")


class TestPytorchSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = small_config()
        cls.result = run_simulation(cls.config, device="cpu")

    def test_output_shapes(self) -> None:
        self.assertEqual(self.result.populations.shape, (3, 2, 3))
        self.assertEqual(self.result.density_matrices.shape, (2, 3, 3, 3))
        self.assertEqual(self.result.leakage.shape, (2, 3))

    def test_trace_hermiticity_and_population_range(self) -> None:
        self.assertLess(self.result.trace_error, 2e-5)
        self.assertLess(self.result.hermiticity_error, 2e-5)
        self.assertGreaterEqual(float(self.result.populations.min()), -2e-5)
        self.assertLessEqual(float(self.result.populations.max()), 1.0 + 2e-5)

    def test_agrees_with_qutip_reference(self) -> None:
        reference = run_qutip_reference(self.config)
        np.testing.assert_allclose(
            self.result.populations, reference.populations, atol=3e-4, rtol=3e-4
        )
        np.testing.assert_allclose(
            self.result.density_matrices,
            reference.density_matrices,
            atol=5e-4,
            rtol=5e-4,
        )

    def test_shared_dispatch_selects_cpu_backend(self) -> None:
        dispatched = run_with_solver(self.config, solver="torch-cpu")
        np.testing.assert_allclose(
            dispatched.density_matrices,
            self.result.density_matrices,
            atol=0.0,
            rtol=0.0,
        )

    def test_superoperator_matches_direct_rhs(self) -> None:
        torch.manual_seed(3)
        rho = torch.randn(2, 3, 3, 2)
        detuning = torch.tensor([-0.2, 0.3])
        drive = torch.tensor([[1.2, -0.4], [0.8, 0.1]])
        direct = lindblad_rhs(
            rho,
            detuning=detuning,
            drive=drive,
            anharmonicity=-20.0,
            inv_t1=0.04,
            inv_t_phi=0.1,
        )
        operators = make_superoperators(
            levels=3,
            anharmonicity=-20.0,
            inv_t1=0.04,
            inv_t_phi=0.1,
            device=torch.device("cpu"),
        )
        linear = linear_lindblad_rhs(
            rho.reshape(2, -1),
            detuning=detuning,
            drive_real=drive[:, 0],
            drive_imag=drive[:, 1],
            superoperators=operators,
        ).reshape_as(rho)
        torch.testing.assert_close(linear, direct, atol=2e-5, rtol=2e-5)

    @unittest.skipUnless(mps_is_available(), "MPS is unavailable")
    def test_mps_agrees_with_pytorch_cpu(self) -> None:
        mps_result = run_simulation(self.config, device="mps")
        self.assertTrue(mps_result.tensor_device.startswith("mps"))
        np.testing.assert_allclose(
            mps_result.density_matrices,
            self.result.density_matrices,
            atol=5e-5,
            rtol=5e-5,
        )


if __name__ == "__main__":
    unittest.main()
