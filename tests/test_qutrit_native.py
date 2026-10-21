"""Independent implementation agreement for the optional fused RK4 kernel."""

import shutil

import numpy as np
import pytest

from echospec.simulation.qutrit import simulate_qutrit_map

pytestmark = pytest.mark.skipif(
    not (shutil.which("clang++") or shutil.which("g++")), reason="C++ compiler unavailable"
)


@pytest.mark.parametrize("window,ramp,power", [("none", 0, 3), ("quintic", .12, 3),
                                             ("polynomial", 0, 3),
                                             ("polynomial", 0, 8)])
@pytest.mark.parametrize("beta", [0, -.5, -1])
def test_native_matches_numpy_with_smooth_echo_and_drag(window, ramp, power, beta):
    settings = dict(duration_us=.4, detuning_mhz=np.array([-1., -.1, 0., .3, .8]),
                    rabi_mhz=np.array([0., 7., 65., 100.]), t1_us=51.24,
                    t_phi_us=7.871480508563624, anharmonicity_mhz=-216.,
                    num_steps_per_half=1000, cutoff=.05, echo=True,
                    echo_transition_shape="sine", echo_transition_us=.016,
                    drag_beta=beta, edge_window_kind=window, edge_ramp_us=ramp,
                    edge_window_power=power)
    reference = simulate_qutrit_map(**settings)
    native = simulate_qutrit_map(**settings, integration_backend="native")
    for name in ["ground", "excited", "second_excited"]:
        np.testing.assert_allclose(getattr(native, name), getattr(reference, name),
                                   atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("cutoff,transition", [(None, 0.), (.02, 0.), (.02, .02)])
def test_native_matches_constant_and_historical_tanh(cutoff, transition):
    settings = dict(duration_us=.2, detuning_mhz=np.array([-.1, 0., .2]),
                    rabi_mhz=np.array([5., 80.]), t1_us=51.24, t_phi_us=7.87,
                    anharmonicity_mhz=-216., num_steps_per_half=1000,
                    cutoff=cutoff, echo=True, echo_transition_us=transition,
                    drag_beta=0. if cutoff is None else .5)
    reference = simulate_qutrit_map(**settings)
    native = simulate_qutrit_map(**settings, integration_backend="native")
    for name in ["ground", "excited", "second_excited"]:
        np.testing.assert_allclose(getattr(native, name), getattr(reference, name),
                                   atol=1e-11, rtol=1e-10)


def test_native_rejects_unsupported_stark_correction():
    with pytest.raises(ValueError, match="zero Stark"):
        simulate_qutrit_map(duration_us=.2, detuning_mhz=np.array([0.]),
                            rabi_mhz=np.array([2.]), t1_us=50., t_phi_us=8.,
                            anharmonicity_mhz=-216., num_steps_per_half=500,
                            integration_backend="native", stark_kappa_mhz_inv=.001)
