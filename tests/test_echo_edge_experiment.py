"""Duration changes preserve physical timing and cannot reuse a 20 us baseline."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import make_echo_edge_ramp_comparison as study


@pytest.fixture(autouse=True)
def isolate_configuration(monkeypatch):
    monkeypatch.setattr(study, "COMMON", study.COMMON.copy())
    monkeypatch.setattr(study, "OUT", study.OUT)


@pytest.mark.parametrize("duration", [20., 30.])
def test_duration_preserves_timestep_and_endpoint_cutoff(duration):
    output = study.configure_experiment(duration)
    assert duration * 1000 / (2 * study.integration_steps()) == .5
    assert duration * 1000 / (2 * study.integration_steps(.25)) == .25
    assert (output.name == "echo_edge_ramp_comparison") == (duration == 20)
    for cutoff in study.CUTOFFS:
        i, _ = study.waveform(-duration / 2, cutoff, "none", 0)
        np.testing.assert_allclose(i, cutoff, atol=1e-15)
        for kind, ramp in [("quintic", 6), ("polynomial", 0)]:
            for endpoint in [-duration / 2, duration / 2]:
                assert study.waveform(endpoint, cutoff, kind, ramp) == (0, 0)
        # The midpoint transition remains exactly 16 ns at either duration.
        assert study.waveform(0, cutoff, "none", 0)[0] == 0
        assert study.waveform(-.008, cutoff, "none", 0)[0] > 0
        assert study.waveform(.008, cutoff, "none", 0)[0] < 0


def test_30us_does_not_reuse_old_baseline_with_same_step_count(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "AMPS", np.array([100.]))
    monkeypatch.setattr(study, "DETUNING", np.array([0.]))
    old_dir = tmp_path / "outputs/sine_echo_cutoff_001_100mhz_comparison"
    old_dir.mkdir(parents=True)
    np.savez(old_dir / "beta-0.5_narrow.npz", rabi_mhz=study.AMPS,
             detuning_mhz=study.DETUNING, populations=np.full((3, 1, 1), 1 / 3))
    calls = []

    def simulate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(ground=np.ones((1, 1)), excited=np.zeros((1, 1)),
                               second_excited=np.zeros((1, 1)))

    monkeypatch.setattr(study, "simulate_qutrit_map", simulate)
    output = study.configure_experiment(30)
    output.mkdir()
    key = study.run_case(.001, "hard", "none", 0, steps=20000)
    assert len(calls) == 1
    assert calls[0]["duration_us"] == 30
    with np.load(output / f"{key}.npz") as saved:
        np.testing.assert_array_equal(saved["populations"][:, 0, 0], [1, 0, 0])
