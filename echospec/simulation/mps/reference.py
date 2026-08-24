"""Compatibility import for the QuTiP backend's former location."""

from echospec.simulation.qutip.runner import run_simulation

run_qutip_reference = run_simulation

__all__ = ["run_qutip_reference"]
