# QuTiP backend

This package contains the adaptive QuTiP `mesolve` implementation of the
finite-level simulation. It uses the same `SimulationConfig`, pulse/DRAG/Stark
functions, returned `SimulationResult`, and final physical diagnostics as the
PyTorch CPU/MPS backend.

```python
from echospec.simulation.config import SimulationConfig
from echospec.simulation.qutip import run_simulation

result = run_simulation(SimulationConfig(
    amplitude_mhz=(0.0, 12.5, 25.0),
    detuning_mhz=(-0.5, 0.0, 0.5),
    duration_us=0.1,
    num_steps_per_half=400,
))
```

The older two-level runner remains at `echospec.simulation.run.Solver` so
existing experiment imports keep working. It is also available from the new
backend folder as `echospec.simulation.qutip.LegacySolver`. New
backend-comparison code should use `run_simulation` and the shared finite-level
configuration.
