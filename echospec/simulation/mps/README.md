# PyTorch MPS transmon simulation

This backend ports the repository's finite-level Duffing-transmon Lindblad
model to a batched PyTorch RK4 solver. Shared configuration and physics live in
`echospec/simulation/config.py`, `model.py`, and `density_matrix.py`. The
adaptive reference implementation lives separately in
`echospec/simulation/qutip/`.

The rotating-frame Hamiltonian is

```text
H = -Delta*n + alpha*n*(n-1)/2 + (drive*a + drive.conj()*a.dag())/2
```

with `Delta = drive frequency - f01`. Relaxation is `sqrt(1/T1)*a`, pure
dephasing is `sqrt(2/T_phi)*n`, and `1/T_phi = 1/T2 - 1/(2*T1)`. The initial
state is `|0><0|`. The Lorentzian, Gaussian, square, echo, zeroed-envelope,
DRAG, and direct AC-Stark detuning conventions match the existing simulation
modules. DRAG uses `Omega_Q = -beta*d(Omega_I)/dt/alpha` and the internal
complex coupling is `Omega_I - i*Omega_Q`.

## Install and run

```bash
python -m pip install -e ".[mps,dev]"
python -m echospec.simulation.mps.benchmark --grid small --full-qutip
python -m echospec.simulation.mps.benchmark --grid full
python -m echospec.simulation.mps.benchmark --grid full --full-qutip
python scripts/run_simulation_backend.py --solver qutip
python scripts/run_simulation_backend.py --solver torch-cpu
python scripts/run_simulation_backend.py --solver mps
```

The last command runs 10,000 separate adaptive QuTiP reference integrations
and can take much longer than the batched PyTorch runs. Without
`--full-qutip`, the benchmark validates QuTiP on representative corner,
resonant/center, and high-amplitude points, then times the full PyTorch grid.
Results are written under `outputs/mps_benchmark/`.

The small validation grid uses a 0.1 us pulse and 400 steps per half. The full
grid uses the production 20 us pulse, 100 by 100 parameter grid, and 30,000
steps per half. `--steps-per-half` can be used for development profiling, but a
reduced-step run must not be reported as the production benchmark.

## Devices and complex values

Pass `device="cpu"` or `device="mps"` to `run_simulation`. Requesting MPS when
`torch.backends.mps.is_available()` is false raises an error. All evolving
tensors are checked to remain on the requested device, and MPS is synchronized
before timing ends.

PyTorch MPS does not support the complex operation set needed here. The solver
therefore stores every density-matrix element as two float32 values (real and
imaginary) and performs complex arithmetic explicitly. It does not silently
move unsupported operations to CPU. Converting the completed state to NumPy
for diagnostics happens only after MPS synchronization and outside the timed
dynamics.

The generated float32 superoperator reconstructs its dependent ground-state
rows so the real and imaginary trace derivatives are exactly zero. Batched
float32 reductions can still accumulate small trace drift over 60,000 steps,
so the returned final density matrix is divided by its complex trace. The
benchmark reports both the raw drift and the post-normalization trace error;
it does not clip populations or hide the pre-normalization diagnostic.
