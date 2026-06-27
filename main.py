import jax.numpy as jnp
import matplotlib.pyplot as plt
import qutip_jax as qj
from diffrax import PIDController, Tsit5
from jax import default_device, devices, grad, jacfwd, jacrev, jit
from qutip import (CoreOptions, about, basis, destroy, lindblad_dissipator,
                   liouvillian, mcsolve, mesolve, projection, qeye, settings,
                   sigmam, sigmax, sigmay, sigmaz, spost, spre, sprepost,
                   steadystate, tensor)


# system parameters
ed = 1
GammaL = 1
GammaR = 1

# simulation parameters
options = {
    "method": "diffrax",
    "normalize_output": False,
    "stepsize_controller": PIDController(rtol=1e-7, atol=1e-7),
    "solver": Tsit5(scan_kind="bounded"),
    "progress_bar": False,
}


print(qeye(3, dtype="jax").dtype.__name__)
print(qeye(3, dtype="jaxdia").dtype.__name__)


qj.set_as_default()
