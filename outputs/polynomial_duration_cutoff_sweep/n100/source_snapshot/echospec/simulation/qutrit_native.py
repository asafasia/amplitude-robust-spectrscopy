"""Optional fused CPU RK4 kernel; same equations and sampling as qutrit.py.

Requires a C++17 compiler. Compiled artifacts are cached in the system temporary
directory by source hash. No external packages beyond NumPy are needed.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_LIBRARY = None


def _kernel():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY.qutrit_rk4
    source = Path(__file__).with_name("qutrit_rk4.cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    library = Path(tempfile.gettempdir()) / f"echospec-qutrit-{digest}.so"
    if not library.exists():
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            raise RuntimeError("The native RK4 backend requires clang++ or g++")
        temporary = library.with_name(library.name + f".{os.getpid()}")
        subprocess.run([compiler, "-std=c++17", "-O3", "-shared", "-fPIC",
                        str(source), "-o", str(temporary)], check=True, capture_output=True)
        temporary.replace(library)
    _LIBRARY = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    _LIBRARY.qutrit_rk4.argtypes = [array, ctypes.c_int, ctypes.c_int, ctypes.c_double,
                                  array, array, array, array,
                                  ctypes.c_double, ctypes.c_double, ctypes.c_double]
    _LIBRARY.qutrit_rk4.restype = None
    return _LIBRARY.qutrit_rk4


def integrate_native(state, *, coordinate_start, coordinate_stop, num_steps,
                     detuning, rabi, anharmonicity, inv_t1, inv_t_phi,
                     drive_scale, time_jacobian, stark_kappa_mhz_inv,
                     stark_correction_mode, accumulated_phase, phase_time_origin):
    """Drop-in integration segment for zero-Stark, uniform-time simulations."""
    if stark_kappa_mhz_inv != 0 or np.any(accumulated_phase != 0):
        raise ValueError("Native integration requires zero Stark correction and phase")
    grid = np.linspace(coordinate_start, coordinate_stop, 2*num_steps+1)
    if not all(time_jacobian(t) == 1. for t in grid):
        raise ValueError("Native integration requires the physical-time coordinate")
    drive = np.array([drive_scale(t) for t in grid], dtype=complex)
    shape = state.shape[1:]
    real_state = np.ascontiguousarray(np.stack([
        state[0].real, state[1].real, state[2].real,
        state[3].real, state[3].imag, state[4].real, state[4].imag,
        state[5].real, state[5].imag]).reshape(9, -1))
    det = np.ascontiguousarray(np.broadcast_to(detuning, shape).ravel())
    amp = np.ascontiguousarray(np.broadcast_to(rabi, shape).ravel())
    _kernel()(real_state.ravel(), det.size, num_steps,
              (coordinate_stop-coordinate_start)/num_steps, det, amp,
              np.ascontiguousarray(drive.real), np.ascontiguousarray(drive.imag),
              anharmonicity, inv_t1, inv_t_phi)
    values = real_state.reshape(9, *shape)
    state[:3] = values[:3]
    for j in range(3):
        state[j+3] = values[2*j+3] + 1j*values[2*j+4]
    return state, accumulated_phase
