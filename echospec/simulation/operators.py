"""Quantum operators used by the simulation backend."""

from __future__ import annotations

import numpy as np
import qutip as qt
from qutip import destroy

N_dim: int = 2

a = destroy(N_dim)
n = qt.num(N_dim)
n2 = a.dag() * a.dag() * a * a
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()


def embed_qubit_op(op2: qt.Qobj) -> qt.Qobj:
    op3 = np.zeros((3, 3), dtype=complex)
    op3[:2, :2] = op2.full()
    return qt.Qobj(op3)


if N_dim > 2:
    sx = embed_qubit_op(sx)
    sy = embed_qubit_op(sy)
    sz = embed_qubit_op(sz)
