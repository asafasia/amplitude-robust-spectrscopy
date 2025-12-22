from typing import Final
import numpy as np


class Units:
    kHz: Final[float] = 1e3
    MHz: Final[float] = 1e6
    GHz: Final[float] = 1e9

    us: Final[float] = 1e-6
    ns: Final[float] = 1e-9
    ps: Final[float] = 1e-12

    pi2: Final[float] = 2 * np.pi
