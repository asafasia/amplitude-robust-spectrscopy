

import h5py
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

current_dir = Path(__file__).parent.parent/"data"
file_name = "lorentzian_echo0.001_10_us.npz"

data = np.load(current_dir/file_name)

x, y, z = data["detunings"], data["amplitudes"], data["matrix"]

plt.pcolormesh(x, y, z, shading="auto")

plt.xlabel("Detuning (Hz)")
plt.ylabel("Rabi Frequency (Hz)")
plt.title("Final Z Expectation Value")
plt.colorbar(label="<Z>")
plt.show()
