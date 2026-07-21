# Detuning convention

All active code, notebooks, plots, and manuscript equations use the
spectroscopy convention

$$
\Delta = \omega_d - \omega_{01},
\qquad
\frac{\Delta}{2\pi} = f_d - f_{01}.
$$

Positive detuning therefore means that the applied drive is above the bare
qubit transition. In the rotating frame, the transmon Hamiltonian is

$$
\frac{H}{\hbar}
= -\Delta a^\dagger a
+ \frac{\alpha}{2}a^{\dagger 2}a^2
+ \frac{\Omega(t)}{2}(a+a^\dagger).
$$

For the project's negative transmon anharmonicity, the direct
$|e\rangle\leftrightarrow|f\rangle$ transition lies at
$\Delta/(2\pi)=\alpha/(2\pi)<0$, and the
$|g\rangle\leftrightarrow|f\rangle$ two-photon feature lies at
$\Delta/(2\pi)=\alpha/(4\pi)<0$. Both must appear to the left of the bare
$|g\rangle\leftrightarrow|e\rangle$ resonance.

Saved OPX1000 `data__detuning` arrays already follow this convention: the
absolute drive frequency is `qubit_f01 + detuning`. Historical archives and
raw measurement bundles are not rewritten; consumers interpret their stored
detuning according to the acquisition metadata.
