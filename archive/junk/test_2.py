import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "legend.fontsize": 9,
})

# Create a custom grid with unequal row/col sizes
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(3, 4, width_ratios=[1.2, 1.6, 1, 2], height_ratios=[1.5, 1, 1.2])

# ------------------ (a) small square ------------------
ax_a = fig.add_subplot(gs[0,0])
ax_a.plot([0,1],[0,1], 'r-')
ax_a.set_title("(a)", loc="left")
ax_a.axis("off")

# ------------------ (b) wide plot ------------------
ax_b = fig.add_subplot(gs[0,1:3])
x = np.linspace(4.3, 5.2, 40)
ax_b.plot(x, np.exp(-(x-4.6)**2/0.05), label="Theory")
ax_b.scatter(x, np.exp(-(x-4.6)**2/0.05)+0.1*np.random.randn(len(x)), facecolors="none", edgecolors="k", label="Data")
ax_b.set_xlabel(r"$\omega_c/2\pi$ (GHz)")
ax_b.set_ylabel(r"$\zeta/2\pi$ (MHz)")
ax_b.legend()
ax_b.set_title("(b)", loc="left")

# ------------------ (c) tall narrow plot ------------------
ax_c = fig.add_subplot(gs[:,3])  # spans all rows
gate_len = np.linspace(20,120,25)
phi = np.sin(gate_len/15)*4
ax_c.scatter(gate_len, phi, color="k")
ax_c.axhline(0, ls="--", color="gray")
ax_c.set_xlabel(r"Gate length $t_G$ (ns)")
ax_c.set_ylabel(r"$\phi_{ZZ}$ (deg)")
ax_c.set_title("(c)", loc="left")

# ------------------ (d) medium panel ------------------
ax_d = fig.add_subplot(gs[1,0:2])
N = np.arange(0, 100, 5)
ax_d.plot(N, 2*N, "o-", label="case 1")
ax_d.plot(N, -N, "o-", label="case 2")
ax_d.legend()
ax_d.set_xlabel(r"$N_{\mathrm{iSWAP}}$")
ax_d.set_ylabel(r"$\phi_{ZZ}$ (deg)")
ax_d.set_title("(d)", loc="left")

# ------------------ (e) long panel ------------------
ax_e = fig.add_subplot(gs[2,0:3])
cliffords = np.arange(0, 90, 5)
fid1 = np.exp(-cliffords/70)
fid2 = np.exp(-cliffords/55)
ax_e.errorbar(cliffords, fid1, yerr=0.02, fmt="o-", label="Standard RB")
ax_e.errorbar(cliffords, fid2, yerr=0.02, fmt="o-", label="Interleaved RB")
ax_e.set_xlabel(r"Number of Cliffords $N_{\mathrm{Clifford}}$")
ax_e.set_ylabel("Sequence fidelity")
ax_e.legend()
ax_e.set_title("(e)", loc="left")

plt.tight_layout()

plt.savefig("custom_grid_figure.png", dpi=300)
plt.show()