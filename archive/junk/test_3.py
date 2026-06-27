import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set publication-quality style
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.major.width'] = 1.2
rcParams['ytick.major.width'] = 1.2
rcParams['xtick.major.size'] = 5
rcParams['ytick.major.size'] = 5

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# ==================== Panel (a) ====================
phi = np.linspace(0, 1, 1000)

# Main curve χ/2π
chi = -15 + 30*phi - 60*phi**2 + 60*phi**3
chi = chi - 5*np.sin(15*np.pi*phi)  # Add oscillations

# Plot main curve
ax1.plot(phi, chi, color='#1f77b4', linewidth=2.5, label=r'$\chi/2\pi$')

# Add vertical lines for Δ₃₁ → 0 (dotted gray)
delta31_positions = [0.25, 0.55]
for pos in delta31_positions:
    ax1.axvline(pos, color='gray', linestyle=':', linewidth=2, alpha=0.8)

# Add vertical lines for Δ₂₀ → 0 (dashed green)
delta20_positions = [0.15, 0.70]
for pos in delta20_positions:
    ax1.axvline(pos, color='#2ca02c', linestyle='--', linewidth=2, alpha=0.8)

# Add RO points (magenta circles)
ro_points = [(0.1, 10), (0.58, -7.5)]
for x, y in ro_points:
    ax1.plot(x, y, 'o', color='#d62728', markersize=9, markeredgewidth=1.5, 
             markeredgecolor='#8B0000', zorder=5)

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#d62728', marker='o', linestyle='', markersize=8, 
           markeredgewidth=1.5, markeredgecolor='#8B0000', label='RO'),
    Line2D([0], [0], color='#1f77b4', linewidth=2.5, label=r'$\chi/2\pi$'),
    Line2D([0], [0], color='gray', linestyle=':', linewidth=2, label=r'$\Delta_{31} \rightarrow 0$'),
    Line2D([0], [0], color='#2ca02c', linestyle='--', linewidth=2, label=r'$\Delta_{20} \rightarrow 0$')
]
ax1.legend(handles=legend_elements, loc='upper left', frameon=True, 
          fancybox=False, edgecolor='black', fontsize=11)

# Formatting panel (a)
ax1.set_xlabel(r'$\Phi_{\mathrm{ext}}/\Phi_0$', fontsize=13)
ax1.set_ylabel(r'$\chi/2\pi$ (MHz)', fontsize=13)
ax1.set_xlim(0, 1)
ax1.set_ylim(-15, 11)
ax1.grid(True, alpha=0.3, linewidth=0.5)
ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontsize=14, 
         fontweight='bold', va='top')

# ==================== Panel (b) ====================
phi2 = np.linspace(0, 1, 1000)

# Create different transition curves
delta30 = 10 - 20*np.cos(2*np.pi*phi2)
delta20 = 6 - 8*np.cos(2*np.pi*phi2) - 1*np.cos(4*np.pi*phi2)
delta31 = 3 - 6*np.cos(2*np.pi*phi2) + 0.5*np.cos(6*np.pi*phi2)
delta10 = 0 - 2*np.cos(2*np.pi*phi2) - 1.5*np.cos(4*np.pi*phi2)
delta21 = -2 - 6*np.cos(2*np.pi*phi2) + 1*np.cos(4*np.pi*phi2)

# Plot curves with specific colors
ax2.plot(phi2, delta30, color='#d62728', linewidth=2.5, label=r'$\Delta_{30}$')
ax2.plot(phi2, delta20, color='#2ca02c', linewidth=2.5, label=r'$\Delta_{20}$')
ax2.plot(phi2, delta31, color='#9467bd', linewidth=2.5, label=r'$\Delta_{31}$')
ax2.plot(phi2, delta10, color='#ff7f0e', linewidth=2.5, label=r'$\Delta_{10}$')
ax2.plot(phi2, delta21, color='#ffbb33', linewidth=2.5, label=r'$\Delta_{21}$')

# Add horizontal dashed line at zero
ax2.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

# Formatting panel (b)
ax2.set_xlabel(r'$\Phi_{\mathrm{ext}}/\Phi_0$', fontsize=13)
ax2.set_ylabel(r'$\Delta_{ij}/2\pi$ (GHz)', fontsize=13)
ax2.set_xlim(0, 1)
ax2.set_ylim(-6, 10.5)
ax2.grid(True, alpha=0.3, linewidth=0.5)
ax2.legend(loc='center left', frameon=True, fancybox=False, 
          edgecolor='black', fontsize=11)
ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontsize=14, 
         fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('scientific_figure.png', dpi=300, bbox_inches='tight')
plt.show()