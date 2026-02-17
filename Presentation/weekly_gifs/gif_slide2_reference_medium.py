"""
Slide 2 GIF — Reference Medium & Polarization Stress
=====================================================
Visualizes the decomposition:
    C(x) = C0 + ΔC(x)
    σ(x) = C0:ε + τ       where τ = ΔC:ε (polarization)

Shows a 2D heterogeneous RVE being split into homogeneous reference
+ perturbation, then equilibrium applied.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 13,
    'mathtext.fontset': 'cm',
})

N = 32  # grid resolution for the RVE
np.random.seed(42)

# ─── Generate a heterogeneous stiffness field (2D Voronoi-like) ───────────────
from scipy.spatial import Voronoi
n_grains = 12
seeds = np.random.rand(n_grains, 2)
# Assign each pixel to nearest seed
xx, yy = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
grain_id = np.zeros((N, N), int)
for iy in range(N):
    for ix in range(N):
        dists = np.sqrt((seeds[:, 0] - xx[iy, ix])**2 + (seeds[:, 1] - yy[iy, ix])**2)
        grain_id[iy, ix] = np.argmin(dists)

# Stiffness per grain (GPa-like)
C_grain = 60 + 120 * np.random.rand(n_grains)  # range 60-180 GPa
C_field = C_grain[grain_id]
C0 = np.mean(C_field)  # reference medium = Voigt average
delta_C = C_field - C0

# ─── Figure setup ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 6.5), facecolor='white')

# Three panels:  C(x)  =  C0  +  ΔC(x)
# Left margin 0.05, right margin 0.05, gap ~0.08 between panels
ax_full = fig.add_axes([0.05, 0.18, 0.24, 0.58])
ax_ref  = fig.add_axes([0.38, 0.18, 0.24, 0.58])
ax_pert = fig.add_axes([0.71, 0.18, 0.24, 0.58])

vmin, vmax = C_field.min(), C_field.max()
dv = max(abs(delta_C.min()), abs(delta_C.max()))

for ax in [ax_full, ax_ref, ax_pert]:
    ax.set_xticks([])
    ax.set_yticks([])

# Static labels — centered over each panel
fig.text(0.17, 0.82, r'$\mathcal{C}(\mathbf{x})$', ha='center', fontsize=20,
         fontweight='bold', color='#1565C0')
fig.text(0.17, 0.10, 'Heterogeneous\nstiffness', ha='center', fontsize=10,
         color='#555')

fig.text(0.50, 0.82, r'$\mathcal{C}^0$', ha='center', fontsize=20,
         fontweight='bold', color='#2E7D32')
fig.text(0.50, 0.10, 'Homogeneous\nreference', ha='center', fontsize=10,
         color='#555')

fig.text(0.83, 0.82, r'$\Delta\mathcal{C}(\mathbf{x})$', ha='center', fontsize=20,
         fontweight='bold', color='#D32F2F')
fig.text(0.83, 0.10, 'Polarization\nperturbation', ha='center', fontsize=10,
         color='#555')

# Operator symbols between panels
eq_text = fig.text(0.34, 0.48, '=', ha='center', va='center', fontsize=28,
                   fontweight='bold', color='black', alpha=0)
plus_text = fig.text(0.66, 0.48, '+', ha='center', va='center', fontsize=28,
                     fontweight='bold', color='black', alpha=0)

# Bottom equation box — safely inside figure
eq_box = fig.text(0.50, 0.03, '', ha='center', va='bottom', fontsize=13,
                  color='#333', fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E1',
                            edgecolor='#FFB300', alpha=0))

# ─── Mutable state ────────────────────────────────────────────────────────────
im_full = ax_full.imshow(np.zeros((N, N)), vmin=vmin, vmax=vmax,
                         cmap='YlOrRd', origin='lower', aspect='equal',
                         extent=[0, 1, 0, 1])
im_ref = ax_ref.imshow(np.full((N, N), C0), vmin=vmin, vmax=vmax,
                       cmap='YlOrRd', origin='lower', aspect='equal',
                       extent=[0, 1, 0, 1])
im_pert = ax_pert.imshow(np.zeros((N, N)), vmin=-dv, vmax=dv,
                         cmap='RdBu_r', origin='lower', aspect='equal',
                         extent=[0, 1, 0, 1])

# Initially hide ref and pert
im_ref.set_alpha(0)
im_pert.set_alpha(0)

# ─── Animation ────────────────────────────────────────────────────────────────
# Phase 1 (0-24):  Build up the heterogeneous field grain by grain
# Phase 2 (25-39): Show "= C0 + ΔC" decomposition
# Phase 3 (40-54): Show equation: σ = C0:ε + τ
# Phase 4 (55-74): Highlight equilibrium: ∇·(C0:ε) + ∇·τ = 0
# Phase 5 (75-89): Hold final
TOTAL = 90

def update(frame):
    # ── Phase 1: reveal grains one by one ──
    if frame <= 24:
        t = frame / 24.0
        n_reveal = int(t * n_grains) + 1
        revealed = np.zeros((N, N))
        for g in range(min(n_reveal, n_grains)):
            revealed[grain_id == g] = C_grain[g]
        # Fade unrevealed to a light gray (map to vmin)
        mask = revealed == 0
        revealed[mask] = vmin * 0.8
        im_full.set_data(revealed)
        im_full.set_alpha(1.0)

        # Draw grain boundaries for revealed grains
        ax_full.set_title(f'Revealing grains ({min(n_reveal, n_grains)}/{n_grains})',
                          fontsize=11, color='#777')

    elif frame <= 39:
        # Phase 2: decomposition appears
        t = (frame - 25) / 14.0
        im_full.set_data(C_field)
        im_full.set_alpha(1.0)
        ax_full.set_title('', fontsize=11)

        im_ref.set_data(np.full((N, N), C0))
        im_ref.set_alpha(min(t * 2, 1.0))
        im_ref.set_clim(vmin, vmax)

        im_pert.set_data(delta_C)
        im_pert.set_alpha(min(t * 2, 1.0))

        eq_text.set_alpha(min(t * 2, 1.0))
        plus_text.set_alpha(min(t * 2, 1.0))

        if t > 0.5:
            ax_ref.set_title(f'$C^0 = {C0:.0f}$ GPa (Voigt avg)', fontsize=10, color='#2E7D32')

    elif frame <= 54:
        # Phase 3: Show polarization stress equation
        t = (frame - 40) / 14.0
        eq_box.set_alpha(min(t * 2, 1.0))
        eq_box.set_text(r'$\sigma = C^0 : \varepsilon + \Delta C : \varepsilon$'
                        r'    (polarization $\tau = \Delta C : \varepsilon$)')
        eq_box.get_bbox_patch().set_alpha(min(t * 2, 1.0))

    elif frame <= 74:
        # Phase 4: Show equilibrium
        t = (frame - 55) / 19.0
        if t > 0.3:
            eq_box.set_text(r'$\nabla \cdot (C^0 : \varepsilon) + \nabla \cdot \tau = 0$'
                            r'   $\rightarrow$ Homogeneous operator + source')

    # Phase 5: hold
    return []


anim = FuncAnimation(fig, update, frames=TOTAL, interval=120, blit=False)
outpath = r'c:\Users\vfn333\Documents\GitHub\FFT_crystals\Presentation\weekly_gifs\slide2_reference_medium.gif'
anim.save(outpath, writer='pillow', fps=8, dpi=130)
plt.close()
print(f'Saved {outpath}')
