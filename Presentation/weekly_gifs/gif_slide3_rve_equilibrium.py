"""
Slide 3 GIF — Periodic RVE: Kinematics & Equilibrium
=====================================================
Shows a periodic 2D RVE with:
  - Voronoi grains tiling periodically
  - Strain field ε = sym(∇u)
  - Equilibrium ∇·σ = 0
  - Macroscopic loading constraint <ε> = E or <σ> = Σ

Animated: the grains appear, then periodic copies tile,
then loading is applied and strain/stress fields emerge.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'mathtext.fontset': 'cm',
})

N = 40
np.random.seed(7)

# ─── Build a Voronoi grain structure ─────────────────────────────────────────
n_grains = 10
seeds = np.random.rand(n_grains, 2)

xx, yy = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))

# Periodic Voronoi: replicate seeds to 3x3, assign, then map back
seeds_periodic = []
for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        seeds_periodic.append(seeds + np.array([dx, dy]))
seeds_periodic = np.vstack(seeds_periodic)

grain_id = np.zeros((N, N), int)
for iy in range(N):
    for ix in range(N):
        dists = np.sqrt((seeds_periodic[:, 0] - xx[iy, ix])**2 +
                        (seeds_periodic[:, 1] - yy[iy, ix])**2)
        nearest = np.argmin(dists)
        grain_id[iy, ix] = nearest % n_grains  # map back to original grain

# Random grain orientations (Euler angle, just for color)
grain_angle = np.random.rand(n_grains) * 180

# Assign a stiffness per grain
C_grain = 80 + 100 * np.random.rand(n_grains)
C_field = C_grain[grain_id]

# Fake strain field: smooth random field + grain-boundary jumps
from scipy.ndimage import gaussian_filter
eps_field = gaussian_filter(np.random.randn(N, N), sigma=3) * 0.01
eps_field += (C_field - np.mean(C_field)) / np.std(C_field) * 0.005

# ─── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 6), facecolor='white')

# Left: periodic RVE (with tiling indication)
ax_rve = fig.add_axes([0.02, 0.10, 0.45, 0.75])
# Right: equation panel
ax_eq = fig.add_axes([0.52, 0.10, 0.46, 0.75])
ax_eq.set_axis_off()

ax_rve.set_xticks([])
ax_rve.set_yticks([])

# Colormap for grain orientation
cmap_grain = plt.cm.tab20
norm_grain = mcolors.Normalize(vmin=0, vmax=n_grains)
grain_colors = grain_angle[grain_id]

im_rve = ax_rve.imshow(grain_colors, cmap='hsv', origin='lower',
                       extent=[0, 1, 0, 1], alpha=0)

# Right panel: equations will appear step by step
equations = [
    (r'$\varepsilon(\mathbf{x}) = \nabla^s \mathbf{u}(\mathbf{x}) = '
     r'\frac{1}{2}\left(\nabla\mathbf{u} + \nabla\mathbf{u}^T\right)$',
     'Kinematics: strain from displacement gradient',
     '#1565C0'),
    (r'$\sigma(\mathbf{x}) = \mathcal{C}(\mathbf{x}) : \varepsilon(\mathbf{x})$',
     'Constitutive: each grain has its own stiffness',
     '#D32F2F'),
    (r'$\nabla \cdot \sigma(\mathbf{x}) = \mathbf{0} \;\;\mathrm{in}\;\;\Omega$',
     'Equilibrium: stress divergence-free',
     '#2E7D32'),
    (r'$\langle\varepsilon\rangle = \mathbf{E}$  or  $\langle\sigma\rangle = \Sigma$',
     'Macroscopic loading constraint',
     '#E65100'),
]

eq_texts = []
desc_texts = []
for i, (eq, desc, col) in enumerate(equations):
    y_pos = 0.82 - i * 0.22
    t1 = ax_eq.text(0.05, y_pos, eq, transform=ax_eq.transAxes,
                    fontsize=15, color=col, va='top', alpha=0)
    t2 = ax_eq.text(0.05, y_pos - 0.08, desc, transform=ax_eq.transAxes,
                    fontsize=10, color='#777', va='top', style='italic', alpha=0)
    eq_texts.append(t1)
    desc_texts.append(t2)

# Title
fig.text(0.50, 0.93, 'Periodic Heterogeneous Elasticity on an RVE',
         ha='center', fontsize=16, fontweight='bold', color='#333')

# Periodic boundary indicators (dashed boxes)
period_lines = []

# ─── Animation ────────────────────────────────────────────────────────────────
# Phase 1 (0-19):  Grain structure appears
# Phase 2 (20-34): Show periodic tiling hints (ghost copies fade in)
# Phase 3 (35-49): Loading arrows appear
# Phase 4 (50-79): Equations appear one by one
# Phase 5 (80-99): Show strain field overlay
TOTAL = 100

ghost_ims = []

def update(frame):
    # Phase 1: grain structure
    if frame <= 19:
        t = frame / 19.0
        im_rve.set_alpha(min(t * 1.5, 1.0))
        ax_rve.set_title('Polycrystalline RVE', fontsize=12, color='#555')

    # Phase 2: periodic copies hint
    elif frame <= 34:
        t = (frame - 20) / 14.0
        if frame == 20:
            # Draw dashed boundary
            for side in ['top', 'bottom', 'left', 'right']:
                ax_rve.spines[side].set_linestyle('--')
                ax_rve.spines[side].set_color('#1565C0')
                ax_rve.spines[side].set_linewidth(2)
            # Add periodic arrows
            ax_rve.annotate('', xy=(1.02, 0.5), xytext=(1.12, 0.5),
                           xycoords='axes fraction',
                           arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))
            ax_rve.annotate('', xy=(-0.02, 0.5), xytext=(-0.12, 0.5),
                           xycoords='axes fraction',
                           arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))
            ax_rve.text(1.07, 0.7, 'periodic', transform=ax_rve.transAxes,
                       fontsize=9, color='#1565C0', rotation=90, ha='center')

        ax_rve.set_title('Periodic boundary conditions', fontsize=12, color='#1565C0')

    # Phase 3: loading arrows
    elif frame <= 49:
        t = (frame - 35) / 14.0
        if frame == 35:
            # Tensile arrows top/bottom
            for yy_pos in [0.0, 1.0]:
                sign = 1 if yy_pos > 0.5 else -1
                for xx_pos in [0.2, 0.5, 0.8]:
                    ax_rve.annotate('', 
                                   xy=(xx_pos, yy_pos + sign * 0.01),
                                   xytext=(xx_pos, yy_pos + sign * 0.12),
                                   arrowprops=dict(arrowstyle='->', color='#D32F2F',
                                                   lw=2.5))
            ax_rve.set_title(r'Apply $\langle\sigma\rangle = \Sigma$', 
                            fontsize=12, color='#D32F2F')

    # Phase 4: equations one by one
    elif frame <= 79:
        eq_idx = min(int((frame - 50) / 7.5), 3)
        for i in range(eq_idx + 1):
            t_eq = min(((frame - 50) - i * 7.5) / 5.0, 1.0)
            if t_eq > 0:
                eq_texts[i].set_alpha(min(t_eq, 1.0))
                desc_texts[i].set_alpha(min(t_eq, 1.0))

    # Phase 5: strain field overlay
    elif frame <= 99:
        t = (frame - 80) / 19.0
        if frame == 80:
            ax_rve.imshow(eps_field, cmap='coolwarm', origin='lower',
                         extent=[0, 1, 0, 1], alpha=0.0)
        # Overlay strain
        blend = min(t * 1.5, 0.6)
        # Redraw with blending
        ax_rve.images[-1].set_alpha(blend) if len(ax_rve.images) > 1 else None
        ax_rve.set_title(r'Strain field $\varepsilon(\mathbf{x})$',
                        fontsize=12, color='#2E7D32')

    return []

anim = FuncAnimation(fig, update, frames=TOTAL, interval=120, blit=False)
outpath = r'c:\Users\vfn333\Documents\GitHub\FFT_crystals\Presentation\weekly_gifs\slide3_rve_equilibrium.gif'
anim.save(outpath, writer='pillow', fps=8, dpi=130)
plt.close()
print(f'Saved {outpath}')
