"""
Slide 5 GIF — Explicit Spectral Green Operator
===============================================
Visualizes the acoustic tensor A(n), its inverse, and how
the Green operator Γ0 is applied at each Fourier mode.

Shows:
  1) Wave direction n = k/|k| on a circle
  2) Building A(n) = μ₀I + (λ₀+μ₀) n⊗n
  3) Inverting A(n)
  4) Computing b = τ·n, u = A⁻¹·b, ε' = sym(n⊗u)
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch, Circle
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'mathtext.fontset': 'cm',
})

# Reference medium parameters
lam0 = 121.4e9   # λ₀ (Pa) - copper-like
mu0 = 75.4e9     # μ₀ (Pa)
alpha_val = (lam0 + mu0) / (mu0 * (lam0 + 2*mu0))

# ─── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 6.5), facecolor='white')

# Left: wave direction visualization (unit circle in 2D)
ax_wave = fig.add_axes([0.02, 0.10, 0.30, 0.75])
# Center: tensor display
ax_tensor = fig.add_axes([0.36, 0.10, 0.30, 0.75])
ax_tensor.set_axis_off()
# Right: computation steps
ax_steps = fig.add_axes([0.70, 0.10, 0.28, 0.75])
ax_steps.set_axis_off()

# ─── Left panel: wave direction ──────────────────────────────────────────────
ax_wave.set_xlim(-1.5, 1.5)
ax_wave.set_ylim(-1.5, 1.5)
ax_wave.set_aspect('equal')
ax_wave.set_xticks([])
ax_wave.set_yticks([])
ax_wave.set_title('Wave direction $\\mathbf{n}$', fontsize=13, fontweight='bold',
                   color='#333')

# Draw unit circle
theta_circle = np.linspace(0, 2*np.pi, 100)
ax_wave.plot(np.cos(theta_circle), np.sin(theta_circle), '-', color='#BBBBBB', lw=1.5)
ax_wave.axhline(0, color='#DDDDDD', lw=0.5)
ax_wave.axvline(0, color='#DDDDDD', lw=0.5)
ax_wave.text(0, -1.35, r'$\mathbf{n} = \mathbf{k}/\|\mathbf{k}\|$',
             ha='center', fontsize=11, color='#666')

# Arrow for current direction (will be updated)
arrow_line, = ax_wave.plot([0, 1], [0, 0], '-', color='#1565C0', lw=3)
arrow_dot, = ax_wave.plot([1], [0], 'o', color='#1565C0', ms=10)

# n⊗n visualization: streak marks on circle
n_trail = ax_wave.scatter([], [], s=30, c='#BBBBBB', alpha=0.4, zorder=0)

# ─── Mutable text ─────────────────────────────────────────────────────────────
tensor_texts = []

# Top equation
fig.text(0.50, 0.93,
    r'Acoustic tensor:  $A_{ik}(\mathbf{n}) = \mu_0\,\delta_{ik} + (\lambda_0+\mu_0)\,n_i n_k$',
    ha='center', fontsize=14, fontweight='bold', color='#1565C0',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#1565C0'))

# ─── Animation ────────────────────────────────────────────────────────────────
TOTAL = 100
trail_xs = []
trail_ys = []

def clear_texts():
    for t in tensor_texts:
        try:
            t.remove()
        except:
            pass
    tensor_texts.clear()

def update(frame):
    clear_texts()

    # Rotate n around the circle
    theta = frame / TOTAL * 2 * np.pi * 1.5  # 1.5 revolutions
    nx = np.cos(theta)
    ny = np.sin(theta)

    # Update arrow
    arrow_line.set_data([0, nx], [0, ny])
    arrow_dot.set_data([nx], [ny])

    # Trail
    trail_xs.append(nx)
    trail_ys.append(ny)
    if len(trail_xs) > 40:
        trail_xs.pop(0)
        trail_ys.pop(0)
    n_trail.set_offsets(np.c_[trail_xs, trail_ys])

    # Direction label
    t1 = ax_wave.text(nx * 1.25, ny * 1.25,
                      f'$\\mathbf{{n}}=({nx:.2f}, {ny:.2f})$',
                      fontsize=10, color='#1565C0', ha='center', fontweight='bold')
    tensor_texts.append(t1)

    # ── Center: show A matrix and A⁻¹ ──
    n = np.array([nx, ny])
    nxn = np.outer(n, n)
    A = mu0 * np.eye(2) + (lam0 + mu0) * nxn
    A_inv = np.eye(2) / mu0 - alpha_val * nxn

    # Normalize for display
    A_display = A / 1e9  # in GPa
    A_inv_display = A_inv * 1e9  # inverse units

    # A matrix
    y_start = 0.85
    t = ax_tensor.text(0.5, y_start, r'$\mathbf{A}(\mathbf{n})$ (GPa)',
                       transform=ax_tensor.transAxes, ha='center',
                       fontsize=13, fontweight='bold', color='#333')
    tensor_texts.append(t)

    for i in range(2):
        for j in range(2):
            val = A_display[i, j]
            color = '#1565C0' if abs(val) > 1 else '#999'
            t = ax_tensor.text(0.3 + j * 0.4, y_start - 0.08 - i * 0.07,
                              f'{val:.1f}',
                              transform=ax_tensor.transAxes, ha='center',
                              fontsize=12, color=color, fontweight='bold',
                              fontfamily='monospace')
            tensor_texts.append(t)

    # A⁻¹ matrix
    y_start2 = 0.55
    t = ax_tensor.text(0.5, y_start2, r'$\mathbf{A}^{-1}(\mathbf{n})$ (GPa$^{-1}$)',
                       transform=ax_tensor.transAxes, ha='center',
                       fontsize=13, fontweight='bold', color='#D32F2F')
    tensor_texts.append(t)

    for i in range(2):
        for j in range(2):
            val = A_inv_display[i, j]
            color = '#D32F2F' if abs(val) > 0.001 else '#999'
            t = ax_tensor.text(0.3 + j * 0.4, y_start2 - 0.08 - i * 0.07,
                              f'{val:.4f}',
                              transform=ax_tensor.transAxes, ha='center',
                              fontsize=11, color=color, fontfamily='monospace')
            tensor_texts.append(t)

    # Closed form
    t = ax_tensor.text(0.5, 0.22,
        r'$A^{-1}_{ik} = \frac{\delta_{ik}}{\mu_0} - \alpha\, n_i n_k$',
        transform=ax_tensor.transAxes, ha='center', fontsize=13,
        color='#D32F2F',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FBE9E7', edgecolor='#D32F2F'))
    tensor_texts.append(t)

    t = ax_tensor.text(0.5, 0.10,
        r'$\alpha = \frac{\lambda_0+\mu_0}{\mu_0(\lambda_0+2\mu_0)}$',
        transform=ax_tensor.transAxes, ha='center', fontsize=12, color='#888')
    tensor_texts.append(t)

    # ── Right panel: computation pipeline ──
    steps = [
        (r'$\mathbf{b} = \hat{\tau}\cdot\mathbf{n}$', 'Project stress', '#1565C0'),
        (r'$\hat{\mathbf{u}} = \mathbf{A}^{-1}\cdot\mathbf{b}$', 'Solve displacement', '#D32F2F'),
        (r'$\hat{\varepsilon}\,' + "'" + r'= \mathrm{sym}(\mathbf{n}\otimes\hat{\mathbf{u}})$',
         'Symmetrize', '#2E7D32'),
        (r'At $\mathbf{k}=0$: $\hat{\varepsilon}=\mathbf{E}$',
         'Zero-mode = macro strain', '#E65100'),
    ]

    phase = min(frame / 20, 4)
    for i, (eq, desc, col) in enumerate(steps):
        if phase > i:
            alpha = min((phase - i) * 2, 1.0)
            y = 0.85 - i * 0.22
            t = ax_steps.text(0.05, y, eq, transform=ax_steps.transAxes,
                             fontsize=13, color=col, va='top', alpha=alpha,
                             fontweight='bold')
            tensor_texts.append(t)
            t = ax_steps.text(0.05, y - 0.07, desc, transform=ax_steps.transAxes,
                             fontsize=9, color='#888', va='top', style='italic',
                             alpha=alpha)
            tensor_texts.append(t)

    return []

anim = FuncAnimation(fig, update, frames=TOTAL, interval=120, blit=False)
outpath = r'c:\Users\vfn333\Documents\GitHub\FFT_crystals\Presentation\weekly_gifs\slide5_green_operator.gif'
anim.save(outpath, writer='pillow', fps=8, dpi=130)
plt.close()
print(f'Saved {outpath}')
