"""
Slide 8 GIF — Convergence: Contrast & Reference Medium
=======================================================
Shows how convergence depends on:
  1) Stiffness contrast ratio (C_max / C_min)
  2) Choice of reference medium C0

Animated comparison of convergence curves for different
contrast levels and reference medium choices.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'mathtext.fontset': 'cm',
})

# ─── Simulated convergence curves ─────────────────────────────────────────────
iters = np.arange(120)

# Different scenarios
curves = {
    'Low contrast\n(C_max/C_min = 2)': {
        'rate': 0.35,
        'color': '#2E7D32',
        'ls': '-',
    },
    'Medium contrast\n(C_max/C_min = 10)': {
        'rate': 0.12,
        'color': '#FF9800',
        'ls': '-',
    },
    'High contrast\n(C_max/C_min = 100)': {
        'rate': 0.03,
        'color': '#D32F2F',
        'ls': '-',
    },
    'High contrast +\nAnderson mixing': {
        'rate': 0.15,
        'color': '#1565C0',
        'ls': '--',
    },
}

# Generate residuals
for key, props in curves.items():
    props['residual'] = 0.5 * np.exp(-props['rate'] * iters)
    # Add some noise
    noise = np.random.RandomState(hash(key) % 2**31).randn(len(iters)) * 0.1
    props['residual'] *= np.exp(noise * 0.2)
    props['residual'] = np.maximum(props['residual'], 1e-10)

# ─── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 6.5), facecolor='white')

# Left: convergence plot
ax_conv = fig.add_axes([0.08, 0.12, 0.55, 0.75])
# Right: explanation panel
ax_info = fig.add_axes([0.68, 0.12, 0.30, 0.75])
ax_info.set_axis_off()

fig.text(0.50, 0.95, 'Convergence Depends on Contrast & Reference Medium',
         ha='center', fontsize=15, fontweight='bold', color='#333')

# ─── Left panel setup ────────────────────────────────────────────────────────
ax_conv.set_xlabel('Iteration', fontsize=12)
ax_conv.set_ylabel(r'Residual $\|\nabla\cdot\sigma\| / \|\sigma\|$', fontsize=12)
ax_conv.set_yscale('log')
ax_conv.set_xlim(0, 120)
ax_conv.set_ylim(1e-8, 1)
ax_conv.grid(True, alpha=0.2, which='both')
ax_conv.spines['top'].set_visible(False)
ax_conv.spines['right'].set_visible(False)

# Tolerance line
ax_conv.axhline(1e-6, color='gray', ls=':', lw=1.5)
ax_conv.text(115, 1.5e-6, 'tol', fontsize=10, color='gray', ha='right')

# Create line objects
lines = {}
for key, props in curves.items():
    l, = ax_conv.plot([], [], props['ls'], color=props['color'], lw=2.5, label=key)
    lines[key] = l

ax_conv.legend(fontsize=9, loc='upper right')

# ─── Right panel: explanations ───────────────────────────────────────────────
info_items = [
    ('Spectral radius:', r'$\rho = \frac{C_{\max}/C_{\min} - 1}{C_{\max}/C_{\min} + 1}$',
     '#333', '#555'),
    ('Convergence rate:', r'$\sim \rho^k$', '#333', '#555'),
    ('Higher contrast', r'$\rightarrow$ slower convergence', '#D32F2F', '#D32F2F'),
    ('Anderson mixing', 'Accelerates by extrapolating\nfrom history of iterates',
     '#1565C0', '#1565C0'),
    ('Reference medium', r'$\mathcal{C}^0$ choice controls $\rho$',
     '#2E7D32', '#2E7D32'),
]

info_txts = []
for i, (label, desc, col1, col2) in enumerate(info_items):
    y = 0.90 - i * 0.19
    t1 = ax_info.text(0.0, y, label, transform=ax_info.transAxes,
                      fontsize=12, color=col1, fontweight='bold', va='top', alpha=0)
    t2 = ax_info.text(0.0, y - 0.06, desc, transform=ax_info.transAxes,
                      fontsize=11, color=col2, va='top', alpha=0)
    info_txts.append((t1, t2))

# Convergence iteration counter
iter_txt = ax_conv.text(0.05, 0.05, '', transform=ax_conv.transAxes,
                        fontsize=14, fontweight='bold', color='#555')

# ─── Animation ────────────────────────────────────────────────────────────────
TOTAL = 110

keys = list(curves.keys())

def update(frame):
    # Phase 1 (0-25): Draw curve 1 (low contrast)
    # Phase 2 (20-45): Draw curve 2 (medium contrast)
    # Phase 3 (40-65): Draw curve 3 (high contrast)
    # Phase 4 (60-85): Draw curve 4 (Anderson)
    # Phase 5 (70-100): Info items appear
    # Phase 6 (95+): Hold

    for idx, key in enumerate(keys):
        start = idx * 20
        end = start + 40
        if frame >= start:
            n_pts = min(int((frame - start) / 40 * 120), 120)
            n_pts = max(n_pts, 1)
            props = curves[key]
            lines[key].set_data(iters[:n_pts], props['residual'][:n_pts])

    # Info items
    for i in range(len(info_items)):
        start_info = 50 + i * 10
        if frame >= start_info:
            t = min((frame - start_info) / 8, 1.0)
            info_txts[i][0].set_alpha(t)
            info_txts[i][1].set_alpha(t)

    # Iteration counter
    if frame < 80:
        max_iter = min(int(frame / 80 * 120), 120)
        iter_txt.set_text(f'Iteration: {max_iter}')

    return []

anim = FuncAnimation(fig, update, frames=TOTAL, interval=120, blit=False)
outpath = r'c:\Users\vfn333\Documents\GitHub\FFT_crystals\Presentation\weekly_gifs\slide8_convergence.gif'
anim.save(outpath, writer='pillow', fps=8, dpi=130)
plt.close()
print(f'Saved {outpath}')
