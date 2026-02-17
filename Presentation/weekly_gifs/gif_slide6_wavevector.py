"""
Slide 6 GIF — Continuous vs Discrete Wavevector
================================================
Compares different discrete derivative schemes:
  - Continuous: k_i = 2π ξ_i / L
  - Finite-difference: k*_i = (2/h) sin(π ξ_i / N)

Shows how the wave vector modifies the Green operator and
its impact on Gibbs ringing near grain boundaries.
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

N = 64
L = 1.0
h = L / N

# Frequency indices
xi = np.arange(N)
xi_shifted = np.where(xi <= N//2, xi, xi - N)

# Continuous wavevector
k_cont = 2 * np.pi * xi_shifted / L

# FD-compatible wavevector
k_fd = (2.0 / h) * np.sin(np.pi * xi_shifted / N)

# Rotated scheme
k_rot = N * np.sin(2 * np.pi * xi_shifted / N)

# ─── Generate a step-function field (high contrast) ──────────────────────────
x = np.linspace(0, L, N, endpoint=False)
# Step function: 0 for x < 0.5, 1 for x >= 0.5
field_step = np.where(x < 0.5, 0.0, 1.0)

# Apply derivative using each scheme
def spectral_deriv(field, k_vec):
    f_hat = np.fft.fft(field)
    df_hat = 1j * k_vec * f_hat
    return np.real(np.fft.ifft(df_hat))

deriv_cont = spectral_deriv(field_step, k_cont)
deriv_fd = spectral_deriv(field_step, k_fd)
deriv_rot = spectral_deriv(field_step, k_rot)

# ─── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 6), facecolor='white')

# Top row: wavevector comparison
ax_k = fig.add_axes([0.07, 0.55, 0.40, 0.35])
# Top right: equations
ax_eq = fig.add_axes([0.55, 0.55, 0.40, 0.35])
ax_eq.set_axis_off()
# Bottom: derivative comparison with Gibbs effect
ax_deriv = fig.add_axes([0.07, 0.10, 0.88, 0.38])

# Title
fig.text(0.50, 0.96, 'Discrete Wavevector Schemes', ha='center',
         fontsize=16, fontweight='bold', color='#333')

# ─── Top left: k vs ξ ─────────────────────────────────────────────────────────
xi_plot = xi_shifted[xi_shifted >= 0]
k_cont_plot = k_cont[xi_shifted >= 0]
k_fd_plot = k_fd[xi_shifted >= 0]
k_rot_plot = k_rot[xi_shifted >= 0]

sort_idx = np.argsort(xi_plot)
xi_plot = xi_plot[sort_idx]
k_cont_plot = k_cont_plot[sort_idx]
k_fd_plot = k_fd_plot[sort_idx]
k_rot_plot = k_rot_plot[sort_idx]

line_cont, = ax_k.plot([], [], '-', color='#1565C0', lw=2.5, label='Continuous')
line_fd, = ax_k.plot([], [], '--', color='#D32F2F', lw=2.5, label='Finite Difference')
line_rot, = ax_k.plot([], [], ':', color='#2E7D32', lw=2.5, label='Rotated')

ax_k.set_xlim(0, N//2)
ax_k.set_ylim(0, max(k_cont_plot) * 1.1)
ax_k.set_xlabel(r'Frequency index $\xi$', fontsize=11)
ax_k.set_ylabel(r'Wavevector $k$', fontsize=11)
ax_k.set_title(r'$k(\xi)$ comparison', fontsize=12, color='#555')
ax_k.legend(fontsize=9, loc='upper left')
ax_k.grid(True, alpha=0.2)
ax_k.spines['top'].set_visible(False)
ax_k.spines['right'].set_visible(False)

# ─── Top right: equations ─────────────────────────────────────────────────────
eqs = [
    (r'Continuous:  $k_i = 2\pi\,\xi_i / L$', '#1565C0'),
    (r'FD:  $k_i^* = \frac{2}{h}\sin\!\left(\frac{\pi\xi_i}{N}\right)$', '#D32F2F'),
    (r'Rotated:  $k_i = N\sin\!\left(\frac{2\pi\xi_i}{N}\right)$', '#2E7D32'),
]
eq_texts = []
for i, (eq, col) in enumerate(eqs):
    t = ax_eq.text(0.05, 0.85 - i * 0.30, eq, transform=ax_eq.transAxes,
                   fontsize=13, color=col, va='top', fontweight='bold', alpha=0)
    eq_texts.append(t)

# ─── Bottom: derivative comparison ───────────────────────────────────────────
ax_deriv.set_xlim(0, L)
y_max = max(abs(deriv_cont.max()), abs(deriv_fd.max()), abs(deriv_rot.max())) * 1.1
ax_deriv.set_ylim(-y_max * 0.3, y_max)
ax_deriv.set_xlabel(r'$x / L$', fontsize=11)
ax_deriv.set_ylabel(r"$\partial f / \partial x$", fontsize=11)
ax_deriv.set_title('Spectral derivative of a step function (Gibbs effect)', fontsize=12, color='#555')
ax_deriv.axhline(0, color='gray', lw=0.5)
ax_deriv.grid(True, alpha=0.2)
ax_deriv.spines['top'].set_visible(False)
ax_deriv.spines['right'].set_visible(False)

dline_cont, = ax_deriv.plot([], [], '-', color='#1565C0', lw=2, label='Continuous')
dline_fd, = ax_deriv.plot([], [], '--', color='#D32F2F', lw=2, label='FD-compatible')
dline_rot, = ax_deriv.plot([], [], ':', color='#2E7D32', lw=2, label='Rotated')
ax_deriv.legend(fontsize=9)

# Annotation for Gibbs
gibbs_txt = ax_deriv.text(0.30, y_max * 0.8, '', fontsize=11, color='#D32F2F',
                          fontweight='bold')

# ─── Animation ────────────────────────────────────────────────────────────────
TOTAL = 90

def update(frame):
    t = frame / TOTAL

    # Phase 1: Draw wavevector curves (0-30)
    if frame <= 30:
        n_pts = max(1, int(t * 3 * len(xi_plot)))
        line_cont.set_data(xi_plot[:n_pts], k_cont_plot[:n_pts])

    # Phase 2: Add FD curve + equation (20-45)
    if frame >= 20 and frame <= 45:
        n_pts = max(1, int((frame - 20) / 25 * len(xi_plot)))
        line_fd.set_data(xi_plot[:n_pts], k_fd_plot[:n_pts])
        eq_texts[0].set_alpha(1.0)
        eq_texts[1].set_alpha(min((frame - 20) / 10, 1.0))

    # Phase 3: Add rotated curve (35-55)
    if frame >= 35 and frame <= 55:
        n_pts = max(1, int((frame - 35) / 20 * len(xi_plot)))
        line_rot.set_data(xi_plot[:n_pts], k_rot_plot[:n_pts])
        eq_texts[2].set_alpha(min((frame - 35) / 10, 1.0))

    # Phase 4: Show derivatives (50-75)
    if frame >= 50:
        n_pts_d = max(1, int((frame - 50) / 25 * len(x)))
        dline_cont.set_data(x[:n_pts_d], deriv_cont[:n_pts_d])
        if frame >= 55:
            dline_fd.set_data(x[:min(n_pts_d, len(x))], deriv_fd[:min(n_pts_d, len(x))])
        if frame >= 60:
            dline_rot.set_data(x[:min(n_pts_d, len(x))], deriv_rot[:min(n_pts_d, len(x))])

    # Phase 5: Gibbs annotation (70+)
    if frame >= 70:
        gibbs_txt.set_text('← Gibbs ringing\n   (FD schemes reduce this!)')

    return []

anim = FuncAnimation(fig, update, frames=TOTAL, interval=120, blit=False)
outpath = r'c:\Users\vfn333\Documents\GitHub\FFT_crystals\Presentation\weekly_gifs\slide6_wavevector.gif'
anim.save(outpath, writer='pillow', fps=8, dpi=130)
plt.close()
print(f'Saved {outpath}')
