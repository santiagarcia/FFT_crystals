"""
Slide 7 GIF — Basic Fixed-Point Algorithm (Moulinec-Suquet)
===========================================================
Animated flowchart of the basic scheme:

  1) Initialize ε(x) = E (uniform)
  2) σ(x) = C(x) : ε(x)         (constitutive, real space)
  3) τ(x) = σ(x) - C0 : ε(x)    (polarization)
  4) FFT(τ) → τ_hat
  5) ε_hat(k) = -Γ0(k) : τ_hat(k) for k≠0,  ε_hat(0) = E
  6) IFFT(ε_hat) → ε(x)
  7) Check convergence ||∇·σ|| < tol → done, else goto 2

Shows the iteration loop with a convergence indicator.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'mathtext.fontset': 'cm',
})

# ─── Figure ───────────────────────────────────────────────────────────────────
fig, (ax_flow, ax_conv) = plt.subplots(1, 2, figsize=(14, 6.5), facecolor='white',
                                        gridspec_kw={'width_ratios': [1.6, 1]})

ax_flow.set_xlim(-0.5, 10.5)
ax_flow.set_ylim(-1.0, 8.5)
ax_flow.set_aspect('equal')
ax_flow.set_axis_off()

# ─── Flowchart boxes ─────────────────────────────────────────────────────────
boxes = [
    # (x, y, w, h, text, facecolor, edgecolor)
    (0.5, 7.0, 4.0, 1.0, r'Initialize $\varepsilon = \mathbf{E}$',
     '#E3F2FD', '#1565C0'),
    (0.5, 5.2, 4.5, 1.0, r'$\sigma(\mathbf{x}) = \mathcal{C}(\mathbf{x}):\varepsilon(\mathbf{x})$',
     '#FBE9E7', '#D84315'),
    (0.5, 3.4, 4.5, 1.0, r'$\tau = \sigma - \mathcal{C}^0:\varepsilon$',
     '#FFF3E0', '#E65100'),
    (0.5, 1.6, 5.5, 1.0,
     r'FFT: $\hat{\varepsilon} = -\hat{\Gamma}^0 : \hat{\tau}$, $\hat{\varepsilon}_0 = \mathbf{E}$',
     '#E8F5E9', '#2E7D32'),
    (0.5, -0.2, 4.5, 1.0, r'Converged? $\|\nabla\!\cdot\!\sigma\| < \mathrm{tol}$',
     '#FCE4EC', '#C62828'),
]

labels = [
    'real space',       # σ
    'real space',       # τ
    'Fourier space',    # FFT + Γ
    '',                 # convergence
    '',                 # init
]

box_patches = []
box_texts = []
label_texts = []
arrow_patches = []

for i, (x, y, w, h, txt, fc, ec) in enumerate(boxes):
    r = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                        facecolor=fc, edgecolor=ec, linewidth=2, alpha=0, zorder=3)
    ax_flow.add_patch(r)
    box_patches.append(r)

    t = ax_flow.text(x + w/2, y + h/2, txt, ha='center', va='center',
                     fontsize=11, color=ec, fontweight='bold', alpha=0, zorder=4)
    box_texts.append(t)

# Side labels
side_labels = [
    (5.2, 7.5, 'real space', '#1565C0'),
    (5.3, 5.7, 'real space', '#D84315'),
    (5.3, 3.9, 'real space', '#E65100'),
    (6.3, 2.1, 'Fourier space', '#2E7D32'),
]
side_txts = []
for (sx, sy, sl, sc) in side_labels:
    t = ax_flow.text(sx, sy, sl, fontsize=9, color=sc, style='italic', alpha=0)
    side_txts.append(t)

# Arrows between boxes
arrow_coords = [
    (2.5, 7.0, 0, -0.7),    # init → σ
    (2.75, 5.2, 0, -0.7),   # σ → τ
    (2.75, 3.4, 0, -0.7),   # τ → FFT
    (3.25, 1.6, 0, -0.7),   # FFT → converge
]

arr_lines = []
for (ax_x, ay, dx, dy) in arrow_coords:
    l, = ax_flow.plot([ax_x, ax_x + dx], [ay, ay + dy], '-',
                      color='#555', lw=2, alpha=0, zorder=2)
    arr_lines.append(l)
    # Arrowhead
    ah, = ax_flow.plot([ax_x + dx], [ay + dy], 'v', color='#555', ms=10, alpha=0, zorder=2)
    arr_lines.append(ah)

# Loop-back arrow (convergence NO → back to σ)
loop_x = [5.0, 7.0, 7.0, 5.0]
loop_y = [0.3, 0.3, 5.7, 5.7]
loop_line, = ax_flow.plot(loop_x, loop_y, '-', color='#C62828', lw=2, alpha=0)
loop_arrow, = ax_flow.plot([5.0], [5.7], '<', color='#C62828', ms=10, alpha=0)
loop_txt = ax_flow.text(7.5, 3.0, 'iterate', fontsize=10, color='#C62828',
                        rotation=90, ha='center', alpha=0, fontweight='bold')

# "Done" text
done_txt = ax_flow.text(2.75, -0.8, '', fontsize=12, color='#2E7D32',
                        ha='center', fontweight='bold')

# ─── Right panel: convergence plot ────────────────────────────────────────────
ax_conv.set_xlabel('Iteration', fontsize=11)
ax_conv.set_ylabel(r'$\|\nabla\cdot\sigma\|$ / $\|\sigma\|$', fontsize=11)
ax_conv.set_title('Convergence', fontsize=13, fontweight='bold', color='#333')
ax_conv.set_yscale('log')
ax_conv.set_xlim(0, 50)
ax_conv.set_ylim(1e-8, 1)
ax_conv.grid(True, alpha=0.2, which='both')
ax_conv.spines['top'].set_visible(False)
ax_conv.spines['right'].set_visible(False)

# Fake convergence curves
iters = np.arange(50)
resid_fast = 0.5 * np.exp(-0.3 * iters)  # good contrast
resid_slow = 0.5 * np.exp(-0.08 * iters)  # bad contrast

conv_line_fast, = ax_conv.plot([], [], '-', color='#2E7D32', lw=2.5, label='Low contrast')
conv_line_slow, = ax_conv.plot([], [], '--', color='#D32F2F', lw=2.5, label='High contrast')
ax_conv.axhline(1e-6, color='gray', ls=':', lw=1, label='Tolerance')
ax_conv.legend(fontsize=9)

tol_txt = ax_conv.text(45, 2e-6, 'tol', fontsize=9, color='gray')

# ─── Animation ────────────────────────────────────────────────────────────────
TOTAL = 100

def update(frame):
    # Phase 1 (0-15): Show init box
    if frame >= 0:
        t = min(frame / 5, 1.0)
        box_patches[0].set_alpha(t)
        box_texts[0].set_alpha(t)

    # Phase 2 (8-18): σ box + arrow
    if frame >= 8:
        t = min((frame - 8) / 5, 1.0)
        arr_lines[0].set_alpha(t)
        arr_lines[1].set_alpha(t)
        box_patches[1].set_alpha(t)
        box_texts[1].set_alpha(t)
        if len(side_txts) > 0:
            side_txts[0].set_alpha(t)

    # Phase 3 (16-26): τ box + arrow
    if frame >= 16:
        t = min((frame - 16) / 5, 1.0)
        arr_lines[2].set_alpha(t)
        arr_lines[3].set_alpha(t)
        box_patches[2].set_alpha(t)
        box_texts[2].set_alpha(t)
        if len(side_txts) > 1:
            side_txts[1].set_alpha(t)
            side_txts[2].set_alpha(t)

    # Phase 4 (24-34): FFT box + arrow
    if frame >= 24:
        t = min((frame - 24) / 5, 1.0)
        arr_lines[4].set_alpha(t)
        arr_lines[5].set_alpha(t)
        box_patches[3].set_alpha(t)
        box_texts[3].set_alpha(t)
        if len(side_txts) > 3:
            side_txts[3].set_alpha(t)

    # Phase 5 (32-42): convergence check + arrow
    if frame >= 32:
        t = min((frame - 32) / 5, 1.0)
        arr_lines[6].set_alpha(t)
        arr_lines[7].set_alpha(t)
        box_patches[4].set_alpha(t)
        box_texts[4].set_alpha(t)

    # Phase 6 (40-50): loop-back arrow
    if frame >= 40:
        t = min((frame - 40) / 8, 1.0)
        loop_line.set_alpha(t)
        loop_arrow.set_alpha(t)
        loop_txt.set_alpha(t)

    # Phase 7 (45-90): animate convergence curves
    if frame >= 45:
        n_pts = min(int((frame - 45) / 45 * 50), 50)
        conv_line_fast.set_data(iters[:n_pts], resid_fast[:n_pts])
        conv_line_slow.set_data(iters[:n_pts], resid_slow[:n_pts])

    # Phase 8 (85+): "Done" label
    if frame >= 85:
        done_txt.set_text(r'$\checkmark$ Converged → $\varepsilon(\mathbf{x}), \sigma(\mathbf{x})$')

    return []

anim = FuncAnimation(fig, update, frames=TOTAL, interval=120, blit=False)
outpath = r'c:\Users\vfn333\Documents\GitHub\FFT_crystals\Presentation\weekly_gifs\slide7_basic_algorithm.gif'
anim.save(outpath, writer='pillow', fps=8, dpi=130)
plt.close()
print(f'Saved {outpath}')
