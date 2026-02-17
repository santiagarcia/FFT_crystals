"""
Slide 9 – EVPFFT: local nonlinear material, global spectral equilibrium
Animated build-up of the EVP extension over the basic FFT scheme.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FuncAnimation

# ── colour palette ──────────────────────────────────────────────
BG       = 'white'
BLUE     = '#1565C0'
GREEN    = '#2E7D32'
RED      = '#D32F2F'
ORANGE   = '#EF6C00'
PURPLE   = '#7B1FA2'
GREY     = '#757575'
DARK     = '#212121'
GOLD     = '#F9A825'
TEAL     = '#00838F'

# ── figure ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8), facecolor=BG)

# ── animation parameters ────────────────────────────────────────
N_FRAMES = 160
FPS      = 8

# Phase boundaries  (cumulative frame counts)
P1_END = 20    # Phase 1: show elastic eq and strike it out
P2_END = 45    # Phase 2: additive split
P3_END = 75    # Phase 3: constitutive update
P4_END = 130   # Phase 4: inner loop diagram (with cycling)
P5_END = 160   # Phase 5: summary

# ── persistent text objects ─────────────────────────────────────
title = fig.text(0.50, 0.94, 'Extension to Elasto-Viscoplasticity (EVP)',
                 ha='center', fontsize=22, fontweight='bold', color=BLUE, alpha=0)

# Phase 1 – elastic equation
elastic_eq = fig.text(0.50, 0.82, r'Elastic:   $\sigma = \mathcal{C} : \varepsilon$',
                      ha='center', fontsize=18, color=DARK, alpha=0)
strike_line = fig.text(0.50, 0.82, '─' * 28,
                       ha='center', fontsize=18, color=RED, alpha=0,
                       fontweight='bold')

# Phase 2 – additive split
split_label = fig.text(0.50, 0.74, 'Additive strain decomposition:',
                       ha='center', fontsize=14, color=GREY, alpha=0)
split_eq1 = fig.text(0.50, 0.68,
                     r'$\varepsilon = \varepsilon^{e} + \varepsilon^{p}$',
                     ha='center', fontsize=20, fontweight='bold', color=DARK, alpha=0)
split_eq2 = fig.text(0.50, 0.61,
                     r'$\sigma = \mathcal{C} : \varepsilon^{e} = \mathcal{C} : (\varepsilon - \varepsilon^{p})$',
                     ha='center', fontsize=20, fontweight='bold', color=GREEN, alpha=0)

# Phase 3 – constitutive update
const_label = fig.text(0.50, 0.52, 'Local constitutive update at each voxel:',
                       ha='center', fontsize=14, color=GREY, alpha=0)
const_eq = fig.text(0.50, 0.44,
                    r'$(\sigma^{n+1},\, \varepsilon^{p,n+1},\, state^{n+1})'
                    r' = \mathcal{G}(\varepsilon^{n+1},\, state^{n},\, \Delta t)$',
                    ha='center', fontsize=18, fontweight='bold', color=PURPLE, alpha=0,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#F3E5F5',
                              edgecolor=PURPLE, alpha=0))

# Phase 4 – inner loop  (drawn with axes below)
loop_title = fig.text(0.50, 0.36, 'Inner loop at each load increment:',
                      ha='center', fontsize=14, color=GREY, alpha=0)

# We'll draw the loop diagram on a dedicated axes
ax_loop = fig.add_axes([0.10, 0.08, 0.80, 0.27])
ax_loop.set_xlim(0, 10)
ax_loop.set_ylim(0, 3)
ax_loop.axis('off')

# Pre-build loop diagram elements (hidden initially)
# Left box: Real-space constitutive update
box_real = mpatches.FancyBboxPatch((0.3, 0.6), 3.4, 1.8,
                                   boxstyle='round,pad=0.15',
                                   facecolor='#E8F5E9', edgecolor=GREEN,
                                   linewidth=2, alpha=0)
ax_loop.add_patch(box_real)
txt_real_title = ax_loop.text(2.0, 2.05, 'Real Space', ha='center', fontsize=13,
                              fontweight='bold', color=GREEN, alpha=0)
txt_real_body = ax_loop.text(2.0, 1.25, r'$\sigma(\mathbf{x}), \varepsilon^{p}(\mathbf{x})$'
                             '\nConstitutive\nupdate',
                             ha='center', va='center', fontsize=11, color=DARK, alpha=0)

# Right box: Fourier-space equilibrium
box_fourier = mpatches.FancyBboxPatch((6.3, 0.6), 3.4, 1.8,
                                      boxstyle='round,pad=0.15',
                                      facecolor='#E3F2FD', edgecolor=BLUE,
                                      linewidth=2, alpha=0)
ax_loop.add_patch(box_fourier)
txt_four_title = ax_loop.text(8.0, 2.05, 'Fourier Space', ha='center', fontsize=13,
                              fontweight='bold', color=BLUE, alpha=0)
txt_four_body = ax_loop.text(8.0, 1.25, r'$\hat{\Gamma}^{0} \cdot \hat{\tau}$'
                             '\nEquilibrium\ncorrection',
                             ha='center', va='center', fontsize=11, color=DARK, alpha=0)

# Arrows between boxes
arrow_top = None   # will be drawn in animate
arrow_bot = None

# Phase 5 – summary
summary_box = fig.text(0.50, 0.03,
                       'EVPFFT in one sentence:  local nonlinear material,  global spectral equilibrium.',
                       ha='center', va='bottom', fontsize=15, fontweight='bold',
                       color=DARK,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1',
                                 edgecolor=GOLD, linewidth=2, alpha=0),
                       alpha=0)

# Step counter
step_txt = fig.text(0.98, 0.97, '', ha='right', va='top', fontsize=10, color=GREY)

# ── helper: smooth fade 0→1 over `dur` frames starting at `start` ──
def fade(frame, start, dur):
    if frame < start:
        return 0.0
    t = (frame - start) / max(dur, 1)
    return min(1.0, t)

# ── helper: pulsate between lo and hi ──
def pulse(frame, period=20, lo=0.5, hi=1.0):
    return lo + (hi - lo) * 0.5 * (1 + np.sin(2 * np.pi * frame / period))

# ── arrows (created once, alpha controlled) ──
arrow_top_obj = ax_loop.annotate('', xy=(6.3, 2.1), xytext=(3.7, 2.1),
                                  arrowprops=dict(arrowstyle='->', color=ORANGE,
                                                  lw=2.5, connectionstyle='arc3,rad=-0.15'),
                                  alpha=0)
arrow_bot_obj = ax_loop.annotate('', xy=(3.7, 0.9), xytext=(6.3, 0.9),
                                  arrowprops=dict(arrowstyle='->', color=ORANGE,
                                                  lw=2.5, connectionstyle='arc3,rad=-0.15'),
                                  alpha=0)
# Arrow labels
arr_top_lbl = ax_loop.text(5.0, 2.55, r'FFT($\tau$)', ha='center', fontsize=11,
                           color=ORANGE, fontweight='bold', alpha=0)
arr_bot_lbl = ax_loop.text(5.0, 0.35, r'IFFT($\hat{\varepsilon}$)', ha='center',
                           fontsize=11, color=ORANGE, fontweight='bold', alpha=0)

# Cycling highlight patches (drawn each frame)
highlight_real = mpatches.FancyBboxPatch((0.2, 0.5), 3.6, 2.0,
                                         boxstyle='round,pad=0.1',
                                         facecolor='none', edgecolor=GOLD,
                                         linewidth=3, alpha=0, linestyle='--')
ax_loop.add_patch(highlight_real)

highlight_four = mpatches.FancyBboxPatch((6.2, 0.5), 3.6, 2.0,
                                         boxstyle='round,pad=0.1',
                                         facecolor='none', edgecolor=GOLD,
                                         linewidth=3, alpha=0, linestyle='--')
ax_loop.add_patch(highlight_four)


def animate(frame):
    artists = []

    # ── Title (always-on after frame 2) ─────────────────────
    title.set_alpha(fade(frame, 0, 5))

    # ── Step label ──────────────────────────────────────────
    if frame < P1_END:
        phase_name = 'Step 1/5 — Elastic limit'
    elif frame < P2_END:
        phase_name = 'Step 2/5 — Additive split'
    elif frame < P3_END:
        phase_name = 'Step 3/5 — Constitutive update'
    elif frame < P4_END:
        phase_name = 'Step 4/5 — Inner loop'
    else:
        phase_name = 'Step 5/5 — Summary'
    step_txt.set_text(phase_name)

    # ── Phase 1: elastic equation + strikethrough ───────────
    elastic_eq.set_alpha(fade(frame, 3, 6))
    strike_alpha = fade(frame, 12, 5)
    strike_line.set_alpha(strike_alpha)
    if strike_alpha > 0:
        elastic_eq.set_color(GREY)

    # ── Phase 2: additive split ─────────────────────────────
    split_label.set_alpha(fade(frame, P1_END, 5))
    split_eq1.set_alpha(fade(frame, P1_END + 5, 6))
    split_eq2.set_alpha(fade(frame, P1_END + 12, 6))

    # ── Phase 3: constitutive update ────────────────────────
    a3 = fade(frame, P2_END, 5)
    const_label.set_alpha(a3)
    a3eq = fade(frame, P2_END + 5, 8)
    const_eq.set_alpha(a3eq)
    if const_eq.get_bbox_patch() is not None:
        const_eq.get_bbox_patch().set_alpha(a3eq)

    # ── Phase 4: inner loop diagram ─────────────────────────
    a4 = fade(frame, P3_END, 8)
    loop_title.set_alpha(a4)

    box_alpha = fade(frame, P3_END + 3, 8)
    box_real.set_alpha(box_alpha)
    txt_real_title.set_alpha(box_alpha)
    txt_real_body.set_alpha(box_alpha)
    box_fourier.set_alpha(box_alpha)
    txt_four_title.set_alpha(box_alpha)
    txt_four_body.set_alpha(box_alpha)

    arr_alpha = fade(frame, P3_END + 10, 6)
    arrow_top_obj.set_alpha(arr_alpha)
    arrow_bot_obj.set_alpha(arr_alpha)
    arr_top_lbl.set_alpha(arr_alpha)
    arr_bot_lbl.set_alpha(arr_alpha)

    # Cycling highlight (alternates every 8 frames once arrows visible)
    if frame >= P3_END + 16 and frame < P4_END:
        cycle = ((frame - P3_END - 16) // 8) % 2
        p = pulse(frame, period=8, lo=0.3, hi=0.9)
        if cycle == 0:
            highlight_real.set_alpha(p)
            highlight_four.set_alpha(0.0)
        else:
            highlight_real.set_alpha(0.0)
            highlight_four.set_alpha(p)
    else:
        highlight_real.set_alpha(0)
        highlight_four.set_alpha(0)

    # ── Phase 5: summary ────────────────────────────────────
    a5 = fade(frame, P4_END, 8)
    summary_box.set_alpha(a5)
    if summary_box.get_bbox_patch() is not None:
        summary_box.get_bbox_patch().set_alpha(a5)

    return artists


anim = FuncAnimation(fig, animate, frames=N_FRAMES, interval=1000 // FPS, blit=False)
out = 'slide9_evpfft.gif'
anim.save(out, writer='pillow', dpi=130)
print(f'Saved {out}')
plt.close(fig)
