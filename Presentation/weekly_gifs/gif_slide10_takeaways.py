"""
Slide 10 – Key Takeaways: Accuracy vs Speed in FFT micromechanics.
Structured build-up: core idea → accuracy pillars → speed pillars → final summary.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FuncAnimation

# ── colours ─────────────────────────────────────────────────────
BG      = 'white'
BLUE    = '#1565C0'
GREEN   = '#2E7D32'
RED     = '#C62828'
ORANGE  = '#EF6C00'
PURPLE  = '#6A1B9A'
TEAL    = '#00838F'
GREY    = '#757575'
DARK    = '#212121'
GOLD    = '#F9A825'
LGREY   = '#F5F5F5'

# ── figure ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8), facecolor=BG)

N_FRAMES = 180
FPS      = 8

# Phase boundaries
P1 = 30     # Core statement
P2 = 80     # Accuracy pillar
P3 = 140    # Speed pillar
P4 = 180    # Hold / summary line

# ── helper ──────────────────────────────────────────────────────
def fade(f, start, dur):
    if f < start:
        return 0.0
    return min(1.0, (f - start) / max(dur, 1))

# ═══════════════════════════════════════════════════════════════
# STATIC LAYOUT  (all start invisible, faded in by animate())
# ═══════════════════════════════════════════════════════════════

# ── Title ───────────────────────────────────────────────────────
title = fig.text(0.50, 0.94, 'Key Takeaways',
                 ha='center', fontsize=24, fontweight='bold', color=BLUE, alpha=0)

# ── Phase 1: core statement ─────────────────────────────────────
core = fig.text(0.50, 0.86,
    'FFT micromechanics solves equilibrium on periodic grids\n'
    'by turning the PDE into the Lippmann-Schwinger convolution equation,\n'
    'which becomes an algebraic multiplication in Fourier space.',
    ha='center', va='top', fontsize=13, color=DARK, alpha=0,
    linespacing=1.5,
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD',
              edgecolor=BLUE, linewidth=1.5, alpha=0))

# ── Accuracy / Speed column headers ────────────────────────────
hdr_acc = fig.text(0.27, 0.66, 'Accuracy',
                   ha='center', fontsize=20, fontweight='bold', color=GREEN, alpha=0)
hdr_spd = fig.text(0.73, 0.66, 'Speed',
                   ha='center', fontsize=20, fontweight='bold', color=ORANGE, alpha=0)

depends = fig.text(0.27, 0.62, 'hinges on', ha='center', fontsize=12, color=GREY, alpha=0)
depends2 = fig.text(0.73, 0.62, 'hinges on', ha='center', fontsize=12, color=GREY, alpha=0)

# Vertical divider coordinates
div_x = 0.50

# ── Accuracy bullets (left column) ──────────────────────────────
acc_items = []

# Item 1: Green operator
acc1_icon = fig.text(0.08, 0.54, '\u25B6', ha='center', fontsize=12, color=GREEN, alpha=0)
acc1_title = fig.text(0.11, 0.54, r'Green operator  $\hat{\Gamma}^{0}$',
                      ha='left', fontsize=15, fontweight='bold', color=DARK, alpha=0)
acc1_desc = fig.text(0.11, 0.505, 'Connects stress polarization to strain correction',
                     ha='left', fontsize=10, color=GREY, alpha=0)
acc_items.append((acc1_icon, acc1_title, acc1_desc))

# Item 2: Wavevector discretization
acc2_icon = fig.text(0.08, 0.44, '\u25B6', ha='center', fontsize=12, color=GREEN, alpha=0)
acc2_title = fig.text(0.11, 0.44, r'Wavevector  $\mathbf{k}$  discretization',
                      ha='left', fontsize=15, fontweight='bold', color=DARK, alpha=0)
acc2_desc = fig.text(0.11, 0.395, 'Continuous vs finite-difference-compatible',
                     ha='left', fontsize=10, color=GREY, alpha=0)
acc_items.append((acc2_icon, acc2_title, acc2_desc))

# Item 3: Zero mode
acc3_icon = fig.text(0.08, 0.34, '\u25B6', ha='center', fontsize=12, color=GREEN, alpha=0)
acc3_title = fig.text(0.11, 0.34, 'Zero-mode handling',
                      ha='left', fontsize=15, fontweight='bold', color=DARK, alpha=0)
acc3_desc = fig.text(0.11, 0.295, 'Correct treatment of the macroscopic average',
                     ha='left', fontsize=10, color=GREY, alpha=0)
acc_items.append((acc3_icon, acc3_title, acc3_desc))

# ── Speed bullets (right column) ────────────────────────────────
spd_items = []

# Item 1: Reference medium
spd1_icon = fig.text(0.54, 0.54, '\u25B6', ha='center', fontsize=12, color=ORANGE, alpha=0)
spd1_title = fig.text(0.57, 0.54, 'Reference medium choice',
                      ha='left', fontsize=15, fontweight='bold', color=DARK, alpha=0)
spd1_desc = fig.text(0.57, 0.505, r'$\mathcal{C}^{0}$ controls conditioning & convergence rate',
                     ha='left', fontsize=10, color=GREY, alpha=0)
spd_items.append((spd1_icon, spd1_title, spd1_desc))

# Item 2: Iteration strategy
spd2_icon = fig.text(0.54, 0.44, '\u25B6', ha='center', fontsize=12, color=ORANGE, alpha=0)
spd2_title = fig.text(0.57, 0.44, 'Iteration strategy',
                      ha='left', fontsize=15, fontweight='bold', color=DARK, alpha=0)
spd2_desc = fig.text(0.57, 0.395, 'Fixed-point  vs  Anderson  vs  Krylov',
                     ha='left', fontsize=10, color=GREY, alpha=0)
spd_items.append((spd2_icon, spd2_title, spd2_desc))

# Item 3: Implementation
spd3_icon = fig.text(0.54, 0.34, '\u25B6', ha='center', fontsize=12, color=ORANGE, alpha=0)
spd3_title = fig.text(0.57, 0.34, 'Implementation details',
                      ha='left', fontsize=15, fontweight='bold', color=DARK, alpha=0)
spd3_desc = fig.text(0.57, 0.295, 'Batched FFTs, preallocation, GPU',
                     ha='left', fontsize=10, color=GREY, alpha=0)
spd_items.append((spd3_icon, spd3_title, spd3_desc))

# ── Decorative axes for divider and boxes ───────────────────────
ax_deco = fig.add_axes([0, 0, 1, 1], zorder=0)
ax_deco.set_xlim(0, 1)
ax_deco.set_ylim(0, 1)
ax_deco.axis('off')

# Vertical divider line
divider = ax_deco.plot([0.50, 0.50], [0.28, 0.68], color='#BDBDBD',
                       linewidth=1.5, linestyle='--', alpha=0)[0]

# Column background boxes
col_acc_bg = mpatches.FancyBboxPatch((0.04, 0.26), 0.44, 0.44,
                                      boxstyle='round,pad=0.02',
                                      facecolor='#E8F5E9', edgecolor=GREEN,
                                      linewidth=1.5, alpha=0)
ax_deco.add_patch(col_acc_bg)

col_spd_bg = mpatches.FancyBboxPatch((0.52, 0.26), 0.44, 0.44,
                                      boxstyle='round,pad=0.02',
                                      facecolor='#FFF3E0', edgecolor=ORANGE,
                                      linewidth=1.5, alpha=0)
ax_deco.add_patch(col_spd_bg)

# ── Bottom summary ──────────────────────────────────────────────
# Small accuracy/speed icons in summary
summary_line = fig.text(0.50, 0.16,
    r'Accuracy  $\longleftrightarrow$  mathematical choices'
    '          '
    r'Speed  $\longleftrightarrow$  algorithmic choices',
    ha='center', fontsize=13, color=DARK, alpha=0)

# Underline emphasis
summary_box = fig.text(0.50, 0.07,
    'Both must be tuned together for a practical EVPFFT solver.',
    ha='center', fontsize=15, fontweight='bold', color=DARK, alpha=0,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1',
              edgecolor=GOLD, linewidth=2, alpha=0))

# ── Step indicator ──────────────────────────────────────────────
step_txt = fig.text(0.98, 0.97, '', ha='right', va='top', fontsize=10, color=GREY)

# ═══════════════════════════════════════════════════════════════
# ANIMATION
# ═══════════════════════════════════════════════════════════════
def animate(frame):
    # Step label
    if frame < P1:
        step_txt.set_text('1/4 — Core idea')
    elif frame < P2:
        step_txt.set_text('2/4 — Accuracy pillars')
    elif frame < P3:
        step_txt.set_text('3/4 — Speed pillars')
    else:
        step_txt.set_text('4/4 — Summary')

    # ── Title ───────────────────────────────────────────────
    title.set_alpha(fade(frame, 0, 6))

    # ── Phase 1: core statement ─────────────────────────────
    a_core = fade(frame, 4, 8)
    core.set_alpha(a_core)
    if core.get_bbox_patch() is not None:
        core.get_bbox_patch().set_alpha(a_core)

    # ── Phase 2: accuracy column ────────────────────────────
    # Column background
    a_col_acc = fade(frame, P1, 6) * 0.35
    col_acc_bg.set_alpha(a_col_acc)

    # Header
    hdr_acc.set_alpha(fade(frame, P1, 6))
    depends.set_alpha(fade(frame, P1 + 3, 5))

    # Divider
    divider.set_alpha(fade(frame, P1, 8) * 0.6)

    # Acc items staggered
    for i, (icon, ttl, desc) in enumerate(acc_items):
        t0 = P1 + 8 + i * 12
        a = fade(frame, t0, 6)
        icon.set_alpha(a)
        ttl.set_alpha(a)
        desc.set_alpha(a)

    # ── Phase 3: speed column ───────────────────────────────
    a_col_spd = fade(frame, P2, 6) * 0.35
    col_spd_bg.set_alpha(a_col_spd)

    hdr_spd.set_alpha(fade(frame, P2, 6))
    depends2.set_alpha(fade(frame, P2 + 3, 5))

    for i, (icon, ttl, desc) in enumerate(spd_items):
        t0 = P2 + 8 + i * 12
        a = fade(frame, t0, 6)
        icon.set_alpha(a)
        ttl.set_alpha(a)
        desc.set_alpha(a)

    # ── Phase 4: summary ────────────────────────────────────
    a_sum = fade(frame, P3, 8)
    summary_line.set_alpha(a_sum)

    a_box = fade(frame, P3 + 10, 8)
    summary_box.set_alpha(a_box)
    if summary_box.get_bbox_patch() is not None:
        summary_box.get_bbox_patch().set_alpha(a_box)

    return []


anim = FuncAnimation(fig, animate, frames=N_FRAMES, interval=1000 // FPS, blit=False)
out = 'slide10_takeaways.gif'
anim.save(out, writer='pillow', dpi=130)
print(f'Saved {out}')
plt.close(fig)
