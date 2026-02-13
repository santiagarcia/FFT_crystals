"""
GIF: Crystal Plasticity Flow Rule — Computational Pipeline (Flowchart)
======================================================================
Each stage:
  1) ZOOM IN — big, centered, with variable-by-variable explanations
  2) ZOOM OUT — shrink into the full pipeline overview with all prior stages
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

BLUE_FACE = '#E3F2FD';  BLUE_EDGE = '#1565C0'
RED_FACE  = '#FBE9E7';  RED_EDGE  = '#D84315'
GRAY_FACE = '#F5F5F5';  GRAY_EDGE = '#757575'
WHITE_FACE = '#FFFFFF';  WHITE_EDGE = '#333333'
GREEN = '#2E7D32'

# ─── Overview layout (small boxes in full pipeline) ───────────────────────────
# Two-row grid: top row flows left→right, bottom row right→left
# Row 1 (y=4.5): input → resolve → driving → slip
# Row 2 (y=1.5): ........................ plastic → fem  (+ feedback)
BW, BH = 2.4, 2.0   # uniform box size
ROW1_Y = 4.8
ROW2_Y = 1.8
OV_BOXES = {
    'input':   (0.4,  ROW1_Y, BW, BH),
    'resolve': (3.6,  ROW1_Y, BW, BH),
    'driving': (6.8,  ROW1_Y, BW, BH),
    'slip':    (10.0, ROW1_Y, BW + 1.0, BH),   # slightly wider for the long title
    'plastic': (10.0, ROW2_Y, BW, BH),
    'fem':     (13.0, ROW2_Y, BW, BH),
}
OV_STYLES = {
    'input':   (WHITE_FACE, WHITE_EDGE),
    'resolve': (BLUE_FACE, BLUE_EDGE),
    'driving': (BLUE_FACE, BLUE_EDGE),
    'slip':    (RED_FACE, RED_EDGE),
    'plastic': (BLUE_FACE, BLUE_EDGE),
    'fem':     (GRAY_FACE, GRAY_EDGE),
}
OV_LABELS = {
    'input':   'INPUTS',
    'resolve': 'RESOLVED\nSHEAR',
    'driving': 'EFFECTIVE\nDRIVING',
    'slip':    'SLIP UPDATE\n+ JACOBIAN',
    'plastic': 'PLASTIC\nSTRAIN',
    'fem':     'FEM /\nSTRESS',
}
OV_ARROWS = [
    ('input', 'resolve'),
    ('resolve', 'driving'),
    ('driving', 'slip'),
    ('slip', 'plastic'),
    ('plastic', 'fem'),
]
STAGE_KEYS = ['input', 'resolve', 'driving', 'slip', 'plastic', 'fem']

# ─── "Zoom" detail content for each stage ─────────────────────────────────────
# Each is a dict with: title, box_color, equations (list of (text, explanation, color))

DETAIL = {}

DETAIL['input'] = {
    'title': 'INPUTS',
    'face': WHITE_FACE, 'edge': WHITE_EDGE,
    'items': [
        ('$\\boldsymbol{\\sigma}$', 'Cauchy stress tensor (from FEM solver)', '#E65100'),
        ('$T$',                     'Temperature in Kelvin', '#E65100'),
        ('$\\chi_i$',               'Backstress on slip system $i$ (kinematic hardening)', '#E65100'),
        ('$r_i^{\\mathrm{th,SSD}}$','Thermal resistance from statistically-stored dislocations', '#E65100'),
        ('$r_i^{\\mathrm{th,CS}}$', 'Thermal resistance from cross-slip', '#E65100'),
        ('$r_i^{\\mathrm{ath}}$',   'Athermal (long-range) resistance', '#E65100'),
        ('$\\Delta t$',             'Time increment from the FEM integrator', '#E65100'),
    ],
}

DETAIL['resolve'] = {
    'title': 'RESOLVED SHEAR STRESS',
    'face': BLUE_FACE, 'edge': BLUE_EDGE,
    'items': [
        ('$\\tau_i = \\mathbf{s}_i \\cdot \\boldsymbol{\\sigma} \\cdot \\mathbf{m}_i$',
         '', '#1565C0'),
        ('$\\mathbf{s}_i$', 'Slip direction unit vector  (12 systems)', '#1565C0'),
        ('$\\boldsymbol{\\sigma}$', 'Applied Cauchy stress tensor', '#E65100'),
        ('$\\mathbf{m}_i$', 'Slip plane normal unit vector', '#1565C0'),
        ('$i = 1 \\ldots 12$', 'Loop over all FCC slip systems', '#555555'),
    ],
}

DETAIL['driving'] = {
    'title': 'EFFECTIVE DRIVING STRESS',
    'face': BLUE_FACE, 'edge': BLUE_EDGE,
    'items': [
        ('$\\tau_i^{\\mathrm{eff}} = |\\tau_i - \\chi_i| - r_i^{\\mathrm{ath}}$',
         '', '#1565C0'),
        ('$\\tau_i$',                'Resolved shear stress (from previous step)', '#1565C0'),
        ('$\\chi_i$',               'Backstress — shifts the yield surface center', '#E65100'),
        ('$r_i^{\\mathrm{ath}}$',   'Athermal resistance — long-range obstacle strength', '#E65100'),
        ('', 'Slip activates only if  $\\tau_i^{\\mathrm{eff}} > 0$', '#D32F2F'),
    ],
}

DETAIL['slip'] = {
    'title': 'SLIP UPDATE  +  JACOBIAN',
    'face': RED_FACE, 'edge': RED_EDGE,
    'items': [
        ('$\\Delta\\gamma_i = \\dot{\\gamma}_0\\,'
         '\\exp\\!\\left[-\\dfrac{G}{k_B T}'
         '\\left(1 - \\left(\\dfrac{\\tau_i^{\\mathrm{eff}}}{r_i^{\\mathrm{th}}}\\right)^p'
         '\\right)^q\\right] \\Delta t$',
         '', '#BF360C'),
        ('$\\dot{\\gamma}_0$',      'Reference slip rate ($10^6$ s$^{-1}$)', '#BF360C'),
        ('$G$',                     'Activation energy for dislocation glide', '#BF360C'),
        ('$k_B T$',                 'Thermal energy (Boltzmann $\\times$ temperature)', '#BF360C'),
        ('$r_i^{\\mathrm{th}}$',    'Total thermal resistance = SSD + cross-slip', '#BF360C'),
        ('$p, q$',                  'Barrier shape exponents (obstacle profile)', '#BF360C'),
        ('',                        'OTIS computes  $\\partial\\Delta\\gamma_i / \\partial\\tau_i$,  '
                                    '$\\partial\\Delta\\gamma_i / \\partial\\chi_i$,  '
                                    '$\\partial\\Delta\\gamma_i / \\partial r_i^{\\mathrm{ath}}$,  '
                                    '$\\partial\\Delta\\gamma_i / \\partial r_i^{\\mathrm{th}}$', GREEN),
    ],
}

DETAIL['plastic'] = {
    'title': 'PLASTIC STRAIN INCREMENT',
    'face': BLUE_FACE, 'edge': BLUE_EDGE,
    'items': [
        ('$\\Delta\\varepsilon^p = \\sum_{i=1}^{12} \\Delta\\gamma_i \\, \\mathbf{P}_i$',
         '', '#1565C0'),
        ('$\\Delta\\gamma_i$',  'Slip increment on system $i$ (from previous step)', '#1565C0'),
        ('$\\mathbf{P}_i$',    'Schmid tensor =  $\\mathrm{sym}(\\mathbf{s}_i \\otimes \\mathbf{m}_i)$',
         '#1565C0'),
        ('', 'Superposition: each system contributes independently', '#555555'),
    ],
}

DETAIL['fem'] = {
    'title': 'FEM / STRESS UPDATE',
    'face': GRAY_FACE, 'edge': GRAY_EDGE,
    'items': [
        ('', 'Elasticity + equilibrium — update stress from strain', '#555555'),
        ('', 'Consistent tangent — uses the Jacobian from OTIS', GREEN),
        ('', 'Newton iteration — converge the constitutive update', '#555555'),
        ('', 'Global FEM solve (outside computeFlowRule)', '#999999'),
        ('', 'Jacobian feeds back: dIncSlipPlastic_dStressResolved', GREEN),
    ],
}


# ─── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 8.5), facecolor='white')
ax.set_aspect('equal')
ax.axis('off')

dynamic = []

def clear():
    for a in dynamic:
        try:
            a.remove()
        except Exception:
            pass
    dynamic.clear()


def draw_rounded_box(ax, x, y, w, h, fc, ec, lw=1.8):
    b = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                                 facecolor=fc, edgecolor=ec, lw=lw, zorder=2)
    ax.add_patch(b)
    return b


# ─── Draw functions ───────────────────────────────────────────────────────────

def draw_detail(key):
    """Draw a zoomed-in detail view centered on screen."""
    d = DETAIL[key]
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)

    # Title
    dynamic.append(ax.text(7.5, 7.8, d['title'], ha='center', va='center',
                           fontsize=26, fontweight='bold', color=d['edge'], zorder=4))

    # Large box
    bx, by, bw, bh = 0.5, 0.4, 14.0, 7.0
    dynamic.append(draw_rounded_box(ax, bx, by, bw, bh, d['face'], d['edge'], lw=2.5))

    # Items
    n = len(d['items'])
    y_start = by + bh - 0.85
    spacing = min(1.0, (bh - 1.0) / max(n, 1))

    for idx, (sym, desc, col) in enumerate(d['items']):
        yy = y_start - idx * spacing
        if sym and desc:
            # Symbol on left, description on right
            dynamic.append(ax.text(1.5, yy, sym, ha='left', va='center',
                                   fontsize=19, color=col, zorder=4))
            dynamic.append(ax.text(5.8, yy, desc, ha='left', va='center',
                                   fontsize=16, color='#333333', zorder=4))
        elif sym and not desc:
            # Equation only — centered, larger
            dynamic.append(ax.text(7.5, yy, sym, ha='center', va='center',
                                   fontsize=20, color=col, zorder=4))
        else:
            # Description only
            dynamic.append(ax.text(1.5, yy, desc, ha='left', va='center',
                                   fontsize=16, color=col, fontstyle='italic',
                                   zorder=4))

    # Step counter
    stage_num = STAGE_KEYS.index(key) + 1
    dynamic.append(ax.text(14.5, 0.15,
                           f'Step {stage_num} / {len(STAGE_KEYS)}',
                           ha='right', va='bottom', fontsize=13, color='#AAAAAA',
                           zorder=4))


def draw_overview(up_to_stage_idx):
    """Draw the small pipeline overview with stages 0..up_to_stage_idx revealed."""
    ax.set_xlim(-0.3, 15.8)
    ax.set_ylim(-0.6, 8.5)

    # Title
    dynamic.append(ax.text(8.0, 8.0,
        'Crystal Plasticity Flow Rule  —  Computational Pipeline',
        ha='center', va='center', fontsize=17, fontweight='bold', color='#222222',
        zorder=4))

    # ── Draw arrows first (behind boxes, zorder=1) ──
    for a_from, a_to in OV_ARROWS:
        idx_from = STAGE_KEYS.index(a_from)
        idx_to = STAGE_KEYS.index(a_to)
        if idx_from <= up_to_stage_idx and idx_to <= up_to_stage_idx:
            bx1 = OV_BOXES[a_from]
            bx2 = OV_BOXES[a_to]

            if a_from == 'slip' and a_to == 'plastic':
                # Down from slip to plastic (same column, different row)
                x1 = bx1[0] + bx1[2]/2
                y1 = bx1[1] - 0.08
                x2 = bx2[0] + bx2[2]/2
                y2 = bx2[1] + bx2[3] + 0.08
            elif a_from == 'plastic' and a_to == 'fem':
                # Right from plastic to fem (same row)
                x1 = bx1[0] + bx1[2] + 0.08
                y1 = bx1[1] + bx1[3]/2
                x2 = bx2[0] - 0.08
                y2 = bx2[1] + bx2[3]/2
            else:
                # Default: right edge → left edge (same row)
                x1 = bx1[0] + bx1[2] + 0.08
                y1 = bx1[1] + bx1[3]/2
                x2 = bx2[0] - 0.08
                y2 = bx2[1] + bx2[3]/2

            a = ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color='#555555', lw=1.8),
                        zorder=1)
            dynamic.append(a)

    # Feedback arrow when FEM is revealed
    if up_to_stage_idx >= 5:
        bx_fem = OV_BOXES['fem']
        bx_slip = OV_BOXES['slip']
        a = ax.annotate('',
            xy=(bx_slip[0] + bx_slip[2], bx_slip[1] + 0.3),
            xytext=(bx_fem[0] + bx_fem[2]/2, bx_fem[1] + bx_fem[3] + 0.08),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.8,
                            connectionstyle='arc3,rad=-0.4', linestyle='dashed'),
            zorder=1)
        dynamic.append(a)

    # ── Draw boxes on top (zorder=3) ──
    for s in range(up_to_stage_idx + 1):
        key = STAGE_KEYS[s]
        x, y, w, h = OV_BOXES[key]
        fc, ec = OV_STYLES[key]

        # Highlight current stage
        lw = 3.0 if s == up_to_stage_idx else 1.5
        alpha = 1.0 if s == up_to_stage_idx else 0.65
        b = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                                     facecolor=fc, edgecolor=ec, lw=lw, zorder=3)
        ax.add_patch(b)
        dynamic.append(b)

        # Label
        dynamic.append(ax.text(x + w/2, y + h/2, OV_LABELS[key],
                               ha='center', va='center', fontsize=12,
                               fontweight='bold', color='#222222',
                               alpha=alpha, zorder=4, linespacing=1.1))


# ─── Animation timing ────────────────────────────────────────────────────────
DETAIL_FRAMES = 40    # ~4.0 s zoomed in
OVERVIEW_FRAMES = 20  # ~2.0 s showing pipeline
FINAL_HOLD = 35       # ~3.5 s final overview

FRAMES_PER_STAGE = DETAIL_FRAMES + OVERVIEW_FRAMES
TOTAL_FRAMES = len(STAGE_KEYS) * FRAMES_PER_STAGE + FINAL_HOLD


def update(frame):
    clear()

    stage_idx = min(frame // FRAMES_PER_STAGE, len(STAGE_KEYS) - 1)
    local = frame - stage_idx * FRAMES_PER_STAGE

    is_final_hold = frame >= len(STAGE_KEYS) * FRAMES_PER_STAGE

    if is_final_hold:
        # Show full pipeline, all stages
        draw_overview(len(STAGE_KEYS) - 1)
    elif local < DETAIL_FRAMES:
        # Zoomed-in detail view
        draw_detail(STAGE_KEYS[stage_idx])
    else:
        # Overview with all stages up to current
        draw_overview(stage_idx)

    return []


anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=100, blit=False)

plt.tight_layout(pad=0.3)
out_path = 'gif_flowchart.gif'
print(f"Rendering {TOTAL_FRAMES} frames "
      f"({len(STAGE_KEYS)} stages × {FRAMES_PER_STAGE} + {FINAL_HOLD} hold) …")
anim.save(out_path, writer='pillow', fps=10, dpi=130,
          savefig_kwargs={'facecolor': 'white'})
print(f"Saved  →  {out_path}")
plt.close()
