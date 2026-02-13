"""
GIF: Hardening — Slip Resistance Evolution (Unit Cell Version)
===============================================================
Visualizes r_i = r_i(gamma_1, gamma_2, ..., gamma_12) using the 3D unit cell.

Two-panel animation:
  Left:  3D FCC unit cell — active system glows, resistance "pulses" propagate
         to other systems (self-hardening = bright, latent = dimmer)
  Right: Clean bar chart of r_i growing step by step

Much more intuitive than the 12x12 matrix approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')

TEXT_FX = [pe.withStroke(linewidth=2, foreground='white')]

# ─── FCC Slip Systems ──────────────────────────────────────────────────────────
PLANES_MILLER = [
    ( 1, 1, 1), ( 1, 1, 1), ( 1, 1, 1),
    (-1, 1, 1), (-1, 1, 1), (-1, 1, 1),
    ( 1,-1, 1), ( 1,-1, 1), ( 1,-1, 1),
    ( 1, 1,-1), ( 1, 1,-1), ( 1, 1,-1),
]
DIRS_MILLER = [
    ( 0, 1,-1), ( 1, 0,-1), ( 1,-1, 0),
    ( 0, 1,-1), ( 1, 0, 1), ( 1, 1, 0),
    ( 0, 1, 1), ( 1, 0,-1), ( 1, 1, 0),
    ( 0, 1, 1), ( 1, 0, 1), ( 1,-1, 0),
]

PLANE_COLORS = {0: '#2196F3', 1: '#E91E63', 2: '#4CAF50', 3: '#FF9800'}
FAMILY_NAMES = ['(111)', r'$(\bar{1}11)$', r'$(1\bar{1}1)$', r'$(11\bar{1})$']


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def build_system(idx):
    m = normalize(np.array(PLANES_MILLER[idx], dtype=float))
    s = normalize(np.array(DIRS_MILLER[idx], dtype=float))
    return s, m


def plane_triangle(normal, scale=0.38, center=np.array([0.5, 0.5, 0.5])):
    n = normalize(np.array(normal, dtype=float))
    t = np.array([1, 0, 0], dtype=float) if abs(n[0]) < 0.9 else np.array([0, 1, 0], dtype=float)
    u = normalize(np.cross(n, t))
    v = normalize(np.cross(n, u))
    return np.array([center + scale * (np.cos(a)*u + np.sin(a)*v)
                     for a in [0, 2*np.pi/3, 4*np.pi/3]])


# ─── Hardening model ─────────────────────────────────────────────────────────
H_SELF = 1.0
H_COPLANAR = 0.2
H_FOREST = 0.6

def hardening_weight(i, j):
    if i == j:
        return H_SELF
    elif i // 3 == j // 3:
        return H_COPLANAR
    else:
        return H_FOREST

R_INITIAL = 300.0  # MPa

# Simplified slip sequence — fewer steps, more impactful
SLIP_SEQUENCE = [
    (0,  0.025),  # Sys 1 slips
    (3,  0.030),  # Sys 4 slips
    (9,  0.028),  # Sys 10 slips
    (0,  0.020),  # Sys 1 again
    (6,  0.015),  # Sys 7 activates
    (11, 0.025),  # Sys 12 slips
]

# Pre-compute resistance histories
R_HISTORY = [np.full(12, R_INITIAL)]
ACTIVE_SYSTEM = []

r_cur = np.full(12, R_INITIAL)
for sys_idx, dg in SLIP_SEQUENCE:
    dr = np.array([hardening_weight(i, sys_idx) * dg * 500.0 for i in range(12)])
    r_cur = r_cur + dr
    R_HISTORY.append(r_cur.copy())
    ACTIVE_SYSTEM.append(sys_idx)

R_MAX = np.max([np.max(r) for r in R_HISTORY]) * 1.15

# ─── Pre-compute arrow positions (spread around center) ──────────────────────
CENTER = np.array([0.5, 0.5, 0.5])
S_ALL = []
M_ALL = []
ARROW_ORIGINS = []
for i in range(12):
    s, m = build_system(i)
    S_ALL.append(s)
    M_ALL.append(m)
    offset = m * 0.12 + s * 0.05 * ((i % 3) - 1)
    ARROW_ORIGINS.append(CENTER + offset)

# ─── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 6.5), facecolor='white')
ax3d = fig.add_subplot(121, projection='3d', facecolor='white')
ax_bar = fig.add_subplot(122, facecolor='white')

# ── 3D static elements ────────────────────────────────────────────────────────
cube_edges = [
    ([0,0,0],[1,0,0]),([0,0,0],[0,1,0]),([0,0,0],[0,0,1]),
    ([1,0,0],[1,1,0]),([1,0,0],[1,0,1]),
    ([0,1,0],[1,1,0]),([0,1,0],[0,1,1]),
    ([0,0,1],[1,0,1]),([0,0,1],[0,1,1]),
    ([1,1,0],[1,1,1]),([1,0,1],[1,1,1]),([0,1,1],[1,1,1]),
]
for st, en in cube_edges:
    ax3d.plot(*zip(st, en), color='#BBBBBB', lw=0.8, alpha=0.4)

corners = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                     [1,1,0],[1,0,1],[0,1,1],[1,1,1]])
ax3d.scatter(*corners.T, s=25, c='#CCCCCC', alpha=0.2, zorder=1)

ax3d.set_xlim(-0.15, 1.15)
ax3d.set_ylim(-0.15, 1.15)
ax3d.set_zlim(-0.15, 1.15)
ax3d.set_axis_off()

title_3d = ax3d.text2D(0.5, 0.97, '', transform=ax3d.transAxes,
                       ha='center', va='top', fontsize=14, fontweight='bold',
                       color='black', path_effects=TEXT_FX)
info_3d = ax3d.text2D(0.5, 0.02, '', transform=ax3d.transAxes,
                      ha='center', va='bottom', fontsize=10, color='#555555',
                      style='italic')

# ── Bar chart setup ───────────────────────────────────────────────────────────
ax_bar.set_xlabel('Slip System $i$', fontsize=11)
ax_bar.set_ylabel('Resistance  $r_i$  (MPa)', fontsize=11)
ax_bar.set_title('$r_i = r_i(\\gamma_1, \\gamma_2, \\ldots, \\gamma_{12})$',
                 fontsize=13, fontweight='bold')
ax_bar.set_xticks(range(1, 13))
ax_bar.set_xlim(0.3, 12.7)
ax_bar.set_ylim(0, R_MAX)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.axhline(R_INITIAL, color='#AAAAAA', ls='--', lw=1, alpha=0.6)
ax_bar.text(12.5, R_INITIAL + 2, '$r_0$', fontsize=9, color='#AAAAAA', va='bottom')

# ─── Mutable ──────────────────────────────────────────────────────────────────
dynamic = []

def clear_dynamic():
    for a in dynamic:
        try:
            a.remove()
        except Exception:
            pass
    dynamic.clear()


# ─── Animation ────────────────────────────────────────────────────────────────
INTRO_FRAMES = 20
FRAMES_PER_STEP = 20
HOLD_FRAMES = 25
TOTAL_FRAMES = INTRO_FRAMES + len(SLIP_SEQUENCE) * FRAMES_PER_STEP + HOLD_FRAMES

VEC_LEN = 0.18


def update(frame):
    clear_dynamic()
    azim = -55 + frame * 0.4
    ax3d.view_init(elev=22, azim=azim)

    # ═══════════════════════════════════════════════════════════════════════════
    # INTRO
    # ═══════════════════════════════════════════════════════════════════════════
    if frame < INTRO_FRAMES:
        title_3d.set_text('12 Slip Systems — Equal Initial Resistance')
        info_3d.set_text('Each system has resistance $r_i$ that will evolve')

        for i in range(12):
            fam = i // 3
            col = PLANE_COLORS[fam]
            origin = ARROW_ORIGINS[i]
            q = ax3d.quiver(*origin, *(S_ALL[i] * VEC_LEN),
                            color=col, alpha=0.5, linewidth=1.5,
                            arrow_length_ratio=0.25)
            dynamic.append(q)
            t = ax3d.text(*(origin + S_ALL[i] * (VEC_LEN + 0.04)),
                          str(i+1), fontsize=7, color=col, alpha=0.7,
                          path_effects=TEXT_FX)
            dynamic.append(t)

        bar_cols = [PLANE_COLORS[i//3] + '88' for i in range(12)]
        bars = ax_bar.bar(range(1, 13), R_HISTORY[0],
                          color=bar_cols,
                          edgecolor=[PLANE_COLORS[i//3] for i in range(12)],
                          linewidth=1.2)
        for b in bars:
            dynamic.append(b)
        return []

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP-BY-STEP HARDENING
    # ═══════════════════════════════════════════════════════════════════════════
    adj = frame - INTRO_FRAMES
    is_hold = adj >= len(SLIP_SEQUENCE) * FRAMES_PER_STEP

    if not is_hold:
        step = adj // FRAMES_PER_STEP
        local_f = adj % FRAMES_PER_STEP
    else:
        step = len(SLIP_SEQUENCE) - 1
        local_f = FRAMES_PER_STEP - 1

    active = ACTIVE_SYSTEM[step]
    fam_active = active // 3
    t_progress = local_f / (FRAMES_PER_STEP - 1)

    # Sub-phases
    if t_progress < 0.3:
        phase = 'slip'
    elif t_progress < 0.7:
        phase = 'propagate'
    else:
        phase = 'result'

    if not is_hold:
        title_3d.set_text(f'Step {step+1}: System {active+1} slips')
    else:
        title_3d.set_text('Systems Are Coupled')

    # Slip plane of active system
    tri = plane_triangle(PLANES_MILLER[active], scale=0.38, center=CENTER)
    poly = Poly3DCollection([tri], alpha=0.20,
                            facecolor=PLANE_COLORS[fam_active],
                            edgecolor=PLANE_COLORS[fam_active], linewidth=1.5)
    ax3d.add_collection3d(poly)
    dynamic.append(poly)

    # Draw all 12 system arrows
    for i in range(12):
        fam = i // 3
        col = PLANE_COLORS[fam]
        origin = ARROW_ORIGINS[i]
        hw = hardening_weight(i, active)

        if i == active:
            pulse = 0.7 + 0.3 * np.sin(local_f * 0.8)
            q = ax3d.quiver(*origin, *(S_ALL[i] * VEC_LEN * 1.3),
                            color=col, alpha=pulse, linewidth=4.0,
                            arrow_length_ratio=0.22)
            dynamic.append(q)

            if phase == 'slip':
                t = ax3d.text(*(origin + S_ALL[i] * (VEC_LEN * 1.3 + 0.06) + M_ALL[i] * 0.04),
                              'SLIP!', fontsize=11, color='#D32F2F',
                              fontweight='bold', path_effects=TEXT_FX)
                dynamic.append(t)

            if phase in ('propagate', 'result'):
                t = ax3d.text(*(origin + M_ALL[i] * 0.10),
                              'Self\nharden', fontsize=7, color='#D32F2F',
                              ha='center', fontweight='bold', path_effects=TEXT_FX,
                              alpha=min(1.0, (t_progress - 0.3) / 0.15))
                dynamic.append(t)
        else:
            if phase == 'slip':
                alpha = 0.2
                lw = 1.0
            elif phase == 'propagate':
                wave_t = (t_progress - 0.3) / 0.4
                alpha = 0.2 + 0.6 * hw * wave_t
                lw = 1.0 + 2.0 * hw * wave_t
            else:
                alpha = 0.2 + 0.5 * hw
                lw = 1.0 + 1.5 * hw

            q = ax3d.quiver(*origin, *(S_ALL[i] * VEC_LEN),
                            color=col, alpha=alpha, linewidth=lw,
                            arrow_length_ratio=0.22)
            dynamic.append(q)

            # Connection lines for forest hardening
            if phase == 'propagate' and hw > 0.3:
                wave_t = (t_progress - 0.3) / 0.4
                if wave_t > 0.2:
                    o_a = ARROW_ORIGINS[active]
                    line, = ax3d.plot([o_a[0], origin[0]],
                                     [o_a[1], origin[1]],
                                     [o_a[2], origin[2]],
                                     color='#D32F2F', ls='--', lw=0.8,
                                     alpha=0.3 * wave_t)
                    dynamic.append(line)

        # System number
        num_alpha = 0.8 if i == active else 0.4
        t = ax3d.text(*(origin + S_ALL[i] * (VEC_LEN + 0.04)),
                      str(i+1), fontsize=7, color=col, alpha=num_alpha,
                      path_effects=TEXT_FX)
        dynamic.append(t)

    # Info text
    if is_hold:
        info_3d.set_text('Kinematics are independent, but hardening couples all systems')
    elif phase == 'slip':
        info_3d.set_text(f'System {active+1} activates — dislocation motion')
    elif phase == 'propagate':
        info_3d.set_text('Resistance grows on ALL systems (forest hardening)')
    else:
        info_3d.set_text('Self-hardening (strong) + Latent hardening (weaker)')

    # ── BAR CHART ──
    r_prev = R_HISTORY[step]
    r_next = R_HISTORY[step + 1]
    r_show = r_prev + t_progress * (r_next - r_prev)

    bar_cols = []
    bar_edges = []
    for i in range(12):
        fam = i // 3
        hw = hardening_weight(i, active)
        if i == active:
            bar_cols.append('#D32F2F')
            bar_edges.append('#B71C1C')
        elif hw > 0.3 and t_progress > 0.3:
            bar_cols.append('#FF8A65')
            bar_edges.append('#E64A19')
        elif hw > 0 and t_progress > 0.3:
            bar_cols.append('#FFCC80')
            bar_edges.append('#FF9800')
        else:
            bar_cols.append(PLANE_COLORS[fam] + '55')
            bar_edges.append(PLANE_COLORS[fam] + '88')

    bars = ax_bar.bar(range(1, 13), r_show,
                      color=bar_cols, edgecolor=bar_edges, linewidth=1.3)
    for b in bars:
        dynamic.append(b)

    # Delta-r annotations
    if t_progress > 0.4:
        dr = r_next[active] - r_prev[active]
        txt = ax_bar.annotate(
            f'+{dr:.0f}\n(self)',
            xy=(active + 1, r_show[active]),
            xytext=(active + 1, r_show[active] + R_MAX * 0.06),
            ha='center', fontsize=8, fontweight='bold', color='#D32F2F',
            arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1))
        dynamic.append(txt)

    if t_progress > 0.5:
        for ex in range(12):
            if ex // 3 != active // 3 and ex != active:
                break
        dr_lat = r_next[ex] - r_prev[ex]
        if dr_lat > 0.5:
            txt2 = ax_bar.annotate(
                f'+{dr_lat:.0f}\n(latent)',
                xy=(ex + 1, r_show[ex]),
                xytext=(ex + 1, r_show[ex] + R_MAX * 0.06),
                ha='center', fontsize=8, fontweight='bold', color='#E64A19',
                arrowprops=dict(arrowstyle='->', color='#E64A19', lw=1))
            dynamic.append(txt2)

    # Color legend (compact, bottom-right, no overlap)
    if t_progress > 0.3:
        leg_items = [
            ('#D32F2F', 'Self-hardening'),
            ('#FF8A65', 'Forest (cross-family)'),
            ('#FFCC80', 'Coplanar (same family)'),
        ]
        for idx_p, (pc, plabel) in enumerate(leg_items):
            xx = 0.62
            yy = 0.18 - idx_p * 0.05
            ax_bar.plot(xx, yy, 's', color=pc, markersize=8,
                        transform=ax_bar.transAxes, clip_on=False)
            t = ax_bar.text(xx + 0.03, yy, plabel, fontsize=8,
                            va='center', color='#333333',
                            transform=ax_bar.transAxes)
            dynamic.append(t)

    # Hold summary box
    if is_hold:
        summary = ax_bar.text(7, R_MAX * 0.55,
            'Slip systems are\nNOT independent\n\n'
            'Slip on one system\nhardens all others',
            ha='center', fontsize=11, color='#333333', fontweight='bold',
            bbox=dict(facecolor='#FFEBEE', edgecolor='#D32F2F',
                      alpha=0.92, boxstyle='round,pad=0.5'))
        dynamic.append(summary)

    return []


anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=140, blit=False)

plt.tight_layout()
out_path = 'gif_hardening.gif'
print(f"Rendering {TOTAL_FRAMES} frames …")
anim.save(out_path, writer='pillow', fps=10, dpi=120,
          savefig_kwargs={'facecolor': 'white'})
print(f"Saved  →  {out_path}")
plt.close()
