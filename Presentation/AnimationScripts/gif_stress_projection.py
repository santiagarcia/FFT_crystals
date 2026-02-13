"""
GIF: Resolved Shear Stress Projection
=======================================
Visualizes how a macroscopic stress tensor sigma is projected onto each
slip system to obtain the resolved shear stress:

    tau_i = s_i . sigma . m_i

Shows the stress tensor as applied arrows, then projects onto each of the
12 FCC slip systems, highlighting that each system sees a different tau_i.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')

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

PLANE_LABELS = [
    "(111)","(111)","(111)",
    "(\\bar{1}11)","(\\bar{1}11)","(\\bar{1}11)",
    "(1\\bar{1}1)","(1\\bar{1}1)","(1\\bar{1}1)",
    "(11\\bar{1})","(11\\bar{1})","(11\\bar{1})",
]

DIR_LABELS = [
    "[0 1 \\bar{1}]","[1 0 \\bar{1}]","[1 \\bar{1} 0]",
    "[0 1 \\bar{1}]","[1 0 1]","[1 1 0]",
    "[0 1 1]","[1 0 \\bar{1}]","[1 1 0]",
    "[0 1 1]","[1 0 1]","[1 \\bar{1} 0]",
]

PLANE_COLORS = {
    0: '#2196F3',  # blue
    1: '#E91E63',  # pink
    2: '#4CAF50',  # green
    3: '#FF9800',  # amber
}


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def build_system(idx):
    m = normalize(np.array(PLANES_MILLER[idx], dtype=float))
    s = normalize(np.array(DIRS_MILLER[idx], dtype=float))
    return s, m


# ─── Applied stress tensor (symmetric, in MPa-like units) ─────────────────────
# Uniaxial-ish tension along [1,0,0] with some shear — gives varied tau_i
SIGMA = np.array([
    [350.0,  50.0, -30.0],
    [ 50.0, -80.0,  20.0],
    [-30.0,  20.0, -60.0]
]) * 1e6  # Pa scale to match driver values

# Pre-compute tau for each system
TAU = np.zeros(12)
for i in range(12):
    s, m = build_system(i)
    TAU[i] = np.dot(s, SIGMA @ m)

TAU_MAX = np.max(np.abs(TAU))


def plane_triangle(normal, scale=0.48, center=np.array([0.5, 0.5, 0.5])):
    n = normalize(np.array(normal, dtype=float))
    if abs(n[0]) < 0.9:
        t = np.array([1, 0, 0], dtype=float)
    else:
        t = np.array([0, 1, 0], dtype=float)
    u = normalize(np.cross(n, t))
    v = normalize(np.cross(n, u))
    angles = [0, 2*np.pi/3, 4*np.pi/3]
    return np.array([center + scale * (np.cos(a)*u + np.sin(a)*v) for a in angles])


# ─── Figure Setup ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 7), facecolor='white')
# Left: 3D crystal view  |  Right: bar chart of tau_i
ax3d = fig.add_subplot(121, projection='3d', facecolor='white')
ax_bar = fig.add_subplot(122, facecolor='white')

TEXT_FX = [pe.withStroke(linewidth=2, foreground='white')]
ARROW_KW = dict(arrow_length_ratio=0.15, linewidth=2.5)

# ── Static elements on 3D axis ────────────────────────────────────────────────
cube_edges = [
    ([0,0,0],[1,0,0]),([0,0,0],[0,1,0]),([0,0,0],[0,0,1]),
    ([1,0,0],[1,1,0]),([1,0,0],[1,0,1]),
    ([0,1,0],[1,1,0]),([0,1,0],[0,1,1]),
    ([0,0,1],[1,0,1]),([0,0,1],[0,1,1]),
    ([1,1,0],[1,1,1]),([1,0,1],[1,1,1]),([0,1,1],[1,1,1]),
]
for s, e in cube_edges:
    ax3d.plot(*zip(s, e), color='#AAAAAA', lw=0.8, alpha=0.5)

# Atoms
corners = np.array([
    [0,0,0],[1,0,0],[0,1,0],[0,0,1],
    [1,1,0],[1,0,1],[0,1,1],[1,1,1]
])
face_centers = np.array([
    [0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],
    [0.5,0.5,1],[0.5,1,0.5],[1,0.5,0.5]
])
ax3d.scatter(*corners.T, s=35, c='#BBBBBB', alpha=0.3, zorder=1)
ax3d.scatter(*face_centers.T, s=25, c='#999999', alpha=0.2, zorder=1)

ax3d.set_xlim(-0.3, 1.3)
ax3d.set_ylim(-0.3, 1.3)
ax3d.set_zlim(-0.3, 1.3)
ax3d.set_axis_off()

# ── Static stress arrows (sigma applied on faces) ─────────────────────────────
# Show as paired arrows on cube faces to suggest applied tension
stress_arrows_static = []
# Tension along x on +x face
sa = ax3d.quiver(1.0, 0.5, 0.5, 0.3, 0, 0, color='#D32F2F', linewidth=2,
                 arrow_length_ratio=0.25, alpha=0.4)
stress_arrows_static.append(sa)
# Compression on -x face
sa = ax3d.quiver(0.0, 0.5, 0.5, -0.3, 0, 0, color='#D32F2F', linewidth=2,
                 arrow_length_ratio=0.25, alpha=0.4)
stress_arrows_static.append(sa)
# Label
ax3d.text(1.35, 0.5, 0.5, '$\\boldsymbol{\\sigma}$', color='#D32F2F',
          fontsize=16, path_effects=TEXT_FX)

# Title
title_3d = ax3d.text2D(0.5, 0.97, '', transform=ax3d.transAxes,
                       ha='center', va='top', fontsize=14, fontweight='bold',
                       color='black', path_effects=TEXT_FX)

# Equation text
eq_text = ax3d.text2D(0.5, 0.02, '', transform=ax3d.transAxes,
                      ha='center', va='bottom', fontsize=12,
                      color='#333333', path_effects=TEXT_FX)

# ── Bar chart setup ───────────────────────────────────────────────────────────
ax_bar.set_xlabel('Slip System $i$', fontsize=11)
ax_bar.set_ylabel('$\\tau_i$ (MPa)', fontsize=11)
ax_bar.set_title('Resolved Shear Stress per System', fontsize=13, fontweight='bold')
ax_bar.set_xticks(range(1, 13))
ax_bar.axhline(0, color='gray', lw=0.5)
ax_bar.set_xlim(0.3, 12.7)
y_lim = TAU_MAX / 1e6 * 1.3
ax_bar.set_ylim(-y_lim, y_lim)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)

# ─── Mutable artists ─────────────────────────────────────────────────────────
dynamic = []


def clear_dynamic():
    for a in dynamic:
        try:
            a.remove()
        except Exception:
            pass
    dynamic.clear()


# ─── Animation ────────────────────────────────────────────────────────────────
FRAMES_PER_SYSTEM = 15
TOTAL_SYSTEMS = 12
# Extra intro frames showing sigma before first projection
INTRO_FRAMES = 20
TOTAL_FRAMES = INTRO_FRAMES + TOTAL_SYSTEMS * FRAMES_PER_SYSTEM

CENTER = np.array([0.5, 0.5, 0.5])
VEC_SCALE = 0.40


def update(frame):
    clear_dynamic()

    # Slow rotation
    azim = -60 + frame * 0.6
    ax3d.view_init(elev=22, azim=azim)

    # ── Intro phase: just show sigma ──
    if frame < INTRO_FRAMES:
        progress = frame / (INTRO_FRAMES - 1)
        title_3d.set_text('Applied Stress Tensor $\\boldsymbol{\\sigma}$')
        eq_text.set_text('$\\tau_i = \\mathbf{s}_i \\cdot \\boldsymbol{\\sigma} \\cdot \\mathbf{m}_i$')

        # Pulsing stress arrows opacity
        pulse = 0.4 + 0.3 * np.sin(frame * 0.5)
        for sa in stress_arrows_static:
            sa.set_alpha(pulse)

        # Empty bar chart — gray placeholders
        bars = ax_bar.bar(range(1, 13), [0]*12, color='#DDDDDD', edgecolor='#CCCCCC')
        for b in bars:
            dynamic.append(b)

        return []

    # ── Projection phase ──
    adj_frame = frame - INTRO_FRAMES
    sys_idx = adj_frame // FRAMES_PER_SYSTEM
    local_f = adj_frame % FRAMES_PER_SYSTEM
    if sys_idx >= TOTAL_SYSTEMS:
        sys_idx = TOTAL_SYSTEMS - 1

    plane_family = sys_idx // 3
    color = PLANE_COLORS[plane_family]

    s_vec, m_vec = build_system(sys_idx)
    tau_i = TAU[sys_idx]

    # Reset stress arrow alpha
    for sa in stress_arrows_static:
        sa.set_alpha(0.3)

    title_3d.set_text(f'Projecting onto System {sys_idx + 1} / 12')

    # ── Phase control within a system ──
    # 0-3:   show slip plane
    # 4-6:   show m_i (normal)
    # 7-9:   show s_i (slip dir)
    # 10-12: show projection arrow (tau vector along s)
    # 13+:   hold, show equation result

    # ── Slip plane ──
    alpha_plane = min(1.0, local_f / 3.0) * 0.25
    tri = plane_triangle(PLANES_MILLER[sys_idx], scale=0.48, center=CENTER)
    poly = Poly3DCollection([tri], alpha=alpha_plane, facecolor=color,
                            edgecolor=color, linewidth=1.2)
    ax3d.add_collection3d(poly)
    dynamic.append(poly)

    # ── m_i (plane normal) ──
    if local_f >= 4:
        alpha_m = min(1.0, (local_f - 4) / 2.0)
        q = ax3d.quiver(*CENTER, *(m_vec * VEC_SCALE),
                        color='#FF5252', alpha=alpha_m, **ARROW_KW)
        dynamic.append(q)
        t = ax3d.text(*(CENTER + m_vec * (VEC_SCALE + 0.07)),
                      '$\\mathbf{m}_i$', color='#FF5252', fontsize=11,
                      alpha=alpha_m, path_effects=TEXT_FX)
        dynamic.append(t)

    # ── s_i (slip direction) ──
    if local_f >= 7:
        alpha_s = min(1.0, (local_f - 7) / 2.0)
        q = ax3d.quiver(*CENTER, *(s_vec * VEC_SCALE),
                        color='#69F0AE', alpha=alpha_s, **ARROW_KW)
        dynamic.append(q)
        t = ax3d.text(*(CENTER + s_vec * (VEC_SCALE + 0.07)),
                      '$\\mathbf{s}_i$', color='#388E3C', fontsize=11,
                      alpha=alpha_s, path_effects=TEXT_FX)
        dynamic.append(t)

    # ── Projection result: tau arrow along s ──
    if local_f >= 10:
        alpha_tau = min(1.0, (local_f - 10) / 2.0)
        # Scale tau arrow proportional to its magnitude
        tau_norm = tau_i / TAU_MAX
        tau_arrow = s_vec * VEC_SCALE * 0.9 * tau_norm
        # Offset slightly from center so it doesn't overlap s_i
        offset = m_vec * 0.08
        q = ax3d.quiver(*(CENTER + offset), *tau_arrow,
                        color='#FF6F00', alpha=alpha_tau,
                        linewidth=3.5, arrow_length_ratio=0.2)
        dynamic.append(q)
        t = ax3d.text(*(CENTER + offset + tau_arrow * 0.5 + m_vec * 0.06),
                      f'$\\tau_{{{sys_idx+1}}}$', color='#E65100', fontsize=12,
                      alpha=alpha_tau, fontweight='bold', path_effects=TEXT_FX)
        dynamic.append(t)

    # ── Equation ──
    if local_f >= 13:
        eq_text.set_text(
            f'$\\tau_{{{sys_idx+1}}} = \\mathbf{{s}}_{{{sys_idx+1}}} '
            f'\\cdot \\boldsymbol{{\\sigma}} \\cdot \\mathbf{{m}}_{{{sys_idx+1}}} '
            f'= {tau_i/1e6:.1f}$ MPa'
        )
    else:
        eq_text.set_text(
            '$\\tau_i = \\mathbf{s}_i \\cdot \\boldsymbol{\\sigma} \\cdot \\mathbf{m}_i$'
        )

    # ── Bar chart: reveal bars up to current system ──
    bar_vals = np.zeros(12)
    bar_colors_list = ['#DDDDDD'] * 12
    bar_edge = ['#CCCCCC'] * 12
    for j in range(sys_idx + 1):
        fam = j // 3
        bar_vals[j] = TAU[j] / 1e6
        if j == sys_idx:
            # Current system — highlighted
            bar_colors_list[j] = PLANE_COLORS[fam]
            bar_edge[j] = '#000000'
        else:
            # Previous systems — muted version of family color
            bar_colors_list[j] = PLANE_COLORS[fam] + '88'  # hex alpha
            bar_edge[j] = PLANE_COLORS[fam]

    bars = ax_bar.bar(range(1, 13), bar_vals,
                      color=bar_colors_list, edgecolor=bar_edge, linewidth=1.2)
    for b in bars:
        dynamic.append(b)

    # Label on current bar
    if local_f >= 10:
        val = TAU[sys_idx] / 1e6
        ypos = val + (5 if val >= 0 else -12)
        t = ax_bar.text(sys_idx + 1, ypos, f'{val:.0f}',
                        ha='center', va='bottom' if val >= 0 else 'top',
                        fontsize=8, fontweight='bold', color='#333333')
        dynamic.append(t)

    return []


anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=120, blit=False)

plt.tight_layout()
out_path = 'gif_stress_projection.gif'
print(f"Rendering {TOTAL_FRAMES} frames …")
anim.save(out_path, writer='pillow', fps=10, dpi=120,
          savefig_kwargs={'facecolor': 'white'})
print(f"Saved  →  {out_path}")
plt.close()
