"""
GIF: Slip System Local Orthonormal Frame
=========================================
Visualizes how each FCC slip system defines a local frame:
  e1 = s_i   (slip direction)
  e3 = m_i   (plane normal)
  e2 = m_i x s_i

Animates through all 12 slip systems, showing the slip plane,
the three basis vectors, and the relationship to the crystal frame.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')

# ─── FCC Slip Systems ──────────────────────────────────────────────────────────
# 4 planes {111} x 3 directions <110> = 12 systems
PLANES_MILLER = [
    ( 1, 1, 1), ( 1, 1, 1), ( 1, 1, 1),
    (-1, 1, 1), (-1, 1, 1), (-1, 1, 1),
    ( 1,-1, 1), ( 1,-1, 1), ( 1,-1, 1),
    ( 1, 1,-1), ( 1, 1,-1), ( 1, 1,-1),
]

DIRS_MILLER = [
    ( 0, 1,-1), ( 1, 0,-1), ( 1,-1, 0),   # on (111)
    ( 0, 1,-1), ( 1, 0, 1), ( 1, 1, 0),   # on (-1,1,1)
    ( 0, 1, 1), ( 1, 0,-1), ( 1, 1, 0),   # on (1,-1,1)
    ( 0, 1, 1), ( 1, 0, 1), ( 1,-1, 0),   # on (1,1,-1)
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
    0: '#2196F3',  # blue  for (111)
    1: '#E91E63',  # pink  for (-1,1,1)
    2: '#4CAF50',  # green for (1,-1,1)
    3: '#FF9800',  # amber for (1,1,-1)
}


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def build_local_frame(plane_idx):
    """Build the local orthonormal frame for slip system `plane_idx`."""
    m = normalize(np.array(PLANES_MILLER[plane_idx], dtype=float))  # plane normal
    s = normalize(np.array(DIRS_MILLER[plane_idx], dtype=float))    # slip direction
    e1 = s                  # slip direction
    e3 = m                  # plane normal
    e2 = np.cross(m, s)     # m x s
    e2 = normalize(e2)
    return e1, e2, e3


def plane_triangle(normal, scale=0.55, center=np.array([0.5, 0.5, 0.5])):
    """Return 3 vertices of an equilateral triangle on the plane with given normal."""
    n = normalize(np.array(normal, dtype=float))
    # Find an arbitrary vector not parallel to n
    if abs(n[0]) < 0.9:
        t = np.array([1, 0, 0], dtype=float)
    else:
        t = np.array([0, 1, 0], dtype=float)
    u = normalize(np.cross(n, t))
    v = normalize(np.cross(n, u))
    # Equilateral triangle vertices
    angles = [0, 2*np.pi/3, 4*np.pi/3]
    verts = []
    for a in angles:
        verts.append(center + scale * (np.cos(a) * u + np.sin(a) * v))
    return np.array(verts)


# ─── Figure Setup ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 8), facecolor='white')
ax = fig.add_subplot(111, projection='3d', facecolor='white')

# Style helpers
TEXT_FX = [pe.withStroke(linewidth=2, foreground='white')]
ARROW_KW = dict(arrow_length_ratio=0.18, linewidth=2.5)
CRYSTAL_ARROW_KW = dict(arrow_length_ratio=0.15, linewidth=1.5, alpha=0.35)

# ─── Static elements: unit cube wireframe + crystal axes ───────────────────────
cube_edges = [
    ([0,0,0],[1,0,0]),([0,0,0],[0,1,0]),([0,0,0],[0,0,1]),
    ([1,0,0],[1,1,0]),([1,0,0],[1,0,1]),
    ([0,1,0],[1,1,0]),([0,1,0],[0,1,1]),
    ([0,0,1],[1,0,1]),([0,0,1],[0,1,1]),
    ([1,1,0],[1,1,1]),([1,0,1],[1,1,1]),([0,1,1],[1,1,1]),
]
for s, e in cube_edges:
    ax.plot(*zip(s, e), color='#555555', lw=0.8, alpha=0.5)

# Crystal (laboratory) frame axes — subtle in background
ax_len = 0.35
origin = np.array([-0.15, -0.15, -0.15])
ax.quiver(*origin, ax_len, 0, 0, color='#666666', **CRYSTAL_ARROW_KW)
ax.quiver(*origin, 0, ax_len, 0, color='#666666', **CRYSTAL_ARROW_KW)
ax.quiver(*origin, 0, 0, ax_len, color='#666666', **CRYSTAL_ARROW_KW)
ax.text(origin[0]+ax_len+0.03, origin[1], origin[2], '$x$ (crystal)',
        color='#888888', fontsize=7, path_effects=TEXT_FX)
ax.text(origin[0], origin[1]+ax_len+0.03, origin[2], '$y$',
        color='#888888', fontsize=7, path_effects=TEXT_FX)
ax.text(origin[0], origin[1], origin[2]+ax_len+0.03, '$z$',
        color='#888888', fontsize=7, path_effects=TEXT_FX)

# Corner atoms for FCC look
corners = np.array([
    [0,0,0],[1,0,0],[0,1,0],[0,0,1],
    [1,1,0],[1,0,1],[0,1,1],[1,1,1]
])
face_centers = np.array([
    [0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],
    [0.5,0.5,1],[0.5,1,0.5],[1,0.5,0.5]
])
ax.scatter(*corners.T, s=40, c='#AAAAAA', alpha=0.3, zorder=1)
ax.scatter(*face_centers.T, s=30, c='#888888', alpha=0.2, zorder=1)

# Axis limits & view
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(-0.3, 1.3)
ax.set_zlim(-0.3, 1.3)
ax.set_axis_off()

# Title
title_text = ax.text2D(0.5, 0.95, '', transform=ax.transAxes,
                       ha='center', va='top', fontsize=16, fontweight='bold',
                       color='black', path_effects=TEXT_FX)

# Equation / info text
info_text = ax.text2D(0.5, 0.03, '', transform=ax.transAxes,
                      ha='center', va='bottom', fontsize=11,
                      color='#333333', path_effects=TEXT_FX,
                      family='monospace')

# Legend-like label in corner
legend_text = ax.text2D(0.02, 0.88, '', transform=ax.transAxes,
                        va='top', fontsize=10, color='black',
                        path_effects=TEXT_FX, linespacing=1.8)

# ─── Mutable artists to clear each frame ──────────────────────────────────────
dynamic_artists = []


def clear_dynamic():
    for a in dynamic_artists:
        try:
            a.remove()
        except Exception:
            pass
    dynamic_artists.clear()


# ─── Animation ────────────────────────────────────────────────────────────────
FRAMES_PER_SYSTEM = 18     # frames per slip system
HOLD_FRAMES = 6            # pause on each system before transitioning
TOTAL_SYSTEMS = 12
TOTAL_FRAMES = TOTAL_SYSTEMS * FRAMES_PER_SYSTEM

CENTER = np.array([0.5, 0.5, 0.5])
VEC_SCALE = 0.45           # length of e1, e2, e3 arrows


def update(frame):
    clear_dynamic()

    # Which system are we showing?
    sys_idx = frame // FRAMES_PER_SYSTEM
    local_f = frame % FRAMES_PER_SYSTEM          # local frame within this system
    if sys_idx >= TOTAL_SYSTEMS:
        sys_idx = TOTAL_SYSTEMS - 1

    plane_family = sys_idx // 3                   # 0..3
    color = PLANE_COLORS[plane_family]

    e1, e2, e3 = build_local_frame(sys_idx)

    # Smooth rotation
    base_azim = -60 + frame * 0.8
    ax.view_init(elev=22, azim=base_azim)

    # ── Phase control ──
    # local_f 0-3 : fade in plane
    # local_f 4-6 : show e3 (normal)
    # local_f 7-9 : show e1 (slip dir)
    # local_f 10-12: show e2 (cross)
    # local_f 13+  : hold all, show equations
    progress = local_f / max(FRAMES_PER_SYSTEM - 1, 1)

    # ── Draw slip plane ──
    alpha_plane = min(1.0, local_f / 3.0) * 0.30
    tri = plane_triangle(PLANES_MILLER[sys_idx], scale=0.55, center=CENTER)
    poly = Poly3DCollection([tri], alpha=alpha_plane, facecolor=color,
                            edgecolor=color, linewidth=1.2)
    ax.add_collection3d(poly)
    dynamic_artists.append(poly)

    # ── Draw e3 = m (plane normal) ──
    if local_f >= 4:
        alpha_e3 = min(1.0, (local_f - 4) / 2.0)
        q = ax.quiver(*CENTER, *(e3 * VEC_SCALE),
                      color='#FF5252', alpha=alpha_e3, **ARROW_KW)
        dynamic_artists.append(q)
        t = ax.text(*(CENTER + e3 * (VEC_SCALE + 0.08)),
                    '$\\mathbf{e}_3 = \\mathbf{m}_i$',
                    color='#FF5252', fontsize=10, alpha=alpha_e3,
                    path_effects=TEXT_FX)
        dynamic_artists.append(t)

    # ── Draw e1 = s (slip direction) ──
    if local_f >= 7:
        alpha_e1 = min(1.0, (local_f - 7) / 2.0)
        q = ax.quiver(*CENTER, *(e1 * VEC_SCALE),
                      color='#69F0AE', alpha=alpha_e1, **ARROW_KW)
        dynamic_artists.append(q)
        t = ax.text(*(CENTER + e1 * (VEC_SCALE + 0.08)),
                    '$\\mathbf{e}_1 = \\mathbf{s}_i$',
                    color='#69F0AE', fontsize=10, alpha=alpha_e1,
                    path_effects=TEXT_FX)
        dynamic_artists.append(t)

    # ── Draw e2 = m x s ──
    if local_f >= 10:
        alpha_e2 = min(1.0, (local_f - 10) / 2.0)
        q = ax.quiver(*CENTER, *(e2 * VEC_SCALE),
                      color='#40C4FF', alpha=alpha_e2, **ARROW_KW)
        dynamic_artists.append(q)
        t = ax.text(*(CENTER + e2 * (VEC_SCALE + 0.08)),
                    '$\\mathbf{e}_2 = \\mathbf{m}_i \\times \\mathbf{s}_i$',
                    color='#40C4FF', fontsize=10, alpha=alpha_e2,
                    path_effects=TEXT_FX)
        dynamic_artists.append(t)

    # ── Dashed lines showing right-angle between e1 & e3 on the plane ──
    if local_f >= 10:
        tip1 = CENTER + e1 * VEC_SCALE * 0.25
        tip3 = CENTER + e3 * VEC_SCALE * 0.25
        corner = CENTER + e1 * VEC_SCALE * 0.25 + e3 * VEC_SCALE * 0.25
        for seg in [(tip1, corner), (corner, tip3)]:
            l, = ax.plot(*zip(*seg), color='white', lw=0.8, ls='--', alpha=0.5)
            dynamic_artists.append(l)

    # ── Title ──
    title_text.set_text(f'Slip System {sys_idx + 1} / 12')

    # ── Info text (bottom) ──
    if local_f >= 13:
        info_text.set_text(
            f'Plane ${PLANE_LABELS[sys_idx]}$    Direction ${DIR_LABELS[sys_idx]}$'
        )
    else:
        info_text.set_text('')

    # ── Legend ──
    legend_text.set_text(
        '$\\mathbf{e}_1 = \\mathbf{s}_i$  slip direction\n'
        '$\\mathbf{e}_3 = \\mathbf{m}_i$  plane normal\n'
        '$\\mathbf{e}_2 = \\mathbf{m}_i \\times \\mathbf{s}_i$'
    )

    return []


anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=120, blit=False)

plt.tight_layout()
out_path = 'gif_local_frame.gif'
print(f"Rendering {TOTAL_FRAMES} frames …")
anim.save(out_path, writer='pillow', fps=10, dpi=120,
          savefig_kwargs={'facecolor': 'white'})
print(f"Saved  →  {out_path}")
plt.close()
