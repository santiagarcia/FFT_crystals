"""
GIF: Macroscopic Plastic Strain Rate — Superposition of Schmid Tensors
=======================================================================
Visualizes:
    eps_dot^p = sum_{i=1}^{12} gamma_dot_i * P_i

where the symmetric Schmid tensor is:
    P_i = 1/2 (s_i ⊗ m_i + m_i ⊗ s_i)

Three-panel animation:
  Left:   3D crystal with slip system arrows accumulating
  Center: 3x3 Schmid tensor heatmap for current system, then the running sum
  Right:  Bar chart of gamma_dot_i contributions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
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

PLANE_COLORS = {
    0: '#2196F3',  # blue
    1: '#E91E63',  # pink
    2: '#4CAF50',  # green
    3: '#FF9800',  # amber
}

PLANE_COLORS_LIGHT = {
    0: '#BBDEFB',
    1: '#F8BBD0',
    2: '#C8E6C9',
    3: '#FFE0B2',
}

TEXT_FX = [pe.withStroke(linewidth=2, foreground='white')]


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def build_system(idx):
    m = normalize(np.array(PLANES_MILLER[idx], dtype=float))
    s = normalize(np.array(DIRS_MILLER[idx], dtype=float))
    return s, m


def schmid_tensor(s, m):
    """P_i = 0.5 * (s ⊗ m + m ⊗ s)"""
    return 0.5 * (np.outer(s, m) + np.outer(m, s))


# ─── Pre-compute Schmid tensors & fake gamma_dots ─────────────────────────────
P_ALL = []
S_ALL = []
M_ALL = []
for i in range(12):
    s, m = build_system(i)
    S_ALL.append(s)
    M_ALL.append(m)
    P_ALL.append(schmid_tensor(s, m))

# Realistic-looking gamma_dot values (from computeFlowRule output pattern)
# Systems 1,4,10,12 are the most active (high resolved stress in the driver)
GAMMA_DOT = np.array([
    5.2e5, -0.3e5, -4.8e5,
    5.5e5, -0.2e5, -5.0e5,
    4.9e5,  0.05e5, -5.1e5,
    5.6e5,  0.03e5, -5.3e5,
])

GAMMA_MAX = np.max(np.abs(GAMMA_DOT))

# Pre-compute the running sums
EPS_DOT_CUMULATIVE = []
running = np.zeros((3, 3))
for i in range(12):
    running = running + GAMMA_DOT[i] * P_ALL[i]
    EPS_DOT_CUMULATIVE.append(running.copy())


def plane_triangle(normal, scale=0.42, center=np.array([0.5, 0.5, 0.5])):
    n = normalize(np.array(normal, dtype=float))
    if abs(n[0]) < 0.9:
        t = np.array([1, 0, 0], dtype=float)
    else:
        t = np.array([0, 1, 0], dtype=float)
    u = normalize(np.cross(n, t))
    v = normalize(np.cross(n, u))
    angles = [0, 2*np.pi/3, 4*np.pi/3]
    return np.array([center + scale * (np.cos(a)*u + np.sin(a)*v) for a in angles])


# ─── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 6.5), facecolor='white')

# Three panels
ax3d = fig.add_subplot(131, projection='3d', facecolor='white')
ax_tensor = fig.add_subplot(132, facecolor='white')
ax_bar = fig.add_subplot(133, facecolor='white')

ARROW_KW = dict(arrow_length_ratio=0.18, linewidth=2.0)

# ── 3D panel: static wireframe ────────────────────────────────────────────────
cube_edges = [
    ([0,0,0],[1,0,0]),([0,0,0],[0,1,0]),([0,0,0],[0,0,1]),
    ([1,0,0],[1,1,0]),([1,0,0],[1,0,1]),
    ([0,1,0],[1,1,0]),([0,1,0],[0,1,1]),
    ([0,0,1],[1,0,1]),([0,0,1],[0,1,1]),
    ([1,1,0],[1,1,1]),([1,0,1],[1,1,1]),([0,1,1],[1,1,1]),
]
for start, end in cube_edges:
    ax3d.plot(*zip(start, end), color='#AAAAAA', lw=0.8, alpha=0.5)

corners = np.array([
    [0,0,0],[1,0,0],[0,1,0],[0,0,1],
    [1,1,0],[1,0,1],[0,1,1],[1,1,1]
])
ax3d.scatter(*corners.T, s=30, c='#BBBBBB', alpha=0.25, zorder=1)

ax3d.set_xlim(-0.2, 1.2)
ax3d.set_ylim(-0.2, 1.2)
ax3d.set_zlim(-0.2, 1.2)
ax3d.set_axis_off()

title_3d = ax3d.text2D(0.5, 0.97, '', transform=ax3d.transAxes,
                       ha='center', va='top', fontsize=13, fontweight='bold',
                       color='black', path_effects=TEXT_FX)

# ── Tensor panel setup ────────────────────────────────────────────────────────
ax_tensor.set_axis_off()
tensor_title = ax_tensor.set_title('', fontsize=13, fontweight='bold')

# ── Bar chart setup ───────────────────────────────────────────────────────────
ax_bar.set_xlabel('Slip System $i$', fontsize=10)
ax_bar.set_ylabel('$\\dot{\\gamma}_i$ (×$10^5$ s$^{-1}$)', fontsize=10)
ax_bar.set_title('Shear Rate Contributions', fontsize=13, fontweight='bold')
ax_bar.set_xticks(range(1, 13))
ax_bar.axhline(0, color='gray', lw=0.5)
ax_bar.set_xlim(0.3, 12.7)
y_lim = GAMMA_MAX / 1e5 * 1.4
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


def draw_tensor_heatmap(ax, tensor, title_str, subtitle='', highlight_color=None):
    """Draw a 3x3 tensor as a colored grid with values."""
    ax.clear()
    ax.set_axis_off()

    vmax = max(np.max(np.abs(tensor)), 1e-5)
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    labels = ['x', 'y', 'z']

    for i in range(3):
        for j in range(3):
            val = tensor[i, j]
            color = cmap(norm(val))
            rect = plt.Rectangle((j, 2 - i), 1, 1, facecolor=color,
                                 edgecolor='#333333', linewidth=1.5)
            ax.add_patch(rect)

            # Value text
            txt_color = 'white' if abs(val) > vmax * 0.55 else 'black'
            ax.text(j + 0.5, 2 - i + 0.5, f'{val:.3f}',
                    ha='center', va='center', fontsize=9,
                    fontweight='bold', color=txt_color)

    # Row/Col labels
    for k in range(3):
        ax.text(k + 0.5, 3.15, labels[k], ha='center', va='bottom',
                fontsize=10, color='#555555')
        ax.text(-0.15, 2 - k + 0.5, labels[k], ha='right', va='center',
                fontsize=10, color='#555555')

    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(-0.8, 3.6)

    ax.set_title(title_str, fontsize=12, fontweight='bold', pad=8)

    if subtitle:
        ax.text(1.5, -0.4, subtitle, ha='center', va='top',
                fontsize=10, color='#555555', style='italic')


# ─── Animation ────────────────────────────────────────────────────────────────
FRAMES_PER_SYSTEM = 12
INTRO_FRAMES = 18
HOLD_FRAMES = 25
TOTAL_SYSTEMS = 12
TOTAL_FRAMES = INTRO_FRAMES + TOTAL_SYSTEMS * FRAMES_PER_SYSTEM + HOLD_FRAMES

CENTER = np.array([0.5, 0.5, 0.5])
VEC_SCALE = 0.35


def update(frame):
    clear_dynamic()

    azim = -50 + frame * 0.5
    ax3d.view_init(elev=20, azim=azim)

    # ═══════════════════════════════════════════════════════════════════════════
    # INTRO: Show equation and empty state
    # ═══════════════════════════════════════════════════════════════════════════
    if frame < INTRO_FRAMES:
        title_3d.set_text('FCC Unit Cell')

        # Show equation in tensor panel
        ax_tensor.clear()
        ax_tensor.set_axis_off()
        ax_tensor.set_xlim(0, 1)
        ax_tensor.set_ylim(0, 1)

        eq1 = ax_tensor.text(0.5, 0.7,
            r'$\dot{\epsilon}^p = \sum_{i=1}^{12} \dot{\gamma}_i \; \mathbf{P}_i$',
            ha='center', va='center', fontsize=18, color='#333333',
            bbox=dict(facecolor='#E3F2FD', edgecolor='#1565C0',
                      alpha=0.9, boxstyle='round,pad=0.5'))
        dynamic.append(eq1)

        eq2 = ax_tensor.text(0.5, 0.35,
            '$\\mathbf{P}_i = \\frac{1}{2}'
            '(\\mathbf{s}_i \\otimes \\mathbf{m}_i + '
            '\\mathbf{m}_i \\otimes \\mathbf{s}_i)$',
            ha='center', va='center', fontsize=15, color='#555555')
        dynamic.append(eq2)

        eq3 = ax_tensor.text(0.5, 0.10,
            'Symmetric Schmid Tensor',
            ha='center', va='center', fontsize=12, color='#888888',
            style='italic')
        dynamic.append(eq3)

        # Empty bars
        bars = ax_bar.bar(range(1, 13), [0]*12, color='#EEEEEE', edgecolor='#CCCCCC')
        for b in bars:
            dynamic.append(b)
        return []

    # ═══════════════════════════════════════════════════════════════════════════
    # ACCUMULATION: Add one system at a time
    # ═══════════════════════════════════════════════════════════════════════════
    adj_frame = frame - INTRO_FRAMES

    if adj_frame < TOTAL_SYSTEMS * FRAMES_PER_SYSTEM:
        sys_idx = adj_frame // FRAMES_PER_SYSTEM
        local_f = adj_frame % FRAMES_PER_SYSTEM
    else:
        # Hold phase
        sys_idx = 11
        local_f = FRAMES_PER_SYSTEM - 1

    plane_family = sys_idx // 3
    color = PLANE_COLORS[plane_family]
    color_light = PLANE_COLORS_LIGHT[plane_family]

    s_vec = S_ALL[sys_idx]
    m_vec = M_ALL[sys_idx]
    gdot_i = GAMMA_DOT[sys_idx]

    title_3d.set_text(f'Adding System {sys_idx + 1} / 12')

    # ── 3D: Draw previously accumulated systems (faded) ──
    for j in range(sys_idx):
        fam_j = j // 3
        s_j, m_j = S_ALL[j], M_ALL[j]
        gdot_j = GAMMA_DOT[j]
        # Arrow along slip direction, scaled by gamma_dot magnitude
        arr_scale = (abs(gdot_j) / GAMMA_MAX) * VEC_SCALE
        c_prev = PLANE_COLORS[fam_j]
        q = ax3d.quiver(*CENTER, *(s_j * arr_scale * np.sign(gdot_j)),
                        color=c_prev, alpha=0.25, arrow_length_ratio=0.2, linewidth=1.2)
        dynamic.append(q)

    # ── 3D: Current system (prominent) ──
    # Phase in: plane, then vectors
    if local_f >= 0:
        alpha_p = min(1.0, local_f / 2.0) * 0.25
        tri = plane_triangle(PLANES_MILLER[sys_idx], scale=0.42, center=CENTER)
        poly = Poly3DCollection([tri], alpha=alpha_p, facecolor=color,
                                edgecolor=color, linewidth=1.5)
        ax3d.add_collection3d(poly)
        dynamic.append(poly)

    if local_f >= 3:
        alpha_v = min(1.0, (local_f - 3) / 2.0)
        # m_i
        q = ax3d.quiver(*CENTER, *(m_vec * VEC_SCALE * 0.7),
                        color='#FF5252', alpha=alpha_v * 0.7,
                        arrow_length_ratio=0.2, linewidth=1.8)
        dynamic.append(q)
        # s_i
        arr_scale = (abs(gdot_i) / GAMMA_MAX) * VEC_SCALE
        q = ax3d.quiver(*CENTER, *(s_vec * arr_scale * np.sign(gdot_i)),
                        color=color, alpha=alpha_v,
                        linewidth=3, arrow_length_ratio=0.2)
        dynamic.append(q)

        t = ax3d.text(*(CENTER + s_vec * (arr_scale * np.sign(gdot_i)) + m_vec * 0.05),
                      f'$\\dot{{\\gamma}}_{{{sys_idx+1}}}\\mathbf{{P}}_{{{sys_idx+1}}}$',
                      color=color, fontsize=9, alpha=alpha_v,
                      fontweight='bold', path_effects=TEXT_FX)
        dynamic.append(t)

    # ── Tensor heatmap ──
    if local_f < 6:
        # Show P_i alone
        draw_tensor_heatmap(ax_tensor, P_ALL[sys_idx],
                            f'$\\mathbf{{P}}_{{{sys_idx+1}}}$ (Schmid Tensor)',
                            f'$\\dot{{\\gamma}}_{{{sys_idx+1}}} = {gdot_i:.1e}$ s$^{{-1}}$')
    else:
        # Show cumulative eps_dot^p
        draw_tensor_heatmap(ax_tensor, EPS_DOT_CUMULATIVE[sys_idx],
                            f'$\dot{{\epsilon}}^p$ after {sys_idx+1} systems',
                            'Superposition of all contributions so far')

    # ── Bar chart ──
    bar_vals = np.zeros(12)
    bar_colors_list = ['#EEEEEE'] * 12
    bar_edge = ['#CCCCCC'] * 12
    for j in range(sys_idx + 1):
        fam_j = j // 3
        bar_vals[j] = GAMMA_DOT[j] / 1e5
        if j == sys_idx:
            bar_colors_list[j] = PLANE_COLORS[fam_j]
            bar_edge[j] = '#000000'
        else:
            bar_colors_list[j] = PLANE_COLORS_LIGHT[fam_j]
            bar_edge[j] = PLANE_COLORS[fam_j]

    bars = ax_bar.bar(range(1, 13), bar_vals,
                      color=bar_colors_list, edgecolor=bar_edge, linewidth=1.2)
    for b in bars:
        dynamic.append(b)

    # ═══════════════════════════════════════════════════════════════════════════
    # HOLD: Final summary
    # ═══════════════════════════════════════════════════════════════════════════
    if adj_frame >= TOTAL_SYSTEMS * FRAMES_PER_SYSTEM:
        title_3d.set_text('All 12 Systems Superposed')

        # Summary box on tensor panel
        summary = ax_tensor.text(1.5, -0.6,
            '• No single flow direction\n'
            '• Captures anisotropy\n'
            '• Multiple simultaneous modes',
            ha='center', va='top', fontsize=10, color='#333333',
            bbox=dict(facecolor='#E8F5E9', edgecolor='#4CAF50',
                      alpha=0.9, boxstyle='round,pad=0.4'))
        dynamic.append(summary)

    return []


anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=140, blit=False)

plt.tight_layout()
out_path = 'gif_superposition.gif'
print(f"Rendering {TOTAL_FRAMES} frames …")
anim.save(out_path, writer='pillow', fps=10, dpi=120,
          savefig_kwargs={'facecolor': 'white'})
print(f"Saved  →  {out_path}")
plt.close()
