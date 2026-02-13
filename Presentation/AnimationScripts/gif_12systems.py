"""
GIF – FCC 12 slip systems: 4 {111} planes × 3 ⟨110⟩ directions.
Progressively highlights each plane and its three slip directions,
building up to the full set of 12 systems.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import fcc_utils

# ── figure ───────────────────────────────────────────────────────────
fig, ax = fcc_utils.setup_plot()
ax.set_title("12 Slip Systems in FCC", color="black", fontsize=14,
             fontweight="bold", pad=10)
ax.view_init(elev=20, azim=-50)

# ── unit cell (centered) ────────────────────────────────────────────
corners = np.array([
    [0,0,0],[1,0,0],[1,1,0],[0,1,0],
    [0,0,1],[1,0,1],[1,1,1],[0,1,1],
], dtype=float) - 0.5

edge_idx = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

face_atoms = np.array([
    [.5,.5,0],[.5,0,.5],[0,.5,.5],
    [.5,.5,1],[.5,1,.5],[1,.5,.5],
]) - 0.5

# ── the 4 planes and 3 directions each ──────────────────────────────
plane_colors = ["#E65100", "#1565C0", "#2E7D32", "#6A1B9A"]  # orange, blue, green, purple
dir_color    = "#D32F2F"

planes_raw = [
    [1, 1, 1],
    [-1, 1, 1],
    [1,-1, 1],
    [1, 1,-1],
]

def find_slip_dirs(n):
    """Find 3 independent ⟨110⟩ directions perpendicular to n."""
    candidates = []
    for a in [-1,0,1]:
        for b in [-1,0,1]:
            for c in [-1,0,1]:
                v = np.array([a,b,c], dtype=float)
                nonzero = np.sum(np.abs(v) > 0)
                if nonzero != 2:
                    continue
                if abs(np.dot(v, n)) > 1e-9:
                    continue
                # Check not a duplicate or negative
                v_norm = v / np.linalg.norm(v)
                is_dup = False
                for existing in candidates:
                    if np.allclose(existing, v_norm) or np.allclose(existing, -v_norm):
                        is_dup = True
                        break
                if not is_dup:
                    candidates.append(v_norm)
    return candidates[:3]

systems = []  # list of (plane_normal, [dir1, dir2, dir3], plane_color)
for i, p in enumerate(planes_raw):
    n = np.array(p, dtype=float)
    n /= np.linalg.norm(n)
    dirs = find_slip_dirs(n)
    systems.append((n, dirs, plane_colors[i]))

# ── plane–cube intersection ─────────────────────────────────────────
def plane_quad(normal):
    c = corners
    cube_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            diff = c[j] - c[i]
            if np.sum(np.abs(diff) > 0.01) == 1:
                cube_edges.append((i,j))
    verts = []
    for i,j in cube_edges:
        d_i, d_j = np.dot(normal, c[i]), np.dot(normal, c[j])
        if d_i * d_j < 0 or abs(d_i) < 1e-9 or abs(d_j) < 1e-9:
            if abs(d_j - d_i) < 1e-12:
                continue
            t = -d_i / (d_j - d_i)
            if -0.01 <= t <= 1.01:
                pt = c[i] + t * (c[j] - c[i])
                if not any(np.linalg.norm(pt - v) < 1e-6 for v in verts):
                    verts.append(pt)
    for i in range(8):
        if abs(np.dot(normal, c[i])) < 1e-9:
            if not any(np.linalg.norm(c[i] - v) < 1e-6 for v in verts):
                verts.append(c[i].copy())
    verts = np.array(verts)
    if len(verts) >= 3:
        ctr = verts.mean(axis=0)
        rel = verts - ctr
        u1 = rel[0] / (np.linalg.norm(rel[0]) + 1e-15)
        u2 = np.cross(normal, u1); u2 /= (np.linalg.norm(u2) + 1e-15)
        angles = np.arctan2(rel @ u2, rel @ u1)
        verts = verts[np.argsort(angles)]
    return verts

plane_verts = [plane_quad(n) for n, _, _ in systems]

# ── static cell drawing ─────────────────────────────────────────────
for i,j in edge_idx:
    p1, p2 = corners[i], corners[j]
    ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]],
            color="#aaaaaa", lw=1.0, zorder=1)

ax.scatter(corners[:,0], corners[:,1], corners[:,2],
           s=50, c="#1565C0", edgecolors="black", linewidths=0.4,
           zorder=3, depthshade=True)
ax.scatter(face_atoms[:,0], face_atoms[:,1], face_atoms[:,2],
           s=35, c="#42A5F5", edgecolors="black", linewidths=0.3,
           zorder=3, depthshade=True)

ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.8, 0.8)
ax.set_zlim(-0.8, 0.8)

# ── dynamic elements ────────────────────────────────────────────────
plane_poly = Poly3DCollection([], alpha=0.0, zorder=2)
ax.add_collection3d(plane_poly)

arrow_artists = []

# Labels
plane_label = ax.text2D(0.02, 0.95, "", transform=ax.transAxes,
                        fontsize=13, fontweight="bold", color="black")
dir_label = ax.text2D(0.02, 0.90, "", transform=ax.transAxes,
                      fontsize=11, color="#D32F2F")
counter_label = ax.text2D(0.98, 0.03, "", transform=ax.transAxes,
                          fontsize=12, ha="right", color="#333333",
                          fontweight="bold")
formula_label = ax.text2D(0.50, -0.04, "", transform=ax.transAxes,
                          fontsize=11, ha="center", color="#888888")

# ── timing ───────────────────────────────────────────────────────────
FPS = 12
# Per plane: fade-in plane (6) + 3 dirs × (show 5 + hold 3) + hold (10) = 40
PLANE_FADE = 6
DIR_SHOW   = 5
DIR_HOLD   = 3
DIR_BLOCK  = DIR_SHOW + DIR_HOLD   # 8
PLANE_HOLD = 10
FRAMES_PER_PLANE = PLANE_FADE + 3 * DIR_BLOCK + PLANE_HOLD  # 40
# After all 4 planes: summary hold
SUMMARY_HOLD = 30
TOTAL = 4 * FRAMES_PER_PLANE + SUMMARY_HOLD

def fmt_miller(v, bracket="()"):
    """Format a Miller index vector."""
    iv = np.round(v * np.linalg.norm(v) / np.min(np.abs(v[np.abs(v)>0.01]))).astype(int)
    chars = []
    for x in iv:
        if x < 0:
            chars.append(f"$\\overline{{{abs(x)}}}$")
        else:
            chars.append(str(x))
    return f"{bracket[0]}{''.join(chars)}{bracket[1]}"

def clear_arrows():
    global arrow_artists
    for a in arrow_artists:
        try:
            a.remove()
        except Exception:
            pass
    arrow_artists = []

def update(frame):
    global arrow_artists

    # How many complete planes before this frame
    completed_planes = min(frame // FRAMES_PER_PLANE, 4)
    running_count = completed_planes * 3

    # ── summary phase ───
    if frame >= 4 * FRAMES_PER_PLANE:
        plane_poly.set_alpha(0.0)
        clear_arrows()
        plane_label.set_text("")
        dir_label.set_text("")
        counter_label.set_text("12 / 12 systems")
        formula_label.set_text(
            "4 planes × 3 directions = 12 slip systems")
        # Show all planes at once
        all_polys = []
        for pi in range(4):
            pv = plane_verts[pi]
            all_polys.append(pv)
        plane_poly.set_verts(all_polys)
        plane_poly.set_facecolor([pc + "44" for _, _, pc in systems])
        plane_poly.set_edgecolor([pc for _, _, pc in systems])
        plane_poly.set_alpha(0.25)
        # draw all 12 arrows
        clear_arrows()
        for pi, (n, dirs, pc) in enumerate(systems):
            ctr = plane_verts[pi].mean(axis=0)
            for d in dirs:
                a = ax.quiver(ctr[0], ctr[1], ctr[2],
                              d[0]*0.3, d[1]*0.3, d[2]*0.3,
                              color=pc, linewidth=1.5,
                              arrow_length_ratio=0.25, zorder=8)
                arrow_artists.append(a)
        return []

    # ── per-plane phase ───
    pi = frame // FRAMES_PER_PLANE
    f  = frame % FRAMES_PER_PLANE
    n, dirs, pc = systems[pi]
    pv = plane_verts[pi]
    miller_plane = fmt_miller(np.array(planes_raw[pi]), "{}")

    formula_label.set_text("")

    if f < PLANE_FADE:
        # Fade in the plane
        alpha = 0.30 * (f + 1) / PLANE_FADE
        plane_poly.set_verts([pv])
        plane_poly.set_facecolor(pc + "55")
        plane_poly.set_edgecolor(pc)
        plane_poly.set_alpha(alpha)
        clear_arrows()
        plane_label.set_text(f"Plane {pi+1}/4:  {miller_plane}")
        plane_label.set_color(pc)
        dir_label.set_text("")
        counter_label.set_text(f"{running_count} / 12 systems")
    else:
        plane_poly.set_verts([pv])
        plane_poly.set_facecolor(pc + "55")
        plane_poly.set_edgecolor(pc)
        plane_poly.set_alpha(0.30)
        plane_label.set_text(f"Plane {pi+1}/4:  {miller_plane}")
        plane_label.set_color(pc)

        dir_f = f - PLANE_FADE
        if dir_f < 3 * DIR_BLOCK:
            di = dir_f // DIR_BLOCK
            df = dir_f % DIR_BLOCK
            d = dirs[di]
            miller_dir = fmt_miller(d, "[]")

            if df == 0:
                # new direction: clear old arrows for this plane, redraw up to this one
                clear_arrows()
                ctr = pv.mean(axis=0)
                for k in range(di + 1):
                    a = ax.quiver(ctr[0], ctr[1], ctr[2],
                                  dirs[k][0]*0.35, dirs[k][1]*0.35, dirs[k][2]*0.35,
                                  color="#D32F2F", linewidth=2.5,
                                  arrow_length_ratio=0.25, zorder=8)
                    arrow_artists.append(a)

            dir_label.set_text(f"  Direction {di+1}/3:  {miller_dir}")
            cur = running_count + di + 1
            counter_label.set_text(f"{cur} / 12 systems")
        else:
            # hold at end of plane
            counter_label.set_text(f"{running_count + 3} / 12 systems")
            dir_label.set_text("")

    return []


anim = FuncAnimation(fig, update, frames=TOTAL,
                     interval=1000//FPS, blit=False)
anim.save("gif_12systems.gif", writer="pillow", fps=FPS)
print(f"Saved gif_12systems.gif  ({TOTAL} frames, {TOTAL/FPS:.1f}s)")
