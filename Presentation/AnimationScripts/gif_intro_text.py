"""
GIF – FCC crystal undergoing slip-system–level plastic deformation.
Shows a unit cell that:
  1. Sits at rest with slip plane highlighted
  2. Dislocation line sweeps across the slip plane
  3. Upper half shears along the slip direction
  4. Resets and repeats on a second slip system
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import fcc_utils

# ── figure ───────────────────────────────────────────────────────────
fig, ax = fcc_utils.setup_plot()
ax.set_title("Dislocation Glide on Slip Planes", color="black", fontsize=14,
             fontweight="bold", pad=10)
ax.view_init(elev=22, azim=-55)

# ── unit-cell geometry (centered at origin) ──────────────────────────
corners0 = np.array([
    [0,0,0], [1,0,0], [1,1,0], [0,1,0],   # bottom face
    [0,0,1], [1,0,1], [1,1,1], [0,1,1],    # top face
], dtype=float) - 0.5

edge_idx = [
    (0,1),(1,2),(2,3),(3,0),   # bottom
    (4,5),(5,6),(6,7),(7,4),   # top
    (0,4),(1,5),(2,6),(3,7),   # verticals
]

# Face-centered atoms
face_atoms0 = np.array([
    [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5],
    [0.5,0.5,1], [0.5,1,0.5], [1,0.5,0.5],
]) - 0.5

# ── slip system data ────────────────────────────────────────────────
nA = np.array([1,1,1], dtype=float); nA /= np.linalg.norm(nA)
sA = np.array([1,-1,0], dtype=float); sA /= np.linalg.norm(sA)

nB = np.array([1,-1,1], dtype=float); nB /= np.linalg.norm(nB)
sB = np.array([1,1,0], dtype=float);  sB /= np.linalg.norm(sB)

# ── slip-plane quad ─────────────────────────────────────────────────
def plane_quad(normal):
    """Vertices where the {111} plane through the origin intersects the cube."""
    c = corners0
    cube_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            diff = c[j] - c[i]
            if np.sum(np.abs(diff) > 0.01) == 1:
                cube_edges.append((i,j))
    verts = []
    for i,j in cube_edges:
        d_i = np.dot(normal, c[i])
        d_j = np.dot(normal, c[j])
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

planeA_verts = plane_quad(nA)
planeB_verts = plane_quad(nB)

# ── drawing objects ──────────────────────────────────────────────────
edge_lines = []
for _ in edge_idx:
    l, = ax.plot([], [], [], color="#333333", lw=1.5, zorder=2)
    edge_lines.append(l)

atom_sc = ax.scatter([], [], [], s=90, c="#1565C0", edgecolors="black",
                     linewidths=0.5, zorder=5, depthshade=True)
face_sc = ax.scatter([], [], [], s=60, c="#42A5F5", edgecolors="black",
                     linewidths=0.4, zorder=5, depthshade=True)

plane_poly = Poly3DCollection([], alpha=0.30, facecolor="#E65100",
                               edgecolor="#E65100", linewidths=1.5, zorder=1)
ax.add_collection3d(plane_poly)

disl_line, = ax.plot([], [], [], color="#D32F2F", lw=3.5, zorder=10)

slip_arrow = ax.quiver(0,0,0, 0,0,0, color="#2E7D32", linewidth=2.5,
                       arrow_length_ratio=0.25, zorder=8)

ax.set_xlim(-0.85, 0.85)
ax.set_ylim(-0.85, 0.85)
ax.set_zlim(-0.85, 0.85)

label = ax.text2D(0.50, 0.03, "", transform=ax.transAxes, ha="center",
                  fontsize=12, color="#333333", fontweight="bold")
sublabel = ax.text2D(0.50, -0.02, "", transform=ax.transAxes, ha="center",
                     fontsize=10, color="#E65100")

# ── timing ───────────────────────────────────────────────────────────
FPS = 15
REST, SWEEP, HOLD = 10, 25, 15
FRAMES_PER_SYS = REST + SWEEP + HOLD   # 50
TOTAL = FRAMES_PER_SYS * 2
MAX_SHEAR = 0.28

def shear_pts(pts, normal, slip_dir, gamma):
    """Shear points above the slip plane by gamma along slip_dir."""
    out = pts.copy()
    above = (out @ normal) > 0.01
    out[above] += gamma * slip_dir
    return out

def disl_endpoints(pv, normal, slip_dir, progress):
    """Dislocation line segment at fractional progress across the plane."""
    ctr = pv.mean(axis=0)
    line_dir = np.cross(normal, slip_dir)
    line_dir /= (np.linalg.norm(line_dir) + 1e-15)
    sweep = slip_dir - np.dot(slip_dir, normal) * normal
    sweep /= (np.linalg.norm(sweep) + 1e-15)
    projs = (pv - ctr) @ sweep
    pos = projs.min() + progress * (projs.max() - projs.min())
    mid = ctr + pos * sweep
    ext = max(abs(((pv - ctr) @ line_dir).min()),
              abs(((pv - ctr) @ line_dir).max())) * 1.05
    return mid - ext * line_dir, mid + ext * line_dir


def update(frame):
    global slip_arrow
    sys_idx = frame // FRAMES_PER_SYS
    f = frame % FRAMES_PER_SYS

    if sys_idx == 0:
        normal, slip_dir, pv = nA, sA, planeA_verts
        sys_name = r"System 1:  (111)[$\overline{1}$10]"
    else:
        normal, slip_dir, pv = nB, sB, planeB_verts
        sys_name = r"System 2:  (1$\overline{1}$1)[110]"

    if f < REST:
        gamma, dp = 0.0, -1
        phase = "slip plane highlighted"
    elif f < REST + SWEEP:
        t = (f - REST) / SWEEP
        gamma, dp = MAX_SHEAR * t, t
        phase = "dislocation gliding  →"
    else:
        gamma, dp = MAX_SHEAR, -1
        phase = "sheared — permanent offset"

    c_def  = shear_pts(corners0, normal, slip_dir, gamma)
    a_def  = shear_pts(face_atoms0, normal, slip_dir, gamma)
    pv_def = shear_pts(pv, normal, slip_dir, gamma)

    for line, (i,j) in zip(edge_lines, edge_idx):
        p1, p2 = c_def[i], c_def[j]
        line.set_data_3d([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]])

    atom_sc._offsets3d = (c_def[:,0], c_def[:,1], c_def[:,2])
    face_sc._offsets3d = (a_def[:,0], a_def[:,1], a_def[:,2])
    plane_poly.set_verts([pv_def])

    if 0 <= dp <= 1:
        p1, p2 = disl_endpoints(pv, normal, slip_dir, dp)
        p1 = shear_pts(p1.reshape(1,3), normal, slip_dir, gamma)[0]
        p2 = shear_pts(p2.reshape(1,3), normal, slip_dir, gamma)[0]
        disl_line.set_data_3d([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]])
        disl_line.set_visible(True)
    else:
        disl_line.set_visible(False)

    slip_arrow.remove()
    origin = pv.mean(axis=0) + gamma * 0.5 * slip_dir
    vec = slip_dir * 0.35
    slip_arrow = ax.quiver(origin[0], origin[1], origin[2],
                           vec[0], vec[1], vec[2],
                           color="#2E7D32", linewidth=2, arrow_length_ratio=0.3,
                           zorder=8)

    label.set_text(sys_name)
    sublabel.set_text(phase)
    return []

anim = FuncAnimation(fig, update, frames=TOTAL,
                     interval=1000//FPS, blit=False)
anim.save("gif_intro_text.gif", writer="pillow", fps=FPS)
print(f"Saved gif_intro_text.gif  ({TOTAL} frames, {TOTAL/FPS:.1f}s)")
