import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import fcc_utils

fig, ax = fcc_utils.setup_plot()
ax.set_title("12 Active Slip Systems", color='black', fontsize=15)

# Static edges
edges = fcc_utils.get_fcc_edges()
for start, end in edges:
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='gray', alpha=0.3)
    
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_zlim(-0.1, 1.1)

# Definition of 4 planes (triangles)
planes_geom = [
    # Normal (1,1,1)
    [[1,0,0], [0,1,0], [0,0,1]],
    # Normal (1,1,-1) -> fits (0,0,0)-(1,0,1)-(0,1,1) ? Normal cross ([1,0,1],[0,1,1]) = (-1, -1, 1). Correct.
    [[1,0,1], [0,1,1], [0,0,0]],
    # Normal (1,-1,1)
    [[1,1,0], [0,0,0], [0,1,1]],
    # Normal (-1,1,1)
    [[0,0,0], [1,1,0], [1,0,1]]
]
plane_colors = ['cyan', 'magenta', 'yellow', 'lime']

# Directions for each plane (start_point -> end_point)
# Just visual proxies
# Plane 1 (1 1 1): (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5).
# arrows connect midpoints of edges of the triangle.
arrows_db = []
for i, tri in enumerate(planes_geom):
    p0, p1, p2 = np.array(tri[0]), np.array(tri[1]), np.array(tri[2])
    # Midpoints
    m01 = (p0+p1)/2
    m12 = (p1+p2)/2
    m20 = (p2+p0)/2
    
    # 3 slip directions connect these midpoints or edges
    # For (111): (1, -1, 0) is along edge (1,0,0)-(0,1,0).
    # So visualization: Arrow ALONG the edge of the triangle.
    
    # Edge 0-1
    arrows_db.append({'p': i, 'start': p0, 'vec': (p1-p0)})
    # Edge 1-2
    arrows_db.append({'p': i, 'start': p1, 'vec': (p2-p1)})
    # Edge 2-0
    arrows_db.append({'p': i, 'start': p2, 'vec': (p0-p2)})

tracker = ax.text2D(0.5, 0.05, "", transform=ax.transAxes, ha='center', color='black', fontsize=14)
poly = None
quiver = None

def update(frame):
    global poly, quiver
    
    # Cycle systems every 5 frames
    sys_idx = (frame // 5) % 12
    plane_idx = arrows_db[sys_idx]['p']
    
    # Clear previous
    if poly: poly.remove()
    if quiver: quiver.remove()
    
    # Draw Plane
    tri = planes_geom[plane_idx]
    tri_x = [v[0] for v in tri]
    tri_y = [v[1] for v in tri]
    tri_z = [v[2] for v in tri]
    poly = ax.plot_trisurf(tri_x, tri_y, tri_z, color=plane_colors[plane_idx], alpha=0.3)
    
    # Draw Arrow
    data = arrows_db[sys_idx]
    # Center arrow?
    s = data['start'] + data['vec'] * 0.2
    v = data['vec'] * 0.6
    
    quiver = ax.quiver(s[0], s[1], s[2], v[0], v[1], v[2], color='red', linewidth=3, length=1.0)
    
    tracker.set_text(f"System {sys_idx+1}/12")
    return []

anim = FuncAnimation(fig, update, frames=60, interval=200, blit=False)
anim.save('gif3_systems.gif', writer='pillow', fps=5)
print("Saved gif3_systems.gif")
