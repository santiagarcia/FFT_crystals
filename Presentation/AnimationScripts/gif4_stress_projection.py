import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import fcc_utils

# Generate 12 systems properly
def get_12_systems():
    # Planes
    planes = [
        np.array([1, 1, 1]), np.array([-1, 1, 1]),
        np.array([1, -1, 1]), np.array([1, 1, -1])
    ]
    systems = []
    for p in planes:
        p = p / np.linalg.norm(p)
        # Find s perp to p
        # Just hardcode for simplicity/correctness
        # (111) -> [0, 1, -1], [1, 0, -1], [1, -1, 0]
        if np.isclose(abs(p[0]), abs(p[1])) and np.isclose(abs(p[1]), abs(p[2])):
            # It is a {111} type
            # Directions are <110> type perps
            cands = []
            if np.isclose(p[0]*1 + p[1]*(-1) + p[2]*0, 0): cands.append(np.array([1, -1, 0]))
            if np.isclose(p[0]*1 + p[1]*0 + p[2]*(-1), 0): cands.append(np.array([1, 0, -1]))
            if np.isclose(p[0]*0 + p[1]*1 + p[2]*(-1), 0): cands.append(np.array([0, 1, -1]))
            # Also negative? Schmid factor is abs? Or signed? Signed.
            for c in cands:
                c = c / np.linalg.norm(c)
                systems.append((p, c))
    # We might get more or less depending on logic, but let's stick to first 12 found
    # actually logic above finds exactly 3 per plane.
    return systems[:12]

systems = get_12_systems()
print(f"Systems found: {len(systems)}")

# Setup
plt.style.use('default')
fig = plt.figure(figsize=(10, 5))
ax3d = fig.add_subplot(121, projection='3d')
axbar = fig.add_subplot(122)

# 3D: Cube + Arrow
fcc_utils.setup_plot() # returns separate fig, ignore
corners = fcc_utils.get_fcc_edges()
for start, end in corners:
    ax3d.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='gray', alpha=0.3)
ax3d.set_xlim(-0.2, 1.2)
ax3d.set_ylim(-0.2, 1.2)
ax3d.set_zlim(-0.2, 1.2)
ax3d.set_axis_off() 
ax3d.set_title("Applied Stress Direction", color='black')

arrow_q = ax3d.quiver(0.5, 0.5, 0.5, 0, 0, 1, color='orange', length=0.8, linewidth=4)

# Bar chart
bars = axbar.bar(range(12), [0]*12, color='cyan')
axbar.set_ylim(-0.5, 0.5)
axbar.set_title(r"Resolved Shear Stress $\tau^\alpha$", color='black')
axbar.set_xlabel("Slip System ID")

def update(frame):
    global arrow_q
    
    # Rotate vector
    angle = frame / 40.0 * np.pi 
    # v rotates in X-Z plane
    vx = 0.5 * np.sin(angle)
    vz = 0.5 * np.cos(angle)
    vy = 0.2 * np.sin(angle*2)
    
    v = np.array([vx, vy, vz])
    v_norm = v / np.linalg.norm(v)
    
    # Update Arrow
    if arrow_q: arrow_q.remove()
    arrow_q = ax3d.quiver(0.5, 0.5, 0.5, v[0], v[1], v[2], color='orange', length=0.6, linewidth=4)
    
    # Update Bars
    # tau = (v . s) * (v . n)
    taus = []
    for (n, s) in systems:
        schmid = np.dot(v_norm, s) * np.dot(v_norm, n)
        taus.append(schmid)
        
    for bar, h in zip(bars, taus):
        bar.set_height(h)
        # Color based on magnitude
        bar.set_color(plt.cm.coolwarm(h + 0.5)) # Map -0.5..0.5 to 0..1
        
    return list(bars) + [arrow_q]

anim = FuncAnimation(fig, update, frames=40, interval=100, blit=False)
anim.save('gif4_schmid.gif', writer='pillow', fps=10)
print("Saved gif4_schmid.gif")
