import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import fcc_utils

fig, ax = fcc_utils.setup_plot()
ax.set_title("FCC Slip Geometry: (111) Plane", color='black', fontsize=15)

nodes = fcc_utils.get_fcc_nodes()
# Filter atoms on (111) plane: x + y + z = 1 (approx tolerance)
plane_nodes = []
other_nodes = []

for p in nodes:
    if abs(np.sum(p) - 1.0) < 1e-5 or abs(np.sum(p) - 2.0) < 1e-5: # Planes at sum=1 and sum=2
        # Focusing on sum=1 main plane
        if abs(np.sum(p) - 1.0) < 1e-5:
            plane_nodes.append(p)
        else:
            other_nodes.append(p)
    else:
        other_nodes.append(p)

plane_nodes = np.array(plane_nodes)
other_nodes = np.array(other_nodes)

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_zlim(-0.1, 1.1)

# Plot "other" atoms
others_scat = ax.scatter(other_nodes[:,0], other_nodes[:,1], other_nodes[:,2], 
                         s=100, c='gray', alpha=0.3, label='Bulk')

# Plot "plane" atoms
plane_scat = ax.scatter(plane_nodes[:,0], plane_nodes[:,1], plane_nodes[:,2], 
                        s=200, c='cyan', alpha=1.0, label='(111) Plane')

# Draw plane surface
# Corners (1,0,0), (0,1,0), (0,0,1) form the bound, but (0.5,0.5,0) etc are inside
# Visual triangle
tri = np.array([[1,0,0], [0,1,0], [0,0,1]])
poly = None

# Arrow
quiver = None

# Text
desc = ax.text2D(0.5, 0.05, "", transform=ax.transAxes, ha='center', color='blue', fontsize=12)

def init():
    return []

def update(frame):
    # Phase 1: Rotate
    ax.view_init(elev=30, azim=45 + frame)
    
    label = "FCC Lattice"
    
    if frame > 20:
        # Show Plane
        if frame == 21:
            global poly
            # Draw semi-transparent triangle
            tri_x = [1, 0, 0]
            tri_y = [0, 1, 0]
            tri_z = [0, 0, 1]
            poly = ax.plot_trisurf(tri_x, tri_y, tri_z, color='cyan', alpha=0.2)
        label = "Slip Plane (111)"
        
    if frame > 40:
        label = "Slip Direction <110>"
        # Show Arrow
        # Vector [1, -1, 0] on the plane.
        # Start at (0, 0.5, 0.5) -> End at (0.5, 0, 0.5) ?
        # (0, 0.5, 0.5) + (1, -1, 0)*0.5 = (0.5, 0, 0.5). Yes.
        # Draw arrow
        if frame == 41:
            ax.quiver(0, 0.5, 0.5, 1, -1, 0, length=0.5, color='red', linewidth=3, arrow_length_ratio=0.3)
            
    desc.set_text(label)
    return [plane_scat, others_scat, desc]

anim = FuncAnimation(fig, update, frames=60, interval=100, blit=False)
anim.save('gif2_geometry.gif', writer='pillow', fps=15)
print("Saved gif2_geometry.gif")
