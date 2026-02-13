import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import fcc_utils

# Setup
fig, ax = fcc_utils.setup_plot()
ax.set_title("Continuum Shear from Multi-Slip", color='black', fontsize=14)

# 1. Define Particles (Atoms) to show bulk flow
# Create a grid of points inside the unit cube
grid_dim = 4
u = np.linspace(0, 1, grid_dim)
v = np.linspace(0, 1, grid_dim)
w = np.linspace(0, 1, grid_dim)
U, V, W = np.meshgrid(u, v, w)
atoms_ref = np.vstack([U.flatten(), V.flatten(), W.flatten()]).T - 0.5 # Centered at origin

# 2. Define Cube Wireframe
cube_nodes_ref = fcc_utils.get_fcc_edges(a=1) - 0.5
# Convert edges to unique lines for animation efficiency
# fcc_utils.get_fcc_edges returns list of pairs of points.
# We'll just transform them on the fly.

# 3. Define Slip Systems (Schmid Tensors)
# System 1: (111) plane, [1 0 -1] direction
n1 = np.array([1, 1, 1]); n1 = n1/np.linalg.norm(n1)
s1 = np.array([1, 0, -1]); s1 = s1/np.linalg.norm(s1)
P1 = np.outer(s1, n1) # Schmide tensor s x n

# System 2: (1 1 -1) plane, [0 1 1] direction
n2 = np.array([1, 1, -1]); n2 = n2/np.linalg.norm(n2)
s2 = np.array([0, 1, 1]); s2 = s2/np.linalg.norm(s2)
P2 = np.outer(s2, n2)

# Graphics Objects
atom_scat = ax.scatter([], [], [], s=20, c='blue', alpha=0.6, label='Lattice Points')
lines = []
for _ in range(12): # 12 edges in a cube
    l, = ax.plot([], [], [], color='black', lw=1.5)
    lines.append(l)

# Slip Plane Traces (visual cues of slip planes)
# Just some static lines that will rotate with the cube?
# Or dynamic planes? Let's use simple arrows for the active systems.
q1 = None
q2 = None

txt = ax.text2D(0.5, 0.02, "", transform=ax.transAxes, ha='center', color='blue', fontsize=12)

# Axis limits - Generous to prevent clipping
limit = 1.0
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_zlim(-limit, limit)

# Animation State
total_F = np.eye(3)

def update(frame):
    global total_F, q1, q2
    
    # Time logic
    # 0-20: System 1 acts
    # 20-40: System 2 acts
    # 40-60: Both act
    
    d_gamma = 0.02 # Incremental shear
    
    active_1 = False
    active_2 = False
    
    if frame < 20:
        # Sys 1
        # F_new = (I + dgamma P1) F_old
        inc = np.eye(3) + d_gamma * P1
        total_F = inc @ total_F
        txt.set_text("System 1 Sliding (Red)")
        active_1 = True
    elif frame < 40:
        # Sys 2
        inc = np.eye(3) + d_gamma * P2
        total_F = inc @ total_F
        txt.set_text("System 2 Sliding (Green)")
        active_2 = True
    else:
        # Both
        inc = np.eye(3) + d_gamma * (P1 + P2)
        total_F = inc @ total_F
        txt.set_text("Combined Multi-Slip = Macroscopic Shear")
        active_1 = True
        active_2 = True
        
    # Apply F to Atoms
    # x_current = F * x_ref
    atoms_curr = (total_F @ atoms_ref.T).T
    atom_scat._offsets3d = (atoms_curr[:,0], atoms_curr[:,1], atoms_curr[:,2])
    
    # Apply F to Cube Edges
    for l, (start, end) in zip(lines, cube_nodes_ref):
        # start, end are reference coordinates
        # Deform them
        p1 = total_F @ start
        p2 = total_F @ end
        l.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        l.set_3d_properties([p1[2], p2[2]])
        
    # Update Vectors (Slip Directions)
    if q1 is not None: 
        q1.remove()
        q1 = None
    if q2 is not None: 
        q2.remove()
        q2 = None
    
    # Show arrows if active
    if active_1:
        # Rotate the vector s1 by current F?
        # Actually in single crystal plasticity, slip directions convect with lattice rotation.
        # Simple viz: just show s vector.
        v1 = s1 * 0.8
        q1 = ax.quiver(0,0,0, v1[0], v1[1], v1[2], color='red', length=0.8, lw=3)
        
    if active_2:
        v2 = s2 * 0.8
        q2 = ax.quiver(0,0,0, v2[0], v2[1], v2[2], color='green', length=0.8, lw=3)
        
    return lines + [atom_scat, txt]

anim = FuncAnimation(fig, update, frames=60, interval=100, blit=False)
anim.save('gif9_multislip.gif', writer='pillow', fps=15)
print("Saved gif9_multislip.gif")
