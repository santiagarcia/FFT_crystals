import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import fcc_utils

fig, ax = fcc_utils.setup_plot()
ax.set_title("Emergent Continuum Shear", color='black')

# Nodes
nodes0 = np.array([
    [0,0,0], [1,0,0], [1,1,0], [0,1,0],
    [0,0,1], [1,0,1], [1,1,1], [0,1,1]
]) - 0.5
edges = fcc_utils.get_fcc_edges(a=1) # Need indices logic, let's reuse logic from gif1/6
edge_indices = [
   (0,1), (0,2), (0,3), (1,4), (1,5), (2,4), (2,6), (3,5), (3,6), (4,7), (5,7), (6,7)
]
# Wait indices in fcc_utils.edges are coordinates not indices.
# Let's map nodes0 to edges manually again
edge_indices = [
    (0,1), (1,2), (2,3), (3,0),
    (4,5), (5,6), (6,7), (7,4),
    (0,4), (1,5), (2,6), (3,7)
]

lines = []
for _ in edge_indices:
    l, = ax.plot([], [], [], color='cyan', lw=2)
    lines.append(l)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

# Slip lines inside (random segments)
slip_segments = []
n_slip = 20
for i in range(n_slip):
    # Random position
    p = np.random.rand(3) - 0.5
    # Random slip dir (mostly x)
    vec = np.array([1.0, 0, 0])
    l, = ax.plot([], [], [], color='black', alpha=0.5, lw=1)
    slip_segments.append({'line': l, 'pos': p, 'vec': vec, 'active': False, 'start_f': np.random.randint(5, 30)})

txt = ax.text2D(0.5, 0.05, "", transform=ax.transAxes, ha='center', color='blue', fontsize=12)

def update(frame):
    # Overall deformation
    total_shear = 0.5 * (frame / 50.0)
    
    # Deform nodes: Simple shear X += Z * shear
    curr_nodes = nodes0.copy()
    curr_nodes[:, 0] += curr_nodes[:, 2] * total_shear
    
    # Update cube
    for line, (i, j) in zip(lines, edge_indices):
        p1 = curr_nodes[i]
        p2 = curr_nodes[j]
        line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        line.set_3d_properties([p1[2], p2[2]])
        
    # Update slip lines
    for i, seg in enumerate(slip_segments):
        if frame > seg['start_f']:
            # Deform position
            # Start position also shears!
            pos = seg['pos'].copy()
            pos[0] += pos[2] * total_shear
            
            # Draw a small segment
            # length 0.2
            v = seg['vec']
            # v might rotate if large deformation, but slip line is material line element?
            # Slip lines are traces of planes.
            # Let's just keep them horizontal visually to show "sliding planes"
            
            p1 = pos - v*0.1
            p2 = pos + v*0.1
            
            seg['line'].set_data([p1[0], p2[0]], [p1[1], p2[1]])
            seg['line'].set_3d_properties([p1[2], p2[2]])
            
    if frame > 40:
        txt.set_text("Result: Smooth Shear (No single plane visible)")
    else:
        txt.set_text("Many discrete slips...")
        
    return lines + [s['line'] for s in slip_segments] + [txt]

anim = FuncAnimation(fig, update, frames=60, interval=50, blit=True)
anim.save('gif8_emergent.gif', writer='pillow', fps=20)
print("Saved gif8_emergent.gif")
