import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import fcc_utils

# Setup
fig, ax = fcc_utils.setup_plot()
ax.set_title("Elastic vs. Plastic Deformation", color='black', fontsize=15)

# Initial Cube
corners = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
]) - 0.5 # Center at origin

# Define edges by indices
edge_indices = [
   (0,1), (0,2), (0,3),
   (1,4), (1,5),
   (2,4), (2,6),
   (3,5), (3,6),
   (4,7), (5,7), (6,7)
]

lines = []
for _ in edge_indices:
    line, = ax.plot([], [], [], color='cyan', lw=2)
    lines.append(line)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

txt = ax.text2D(0.5, 0.05, "", transform=ax.transAxes, ha='center', color='blue', fontsize=12)

def update(frame):
    current_corners = corners.copy()
    
    # Elastic Phase (0-30)
    if frame < 30:
        strain = 0.3 * np.sin(frame / 30 * np.pi) # Goes up and down
        # Apply tensile strain in Z
        # z -> z * (1 + strain)
        # x -> x * (1 - 0.3*strain) Poisson
        current_corners[:, 2] *= (1 + strain)
        current_corners[:, 0] *= (1 - 0.3*strain)
        current_corners[:, 1] *= (1 - 0.3*strain)
        label = "Elastic: Stretch & Recover"
        
    # Plastic Phase (30-60)
    else:
        gamma = 0.5 * (frame - 30) / 30
        # Simple shear: x -> x + gamma * z
        # But we want to simulate slip plane, let's keep it simple simple shear
        current_corners[:, 0] += gamma * (current_corners[:, 2] + 0.5) # Shear relative to bottom
        label = "Plastic: Permanent Shear"

    # Update visual lines
    for line, (i, j) in zip(lines, edge_indices):
        p1 = current_corners[i]
        p2 = current_corners[j]
        line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        line.set_3d_properties([p1[2], p2[2]])
        
    txt.set_text(label)
    return lines + [txt]

anim = FuncAnimation(fig, update, frames=60, interval=50, blit=True)
anim.save('gif1_plasticity.gif', writer='pillow', fps=20)
print("Saved gif1_plasticity.gif")
