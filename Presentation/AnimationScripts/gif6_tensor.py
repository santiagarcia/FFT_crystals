import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import fcc_utils

fig, ax = fcc_utils.setup_plot()
ax.set_title("Summing Slip Tensors", color='black')

# Nodes of a cube
nodes = np.array([
    [0,0,0], [1,0,0], [1,1,0], [0,1,0],
    [0,0,1], [1,0,1], [1,1,1], [0,1,1]
]) - 0.5

edges = [
    (0,1), (1,2), (2,3), (3,0),
    (4,5), (5,6), (6,7), (7,4),
    (0,4), (1,5), (2,6), (3,7)
]

def plot_cube(curr_nodes, color='cyan', alpha=1.0):
    lines = []
    for i, j in edges:
        p1 = curr_nodes[i]
        p2 = curr_nodes[j]
        l, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, alpha=alpha, lw=2)
        lines.append(l)
    return lines

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)

lines_obj = plot_cube(nodes)

# Tensors
# 1. Shear XY
# 2. Shear YZ
# 3. Shear ZX
def shear_mat(dim1, dim2, amount):
    # s along dim1, n along dim2
    # F = I + amount * (e1 x e2)
    # x_new = x + amount * y (if dim1=0, dim2=1)
    # This is a deformation gradient F. x_new = F * x
    M = np.eye(3)
    M[dim1, dim2] = amount
    return M

tensors = [
    shear_mat(0, 1, 0.2), # xy shear
    shear_mat(1, 2, 0.2), # yz shear
    shear_mat(2, 0, 0.2), # zx shear
    shear_mat(0, 2, -0.2),
    shear_mat(1, 0, -0.2)
]

current_F = np.eye(3)
step = 0
frame_steps = 10

txt = ax.text2D(0.5, 0.05, "", transform=ax.transAxes, ha='center', color='blue', fontsize=12)

def update(frame):
    global current_F, step
    
    # Calculate target based on step
    # 0-10: Tensor 1
    # 10-20: Tensor 2
    
    stage = frame // 10
    local_t = (frame % 10) / 10.0
    
    if stage < len(tensors):
        # Interpolate effect? 
        # Actually easier to just accumulate.
        # F_new = (I + dgamma * P) * F_old
        # Let's just say we deform linearly towards target.
        
        # We want "Discrete additions".
        # Initial state of this stage: current_F_base
        # Target state: tensors[stage] @ current_F_base ?
        # Small deformation approx: F = (I + gamma P) F0
        
        # To make it smooth:
        # P = tensors[stage] - I
        # dP = P * local_t
        # F(t) = (I + dP) * current_F_accumulated_before
        
        # Simplified:
        # Just evolve the nodes directly.
        pass
    
    # Re-impl:
    # Just apply cumulative total from 0 to frame.
    
    total_deform = np.eye(3)
    
    desc = ""
    for i in range(len(tensors)):
        target_F = tensors[i] # This is I + gamma P
        # how much of this tensor to apply?
        if stage > i:
            f = 1.0
        elif stage == i:
            f = local_t
        else:
            f = 0.0
            
        if f > 0:
            # incremental F = I + f * (target - I)
            inc = np.eye(3) + f * (target_F - np.eye(3))
            total_deform = inc @ total_deform
            
    if stage < len(tensors):
        desc = f"Adding Slip System {stage+1}"
    else:
        desc = "Total Plastic Strain"
        
    # Apply to nodes
    new_nodes = (total_deform @ nodes.T).T
    
    # Update lines
    for line, (i, j) in zip(lines_obj, edges):
        p1 = new_nodes[i]
        p2 = new_nodes[j]
        line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        line.set_3d_properties([p1[2], p2[2]])
        
    txt.set_text(desc)
    return lines_obj + [txt]

anim = FuncAnimation(fig, update, frames=55, interval=100, blit=False)
anim.save('gif6_tensor.gif', writer='pillow', fps=10)
print("Saved gif6_tensor.gif")
