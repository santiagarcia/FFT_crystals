import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import fcc_utils

plt.style.use('default')
fig = plt.figure(figsize=(10, 5))
ax3d = fig.add_subplot(121, projection='3d')
axbar = fig.add_subplot(122)

fcc_utils.setup_plot() # returns separate ignored
# Setup Cube
corners = fcc_utils.get_fcc_edges()
for start, end in corners:
    ax3d.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='gray', alpha=0.3)
ax3d.set_axis_off()
ax3d.set_xlim(-0.2, 1.2)
ax3d.set_ylim(-0.2, 1.2)
ax3d.set_zlim(-0.2, 1.2)
ax3d.set_title("Slip Event on System A", color='black')

# Bars
bar_labels = ['System A', 'System B']
bar_vals = [0.2, 0.2]
bars = axbar.bar(bar_labels, bar_vals, color=['cyan', 'magenta'])
axbar.set_ylim(0, 1.0)
axbar.set_title("Slip Resistance $S$", color='black')

# Arrow for slip
arrow_q = ax3d.quiver(0, 0.5, 0.5, 1, 0, 0, color='cyan', length=0, linewidth=4) # Hidden initially

def update(frame):
    # Frames 0-10: nothing
    # Frames 10-40: Slip happens (arrow grows) -> Resistance grows
    
    global arrow_q, bars
    
    if frame < 10:
        return list(bars) + [arrow_q]
        
    progress = (frame - 10) / 30.0
    if progress > 1: progress = 1
    
    # Arrow grows
    if arrow_q: arrow_q.remove()
    arrow_q = ax3d.quiver(0, 0.5, 0.5, 1, 0, 0, color='cyan', length=1.0*progress, linewidth=4)
    
    # Resistance grows
    # A grows by 0.5 * p
    # B grows by 0.6 * p (latent often stronger -> 1.4 ratio?)
    
    new_h_a = 0.2 + 0.3 * progress
    new_h_b = 0.2 + 0.4 * progress
    
    bars[0].set_height(new_h_a)
    bars[1].set_height(new_h_b)
    
    # Color shift to red as it hardens?
    bars[0].set_color(plt.cm.cool(progress))
    bars[1].set_color(plt.cm.cool(progress))
    
    if progress > 0.5:
        axbar.set_title("Latent Hardening (B hardens too!)", color='blue')
        
    return list(bars) + [arrow_q]

anim = FuncAnimation(fig, update, frames=50, interval=100, blit=False)
anim.save('gif7_hardening.gif', writer='pillow', fps=10)
print("Saved gif7_hardening.gif")
