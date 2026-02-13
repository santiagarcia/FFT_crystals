import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('default')
fig, ax = plt.subplots(figsize=(6, 4))

tau = np.linspace(-2, 2, 200)
# Flow rule: gamma_dot = sign(tau) * |tau|^5 (power law proxy for exponential)
gamma_dot = np.sign(tau) * (np.abs(tau)**10) * 1e-2

ax.plot(tau, gamma_dot, 'cyan', lw=2, label='Activaton Curve')
ax.set_xlabel(r'Resolved Shear Stress $\tau$')
ax.set_ylabel(r'Slip Rate $\dot{\gamma}$')
ax.set_title('Slip Rate Activation', color='black')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.set_ylim(-2, 2)
ax.grid(True, alpha=0.3)

# Systems
dots, = ax.plot([], [], 'o', color='red', markersize=8)

# Simulation: 
# Stress states of 12 systems.
# Simply assume random distributions that scale with time.
# t in 0..1. Taus = initial_random * t * 2.

taus_base = np.random.uniform(-1, 1, 12)

def update(frame):
    scale = (frame / 50.0) * 1.8 # Ramp up load
    
    current_taus = taus_base * scale
    current_gamma = np.sign(current_taus) * (np.abs(current_taus)**10) * 1e-2
    
    dots.set_data(current_taus, current_gamma)
    return [dots]

anim = FuncAnimation(fig, update, frames=60, interval=50, blit=True)
anim.save('gif5_activation.gif', writer='pillow', fps=20)
print("Saved gif5_activation.gif")
