import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use('default') 
# Update params to look better in beamer (larger fonts)
plt.rcParams.update({
    'font.size': 12, 
    'lines.linewidth': 2.5,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10
})

# Constants
k_B = 1.3806e-23 # J/K
T = 300.0 # K
mu = 1.602e-19 # eV to J conversion factor for plots label if needed, or just work in eV

# Plot 1: Energy Barrier Shape
def energy_profile(x, p, q):
    # x is tau/S_thermal
    # Ensure x is within [0, 1]
    x_clip = np.clip(x, 0, 1)
    return (1 - x_clip**p)**q

x = np.linspace(0, 1, 200)

plt.figure(figsize=(6, 4))
plt.plot(x, energy_profile(x, 0.5, 1.0), label=r'$p=0.5, q=1.0$')
plt.plot(x, energy_profile(x, 1.0, 1.0), label=r'$p=1.0, q=1.0$ (Box)')
plt.plot(x, energy_profile(x, 0.78, 1.15), label=r'$p=0.78, q=1.15$ (FCC)', linestyle='--', color='red')
plt.xlabel(r'Normalized Stress $\tau / S_{th}$')
plt.ylabel(r'Barrier Energy $\Delta G / \Delta G_0$')
plt.title('Obstacle Profile Shape')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('energy_barrier.pdf')
plt.close()

# Plot 2: Flow Rule Sensitivity (Stiffness)
# gamma_dot = ref * exp( -DG0/kT * (1 - (tau/S)^p)^q )
gamma_ref = 1.0e7
DG0_eV = 1.5 # eV typical for cross-slip
DG0 = DG0_eV * 1.602e-19 # J
S_th = 200.0 # MPa
p = 0.78
q = 1.15
kT = k_B * T

tau = np.linspace(0, S_th*1.1, 500)
eff_stress_ratio = np.clip(tau/S_th, 0, 1) 
exponent = -(DG0/kT) * (1 - eff_stress_ratio**p)**q
# If tau > S_th, exponent is 0 (or close to it physically, barrier overcome)
# Actually, the model typically switches to a power law or just saturates, 
# but for the Arrhenius part, it goes to exp(0) = 1. 0 barrier.
# Let's simple clip it.

gamma_dot = gamma_ref * np.exp(exponent)

plt.figure(figsize=(6, 4))
plt.plot(tau, gamma_dot, color='darkred')
plt.yscale('log')
plt.xlabel(r'Applied Stress $\tau$ (MPa)')
plt.ylabel(r'Slip Rate $\dot{\gamma}$ ($s^{-1}$)')
plt.title(r'Rate Sensitivity ($T=300K$)')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.axvline(x=S_th, color='black', linestyle='--', label=r'$S_{th}$')
plt.text(S_th*0.1, 1e-5, r'$\Delta G \gg k_B T$' + '\n(No Slip)', fontsize=12)
plt.text(S_th*0.65, 1e2, r'$\tau \approx S_{th}$', fontsize=12)
plt.ylim(1e-15, 1e8)
plt.tight_layout()
plt.savefig('flow_sensitivity.pdf')
plt.close()

# Plot 3: Hardening (Taylor)
rho = np.linspace(0, 1e16, 100) # m^-2
alpha = 0.35
b = 2.86e-10 # m (Aluminum)
shear_modulus = 26e9 # Pa 
taylor_stress = alpha * shear_modulus * b * np.sqrt(rho) / 1e6 # MPa

plt.figure(figsize=(6, 4))
plt.plot(rho/1e14, taylor_stress, color='darkblue')
plt.xlabel(r'Dislocation Density $\rho$ ($10^{14} m^{-2}$)')
plt.ylabel(r'Resistance Strength $S$ (MPa)')
plt.title(r'Taylor Hardening Law $S \propto \sqrt{\rho}$')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hardening.pdf')
plt.close()

print("Plots generated: energy_barrier.pdf, flow_sensitivity.pdf, hardening.pdf")
