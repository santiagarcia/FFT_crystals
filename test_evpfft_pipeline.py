"""Quick test of the full EVPFFT pipeline including snapshot storage."""
import numpy as np
from microstructure import generate_voronoi_microstructure
import constitutive

N, n_grains = 8, 4
grain_ids, centers, euler_angles = generate_voronoi_microstructure(N, n_grains, seed=42)

# Use Copper preset (low yield → should see plasticity easily)
hardening = constitutive.VoceHardening(tau0=16e6, tau_s=148e6, h0=180e6)
state = constitutive.CrystalPlasticityState(
    grain_ids, euler_angles, hardening=hardening,
    temperature=1123, rate_slip_ref=1e7)

import time
S_macro = np.array([5000e6, 0, 0, 0, 0, 0])  # 5 GPa uniaxial
t0 = time.perf_counter()
history = constitutive.solve_evpfft(state, S_macro, n_increments=5, dt=0.1,
    tol_fft=1e-3, tol_stress=1e-2, max_iter_fft=200, verbose=True)
elapsed = time.perf_counter() - t0

print(f'\n=== RESULTS ({elapsed:.2f}s) ===')
print(f'History steps: {len(history)}')
print(f'Snapshots stored: {len(state._history_snapshots)}')

for i, snap in enumerate(state._history_snapshots):
    ep = snap['ep_eq']
    acc = snap['acc_slip']
    sres = snap['slip_res']
    print(f'  Step {i+1}: ep_eq max={np.max(ep):.6f}  '
          f'acc_slip max={np.max(acc):.6f}  '
          f'slip_res mean={np.mean(sres):.1f} MPa')

ep_final = state.get_von_mises_plastic_strain_field()
print(f'Final ep_eq max: {np.max(ep_final):.6f}')
print(f'Final plastic_strain norm: {np.linalg.norm(state.plastic_strain):.6f}')
print(f'Final plastic_strain[:3, :3]:\n{state.plastic_strain[:3, :3]}')
print(f'Final plastic_strain[:3, 3:]:\n{state.plastic_strain[:3, 3:]}')
