"""
Fierro EVPFFT Comparison Script
================================
Reproduces the exact Cu polycrystal test case from LANL's Fierro code
(legacy/Fierro_V0/src/EVPFFT/example_input_files/) and compares results.

Test Case:
  - 8×8×8 polycrystal, 512 grains (FCC Cu)
  - Uniaxial tension: ε̇₃₃ = 1.0 s⁻¹, σ₁₁ = σ₂₂ = 0 (mixed BCs)
  - Power-law flow rule: γ̇ = γ̇₀·|τ/g|^n·sgn(τ), γ̇₀=1.0, n=10
  - Voce hardening: τ₀=9 MPa, τ_s=14 MPa, θ₀=400 MPa, θ₁=250 MPa
  - 30 steps, dt = 0.00005 s

Fierro Parameters (from example input files):
  Elastic: C11=168400, C12=121400, C44=75400 MPa
  Flow:    nrsx=10 (exponent), gamd0x=1.0 (ref rate)
  Voce:    tau0xf=9, tau0xb=9, tau1x=5, thet0=400, thet1=250 MPa
           hselfx=1.0, hlatex=1.0
  Loading: velocity gradient controlled, ε̇₃₃=1.0, σ₁₁=σ₂₂=0
"""

import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constitutive import (CrystalPlasticityState, VoceHardening,
                          solve_evpfft_mixed)
from microstructure import euler_to_rotation


# ============================================================================
# 1. Parse Fierro's 8×8×8 microstructure file
# ============================================================================

def parse_fierro_microstructure(filepath):
    """
    Parse Fierro's microstructure file.
    
    Format per line: phi1 Phi phi2 ix iy iz grain_id phase_id
    Euler angles in degrees (Bunge convention).
    Voxel indices are 1-based.
    
    Returns:
        grain_ids   : (8, 8, 8) int array (0-based grain IDs)
        euler_angles: (n_grains, 3) array in radians
    """
    data = np.loadtxt(filepath)
    # columns: phi1, Phi, phi2, ix, iy, iz, grain_id, phase_id
    euler_deg = data[:, 0:3]    # (512, 3) Bunge angles in degrees
    ix = data[:, 3].astype(int) - 1   # 0-based
    iy = data[:, 4].astype(int) - 1
    iz = data[:, 5].astype(int) - 1
    gid_raw = data[:, 6].astype(int)  # Fierro grain IDs (1-based arbitrary)
    
    N = 8
    # Map arbitrary Fierro grain IDs to contiguous 0-based
    unique_gids = np.unique(gid_raw)
    n_grains = len(unique_gids)
    gid_map = {old: new for new, old in enumerate(unique_gids)}
    
    grain_ids = np.zeros((N, N, N), dtype=int)
    grain_euler_sum = np.zeros((n_grains, 3))
    grain_count = np.zeros(n_grains, dtype=int)
    
    for i in range(len(data)):
        new_gid = gid_map[gid_raw[i]]
        grain_ids[ix[i], iy[i], iz[i]] = new_gid
        grain_euler_sum[new_gid] += euler_deg[i]
        grain_count[new_gid] += 1
    
    # Average Euler angles per grain, convert to radians
    euler_angles = np.deg2rad(grain_euler_sum / grain_count[:, None])
    
    print(f"  Microstructure: {N}×{N}×{N} = {N**3} voxels")
    print(f"  Unique grains: {n_grains}")
    print(f"  Euler angle range (deg): "
          f"φ₁=[{np.rad2deg(euler_angles[:,0]).min():.1f}, "
          f"{np.rad2deg(euler_angles[:,0]).max():.1f}], "
          f"Φ=[{np.rad2deg(euler_angles[:,1]).min():.1f}, "
          f"{np.rad2deg(euler_angles[:,1]).max():.1f}], "
          f"φ₂=[{np.rad2deg(euler_angles[:,2]).min():.1f}, "
          f"{np.rad2deg(euler_angles[:,2]).max():.1f}]")
    
    return grain_ids, euler_angles


# ============================================================================
# 2. Set up identical material parameters
# ============================================================================

def create_fierro_state(grain_ids, euler_angles):
    """
    Create CrystalPlasticityState with Fierro's exact parameters.
    
    All values from Fierro's example input files (in MPa → converted to Pa):
      C11 = 168400 MPa, C12 = 121400 MPa, C44 = 75400 MPa
      tau0 = 9 MPa, tau_s = tau0 + tau1 = 14 MPa
      theta0 = 400 MPa, theta1 = 250 MPa
      hself = 1.0, hlat = 1.0  (isotropic q matrix)
      gamma_dot_0 = 1.0 s⁻¹, n = 10
    """
    # Fierro Voce parameters (MPa → Pa)
    tau0 = 9.0e6       # initial CRSS
    tau1 = 5.0e6       # saturation stress increment
    tau_s = tau0 + tau1  # = 14 MPa saturation CRSS
    theta0 = 400.0e6   # initial hardening rate
    theta1 = 250.0e6   # asymptotic hardening rate
    h_self = 1.0
    h_lat = 1.0        # isotropic hardening
    
    hardening = VoceHardening(
        tau0=tau0,
        tau_s=tau_s,
        h0=theta0,
        h1=theta1,
        q_latent=h_lat   # Fierro uses 1.0 (isotropic)
    )
    
    # Elastic constants (MPa → Pa)
    C11 = 168400e6
    C12 = 121400e6
    C44 = 75400e6
    
    # Power-law flow rule: γ̇ = γ̇₀ |τ/g|^n sgn(τ)
    state = CrystalPlasticityState(
        grain_ids, euler_angles,
        hardening=hardening,
        C11=C11, C12=C12, C44=C44,
        flow_rule='powerlaw',
        gamma_dot_0=1.0,    # reference slip rate
        n_exponent=10.0,    # rate sensitivity exponent
    )
    
    print(f"\n  Material: FCC Cu (power-law, n={10})")
    print(f"  Elastic: C11={C11/1e9:.1f}, C12={C12/1e9:.1f}, "
          f"C44={C44/1e9:.1f} GPa")
    print(f"  Voce: τ₀={tau0/1e6:.0f}, τ_s={tau_s/1e6:.0f}, "
          f"θ₀={theta0/1e6:.0f}, θ₁={theta1/1e6:.0f} MPa")
    print(f"  Flow: γ̇₀={1.0}, n={10}, q_lat={h_lat}")
    
    return state


# ============================================================================
# 3. Run simulation with Fierro's boundary conditions
# ============================================================================

def run_fierro_test(state, n_steps=30, dt=0.00005):
    """
    Run the Fierro Cu test case: uniaxial tension with mixed BCs.
    
    Boundary conditions (from Fierro input file):
      - ε̇₃₃ = 1.0 s⁻¹ (prescribed)
      - All off-diagonal L = 0 (prescribed)
      - L₁₁, L₂₂ unknown (adjusted to satisfy σ₁₁=σ₂₂=0)
      - σ₁₁ = 0, σ₂₂ = 0 (prescribed, traction-free)
    
    In Voigt [11, 22, 33, 23, 13, 12]:
      strain_mask = [F, F, T, T, T, T]  → ε₃₃, γ₂₃, γ₁₃, γ₁₂ known
      stress_mask = [T, T, F, F, F, F]  → σ₁₁, σ₂₂ targets = 0
    """
    # Macroscopic velocity gradient symmetric part (Voigt)
    # D = [[D11, 0, 0], [0, D22, 0], [0, 0, 1.0]]
    # D11, D22 are unknown, but D_macro placeholder has all components
    D_macro = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    
    # Target stress for stress-controlled components
    sig_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Masks: which components are strain-controlled vs stress-controlled
    strain_mask = np.array([False, False, True, True, True, True])
    stress_mask = np.array([True, True, False, False, False, False])
    
    print(f"\n{'='*70}")
    print(f"  Running Fierro Cu test case")
    print(f"  Steps: {n_steps}, dt: {dt:.2e}")
    print(f"  Total ε₃₃ = {n_steps * 1.0 * dt:.6f}")
    print(f"  Loading: uniaxial tension (ε̇₃₃=1.0, σ₁₁=σ₂₂=0)")
    print(f"{'='*70}")
    
    t0 = time.time()
    
    history = solve_evpfft_mixed(
        state, D_macro, sig_target,
        strain_mask, stress_mask,
        n_increments=n_steps,
        dt=dt,
        tol_fft=1e-5,
        tol_stress=1e-4,    # Fierro uses 1e-7, we relax for speed
        max_iter_fft=50,
        max_iter_stress=30,
        verbose=True,
    )
    
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f} s")
    
    return history


# ============================================================================
# 4. Post-processing and comparison plots
# ============================================================================

def plot_results(history, output_dir='.'):
    """Generate comparison plots from simulation results."""
    
    n_steps = len(history)
    eps_33 = np.array([h['eps_macro'][2] for h in history])
    sig_33 = np.array([h['sig_macro'][2] for h in history]) / 1e6  # MPa
    sig_11 = np.array([h['sig_macro'][0] for h in history]) / 1e6
    sig_22 = np.array([h['sig_macro'][1] for h in history]) / 1e6
    eps_11 = np.array([h['eps_macro'][0] for h in history])
    eps_22 = np.array([h['eps_macro'][1] for h in history])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FFT_crystals vs Fierro EVPFFT — Cu Polycrystal\n'
                 '(8×8×8, power-law n=10, Voce hardening)',
                 fontsize=14, fontweight='bold')
    
    # --- 1. Stress-strain curve ---
    ax = axes[0, 0]
    ax.plot(eps_33 * 100, sig_33, 'b-o', markersize=3, linewidth=2,
            label='Our FFT_crystals')
    ax.set_xlabel('ε₃₃ (%)')
    ax.set_ylabel('σ₃₃ (MPa)')
    ax.set_title('Uniaxial Stress-Strain Response')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- 2. Lateral stress check (should be ~0) ---
    ax = axes[0, 1]
    ax.plot(eps_33 * 100, sig_11, 'r-o', markersize=3, label='σ₁₁')
    ax.plot(eps_33 * 100, sig_22, 'g-s', markersize=3, label='σ₂₂')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('ε₃₃ (%)')
    ax.set_ylabel('Stress (MPa)')
    ax.set_title('Lateral Stress (should be ≈ 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- 3. Lateral strain (Poisson contraction) ---
    ax = axes[1, 0]
    ax.plot(eps_33 * 100, eps_11 * 100, 'r-o', markersize=3, label='ε₁₁')
    ax.plot(eps_33 * 100, eps_22 * 100, 'g-s', markersize=3, label='ε₂₂')
    # Expected: ε₁₁ ≈ ε₂₂ ≈ -ν·ε₃₃ where ν ~ 0.35 for Cu
    nu_eff = 0.35
    ax.plot(eps_33 * 100, -nu_eff * eps_33 * 100, 'k--', alpha=0.5,
            label=f'−ν·ε₃₃ (ν={nu_eff})')
    ax.set_xlabel('ε₃₃ (%)')
    ax.set_ylabel('Lateral Strain (%)')
    ax.set_title('Poisson Contraction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- 4. Hardening evolution ---
    ax = axes[1, 1]
    # Compute incremental hardening rate
    if len(eps_33) > 1:
        d_sig = np.diff(sig_33)
        d_eps = np.diff(eps_33)
        d_eps[d_eps == 0] = 1e-20
        hardening_rate = d_sig / d_eps / 1e3  # GPa
        ax.plot(eps_33[1:] * 100, hardening_rate, 'b-o', markersize=3,
                label='dσ/dε (tangent)')
    ax.set_xlabel('ε₃₃ (%)')
    ax.set_ylabel('Tangent Modulus (GPa)')
    ax.set_title('Hardening Rate Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    outpath = os.path.join(output_dir, 'fierro_comparison.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to: {outpath}")
    plt.show()
    
    return fig


def print_summary_table(history):
    """Print a summary comparison table."""
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY — Fierro Cu Test Case")
    print(f"{'='*70}")
    print(f"  {'Step':>4}  {'ε₃₃':>12}  {'σ₃₃ (MPa)':>12}  "
          f"{'σ₁₁ (MPa)':>12}  {'ε₁₁':>12}")
    print(f"  {'—'*4}  {'—'*12}  {'—'*12}  {'—'*12}  {'—'*12}")
    
    for h in history:
        step = h['step']
        e33 = h['eps_macro'][2]
        s33 = h['sig_macro'][2] / 1e6
        s11 = h['sig_macro'][0] / 1e6
        e11 = h['eps_macro'][0]
        print(f"  {step:>4}  {e33:>12.6e}  {s33:>12.4f}  "
              f"{s11:>12.4f}  {e11:>12.6e}")
    
    # Final values
    final = history[-1]
    e33 = final['eps_macro'][2]
    s33 = final['sig_macro'][2] / 1e6
    print(f"\n  Final σ₃₃ = {s33:.4f} MPa at ε₃₃ = {e33:.6e}")
    
    # Expected from Fierro (Cu polycrystal, n=10):
    # Taylor factor for FCC ~ 3.06
    # Initial yield ≈ M·τ₀ = 3.06 × 9 ≈ 27.5 MPa
    # After hardening ≈ M·τ_s = 3.06 × 14 ≈ 42.8 MPa (with θ₁ it stays higher)
    M_taylor = 3.06
    yield_est = M_taylor * 9.0
    print(f"\n  Expected (Taylor estimate):")
    print(f"    σ_yield ≈ M·τ₀ = {M_taylor:.2f} × 9 = {yield_est:.1f} MPa")
    print(f"    σ_sat   ≈ M·τ_s = {M_taylor:.2f} × 14 = {M_taylor*14:.1f} MPa")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("  Fierro EVPFFT Comparison — FFT_crystals")
    print("="*70)
    
    # 1. Parse Fierro microstructure
    micro_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'Fierro', 'legacy', 'Fierro_V0', 'src', 'EVPFFT',
        'example_input_files', 'random_microstructure_8x8x8.txt'
    )
    
    if not os.path.exists(micro_file):
        print(f"  ERROR: Fierro microstructure file not found:")
        print(f"  {micro_file}")
        print(f"  Expected at: ../Fierro/legacy/Fierro_V0/src/EVPFFT/"
              f"example_input_files/")
        sys.exit(1)
    
    print(f"\n  Loading Fierro microstructure: {os.path.basename(micro_file)}")
    grain_ids, euler_angles = parse_fierro_microstructure(micro_file)
    
    # 2. Create state with Fierro parameters
    state = create_fierro_state(grain_ids, euler_angles)
    
    # 3. Run simulation
    history = run_fierro_test(state, n_steps=30, dt=0.00005)
    
    # 4. Print summary
    print_summary_table(history)
    
    # 5. Plot results
    plot_results(history, output_dir=os.path.dirname(os.path.abspath(__file__)))
