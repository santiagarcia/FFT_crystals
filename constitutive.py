"""
Crystal Plasticity Constitutive Model for Polycrystal FFT Solver
=================================================================
Bridges the OTIS single-crystal flow rule (physics_engine.py) with the
FFT-based homogenization solver for full-field polycrystal simulations.

Implements:
  - Elasto-viscoplastic constitutive update at each voxel
  - Rotation of slip systems by grain orientation
  - Hardening law (Voce-type)
  - Incremental strain decomposition: ε = εᵉ + εᵖ
  - State variable management (slip resistances, accumulated slip)

References:
  Lebensohn (2001) — VPFFT formulation
  Lebensohn et al. (2012) — EVPFFT formulation
  Lebensohn et al. (2016) — Nonlocal extensions
"""

import numpy as np
from microstructure import euler_to_rotation, cubic_stiffness_tensor, rotate_stiffness_voigt
import physics_engine


# ============================================================================
# FCC Slip System Geometry (12 {111}<110> systems)
# ============================================================================

# Slip plane normals (FCC)
SLIP_NORMALS_REF = np.array([
    [ 1, 1, 1], [ 1, 1, 1], [ 1, 1, 1],
    [-1, 1, 1], [-1, 1, 1], [-1, 1, 1],
    [ 1,-1, 1], [ 1,-1, 1], [ 1,-1, 1],
    [ 1, 1,-1], [ 1, 1,-1], [ 1, 1,-1],
], dtype=float) / np.sqrt(3)

# Slip directions (FCC)
SLIP_DIRS_REF = np.array([
    [ 0, 1,-1], [-1, 0, 1], [ 1,-1, 0],
    [ 0, 1,-1], [ 1, 0, 1], [-1,-1, 0],
    [ 0, 1, 1], [-1, 0, 1], [ 1, 1, 0],
    [ 0, 1, 1], [ 1, 0, 1], [-1, 1, 0],
], dtype=float) / np.sqrt(2)

N_SLIP = 12


def rotate_slip_systems(R):
    """
    Rotate the 12 FCC slip systems by rotation matrix R.
    Returns rotated normals and directions.
    """
    normals = (R @ SLIP_NORMALS_REF.T).T  # (12, 3)
    dirs = (R @ SLIP_DIRS_REF.T).T        # (12, 3)
    return normals, dirs


def schmid_tensors(normals, dirs):
    """
    Compute the Schmid tensors P^α = 0.5 * (s^α ⊗ n^α + n^α ⊗ s^α)
    for symmetric part, and W^α = 0.5 * (s^α ⊗ n^α - n^α ⊗ s^α) 
    for the skew part (plastic spin).
    
    Returns:
        P : (12, 3, 3) — symmetric Schmid tensors
        W : (12, 3, 3) — skew Schmid tensors (plastic spin)
    """
    P = np.zeros((N_SLIP, 3, 3))
    W = np.zeros((N_SLIP, 3, 3))
    for alpha in range(N_SLIP):
        sn = np.outer(dirs[alpha], normals[alpha])
        P[alpha] = 0.5 * (sn + sn.T)
        W[alpha] = 0.5 * (sn - sn.T)
    return P, W


# ============================================================================
# Voce Hardening Law
# ============================================================================

class VoceHardening:
    """
    Voce-type isotropic hardening for slip resistance evolution.
    
    ṡ^α = h₀(1 - s^α/s_s) * |γ̇^α|   (self-hardening, simplified)
    
    With latent hardening:
    ṡ^α = Σ_β h_{αβ} |γ̇^β|
    h_{αβ} = q_{αβ} * h₀(1 - s^β/s_s)
    
    where q_αα = 1 (self), q_αβ = q_lat (latent, default 1.4)
    """
    
    def __init__(self, tau0=50e6, tau_s=200e6, h0=500e6, q_latent=1.4):
        """
        Parameters:
            tau0     : initial slip resistance (Pa)
            tau_s    : saturation slip resistance (Pa)
            h0       : initial hardening rate (Pa)
            q_latent : latent hardening ratio
        """
        self.tau0 = tau0
        self.tau_s = tau_s
        self.h0 = h0
        self.q_latent = q_latent
    
    def initial_resistance(self):
        """Return initial slip resistance for all 12 systems."""
        return np.full(N_SLIP, self.tau0)
    
    def update(self, slip_resistance, delta_gamma, dt):
        """
        Update slip resistance based on accumulated slip increments.
        
        Parameters:
            slip_resistance : (12,) current resistance
            delta_gamma     : (12,) slip increments (signed)
            dt              : time increment
        Returns:
            new_resistance  : (12,) updated resistance
        """
        abs_dg = np.abs(delta_gamma)
        
        new_res = slip_resistance.copy()
        for alpha in range(N_SLIP):
            # Self + latent hardening
            total_rate = 0.0
            for beta in range(N_SLIP):
                q = 1.0 if alpha == beta else self.q_latent
                h = self.h0 * max(0, 1.0 - slip_resistance[beta] / self.tau_s)
                total_rate += q * h * abs_dg[beta]
            new_res[alpha] += total_rate
        
        return new_res


# ============================================================================
# Material state (per voxel)
# ============================================================================

class PolycrystalState:
    """
    Manages the state variables for a polycrystalline RVE.
    
    Per voxel:
      - Slip resistance (12 values)
      - Accumulated plastic strain tensor
      - Accumulated slip per system
      - Backstress (for kinematic hardening, optional)
    """
    
    def __init__(self, grain_ids, euler_angles, hardening=None,
                 C11=168.4e9, C12=121.4e9, C44=75.4e9):
        """
        Parameters:
            grain_ids    : (N,N,N) int array
            euler_angles : (n_grains, 3) array
            hardening    : VoceHardening instance (or None for defaults)
            C11, C12, C44: elastic constants
        """
        self.grain_ids = grain_ids
        self.euler_angles = euler_angles
        self.shape = grain_ids.shape
        self.n_voxels = grain_ids.size
        self.n_grains = euler_angles.shape[0]
        
        self.C11, self.C12, self.C44 = C11, C12, C44
        
        if hardening is None:
            hardening = VoceHardening()
        self.hardening = hardening
        
        # Pre-compute per-grain data
        self._precompute_grain_data()
        
        # State variables (flat voxel indexing)
        self.slip_resistance = np.zeros((self.n_voxels, N_SLIP))
        self.accumulated_slip = np.zeros((self.n_voxels, N_SLIP))
        self.plastic_strain = np.zeros((self.n_voxels, 6))  # Voigt
        
        # Initialize resistance
        tau0 = hardening.initial_resistance()
        self.slip_resistance[:] = tau0
        
        # Temperature (can be spatially varying)
        self.temperature = np.full(self.n_voxels, 300.0)  # K
    
    def _precompute_grain_data(self):
        """Pre-compute rotation matrices, rotated slip systems, stiffness per grain."""
        n_grains = self.n_grains
        
        self.rotations = np.zeros((n_grains, 3, 3))
        self.normals_rot = np.zeros((n_grains, N_SLIP, 3))
        self.dirs_rot = np.zeros((n_grains, N_SLIP, 3))
        self.schmid_P = np.zeros((n_grains, N_SLIP, 3, 3))
        self.schmid_W = np.zeros((n_grains, N_SLIP, 3, 3))
        self.C_voigt = np.zeros((n_grains, 6, 6))
        
        C_base = cubic_stiffness_tensor(self.C11, self.C12, self.C44)
        
        for g in range(n_grains):
            R = euler_to_rotation(*self.euler_angles[g])
            self.rotations[g] = R
            
            normals, dirs = rotate_slip_systems(R)
            self.normals_rot[g] = normals
            self.dirs_rot[g] = dirs
            
            P, W = schmid_tensors(normals, dirs)
            self.schmid_P[g] = P
            self.schmid_W[g] = W
            
            self.C_voigt[g] = rotate_stiffness_voigt(C_base, R)
    
    def get_stiffness_field(self):
        """Return the full (N,N,N,6,6) stiffness field."""
        return self.C_voigt[self.grain_ids]
    
    def compute_resolved_shear_stress(self, stress_tensor, voxel_idx):
        """
        Compute resolved shear stress on all slip systems for a single voxel.
        
        Parameters:
            stress_tensor : (3,3) Cauchy stress
            voxel_idx     : flat voxel index
        Returns:
            rss : (12,) resolved shear stress
        """
        grain = self.grain_ids.flat[voxel_idx]
        P = self.schmid_P[grain]  # (12, 3, 3)
        
        # RSS = P : σ = tr(P^T σ)
        rss = np.einsum('aij,ij->a', P, stress_tensor)
        return rss
    
    def constitutive_update_voxel(self, strain_increment_voigt, voxel_idx, dt=1.0):
        """
        Perform elasto-viscoplastic constitutive update for a single voxel.
        
        Uses the OTIS flow rule from physics_engine.py to compute plastic slip rates,
        then updates plastic strain and hardening.
        
        Parameters:
            strain_increment_voigt : (6,) total strain increment (Voigt)
            voxel_idx              : flat index
            dt                     : time increment
        
        Returns:
            stress_voigt : (6,) updated stress (Voigt)
            plastic_strain_inc_voigt : (6,) plastic strain increment
        """
        grain = self.grain_ids.flat[voxel_idx]
        C = self.C_voigt[grain]  # (6,6)
        
        # Trial elastic stress
        elastic_strain_inc = strain_increment_voigt - self.plastic_strain[voxel_idx]
        # Actually, we should use elastic predictor: trial stress from total strain
        # σ_trial = C : (ε_total - ε_plastic)
        # For now, use the simpler approach: compute trial stress from strain increment
        stress_trial_voigt = C @ strain_increment_voigt
        
        # Convert to tensor for RSS calculation
        stress_tensor = np.array([
            [stress_trial_voigt[0], stress_trial_voigt[5], stress_trial_voigt[4]],
            [stress_trial_voigt[5], stress_trial_voigt[1], stress_trial_voigt[3]],
            [stress_trial_voigt[4], stress_trial_voigt[3], stress_trial_voigt[2]],
        ])
        
        # Resolved shear stress on each slip system
        rss = self.compute_resolved_shear_stress(stress_tensor, voxel_idx)
        
        # Compute flow rule using OTIS engine
        T = self.temperature[voxel_idx]
        slip_res = self.slip_resistance[voxel_idx]
        
        results = physics_engine.compute_flow_rule(
            rss, T,
            res_thermal_ssd_undamaged=slip_res,
            increment_time=dt
        )
        
        delta_gamma = results['Increment_Slip_Plastic']
        
        # Compute plastic strain increment: Δεᵖ = Σ_α Δγ^α * P^α
        P = self.schmid_P[grain]
        eps_plastic_inc = np.zeros((3, 3))
        for alpha in range(N_SLIP):
            eps_plastic_inc += delta_gamma[alpha] * P[alpha]
        
        # Convert to Voigt
        eps_p_voigt = np.array([
            eps_plastic_inc[0,0], eps_plastic_inc[1,1], eps_plastic_inc[2,2],
            2*eps_plastic_inc[1,2], 2*eps_plastic_inc[0,2], 2*eps_plastic_inc[0,1]
        ])
        
        # Update plastic strain
        self.plastic_strain[voxel_idx] += eps_p_voigt
        
        # Update hardening
        self.slip_resistance[voxel_idx] = self.hardening.update(
            slip_res, delta_gamma, dt
        )
        
        # Update accumulated slip
        self.accumulated_slip[voxel_idx] += np.abs(delta_gamma)
        
        # Corrected stress: σ = C : (ε - εᵖ)
        elastic_strain = strain_increment_voigt - self.plastic_strain[voxel_idx]
        stress_voigt = C @ elastic_strain
        
        return stress_voigt, eps_p_voigt
    
    def get_total_accumulated_slip_field(self):
        """Return total accumulated slip per voxel, reshaped to (N,N,N)."""
        total = np.sum(self.accumulated_slip, axis=1)
        return total.reshape(self.shape)
    
    def get_von_mises_plastic_strain_field(self):
        """Return equivalent plastic strain per voxel, reshaped to (N,N,N)."""
        ep = self.plastic_strain
        e11 = ep[:, 0]; e22 = ep[:, 1]; e33 = ep[:, 2]
        e23 = ep[:, 3]/2; e13 = ep[:, 4]/2; e12 = ep[:, 5]/2
        
        vm = np.sqrt(2.0/3.0 * ((e11-e22)**2 + (e22-e33)**2 + (e33-e11)**2 +
                                  6*(e23**2 + e13**2 + e12**2)))
        return vm.reshape(self.shape)


# ============================================================================
# EVP-FFT Solver (Elasto-Viscoplastic FFT)
# ============================================================================

def solve_evpfft(state, E_macro, n_increments=10, dt=0.1,
                  tol=1e-5, max_iter_fft=100, max_iter_cp=20,
                  verbose=True, callback=None):
    """
    Elasto-Viscoplastic FFT solver following Lebensohn (2001, 2012).
    
    Performs an incremental loading simulation:
      - At each load increment, apply macroscopic strain
      - Solve the equilibrium problem iteratively
      - At each voxel, perform the crystal plasticity constitutive update
      - Iterate until stress equilibrium and constitutive consistency
    
    Parameters:
        state         : PolycrystalState — the RVE state
        E_macro       : (6,) total macroscopic strain to reach
        n_increments  : int — number of load increments
        dt            : float — time increment per step
        tol           : float — equilibrium tolerance
        max_iter_fft  : max FFT iterations per increment
        max_iter_cp   : max constitutive iterations per FFT iteration
        verbose       : print progress
        callback      : callable(increment, eps_field, sig_field, state) per increment
    
    Returns:
        history : list of dicts with per-increment results
    """
    from fft_solver import (build_green_data, apply_green_operator_tensor,
                            fft_3x3, ifft_3x3, _equilibrium_error, _strain_change,
                            apply_stiffness, strain_field_v2t, stress_field_v2t,
                            strain_field_t2v, stress_field_t2v, strain_v2t)
    from fft_solver import compute_reference_medium
    
    shape = state.shape
    N = shape[0]
    n_vox = state.n_voxels
    
    # Get stiffness field
    C_field = state.get_stiffness_field()
    
    # Reference medium
    C0, lam0, mu0 = compute_reference_medium(C_field)
    C0_broadcast = C0[np.newaxis, np.newaxis, np.newaxis]
    
    n_field, Ainv = build_green_data(N, lam0, mu0)
    
    dE = E_macro / n_increments  # strain increment per step
    E_bar_t = strain_v2t(dE)  # tensor form of increment
    
    # Current total strain field (tensor form for the FFT solver)
    eps_field = np.zeros(shape + (6,))
    sig_field = np.zeros(shape + (6,))
    
    history = []
    
    for inc in range(n_increments):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Load Increment {inc+1}/{n_increments}")
            print(f"{'='*50}")
        
        # Prescribed macro strain for this increment
        E_current = dE * (inc + 1)
        E_current_t = strain_v2t(E_current)
        
        # Update strain field (add increment uniformly as initial guess)
        eps_field[...] += dE
        
        # FFT iteration for equilibrium
        for fft_iter in range(max_iter_fft):
            eps_old = eps_field.copy()
            
            # 1. Constitutive update at each voxel
            sig_field_flat = sig_field.reshape(-1, 6)
            eps_field_flat = eps_field.reshape(-1, 6)
            
            for vox in range(n_vox):
                sig_v, _ = state.constitutive_update_voxel(
                    eps_field_flat[vox], vox, dt=dt
                )
                sig_field_flat[vox] = sig_v
            
            sig_field = sig_field_flat.reshape(shape + (6,))
            
            # 2. Compute polarization in tensor form: τ = σ - C⁰ : ε
            eps_tensor = strain_field_v2t(eps_field)
            sig_tensor = stress_field_v2t(sig_field)
            sig0_tensor = apply_stiffness(C0_broadcast, eps_tensor)
            tau_tensor = sig_tensor - sig0_tensor
            
            # 3. FFT Green's operator (tensor form)
            tau_hat = fft_3x3(tau_tensor)
            gamma_tau_hat = apply_green_operator_tensor(tau_hat, n_field, Ainv)
            gamma_tau_hat[0, 0, 0, :, :] = 0.0
            
            gamma_tau = ifft_3x3(gamma_tau_hat)
            
            # 4. Update strain: eps = E_bar - Gamma * tau (Lippmann-Schwinger)
            eps_new_tensor = E_current_t - gamma_tau
            eps_field = strain_field_t2v(eps_new_tensor)
            
            # 5. Check convergence (strain change)
            err = _strain_change(eps_new_tensor, strain_field_v2t(eps_old))
            
            if verbose and (fft_iter % 5 == 0 or err < tol):
                sig_mean = np.mean(sig_field.reshape(-1, 6), axis=0)
                print(f"  FFT iter {fft_iter:3d}: Δε = {err:.4e}, "
                      f"<σ11>={sig_mean[0]/1e6:.1f} MPa")
            
            if err < tol:
                if verbose:
                    print(f"  Converged at FFT iteration {fft_iter}")
                break
        
        # Record history
        sig_mean = np.mean(sig_field.reshape(-1, 6), axis=0)
        eps_mean = np.mean(eps_field.reshape(-1, 6), axis=0)
        
        inc_result = {
            'increment': inc + 1,
            'macro_strain': E_current.copy(),
            'macro_stress': sig_mean.copy(),
            'eps_field': eps_field.copy(),
            'sig_field': sig_field.copy(),
            'accumulated_slip': state.get_total_accumulated_slip_field().copy(),
            'eq_plastic_strain': state.get_von_mises_plastic_strain_field().copy(),
            'fft_iterations': fft_iter + 1,
            'equilibrium_error': err,
        }
        history.append(inc_result)
        
        if callback:
            callback(inc, eps_field, sig_field, state)
    
    return history


# ============================================================================
# Simplified elastic polycrystal solver (for quick demos)
# ============================================================================

def solve_elastic_polycrystal(grain_ids, euler_angles, E_macro,
                               C11=168.4e9, C12=121.4e9, C44=75.4e9,
                               solver='cg', tol=1e-6, max_iter=200, verbose=True):
    """
    Quick wrapper: generate stiffness field + solve elastic problem.
    
    Parameters:
        grain_ids    : (N,N,N) grain map
        euler_angles : (n_grains, 3) Euler angles
        E_macro      : (6,) macroscopic strain (Voigt)
        C11, C12, C44: cubic elastic constants
        solver       : 'basic' or 'cg'
    
    Returns:
        eps_field, sig_field, info
    """
    from microstructure import build_local_stiffness_field_fast
    from fft_solver import solve_basic_scheme, solve_conjugate_gradient
    
    C_field = build_local_stiffness_field_fast(grain_ids, euler_angles, C11, C12, C44)
    
    solve_fn = solve_conjugate_gradient if solver == 'cg' else solve_basic_scheme
    eps, sig, info = solve_fn(C_field, E_macro, tol=tol, max_iter=max_iter, verbose=verbose)
    
    return eps, sig, info


# ============================================================================
# Quick test
# ============================================================================

if __name__ == '__main__':
    from microstructure import generate_voronoi_microstructure
    
    print("Polycrystal constitutive model test...")
    
    N = 8
    n_grains = 4
    grain_ids, centers, euler_angles = generate_voronoi_microstructure(N, n_grains, seed=42)
    
    # Create state
    state = PolycrystalState(grain_ids, euler_angles)
    
    print(f"  Grid: {state.shape}")
    print(f"  Grains: {state.n_grains}")
    print(f"  Total voxels: {state.n_voxels}")
    print(f"  Initial slip resistance: {state.slip_resistance[0, 0]/1e6:.1f} MPa")
    
    # Quick elastic solve
    E_macro = np.array([0.001, 0, 0, 0, 0, 0])
    eps, sig, info = solve_elastic_polycrystal(
        grain_ids, euler_angles, E_macro, solver='cg', tol=1e-4, verbose=True
    )
    
    from fft_solver import von_mises_stress
    vm = von_mises_stress(sig)
    print(f"\n  Max von Mises stress: {np.max(vm)/1e6:.1f} MPa")
    print(f"  Min von Mises stress: {np.min(vm)/1e6:.1f} MPa")
    print("  Done.")
