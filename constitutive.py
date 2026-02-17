"""
Crystal Plasticity Constitutive Model for Polycrystal FFT Solver
=================================================================
Bridges the crystal plasticity flow rule with the FFT-based 
homogenization solver for full-field polycrystal simulations.

Implements:
  - Elasto-viscoplastic constitutive update (vectorized over all voxels)
  - Rotation of slip systems by grain orientation
  - Hardening law (Voce-type)
  - Power-law viscoplastic flow rule (Lebensohn 2001)
  - Incremental strain decomposition: ε = εᵉ + εᵖ
  - State variable management (slip resistances, accumulated slip)
  - GPU-accelerated EVPFFT solver

References:
  Lebensohn (2001) — VPFFT formulation
  Lebensohn et al. (2012) — EVPFFT formulation
  Peirce, Asaro, Needleman (1983) — Rate-dependent CP
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
    """Rotate the 12 FCC slip systems by rotation matrix R."""
    normals = (R @ SLIP_NORMALS_REF.T).T  # (12, 3)
    dirs = (R @ SLIP_DIRS_REF.T).T        # (12, 3)
    return normals, dirs


def schmid_tensors(normals, dirs):
    """
    Compute symmetric Schmid tensors P^α = 0.5*(s⊗n + n⊗s) in Voigt.
    Returns P_voigt: (12, 6) — each row is Voigt [P11,P22,P33,P23,P13,P12].
    """
    P_voigt = np.zeros((N_SLIP, 6))
    for alpha in range(N_SLIP):
        s = dirs[alpha]
        n = normals[alpha]
        sn = np.outer(s, n)
        P = 0.5 * (sn + sn.T)
        P_voigt[alpha] = [P[0,0], P[1,1], P[2,2],
                          P[1,2], P[0,2], P[0,1]]
    return P_voigt


# ============================================================================
# Voce Hardening Law
# ============================================================================

class VoceHardening:
    """
    Voce-type hardening for slip resistance evolution.
    
    ṡ^α = Σ_β h_{αβ} |γ̇^β|
    h_{αβ} = q_{αβ} · [h₁ + (h₀ − h₁) · max(0, 1 − s^β/s_s)]
    q_αα = 1 (self), q_αβ = q_lat (latent)
    
    When h1=0 this reduces to the standard simplified Voce law.
    When h1>0 it matches the extended Voce used by Fierro/Lebensohn
    with asymptotic hardening rate θ₁.
    """
    
    def __init__(self, tau0=50e6, tau_s=200e6, h0=500e6, q_latent=1.4,
                 h1=0.0):
        self.tau0 = tau0
        self.tau_s = tau_s
        self.h0 = h0
        self.h1 = h1        # asymptotic hardening rate (θ₁)
        self.q_latent = q_latent
    
    def initial_resistance(self):
        """Return initial slip resistance for all 12 systems."""
        return np.full(N_SLIP, self.tau0)
    
    def update_vectorized(self, slip_resistance, delta_gamma):
        """
        Vectorized update of slip resistance for all voxels.
        
        Parameters:
            slip_resistance : (n_vox, 12) current resistance
            delta_gamma     : (n_vox, 12) slip increments (signed)
        Returns:
            new_resistance  : (n_vox, 12) updated resistance
        """
        abs_dg = np.abs(delta_gamma)  # (n_vox, 12)
        
        # h_β = h₁ + (h₀ − h₁) · max(0, 1 − s^β / s_s)
        h_beta = self.h1 + (self.h0 - self.h1) * np.maximum(
            0, 1.0 - slip_resistance / self.tau_s)
        # (n_vox, 12)
        
        # q_{αβ} · h_β · |Δγ^β|  summed over β
        # Self-hardening (α = β): q = 1 → contribution = h_β · |Δγ^β|
        # Latent (α ≠ β): q = q_lat → contribution = q_lat · h_β · |Δγ^β|
        
        # Total rate = q_lat·Σ_β(h_β·|Δγ^β|) + (1−q_lat)·h_α·|Δγ^α|
        total_sum = np.sum(h_beta * abs_dg, axis=1, keepdims=True)  # (n_vox, 1)
        self_term = h_beta * abs_dg  # (n_vox, 12)
        
        hardening_inc = self.q_latent * total_sum + (1.0 - self.q_latent) * self_term
        
        return slip_resistance + hardening_inc


# ============================================================================
# Vectorized OTIS Thermally-Activated Flow Rule
# (matches physics_engine.compute_flow_rule but numpy-vectorized)
# ============================================================================

def compute_slip_powerlaw(rss, slip_resistance, dt,
                          gamma_dot_0=1.0, n_exponent=10.0,
                          max_dgamma=0.01):
    """
    Power-law viscoplastic flow rule (Lebensohn 2001, Fierro EVPFFT).

    Δγ^α = γ̇₀ · |τ^α / g^α|^n · sign(τ^α) · dt

    Parameters
    ----------
    rss             : (n_vox, 12) resolved shear stress (Pa)
    slip_resistance : (n_vox, 12) slip resistance / CRSS (Pa)
    dt              : float  time increment
    gamma_dot_0     : float  reference slip rate (s⁻¹) [default 1.0]
    n_exponent      : float  rate sensitivity exponent [default 10]
    max_dgamma      : float  clamp for numerical safety

    Returns
    -------
    dgamma : (n_vox, 12) slip increments
    """
    sign_tau = np.sign(rss)
    tau_abs = np.abs(rss)
    s = np.maximum(np.abs(slip_resistance), 1e-10)

    ratio = tau_abs / s
    # |τ/g|^n  (clamp ratio to avoid overflow for large n)
    ratio_clamped = np.minimum(ratio, 100.0)
    rate = gamma_dot_0 * ratio_clamped ** n_exponent

    dgamma = rate * sign_tau * dt
    return np.clip(dgamma, -max_dgamma, max_dgamma)


def compute_slip_otis(rss, slip_resistance, temperature, dt,
                      rate_slip_ref=1e7, Q_activation=9.5e-19,
                      p_barrier=0.78, q_barrier=1.15,
                      max_dgamma=0.01):
    """
    Vectorised OTIS thermally-activated flow rule.

    Δγ^α = γ̇₀_ref · exp(-Q/kT · (1 - (|τ|/s)^p)^q) · sign(τ) · dt

    Parameters
    ----------
    rss             : (n_vox, 12) resolved shear stress (Pa)
    slip_resistance : (n_vox, 12) thermal resistance (Pa)
    temperature     : float  (K)
    dt              : float  time increment
    rate_slip_ref   : float  reference slip rate  (s⁻¹)  [default 1e7]
    Q_activation    : float  activation energy    (J)    [default 9.5e-19]
    p_barrier       : float  barrier shape p             [default 0.78]
    q_barrier       : float  barrier shape q             [default 1.15]
    max_dgamma      : float  clamp for numerical safety

    Returns
    -------
    dgamma : (n_vox, 12) slip increments
    """
    kT = 1.3806e-23 * max(temperature, 1.0)

    sign_tau = np.sign(rss)
    tau_abs = np.abs(rss)
    s = np.maximum(np.abs(slip_resistance), 1e-10)

    ratio = tau_abs / s

    dgamma = np.zeros_like(rss)

    active = tau_abs > 1e-10

    # Saturated: |τ| ≥ s  →  max slip rate
    saturated = active & (ratio >= 1.0)
    dgamma[saturated] = rate_slip_ref * sign_tau[saturated] * dt

    # Thermally activated: 0 < ratio < 1
    normal = active & (ratio < 1.0)
    if np.any(normal):
        r = ratio[normal]
        bracket = (1.0 - r ** p_barrier) ** q_barrier
        exponent = np.maximum(-Q_activation / kT * bracket, -50.0)
        dgamma[normal] = rate_slip_ref * np.exp(exponent) * sign_tau[normal] * dt

    return np.clip(dgamma, -max_dgamma, max_dgamma)


# ============================================================================
# Material state (per voxel)
# ============================================================================

class CrystalPlasticityState:
    """
    Vectorized state for polycrystalline EVPFFT.
    
    Pre-computes per-grain Schmid tensors in Voigt form for fast
    resolved shear stress and plastic strain computation.
    """
    
    def __init__(self, grain_ids, euler_angles, hardening=None,
                 C11=168.4e9, C12=121.4e9, C44=75.4e9,
                 temperature=293.0,
                 rate_slip_ref=1e7, Q_activation=9.5e-19,
                 p_barrier=0.78, q_barrier=1.15,
                 flow_rule='otis', gamma_dot_0=1.0, n_exponent=10.0):
        """
        Parameters:
            grain_ids     : (N,N,N) int array
            euler_angles  : (n_grains, 3) Euler angles
            hardening     : VoceHardening instance
            temperature   : temperature in K (for OTIS flow rule)
            rate_slip_ref : reference slip rate γ̇₀ (s⁻¹)
            Q_activation  : activation energy (J)
            p_barrier     : barrier shape parameter p
            q_barrier     : barrier shape parameter q
            flow_rule     : 'otis' or 'powerlaw'
            gamma_dot_0   : power-law reference slip rate (s⁻¹)
            n_exponent    : power-law rate sensitivity exponent
        """
        self.grain_ids = grain_ids
        self.shape = grain_ids.shape
        N = grain_ids.shape[0]
        self.n_voxels = N**3
        self.n_grains = euler_angles.shape[0]
        self.C11, self.C12, self.C44 = C11, C12, C44

        # OTIS flow rule parameters
        self.temperature = temperature
        self.rate_slip_ref = rate_slip_ref
        self.Q_activation = Q_activation
        self.p_barrier = p_barrier
        self.q_barrier = q_barrier

        # Power-law flow rule parameters
        self.flow_rule = flow_rule      # 'otis' or 'powerlaw'
        self.gamma_dot_0 = gamma_dot_0  # reference slip rate (s⁻¹)
        self.n_exponent = n_exponent    # rate sensitivity exponent
        
        if hardening is None:
            hardening = VoceHardening()
        self.hardening = hardening
        
        # Pre-compute Schmid tensors in Voigt: (n_grains, 12, 6)
        self.P_voigt = np.zeros((self.n_grains, N_SLIP, 6))
        self.C_voigt = np.zeros((self.n_grains, 6, 6))
        C_base = cubic_stiffness_tensor(C11, C12, C44)
        
        for g in range(self.n_grains):
            R = euler_to_rotation(*euler_angles[g])
            normals, dirs = rotate_slip_systems(R)
            self.P_voigt[g] = schmid_tensors(normals, dirs)
            self.C_voigt[g] = rotate_stiffness_voigt(C_base, R)
        
        # Map grain-level data to voxel-level using grain_ids
        gids_flat = grain_ids.ravel()
        self.P_vox = self.P_voigt[gids_flat]        # (n_vox, 12, 6)
        self.C_vox = self.C_voigt[gids_flat]         # (n_vox, 6, 6)
        
        # State variables (all voxels × 12 slip systems)
        self.slip_resistance = np.full((self.n_voxels, N_SLIP), hardening.tau0)
        self.accumulated_slip = np.zeros((self.n_voxels, N_SLIP))
        self.plastic_strain = np.zeros((self.n_voxels, 6))  # Voigt
    
    def get_stiffness_field(self):
        """Return (N,N,N,6,6) stiffness field."""
        return self.C_voigt[self.grain_ids]
    
    def resolved_shear_stress(self, stress_voigt_flat):
        """
        Vectorized RSS computation.
        
        Parameters:
            stress_voigt_flat : (n_vox, 6) stress in Voigt
        Returns:
            rss : (n_vox, 12) resolved shear stress on each system
        """
        # RSS^α = P^α : σ  (contracted in Voigt with proper weights)
        # P_voigt is [P11,P22,P33,P23,P13,P12]
        # σ_voigt is [σ11,σ22,σ33,σ23,σ13,σ12]
        # RSS = P11·σ11 + P22·σ22 + P33·σ33 + 2·P23·σ23 + 2·P13·σ13 + 2·P12·σ12
        weights = np.array([1, 1, 1, 2, 2, 2], dtype=float)
        rss = np.einsum('va,va->v', 
                        stress_voigt_flat[:, None, :] * weights,
                        self.P_vox[:, :, :].reshape(-1, 6)).reshape(-1, N_SLIP)
        # Above is not quite right, let me do it properly:
        rss = np.einsum('vas,vs->va', self.P_vox, 
                        stress_voigt_flat * weights)
        return rss
    
    def compute_slip_rates(self, rss, dt=1.0):
        """
        Flow rule dispatch (vectorized).

        Supports:
          'otis'     : OTIS thermally-activated
          'powerlaw' : Power-law |τ/g|^n (Lebensohn 2001)

        Parameters:
            rss : (n_vox, 12) resolved shear stress
            dt  : time increment
        Returns:
            delta_gamma : (n_vox, 12) slip increments
        """
        if self.flow_rule == 'powerlaw':
            return compute_slip_powerlaw(
                rss, self.slip_resistance, dt,
                gamma_dot_0=self.gamma_dot_0,
                n_exponent=self.n_exponent)
        else:
            return compute_slip_otis(
                rss, self.slip_resistance, self.temperature, dt,
                rate_slip_ref=self.rate_slip_ref,
                Q_activation=self.Q_activation,
                p_barrier=self.p_barrier,
                q_barrier=self.q_barrier)
    
    def plastic_strain_increment(self, delta_gamma):
        """
        Δεᵖ = Σ_α Δγ^α · P^α   (vectorized).

        Returns engineering-shear Voigt format [ε11,ε22,ε33,γ23,γ13,γ12]
        to match the FFT solver convention (γ_ij = 2·ε_ij for shear).

        Parameters:
            delta_gamma : (n_vox, 12) slip increments
        Returns:
            deps_p : (n_vox, 6) plastic strain increment (Voigt, eng. shear)
        """
        # P_vox stores tensor-shear Schmid components
        deps_p = np.einsum('va,vas->vs', delta_gamma, self.P_vox)
        # Convert shear components to engineering shear (×2)
        deps_p[:, 3:] *= 2
        return deps_p
    
    def constitutive_update(self, eps_voigt_flat, dt=1.0):
        """
        Single-pass stress evaluation using the OTIS flow rule.
        Does NOT mutate state.

        Given total strain ε, computes stress using the frozen plastic state:
            σ = C:(ε − εᵖ_frozen − Δεᵖ(σ_trial))
        The slip increment Δγ is computed from the OTIS thermally-activated
        flow rule (physics_engine.compute_flow_rule).

        Parameters:
            eps_voigt_flat : (n_vox, 6) total strain
            dt             : time increment
        Returns:
            sig : (n_vox, 6) stress
            delta_gamma : (n_vox, 12) slip increments
        """
        weights = np.array([1, 1, 1, 2, 2, 2], dtype=float)

        # Trial stress from frozen plastic strain
        e_trial = eps_voigt_flat - self.plastic_strain
        sig_trial = np.einsum('vij,vj->vi', self.C_vox, e_trial)

        # Resolved shear stress from trial stress
        rss = np.einsum('vas,vs->va', self.P_vox, sig_trial * weights)

        # Slip increments — dispatch to active flow rule
        if self.flow_rule == 'powerlaw':
            dgamma = compute_slip_powerlaw(
                rss, self.slip_resistance, dt,
                gamma_dot_0=self.gamma_dot_0,
                n_exponent=self.n_exponent,
                max_dgamma=0.005)
        else:
            dgamma = compute_slip_otis(
                rss, self.slip_resistance, self.temperature, dt,
                rate_slip_ref=self.rate_slip_ref,
                Q_activation=self.Q_activation,
                p_barrier=self.p_barrier,
                q_barrier=self.q_barrier,
                max_dgamma=0.005)

        # Plastic strain increment (engineering-shear Voigt, NOT accumulated)
        deps_p = np.einsum('va,vas->vs', dgamma, self.P_vox)
        deps_p[:, 3:] *= 2  # tensor → engineering shear

        # Stress consistent with plastic correction
        sig = np.einsum('vij,vj->vi', self.C_vox, e_trial - deps_p)

        return sig, dgamma

    def commit_state(self, delta_gamma):
        """Permanently update internal state after a converged increment."""
        deps_p = self.plastic_strain_increment(delta_gamma)
        self.plastic_strain += deps_p
        self.slip_resistance = self.hardening.update_vectorized(
            self.slip_resistance, delta_gamma)
        self.accumulated_slip += np.abs(delta_gamma)
    
    def get_total_accumulated_slip_field(self):
        """Return total accumulated slip per voxel, reshaped to (N,N,N)."""
        return np.sum(self.accumulated_slip, axis=1).reshape(self.shape)
    
    def get_von_mises_plastic_strain_field(self):
        """Return equivalent plastic strain per voxel, reshaped to (N,N,N).

        plastic_strain is stored in engineering-shear Voigt
        [εp11, εp22, εp33, γp23, γp13, γp12] where γ=2ε.
        """
        ep = self.plastic_strain
        e11, e22, e33 = ep[:, 0], ep[:, 1], ep[:, 2]
        # Convert engineering shear back to tensor shear for VM formula
        e23, e13, e12 = ep[:, 3] / 2, ep[:, 4] / 2, ep[:, 5] / 2
        vm = np.sqrt(2.0 / 3.0 * ((e11 - e22)**2 + (e22 - e33)**2 +
                                    (e33 - e11)**2 +
                                    6 * (e23**2 + e13**2 + e12**2)))
        return vm.reshape(self.shape)
    
    def get_slip_resistance_field(self):
        """Return mean slip resistance per voxel (N,N,N) in MPa."""
        return np.mean(self.slip_resistance, axis=1).reshape(self.shape) / 1e6


# ============================================================================
# Vectorized EVPFFT Solver
# ============================================================================

def solve_evpfft(state, S_macro_voigt, n_increments=10, dt=0.1,
                 tol_fft=1e-5, tol_stress=1e-4,
                 max_iter_fft=100, max_iter_stress=30,
                 verbose=True, callback=None,
                 derivative_scheme='continuous', ref_medium_mode='mean'):
    """
    Elasto-Viscoplastic FFT solver (Lebensohn 2001 / 2012).
    
    Stress-controlled: ramps macroscopic stress from 0 → S_macro in
    n_increments steps.  At each step, iterates between FFT equilibrium
    and the crystal plasticity constitutive update.
    
    Parameters:
        state          : CrystalPlasticityState
        S_macro_voigt  : (6,) target macroscopic stress (Pa, Voigt)
        n_increments   : number of load increments
        dt             : time per increment (for rate-dependent flow)
        tol_fft        : strain equilibrium tolerance
        tol_stress     : macroscopic stress tolerance
        max_iter_fft   : max FFT iterations per increment
        max_iter_stress: max Newton iterations for stress control
        verbose        : print progress
        callback       : callable(increment, eps_field, sig_field, state)
        derivative_scheme : 'continuous' | 'finite_difference' | 'rotated'
        ref_medium_mode   : 'mean' | 'contrast_aware'
        
    Returns:
        history : list of (eps_field, sig_field, frac) per increment
    """
    from fft_solver import (build_green_data, apply_green_operator_tensor,
                            fft_3x3, ifft_3x3, _strain_change,
                            apply_stiffness, strain_field_v2t, stress_field_v2t,
                            strain_field_t2v, stress_field_t2v, strain_v2t,
                            compute_reference_medium, von_mises_stress,
                            _to_gpu, _to_cpu, HAS_GPU)
    
    shape = state.shape
    N = shape[0]
    n_vox = state.n_voxels
    
    # Initialize history snapshots list for per-step plastic fields
    state._history_snapshots = []
    
    # Stiffness field & reference medium
    C_field = state.get_stiffness_field()
    C0, lam0, mu0 = compute_reference_medium(C_field, mode=ref_medium_mode)
    C0_6x6 = C0.copy()
    
    # Green's operator data
    use_gpu = HAS_GPU
    n_field, Ainv = build_green_data(N, lam0, mu0, on_gpu=use_gpu,
                                      derivative_scheme=derivative_scheme)
    
    # Initialize fields
    eps_field = np.zeros(shape + (6,))
    sig_field = np.zeros(shape + (6,))
    
    # Compliance for initial strain guess (isotropic approximation)
    S0_inv = np.linalg.inv(C0_6x6)
    
    history = []
    
    for inc in range(1, n_increments + 1):
        frac = inc / n_increments
        S_target = S_macro_voigt * frac
        
        if verbose:
            s_mpa = S_target / 1e6
            print(f"\nEVPFFT Step {inc}/{n_increments}  "
                  f"σ_target (MPa): [{s_mpa[0]:.0f}, {s_mpa[1]:.0f}, "
                  f"{s_mpa[2]:.0f}, {s_mpa[3]:.0f}, {s_mpa[4]:.0f}, {s_mpa[5]:.0f}]")
        
        # --- Strain-controlled target for this step ---
        E_target = S0_inv @ S_target
        E_bar = E_target.copy()

        # Initial strain field
        if inc == 1:
            eps_field[:] = E_bar
        else:
            # Shift from previous converged state
            E_prev = np.mean(eps_field.reshape(-1, 6), axis=0)
            eps_field += (E_bar - E_prev)

        last_delta_gamma = None

        # Secant compliance (updated after first Newton step)
        S_eff = S0_inv.copy()
        E_bar_prev = None
        sig_mean_prev = None

        # Cache cupy zero once
        _gpu_zero = None
        if use_gpu:
            import cupy as _cp
            _gpu_zero = _cp.zeros((3, 3))

        C0_bc = C0_6x6[np.newaxis, np.newaxis, np.newaxis]

        # --- Newton loop for stress control ---
        for newton in range(max_iter_stress):
            # Limit inner FFT iters: fewer early on, more near convergence
            inner_cap = min(max_iter_fft, 5 if newton < 3 else 15)

            # --- Inner FFT equilibrium loop ---
            for fft_iter in range(inner_cap):
                eps_old_t = strain_field_v2t(eps_field)

                # Constitutive response (does NOT mutate state)
                eps_flat = eps_field.reshape(-1, 6)
                sig_flat, last_delta_gamma = state.constitutive_update(
                    eps_flat, dt=dt)
                sig_field = sig_flat.reshape(shape + (6,))

                # Polarization: τ = σ − C⁰:ε
                sig_tensor = stress_field_v2t(sig_field)
                sig0_tensor = apply_stiffness(C0_bc, eps_old_t)
                tau_tensor = sig_tensor - sig0_tensor

                # FFT Green's operator
                if use_gpu:
                    tau_g = _to_gpu(tau_tensor)
                    tau_hat = fft_3x3(tau_g)
                    gamma_tau_hat = apply_green_operator_tensor(
                        tau_hat, n_field, Ainv)
                    gamma_tau_hat[0, 0, 0, :, :] = _gpu_zero
                    gamma_tau = _to_cpu(ifft_3x3(gamma_tau_hat))
                else:
                    tau_hat = fft_3x3(tau_tensor)
                    gamma_tau_hat = apply_green_operator_tensor(
                        tau_hat, n_field, Ainv)
                    gamma_tau_hat[0, 0, 0, :, :] = 0.0
                    gamma_tau = ifft_3x3(gamma_tau_hat)

                # Lippmann-Schwinger: ε = Ē − Γ⁰:τ
                E_bar_t = strain_v2t(E_bar)
                eps_new_tensor = E_bar_t - gamma_tau
                eps_field = strain_field_t2v(eps_new_tensor)

                # Check convergence
                err_fft = _strain_change(
                    strain_field_v2t(eps_field), eps_old_t)
                if err_fft < tol_fft:
                    break

            # Macroscopic stress check
            sig_mean = np.mean(sig_field.reshape(-1, 6), axis=0)
            stress_err = np.linalg.norm(sig_mean - S_target) / \
                         max(np.linalg.norm(S_target), 1e-10)

            if verbose:
                print(f"  Newton {newton}: |ΔS|/|S|={stress_err:.4e}  "
                      f"<σ₁₁>={sig_mean[0]/1e6:.1f} MPa  "
                      f"(FFT: {fft_iter+1} iters, Δε={err_fft:.2e})")

            if stress_err < tol_stress:
                if verbose:
                    print(f"  Converged at Newton iter {newton}")
                break

            # Secant update for effective compliance
            if E_bar_prev is not None and sig_mean_prev is not None:
                dE = E_bar - E_bar_prev
                dS = sig_mean - sig_mean_prev
                dS_norm2 = np.dot(dS, dS)
                if dS_norm2 > 1e-30:
                    # Broyden-type rank-1 update
                    S_eff += np.outer(dE - S_eff @ dS, dS) / dS_norm2

            E_bar_prev = E_bar.copy()
            sig_mean_prev = sig_mean.copy()

            # Full Newton correction with secant compliance
            dS_corr = S_target - sig_mean
            dE_corr = S_eff @ dS_corr
            E_bar += dE_corr
            eps_field += dE_corr
        
        # --- Commit state update after converged increment ---
        if last_delta_gamma is not None:
            state.commit_state(last_delta_gamma)
        
        # Store results
        history.append((eps_field.copy(), sig_field.copy(), frac))
        
        # Snapshot plastic fields for visualization
        state._history_snapshots.append({
            'ep_eq': state.get_von_mises_plastic_strain_field().copy(),
            'acc_slip': state.get_total_accumulated_slip_field().copy(),
            'slip_res': state.get_slip_resistance_field().copy(),
        })
        
        if callback:
            callback(inc, eps_field, sig_field, state)
    
    return history


# ============================================================================
# Mixed-BC EVPFFT Solver (strain-rate + stress controlled)
# ============================================================================

def solve_evpfft_mixed(state, D_macro, sig_target,
                       strain_mask, stress_mask,
                       n_increments=30, dt=0.00005,
                       tol_fft=1e-5, tol_stress=1e-7,
                       max_iter_fft=50, max_iter_stress=30,
                       verbose=True, callback=None,
                       derivative_scheme='continuous',
                       ref_medium_mode='mean'):
    """
    Mixed-boundary-condition EVPFFT solver (Lebensohn 2001).

    Supports mixed loading where some strain-rate components are prescribed
    and the complementary stress components are prescribed. This is the
    standard formulation used in VPFFT/EVPFFT codes (Fierro, DAMASK).

    Example — uniaxial tension in z:
        D_macro    = [0, 0, 1.0, 0, 0, 0]   (rate of deformation, Voigt)
        sig_target = [0, 0, 0,   0, 0, 0]   (target stress, Pa)
        strain_mask = [False, False, True, True, True, True]
            → ε̇₃₃=1.0, γ̇₂₃=γ̇₁₃=γ̇₁₂=0  prescribed
        stress_mask = [True, True, False, False, False, False]
            → σ₁₁=0, σ₂₂=0  prescribed (traction-free)

    Parameters
    ----------
    state           : CrystalPlasticityState
    D_macro         : (6,) macroscopic strain-rate (Voigt, eng. shear)
    sig_target      : (6,) target macroscopic stress for stress-controlled components
    strain_mask     : (6,) bool — True where strain-rate is prescribed
    stress_mask     : (6,) bool — True where stress is prescribed
    n_increments    : int
    dt              : float — time increment per step
    tol_fft         : float — FFT equilibrium tolerance
    tol_stress      : float — stress convergence tolerance (relative)
    max_iter_fft    : int
    max_iter_stress : int
    verbose         : bool
    callback        : callable(inc, eps_field, sig_field, state)
    derivative_scheme : str
    ref_medium_mode   : str

    Returns
    -------
    history : list of dicts with keys 'eps_field', 'sig_field', 'sig_macro',
              'eps_macro', 'step'
    """
    from fft_solver import (build_green_data, apply_green_operator_tensor,
                            fft_3x3, ifft_3x3, _strain_change,
                            apply_stiffness, strain_field_v2t, stress_field_v2t,
                            strain_field_t2v, stress_field_t2v, strain_v2t,
                            compute_reference_medium, von_mises_stress,
                            _to_gpu, _to_cpu, HAS_GPU)

    strain_mask = np.asarray(strain_mask, dtype=bool)
    stress_mask = np.asarray(stress_mask, dtype=bool)
    D_macro = np.asarray(D_macro, dtype=float)
    sig_target = np.asarray(sig_target, dtype=float)

    shape = state.shape
    N = shape[0]
    n_vox = state.n_voxels

    # Initialize history
    state._history_snapshots = []

    # Stiffness field & reference medium
    C_field = state.get_stiffness_field()
    C0, lam0, mu0 = compute_reference_medium(C_field, mode=ref_medium_mode)
    C0_6x6 = C0.copy()

    # Green's operator
    use_gpu = HAS_GPU
    n_field, Ainv = build_green_data(N, lam0, mu0, on_gpu=use_gpu,
                                      derivative_scheme=derivative_scheme)

    # Compliance for initial guesses
    S0_inv = np.linalg.inv(C0_6x6)

    # Initialize fields
    eps_field = np.zeros(shape + (6,))
    sig_field = np.zeros(shape + (6,))

    # Indices for unknown strain and known stress
    idx_strain_unknown = np.where(stress_mask)[0]   # these strain components float
    idx_stress_known   = np.where(stress_mask)[0]   # these stress targets are prescribed

    # Cumulative macroscopic strain
    E_bar = np.zeros(6)

    # Track previous step's converged macroscopic quantities for extrapolation
    prev_E_bar = None
    prev_sig_mean = None

    _gpu_zero = None
    if use_gpu:
        import cupy as _cp
        _gpu_zero = _cp.zeros((3, 3))

    C0_bc = C0_6x6[np.newaxis, np.newaxis, np.newaxis]

    history = []

    for inc in range(1, n_increments + 1):
        # Prescribed strain increment for known components
        dE_prescribed = D_macro * dt

        # Update macroscopic strain (prescribed components)
        E_bar[strain_mask] += dE_prescribed[strain_mask]

        # Estimate unknown strain components for this step
        if inc == 1:
            # First step: use Poisson estimate
            E_bar[idx_strain_unknown] = -0.35 * np.max(
                np.abs(E_bar[strain_mask]))
            # If target stress has value, use compliance
            for k, idx in enumerate(idx_stress_known):
                if abs(sig_target[idx]) > 1e-10:
                    E_bar[idx_strain_unknown] += S0_inv[
                        np.ix_(idx_strain_unknown, [idx])] @ [sig_target[idx]]
        else:
            # Extrapolate unknown strains from previous step's ratio
            # ε_unknown(n+1) ≈ ε_unknown(n) + dε_unknown/dε_known × dε_known
            if prev_E_bar is not None:
                # Scale unknown strains proportionally to prescribed increment
                for i_unk in idx_strain_unknown:
                    # Estimate from previous Poisson-like ratio
                    best_known = 2  # Use ε₃₃ (primary loading direction)
                    if abs(prev_E_bar[best_known]) > 1e-20:
                        ratio = prev_E_bar[i_unk] / prev_E_bar[best_known]
                        E_bar[i_unk] = prev_E_bar[i_unk] + \
                            ratio * dE_prescribed[best_known]

        # Apply strain increment to field
        if inc == 1:
            eps_field[:] = E_bar
        else:
            eps_flat_mean = np.mean(eps_field.reshape(-1, 6), axis=0)
            eps_field += (E_bar - eps_flat_mean)

        if verbose:
            print(f"\nEVPFFT Mixed-BC Step {inc}/{n_increments}  "
                  f"dt={dt:.2e}  E_bar_33={E_bar[2]:.6e}")

        last_delta_gamma = None

        # Secant compliance for this increment (updated during Newton)
        S_eff = S0_inv.copy()
        E_bar_newton_prev = None
        sig_newton_prev = None

        # --- Newton loop for stress control on unknown strain components ---
        for newton in range(max_iter_stress):
            inner_cap = min(max_iter_fft, 10 if newton < 3 else max_iter_fft)

            # --- Inner FFT equilibrium loop ---
            for fft_iter in range(inner_cap):
                eps_old_t = strain_field_v2t(eps_field)

                # Constitutive response
                eps_flat = eps_field.reshape(-1, 6)
                sig_flat, last_delta_gamma = state.constitutive_update(
                    eps_flat, dt=dt)
                sig_field = sig_flat.reshape(shape + (6,))

                # Polarization
                sig_tensor = stress_field_v2t(sig_field)
                sig0_tensor = apply_stiffness(C0_bc, eps_old_t)
                tau_tensor = sig_tensor - sig0_tensor

                # FFT Green's operator
                if use_gpu:
                    tau_g = _to_gpu(tau_tensor)
                    tau_hat = fft_3x3(tau_g)
                    gamma_tau_hat = apply_green_operator_tensor(
                        tau_hat, n_field, Ainv)
                    gamma_tau_hat[0, 0, 0, :, :] = _gpu_zero
                    gamma_tau = _to_cpu(ifft_3x3(gamma_tau_hat))
                else:
                    tau_hat = fft_3x3(tau_tensor)
                    gamma_tau_hat = apply_green_operator_tensor(
                        tau_hat, n_field, Ainv)
                    gamma_tau_hat[0, 0, 0, :, :] = 0.0
                    gamma_tau = ifft_3x3(gamma_tau_hat)

                # Lippmann-Schwinger
                E_bar_t = strain_v2t(E_bar)
                eps_new_tensor = E_bar_t - gamma_tau
                eps_field = strain_field_t2v(eps_new_tensor)

                err_fft = _strain_change(
                    strain_field_v2t(eps_field), eps_old_t)
                if err_fft < tol_fft:
                    break

            # Macroscopic stress
            sig_mean = np.mean(sig_field.reshape(-1, 6), axis=0)

            # Check stress convergence on controlled components
            if len(idx_stress_known) > 0:
                stress_residual = sig_mean[idx_stress_known] - \
                                  sig_target[idx_stress_known]
                stress_norm = max(np.linalg.norm(sig_mean), 1e-10)
                stress_err = np.linalg.norm(stress_residual) / stress_norm
            else:
                stress_err = 0.0

            if verbose:
                print(f"  Newton {newton}: |Δσ|/|σ|={stress_err:.4e}  "
                      f"<σ₃₃>={sig_mean[2]/1e6:.2f} MPa  "
                      f"<σ₁₁>={sig_mean[0]/1e6:.2f}  "
                      f"<σ₂₂>={sig_mean[1]/1e6:.2f}  "
                      f"(FFT: {fft_iter+1} iters, Δε={err_fft:.2e})")

            if stress_err < tol_stress:
                if verbose:
                    print(f"  Converged at Newton iter {newton}")
                break

            # Newton correction: adjust unknown strain to reduce stress error
            if len(idx_strain_unknown) > 0:
                # Secant update (Broyden rank-1)
                if E_bar_newton_prev is not None and \
                   sig_newton_prev is not None:
                    dE = E_bar[idx_strain_unknown] - E_bar_newton_prev
                    dS = sig_mean[idx_stress_known] - sig_newton_prev
                    dS_norm2 = np.dot(dS, dS)
                    if dS_norm2 > 1e-30:
                        S_sub_eff = S_eff[np.ix_(idx_strain_unknown,
                                                  idx_stress_known)]
                        S_sub_eff += np.outer(
                            dE - S_sub_eff @ dS, dS) / dS_norm2
                        S_eff[np.ix_(idx_strain_unknown,
                                     idx_stress_known)] = S_sub_eff

                E_bar_newton_prev = E_bar[idx_strain_unknown].copy()
                sig_newton_prev = sig_mean[idx_stress_known].copy()

                # Correction from effective compliance
                S_sub = S_eff[np.ix_(idx_strain_unknown, idx_stress_known)]
                dE_corr = S_sub @ (-stress_residual)

                # Damping to prevent oscillation
                alpha = 0.8 if newton < 5 else 1.0
                dE_corr *= alpha

                E_bar[idx_strain_unknown] += dE_corr
                # Broadcast to full 6-component and add to field
                dE_full = np.zeros(6)
                dE_full[idx_strain_unknown] = dE_corr
                eps_field += dE_full

        # Store converged E_bar for next step extrapolation
        prev_E_bar = E_bar.copy()
        prev_sig_mean = sig_mean.copy()

        # Commit state
        if last_delta_gamma is not None:
            state.commit_state(last_delta_gamma)

        # Record macroscopic quantities
        sig_mean = np.mean(sig_field.reshape(-1, 6), axis=0)
        eps_mean = np.mean(eps_field.reshape(-1, 6), axis=0)

        history.append({
            'eps_field': eps_field.copy(),
            'sig_field': sig_field.copy(),
            'sig_macro': sig_mean.copy(),
            'eps_macro': eps_mean.copy(),
            'step': inc,
        })

        # Snapshot plastic fields
        state._history_snapshots.append({
            'ep_eq': state.get_von_mises_plastic_strain_field().copy(),
            'acc_slip': state.get_total_accumulated_slip_field().copy(),
            'slip_res': state.get_slip_resistance_field().copy(),
        })

        if verbose:
            print(f"  → σ₃₃ = {sig_mean[2]/1e6:.2f} MPa,  "
                  f"ε₃₃ = {eps_mean[2]:.6e},  "
                  f"ε₁₁ = {eps_mean[0]:.6e},  ε₂₂ = {eps_mean[1]:.6e}")

        if callback:
            callback(inc, eps_field, sig_field, state)

    return history


# ============================================================================
# Quick test
# ============================================================================

if __name__ == '__main__':
    from microstructure import generate_voronoi_microstructure
    
    print("Crystal Plasticity EVPFFT test...")
    
    N = 8
    n_grains = 4
    grain_ids, centers, euler_angles = generate_voronoi_microstructure(N, n_grains, seed=42)
    
    # Create state with plasticity
    hardening = VoceHardening(tau0=50e6, tau_s=200e6, h0=500e6)
    state = CrystalPlasticityState(grain_ids, euler_angles, hardening=hardening)
    
    print(f"  Grid: {state.shape}")
    print(f"  Grains: {state.n_grains}")
    print(f"  Initial slip resistance: {state.slip_resistance[0, 0]/1e6:.1f} MPa")
    
    # Run EVPFFT
    S_macro = np.array([200e6, 0, 0, 0, 0, 0])  # 200 MPa uniaxial
    history = solve_evpfft(state, S_macro, n_increments=3, dt=0.1,
                           tol_fft=1e-4, tol_stress=1e-3,
                           max_iter_fft=50, verbose=True)
    
    from fft_solver import von_mises_stress
    eps_final, sig_final, _ = history[-1]
    vm = von_mises_stress(sig_final)
    print(f"\n  Max VM stress: {np.max(vm)/1e6:.1f} MPa")
    ep = state.get_von_mises_plastic_strain_field()
    print(f"  Max eq. plastic strain: {np.max(ep):.6f}")
    print(f"  Mean slip resistance: {np.mean(state.slip_resistance)/1e6:.1f} MPa")
    print("  Done.")
