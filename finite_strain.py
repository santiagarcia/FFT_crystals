"""
Finite-Strain Crystal Plasticity for FFT Polycrystal Simulations
================================================================
NASA-grade implementation of advanced crystal plasticity features:

  1. Finite-strain kinematics:  F = Fe · Fp  (multiplicative decomposition)
  2. Kocks–Mecking dislocation-density hardening  (replaces Voce)
  3. Armstrong–Frederick kinematic hardening  (back-stress / Bauschinger)
  4. Fully implicit per-voxel Newton–Raphson constitutive update
  5. GND hardening from Nye tensor  (non-local, via curl(Fp))
  6. Texture evolution tracking  (lattice rotation / Euler-angle update)

Stress measures:
  - 2nd Piola–Kirchhoff  S  in intermediate configuration
  - 1st Piola–Kirchhoff  P  in reference configuration  (for equilibrium)
  - Mandel stress  M = Ce·S  (for resolved shear stress)

References:
  Roters et al., Acta Mater. 58:1152-1211, 2010
  Eisenlohr et al., IJOP 46:37-53, 2013  (DAMASK)
  Lebensohn et al., MSMSE 20:024009, 2012  (EVPFFT)
  Kocks & Mecking, PTMS 48:171-292, 2003
  Armstrong & Frederick, CEGB Report RD/B/N731, 1966
  Arsenlis & Parks, JMPS 47:1597-1635, 1999  (GND)
  Nye, Acta Metall. 1:153-162, 1953
"""

import numpy as np
from numpy.linalg import inv, det, norm, solve, eigh
import numba as nb

from constitutive import (
    SLIP_NORMALS_REF, SLIP_DIRS_REF, N_SLIP,
    rotate_slip_systems, schmid_tensors,
    compute_slip_otis, VoceHardening,
)
from microstructure import (
    euler_to_rotation, cubic_stiffness_tensor,
    rotate_stiffness_voigt, random_orientations,
)

# Boltzmann constant (J/K)
KB = 1.3806e-23


# ============================================================================
#  0.  Numba-accelerated kernels  (compiled once, reused)
# ============================================================================

@nb.njit(cache=True, fastmath=True, parallel=True)
def _otis_kernel(rss, slip_resistance, kT, dt, rate_ref,
                 Q_act, p, q, max_dg):
    """Numba-JIT OTIS flow rule — ~10× faster than pure NumPy."""
    nv, ns = rss.shape
    dgamma = np.zeros((nv, ns), dtype=np.float64)
    QkT = Q_act / kT
    for v in nb.prange(nv):
        for a in range(ns):
            tau = rss[v, a]
            s = abs(slip_resistance[v, a])
            if s < 1e-10:
                s = 1e-10
            tau_abs = abs(tau)
            if tau_abs < 1e-10:
                continue
            sgn = 1.0 if tau > 0.0 else -1.0
            r = tau_abs / s
            if r >= 1.0:
                dg = rate_ref * sgn * dt
            else:
                rp = r ** p
                bracket = (1.0 - rp) ** q
                exponent = -QkT * bracket
                if exponent < -50.0:
                    exponent = -50.0
                dg = rate_ref * np.exp(exponent) * sgn * dt
            if dg > max_dg:
                dg = max_dg
            elif dg < -max_dg:
                dg = -max_dg
            dgamma[v, a] = dg
    return dgamma


@nb.njit(cache=True, fastmath=True, parallel=True)
def _otis_with_tangent_kernel(rss, slip_resistance, kT, dt, rate_ref,
                              Q_act, p, q, max_dg):
    """Numba-JIT OTIS flow rule returning Δγ AND ∂Δγ/∂τ."""
    nv, ns = rss.shape
    dgamma = np.zeros((nv, ns), dtype=np.float64)
    ddg_dtau = np.zeros((nv, ns), dtype=np.float64)
    QkT = Q_act / kT
    for v in nb.prange(nv):
        for a in range(ns):
            tau = rss[v, a]
            s = abs(slip_resistance[v, a])
            if s < 1e-10:
                s = 1e-10
            tau_abs = abs(tau)
            if tau_abs < 1e-10:
                continue
            sgn = 1.0 if tau > 0.0 else -1.0
            r = tau_abs / s
            if r >= 1.0 - 1e-12:
                dg = rate_ref * sgn * dt
                # tangent = 0 at saturation
            else:
                rp = r ** p
                bracket_q = (1.0 - rp) ** q
                exponent = -QkT * bracket_q
                if exponent < -50.0:
                    exponent = -50.0
                exp_val = np.exp(exponent)
                dg = rate_ref * exp_val * sgn * dt
                # tangent
                bracket_qm1 = (1.0 - rp) ** (q - 1.0) if (1.0 - rp) > 1e-30 else 0.0
                rp_m1 = r ** (p - 1.0) if r > 1e-30 else 0.0
                ddg_dtau[v, a] = rate_ref * dt * exp_val * QkT * p * q * rp_m1 / s * bracket_qm1
            if dg > max_dg:
                dg = max_dg
            elif dg < -max_dg:
                dg = -max_dg
            dgamma[v, a] = dg
    return dgamma, ddg_dtau


@nb.njit(cache=True, fastmath=True, parallel=True)
def _compute_rss_nb(M, normals_vox, dirs_vox):
    """Resolved shear stress for all 12 systems — Numba-parallel."""
    nv = M.shape[0]
    ns = normals_vox.shape[1]
    rss = np.empty((nv, ns), dtype=np.float64)
    for v in nb.prange(nv):
        for a in range(ns):
            val = 0.0
            for i in range(3):
                for j in range(3):
                    val += M[v, i, j] * dirs_vox[v, a, i] * normals_vox[v, a, j]
            rss[v, a] = val
    return rss


@nb.njit(cache=True, fastmath=True, parallel=True)
def _compute_dLp_nb(dgamma, dirs_vox, normals_vox):
    """Plastic velocity gradient  ΔLp = Σ_α Δγ^α (s^α ⊗ n^α) — Numba."""
    nv = dgamma.shape[0]
    ns = dgamma.shape[1]
    dLp = np.zeros((nv, 3, 3), dtype=np.float64)
    for v in nb.prange(nv):
        for a in range(ns):
            dg = dgamma[v, a]
            if abs(dg) < 1e-30:
                continue
            for i in range(3):
                for j in range(3):
                    dLp[v, i, j] += dg * dirs_vox[v, a, i] * normals_vox[v, a, j]
    return dLp


@nb.njit(cache=True, fastmath=True, parallel=True)
def _batched_solve12(J, R):
    """Solve J·x = -R for batched (nv, 12, 12) J and (nv, 12) R.
    Uses in-place Gaussian elimination — ~5× faster than np.linalg.solve
    for small fixed-size systems."""
    nv = J.shape[0]
    n = J.shape[1]
    x = np.empty((nv, n), dtype=np.float64)
    for v in nb.prange(nv):
        # Copy J and -R into augmented matrix
        A = np.empty((n, n + 1), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                A[i, j] = J[v, i, j]
            A[i, n] = -R[v, i]
        # Forward elimination with partial pivoting
        for col in range(n):
            # Find pivot
            max_val = abs(A[col, col])
            max_row = col
            for row in range(col + 1, n):
                val = abs(A[row, col])
                if val > max_val:
                    max_val = val
                    max_row = row
            if max_row != col:
                for j in range(n + 1):
                    tmp = A[col, j]
                    A[col, j] = A[max_row, j]
                    A[max_row, j] = tmp
            pivot = A[col, col]
            if abs(pivot) < 1e-30:
                # Singular — use damped step
                for i in range(n):
                    x[v, i] = -R[v, i] * 0.5
                break
            inv_pivot = 1.0 / pivot
            for row in range(col + 1, n):
                factor = A[row, col] * inv_pivot
                for j in range(col + 1, n + 1):
                    A[row, j] -= factor * A[col, j]
                A[row, col] = 0.0
        else:
            # Back substitution
            for i in range(n - 1, -1, -1):
                val = A[i, n]
                for j in range(i + 1, n):
                    val -= A[i, j] * x[v, j]
                x[v, i] = val / A[i, i]
    return x


def _warm_up_numba():
    """Trigger Numba compilation on tiny arrays (called once at import)."""
    _r = np.zeros((1, 12))
    _s = np.ones((1, 12))
    _M = np.zeros((1, 3, 3))
    _n = np.zeros((1, 12, 3))
    _d = np.zeros((1, 12, 3))
    _J = np.eye(12).reshape(1, 12, 12)
    try:
        _otis_kernel(_r, _s, 300.0, 0.01, 1e7, 9.5e-19, 0.78, 1.15, 0.005)
        _otis_with_tangent_kernel(_r, _s, 300.0, 0.01, 1e7, 9.5e-19, 0.78, 1.15, 0.005)
        _compute_rss_nb(_M, _n, _d)
        _compute_dLp_nb(_r, _d, _n)
        _batched_solve12(_J, _r)
    except Exception:
        pass

# Compile at import time so first call is fast
_warm_up_numba()


# ============================================================================
#  1.  Matrix Utilities
# ============================================================================

def _eye3():
    return np.eye(3, dtype=np.float64)


def exp_skew(W):
    """
    Matrix exponential of a 3×3 skew-symmetric tensor via Rodrigues.
    For small |W|, this is equivalent to a rotation by angle |w| about w/|w|.
    """
    w = np.array([W[2, 1], W[0, 2], W[1, 0]])
    theta = norm(w)
    if theta < 1e-14:
        return _eye3() + W + 0.5 * W @ W
    n = w / theta
    K = W / theta  # normalized skew
    return _eye3() + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def polar_decomposition_right(F):
    """
    Right polar decomposition  F = R U.
    Returns R (rotation) and U (right stretch, symmetric positive-definite).
    Works for single (3,3) or batched (...,3,3).
    """
    C = np.einsum('...ji,...jk->...ik', F, F)   # F^T F
    eigvals, eigvecs = eigh(C)                   # eigenvalues ≥ 0
    eigvals = np.maximum(eigvals, 1e-30)
    sqrt_eigvals = np.sqrt(eigvals)
    U = np.einsum('...ij,...j,...kj->...ik', eigvecs, sqrt_eigvals, eigvecs)
    U_inv = np.einsum('...ij,...j,...kj->...ik', eigvecs, 1.0 / sqrt_eigvals, eigvecs)
    R = F @ U_inv
    return R, U


def _sym(A):
    """Return sym(A) = 0.5*(A + A^T), supports batched (...,3,3)."""
    return 0.5 * (A + np.swapaxes(A, -2, -1))


def _skew(A):
    """Return skew(A) = 0.5*(A - A^T), supports batched (...,3,3)."""
    return 0.5 * (A - np.swapaxes(A, -2, -1))


def _batched_inv3(A):
    """Fast inverse of batched (...,3,3) matrices using adjugate formula.
    ~3× faster than np.linalg.inv for 3×3."""
    a = A
    # Cofactors
    c00 = a[...,1,1]*a[...,2,2] - a[...,1,2]*a[...,2,1]
    c01 = a[...,1,2]*a[...,2,0] - a[...,1,0]*a[...,2,2]
    c02 = a[...,1,0]*a[...,2,1] - a[...,1,1]*a[...,2,0]
    c10 = a[...,0,2]*a[...,2,1] - a[...,0,1]*a[...,2,2]
    c11 = a[...,0,0]*a[...,2,2] - a[...,0,2]*a[...,2,0]
    c12 = a[...,0,1]*a[...,2,0] - a[...,0,0]*a[...,2,1]
    c20 = a[...,0,1]*a[...,1,2] - a[...,0,2]*a[...,1,1]
    c21 = a[...,0,2]*a[...,1,0] - a[...,0,0]*a[...,1,2]
    c22 = a[...,0,0]*a[...,1,1] - a[...,0,1]*a[...,1,0]
    det = a[...,0,0]*c00 + a[...,0,1]*c01 + a[...,0,2]*c02
    inv_det = 1.0 / det[..., None, None]
    out = np.empty_like(A)
    out[...,0,0] = c00; out[...,0,1] = c10; out[...,0,2] = c20
    out[...,1,0] = c01; out[...,1,1] = c11; out[...,1,2] = c21
    out[...,2,0] = c02; out[...,2,1] = c12; out[...,2,2] = c22
    return out * inv_det


def _batched_inv(A):
    """Inverse of batched (...,3,3) matrices — uses fast 3×3 formula."""
    if A.shape[-1] == 3 and A.shape[-2] == 3:
        return _batched_inv3(A)
    return np.linalg.inv(A)


def _batched_det3(A):
    """Fast determinant of batched (...,3,3) matrices."""
    a = A
    return (a[...,0,0]*(a[...,1,1]*a[...,2,2] - a[...,1,2]*a[...,2,1])
          - a[...,0,1]*(a[...,1,0]*a[...,2,2] - a[...,1,2]*a[...,2,0])
          + a[...,0,2]*(a[...,1,0]*a[...,2,1] - a[...,1,1]*a[...,2,0]))


def _batched_det(A):
    """Determinant of batched (...,3,3) matrices."""
    if A.shape[-1] == 3 and A.shape[-2] == 3:
        return _batched_det3(A)
    return np.linalg.det(A)


# ============================================================================
#  2.  Kocks–Mecking Dislocation-Density Hardening
# ============================================================================

class KocksMeckingHardening:
    r"""
    Dislocation-density-based hardening (Kocks & Mecking, 2003).

    The slip resistance is computed from the total dislocation density:

        s^α = s_0 + α_T · μ · b · √(∑_β q_{αβ} · ρ^β)

    The SSD density evolves per system:

        dρ^α/dΓ = k₁√(∑ρ) − k₂ρ^α

    Parameters
    ----------
    k1           : float   storage coeff (m⁻¹)         [default 7e8  for Cu]
    k2           : float   recovery coeff (−)           [default 10   for Cu]
    alpha_taylor : float   Taylor coefficient            [default 0.3]
    burgers      : float   Burgers vector (m)            [default 2.56e-10 Cu]
    mu           : float   shear modulus (Pa)             [default 48e9 for poly-Cu]
    rho0         : float   initial SSD density (m⁻²)     [default 1e12]
    s0           : float   lattice friction (Pa)          [default 10e6]
    q_latent     : float   latent hardening ratio         [default 1.4]
    """

    def __init__(self, k1=7e8, k2=10.0, alpha_taylor=0.3,
                 burgers=2.56e-10, mu=48e9, rho0=1e12, s0=10e6,
                 q_latent=1.4):
        self.k1 = k1
        self.k2 = k2
        self.alpha_taylor = alpha_taylor
        self.burgers = burgers
        self.mu = mu
        self.rho0 = rho0
        self.s0 = s0
        self.q_latent = q_latent

    def initial_density(self):
        """Initial SSD density for all 12 systems (m⁻²)."""
        return np.full(N_SLIP, self.rho0)

    def resistance_from_density(self, rho):
        """
        Compute slip resistance from dislocation density.

        Parameters
        ----------
        rho : (..., 12)  SSD + GND density per system (m⁻²)

        Returns
        -------
        s   : (..., 12)  slip resistance (Pa)
        """
        # q-weighted density sum:  ρ_eff^α = Σ_β q_{αβ} ρ^β
        # Self: q=1, Latent: q=q_lat
        rho_total = np.sum(rho, axis=-1, keepdims=True)  # (..., 1)
        rho_self = rho                                     # (..., 12)
        rho_eff = self.q_latent * rho_total + (1.0 - self.q_latent) * rho_self

        s = self.s0 + self.alpha_taylor * self.mu * self.burgers * np.sqrt(
            np.maximum(rho_eff, 0.0))
        return s

    def update_density(self, rho, delta_gamma):
        """
        Update SSD density after a converged increment.

        dρ^α = (k₁√(Σρ) − k₂ρ^α) · |Δγ^α|

        Parameters
        ----------
        rho         : (n_vox, 12) current density
        delta_gamma : (n_vox, 12) slip increments (signed)

        Returns
        -------
        rho_new : (n_vox, 12) updated density (≥ 0)
        """
        rho_total = np.sum(np.maximum(rho, 0.0), axis=-1, keepdims=True)
        abs_dg = np.abs(delta_gamma)
        drho = (self.k1 * np.sqrt(rho_total) - self.k2 * rho) * abs_dg
        return np.maximum(rho + drho, 0.0)


# ============================================================================
#  3.  Armstrong–Frederick Kinematic Hardening (Back-Stress)
# ============================================================================

class ArmstrongFrederickBackstress:
    r"""
    Armstrong–Frederick evolution of back-stress (Bauschinger effect).

        dχ^α = c₁ dγ^α − c₂ χ^α |dγ^α|

    Saturation back-stress: χ_sat = c₁/c₂

    Parameters
    ----------
    c1 : float  direct hardening modulus (Pa)  [default 1e9  for Cu]
    c2 : float  dynamic recovery coeff  (−)    [default 10   for Cu]
    """

    def __init__(self, c1=1.0e9, c2=10.0):
        self.c1 = c1
        self.c2 = c2

    @property
    def saturation(self):
        return self.c1 / self.c2 if self.c2 > 0 else np.inf

    def update(self, chi, delta_gamma):
        """
        Update back-stress.

        Parameters
        ----------
        chi         : (n_vox, 12) current back-stress (Pa)
        delta_gamma : (n_vox, 12) slip increments (signed)

        Returns
        -------
        chi_new : (n_vox, 12) updated back-stress
        """
        abs_dg = np.abs(delta_gamma)
        dchi = self.c1 * delta_gamma - self.c2 * chi * abs_dg
        return chi + dchi


# ============================================================================
#  4a.  Voigt  ↔  Tensor helpers for finite-strain stress
# ============================================================================

# Voigt map: (0,0)->0  (1,1)->1  (2,2)->2  (1,2)->3  (0,2)->4  (0,1)->5
_VOIGT_IJ = np.array([[0,5,4],[5,1,3],[4,3,2]])   # tensor (i,j) → Voigt index

def _Ee_tensor_to_voigt(Ee):
    """Convert Ee (...,3,3) symmetric tensor to strain-Voigt (...,6).
    Voigt strain convention: factor 2 on shear."""
    v = np.empty(Ee.shape[:-2] + (6,), dtype=Ee.dtype)
    v[..., 0] = Ee[..., 0, 0]
    v[..., 1] = Ee[..., 1, 1]
    v[..., 2] = Ee[..., 2, 2]
    v[..., 3] = 2.0 * Ee[..., 1, 2]
    v[..., 4] = 2.0 * Ee[..., 0, 2]
    v[..., 5] = 2.0 * Ee[..., 0, 1]
    return v

def _S_voigt_to_tensor(Sv):
    """Convert stress-Voigt (...,6) to symmetric tensor (...,3,3)."""
    t = np.empty(Sv.shape[:-1] + (3, 3), dtype=Sv.dtype)
    t[..., 0, 0] = Sv[..., 0]
    t[..., 1, 1] = Sv[..., 1]
    t[..., 2, 2] = Sv[..., 2]
    t[..., 1, 2] = t[..., 2, 1] = Sv[..., 3]
    t[..., 0, 2] = t[..., 2, 0] = Sv[..., 4]
    t[..., 0, 1] = t[..., 1, 0] = Sv[..., 5]
    return t

def _stress_voigt(C_vox, Ee):
    """S = C:Ee via 6×6 Voigt contraction.  ~10× faster than 3333 einsum.
    C_vox: (...,6,6), Ee: (...,3,3) → S: (...,3,3)."""
    Ee_v = _Ee_tensor_to_voigt(Ee)                      # (...,6)
    Sv = (C_vox @ Ee_v[..., None])[..., 0]              # (...,6) via matmul
    return _S_voigt_to_tensor(Sv)                        # (...,3,3)


# ============================================================================
#  4b.  OTIS Flow Rule with Analytical Tangent (vectorized)
# ============================================================================

def compute_slip_otis_with_tangent(rss, slip_resistance, temperature, dt,
                                   rate_slip_ref=1e7, Q_activation=9.5e-19,
                                   p_barrier=0.78, q_barrier=1.15,
                                   max_dgamma=0.005):
    """
    Vectorized OTIS flow rule returning both Δγ AND ∂Δγ/∂τ.

    Parameters
    ----------
    rss, slip_resistance : (n_vox, 12)
    temperature          : float (K)
    dt                   : float

    Returns
    -------
    dgamma   : (n_vox, 12)  slip increments
    ddg_dtau : (n_vox, 12)  tangent  ∂Δγ/∂τ  (always ≥ 0)
    """
    kT = KB * max(temperature, 1.0)
    p, q = p_barrier, q_barrier
    QkT = Q_activation / kT

    sign_tau = np.sign(rss)
    tau_abs = np.abs(rss)
    s = np.maximum(np.abs(slip_resistance), 1e-10)
    ratio = np.minimum(tau_abs / s, 1.0 - 1e-12)

    dgamma = np.zeros_like(rss)
    ddg_dtau = np.zeros_like(rss)

    active = tau_abs > 1e-10

    # Saturated regime: |τ| ≥ s  →  max slip rate, zero tangent
    saturated = active & (ratio >= 1.0 - 1e-12)
    dgamma[saturated] = rate_slip_ref * sign_tau[saturated] * dt

    # Thermally activated regime
    normal = active & ~saturated
    if np.any(normal):
        r = ratio[normal]
        rp = r ** p
        bracket = (1.0 - rp) ** q
        exponent = np.maximum(-QkT * bracket, -50.0)
        exp_val = np.exp(exponent)

        dgamma[normal] = rate_slip_ref * exp_val * sign_tau[normal] * dt

        # Analytical tangent: chain rule
        #   ∂Δγ/∂τ = γ̇_ref·Δt · exp(·) · (QkT·p·q) · r^{p-1}/s · (1-r^p)^{q-1}
        bracket_m1 = np.maximum((1.0 - rp), 1e-30) ** (q - 1.0)
        rp_m1 = np.maximum(r, 1e-30) ** (p - 1.0)
        tang = rate_slip_ref * dt * exp_val * QkT * p * q * rp_m1 / s[normal] * bracket_m1
        ddg_dtau[normal] = tang

    dgamma = np.clip(dgamma, -max_dgamma, max_dgamma)
    return dgamma, ddg_dtau


# ============================================================================
#  5.  Finite-Strain Constitutive State
# ============================================================================

class FiniteStrainState:
    """
    Per-voxel state for finite-strain elasto-viscoplastic FFT.

    Tracks:
      - Fp          (n_vox, 3, 3)   plastic deformation gradient
      - rho_ssd     (n_vox, 12)     SSD density  (Kocks–Mecking)
      - chi         (n_vox, 12)     back-stress   (Armstrong–Frederick)
      - acc_slip    (n_vox, 12)     accumulated |γ^α|
      - R_current   (n_grains, 3, 3) current crystal orientations
      - euler_current (n_grains, 3)  current Euler angles
    """

    def __init__(self, grain_ids, euler_angles,
                 C11=168.4e9, C12=121.4e9, C44=75.4e9,
                 temperature=293.0,
                 rate_slip_ref=1e7, Q_activation=9.5e-19,
                 p_barrier=0.78, q_barrier=1.15,
                 hardening_model='kocks_mecking',
                 km_params=None, af_params=None,
                 voce_params=None,
                 enable_backstress=True,
                 enable_gnd=True):
        """
        Parameters
        ----------
        grain_ids     : (N,N,N)  int grain ID per voxel
        euler_angles  : (n_grains, 3) initial Euler angles (rad)
        hardening_model : 'kocks_mecking' or 'voce'
        km_params     : dict of KocksMeckingHardening kwargs  (or None for defaults)
        af_params     : dict of ArmstrongFrederickBackstress kwargs  (or None)
        voce_params   : dict of VoceHardening kwargs (used when hardening_model='voce')
        enable_backstress : bool  Armstrong–Frederick back-stress
        enable_gnd    : bool  GND hardening from curl(Fp)
        """
        self.grain_ids = grain_ids
        self.shape = grain_ids.shape
        N = grain_ids.shape[0]
        self.N = N
        self.n_voxels = N ** 3
        self.n_grains = euler_angles.shape[0]

        # Elastic constants
        self.C11, self.C12, self.C44 = C11, C12, C44

        # OTIS parameters
        self.temperature = temperature
        self.rate_slip_ref = rate_slip_ref
        self.Q_activation = Q_activation
        self.p_barrier = p_barrier
        self.q_barrier = q_barrier

        # Feature flags
        self.enable_backstress = enable_backstress
        self.enable_gnd = enable_gnd
        self.hardening_model = hardening_model

        # --- Hardening models ---
        if hardening_model == 'kocks_mecking':
            self.km = KocksMeckingHardening(**(km_params or {}))
        else:
            self.km = None
        if hardening_model == 'voce':
            self.voce = VoceHardening(**(voce_params or {}))
        else:
            self.voce = None

        # Back-stress
        self.af = ArmstrongFrederickBackstress(**(af_params or {})) \
            if enable_backstress else None

        # --- Crystal geometry (reference frame) ---
        self.euler_initial = euler_angles.copy()
        self.euler_current = euler_angles.copy()

        # Build rotation matrices, Schmid tensors, stiffness per grain
        C_base = cubic_stiffness_tensor(C11, C12, C44)
        self.P_voigt = np.zeros((self.n_grains, N_SLIP, 6))
        self.C_voigt = np.zeros((self.n_grains, 6, 6))
        self.slip_normals = np.zeros((self.n_grains, N_SLIP, 3))
        self.slip_dirs = np.zeros((self.n_grains, N_SLIP, 3))
        self.R_initial = np.zeros((self.n_grains, 3, 3))
        self.R_current = np.zeros((self.n_grains, 3, 3))

        for g in range(self.n_grains):
            R = euler_to_rotation(*euler_angles[g])
            self.R_initial[g] = R
            self.R_current[g] = R.copy()
            normals, dirs = rotate_slip_systems(R)
            self.slip_normals[g] = normals
            self.slip_dirs[g] = dirs
            self.P_voigt[g] = schmid_tensors(normals, dirs)
            self.C_voigt[g] = rotate_stiffness_voigt(C_base, R)

        # Precompute interaction matrix H_αβ per grain (constant!)
        # H_{αβ} = P^α_I C_{IJ} P^β_J  (Voigt contraction)
        # This is ~100× faster than computing per-voxel with C_3333
        self.H_grain = np.zeros((self.n_grains, N_SLIP, N_SLIP))
        for g in range(self.n_grains):
            # P_voigt[g] is (12, 6), C_voigt[g] is (6,6)
            CP = self.C_voigt[g] @ self.P_voigt[g].T    # (6, 12)
            self.H_grain[g] = self.P_voigt[g] @ CP       # (12, 12)

        # Map to voxels
        gids = grain_ids.ravel()
        self.P_vox = self.P_voigt[gids]                # (n_vox, 12, 6)
        self.C_vox = self.C_voigt[gids]                 # (n_vox, 6, 6)
        self.normals_vox = self.slip_normals[gids]      # (n_vox, 12, 3)
        self.dirs_vox = self.slip_dirs[gids]            # (n_vox, 12, 3)
        self.H_vox = self.H_grain[gids]                 # (n_vox, 12, 12)

        # --- State variables ---
        I3 = np.eye(3, dtype=np.float64)
        self.Fp = np.tile(I3, (self.n_voxels, 1, 1))            # (n_vox, 3, 3)
        self.accumulated_slip = np.zeros((self.n_voxels, N_SLIP))

        # Kocks-Mecking
        if self.km is not None:
            self.rho_ssd = np.full((self.n_voxels, N_SLIP), self.km.rho0)
            self.slip_resistance = self.km.resistance_from_density(self.rho_ssd)
        else:
            self.rho_ssd = None
            self.slip_resistance = np.full((self.n_voxels, N_SLIP),
                                           self.voce.tau0 if self.voce else 50e6)

        # Armstrong-Frederick
        if self.af is not None:
            self.chi = np.zeros((self.n_voxels, N_SLIP))
        else:
            self.chi = None

        # GND density (computed externally, stored here)
        self.rho_gnd = np.zeros((self.n_voxels, N_SLIP))

        # Plastic strain (Voigt, for visualization compatibility)
        self.plastic_strain = np.zeros((self.n_voxels, 6))

        # History for GUI
        self._history_snapshots = []

    # ------------------------------------------------------------------
    #  Stiffness field for FFT solver
    # ------------------------------------------------------------------
    def get_stiffness_field(self):
        """Return (N,N,N,6,6) Voigt stiffness field."""
        return self.C_voigt[self.grain_ids]

    # ------------------------------------------------------------------
    #  Finite-strain constitutive update  (vectorized, single-pass)
    # ------------------------------------------------------------------
    def constitutive_update(self, F_field_flat, dt=1.0):
        """
        Given total deformation gradient F at each voxel (frozen Fp),
        compute 1st Piola–Kirchhoff stress P and slip increments Δγ.

        Parameters
        ----------
        F_field_flat : (n_vox, 3, 3) deformation gradient
        dt           : float  time increment

        Returns
        -------
        P_flat      : (n_vox, 3, 3) 1st Piola–Kirchhoff stress
        dgamma      : (n_vox, 12) slip increments
        """
        nv = self.n_voxels
        I3 = np.eye(3)

        # 1. Elastic deformation gradient  Fe = F · Fp⁻¹
        Fp_inv = _batched_inv(self.Fp)                    # (nv, 3, 3)
        Fe = F_field_flat @ Fp_inv                         # (nv, 3, 3)

        # 2-3. Green–Lagrange elastic strain  Ee = ½(Fe^T Fe − I)
        FeT = Fe.swapaxes(-2, -1)
        Ce = FeT @ Fe                                      # (nv, 3, 3)
        Ee = 0.5 * (Ce - I3)                               # (nv, 3, 3)

        # 4. 2nd Piola–Kirchhoff stress  S = C : Ee  (Voigt — fast)
        S = _stress_voigt(self.C_vox, Ee)                  # (nv, 3, 3)

        # 5. Mandel stress  M = Ce · S
        M = Ce @ S                                         # (nv, 3, 3)

        # 6. Resolved shear stress — Numba kernel
        rss = _compute_rss_nb(M, self.normals_vox, self.dirs_vox)  # (nv, 12)

        # 7. Effective stress (subtract back-stress)
        rss_eff = rss - (self.chi if self.chi is not None else 0.0)

        # 8. Slip increments — Numba-JIT OTIS flow rule
        kT = KB * max(self.temperature, 1.0)
        dgamma = _otis_kernel(
            rss_eff, self.slip_resistance, kT, dt,
            self.rate_slip_ref, self.Q_activation,
            self.p_barrier, self.q_barrier, 0.005)

        # 9. Plastic correction → updated Fe, stress, P  (single pass)
        dLp = _compute_dLp_nb(dgamma, self.dirs_vox, self.normals_vox)
        Fp_trial = (I3 + dLp) @ self.Fp
        Fp_trial_inv = _batched_inv(Fp_trial)
        Fe_t = F_field_flat @ Fp_trial_inv
        Ce_t = Fe_t.swapaxes(-2, -1) @ Fe_t
        Ee_t = 0.5 * (Ce_t - I3)
        S_t = _stress_voigt(self.C_vox, Ee_t)

        # 10. 1st Piola–Kirchhoff stress  P = Fe · S · Fp⁻ᵀ
        P = (Fe_t @ S_t) @ Fp_trial_inv.swapaxes(-2, -1)

        return P, dgamma

    # ------------------------------------------------------------------
    #  Implicit constitutive update (per-voxel Newton-Raphson)
    # ------------------------------------------------------------------
    def constitutive_update_implicit(self, F_field_flat, dt=1.0,
                                     max_newton=10, tol_newton=1e-8):
        """
        Fully implicit constitutive update with Newton-Raphson.

        Solves the 12-dimensional nonlinear system per voxel:
            R^α = Δγ^α − Δγ^α_OTIS(τ_eff^α(Δγ), s, T, Δt) = 0

        Parameters
        ----------
        F_field_flat : (n_vox, 3, 3) deformation gradient
        dt, max_newton, tol_newton

        Returns
        -------
        P_flat : (n_vox, 3, 3) 1st PK stress
        dgamma : (n_vox, 12) converged slip increments
        """
        nv = self.n_voxels
        I3 = np.eye(3)

        Fp_inv = _batched_inv(self.Fp)

        # Interaction matrix — precomputed per grain, indexed to voxels
        H = self.H_vox  # (nv, 12, 12) — already built in __init__
        kT = KB * max(self.temperature, 1.0)
        chi_val = self.chi if self.chi is not None else 0.0
        I12 = np.eye(N_SLIP)

        # Initial guess: explicit dgamma
        dgamma = np.zeros((nv, N_SLIP))

        for newton_it in range(max_newton):
            # Build Fp_trial from current dgamma
            dLp = _compute_dLp_nb(dgamma, self.dirs_vox, self.normals_vox)
            Fp_trial = (I3 + dLp) @ self.Fp
            Fp_trial_inv = _batched_inv(Fp_trial)
            Fe = F_field_flat @ Fp_trial_inv

            # Stress (Voigt + matmul)
            Ce = Fe.swapaxes(-2, -1) @ Fe
            Ee = 0.5 * (Ce - I3)
            S = _stress_voigt(self.C_vox, Ee)
            M = Ce @ S

            # Resolved shear stress — Numba
            rss = _compute_rss_nb(M, self.normals_vox, self.dirs_vox)
            rss_eff = rss - chi_val

            # OTIS with tangent — Numba
            dgamma_otis, ddg_dtau = _otis_with_tangent_kernel(
                rss_eff, self.slip_resistance, kT, dt,
                self.rate_slip_ref, self.Q_activation,
                self.p_barrier, self.q_barrier, 0.005)

            # Residual
            R = dgamma - dgamma_otis                       # (nv, 12)
            res_norm = np.max(np.abs(R))
            if res_norm < tol_newton:
                break

            # Jacobian  J = I + diag(∂Δγ/∂τ) · H
            J = I12 + ddg_dtau[:, :, None] * H             # (nv, 12, 12)

            # Newton correction: solve J·dx = -R per voxel (Numba GE)
            dx = _batched_solve12(J, R)                    # (nv, 12)

            dgamma = dgamma + dx

        # Final stress computation with converged dgamma
        dLp = _compute_dLp_nb(dgamma, self.dirs_vox, self.normals_vox)
        Fp_final = (I3 + dLp) @ self.Fp
        Fp_final_inv = _batched_inv(Fp_final)
        Fe_final = F_field_flat @ Fp_final_inv
        Ce_f = Fe_final.swapaxes(-2, -1) @ Fe_final
        Ee_f = 0.5 * (Ce_f - I3)
        S_f = _stress_voigt(self.C_vox, Ee_f)
        P = (Fe_final @ S_f) @ Fp_final_inv.swapaxes(-2, -1)

        return P, dgamma

    # ------------------------------------------------------------------
    #  Commit state after converged increment
    # ------------------------------------------------------------------
    def commit_state(self, dgamma, F_field_flat=None):
        """
        Permanently update internal state after a converged load step.

        Parameters
        ----------
        dgamma       : (n_vox, 12) converged slip increments
        F_field_flat : (n_vox, 3, 3) deformation gradient  (for texture update)
        """
        I3 = np.eye(3)

        # 1. Update Fp
        dLp = _compute_dLp_nb(dgamma, self.dirs_vox, self.normals_vox)
        self.Fp = (I3 + dLp) @ self.Fp

        # 2. Accumulated slip
        self.accumulated_slip += np.abs(dgamma)

        # 3. Update hardening (SSD density and slip resistance)
        if self.km is not None:
            self.rho_ssd = self.km.update_density(self.rho_ssd, dgamma)
            total_rho = self.rho_ssd.copy()
            if self.enable_gnd:
                total_rho = total_rho + self.rho_gnd
            self.slip_resistance = self.km.resistance_from_density(total_rho)
        elif self.voce is not None:
            self.slip_resistance = self.voce.update_vectorized(
                self.slip_resistance, dgamma)

        # 4. Update back-stress
        if self.af is not None:
            self.chi = self.af.update(self.chi, dgamma)

        # 5. Update equivalent plastic strain (for visualization)
        deps_p = np.einsum('va,vas->vs', dgamma, self.P_vox)
        deps_p[:, 3:] *= 2  # tensor → engineering shear
        self.plastic_strain += deps_p

        # 6. Update texture  (crystal orientation from polar decomp of Fe)
        if F_field_flat is not None:
            self._update_texture(F_field_flat)

    def _update_texture(self, F_field_flat):
        """Update crystal orientations from current Fe = F · Fp⁻¹."""
        Fp_inv = _batched_inv(self.Fp)
        Fe = F_field_flat @ Fp_inv

        # Per-grain average Fe → extract rotation
        for g in range(self.n_grains):
            mask = (self.grain_ids.ravel() == g)
            if not np.any(mask):
                continue
            Fe_grain = np.mean(Fe[mask], axis=0)  # average over voxels in grain
            R_e, _ = polar_decomposition_right(Fe_grain)

            # New orientation: R_current = R_e · R_initial
            self.R_current[g] = R_e @ self.R_initial[g]

            # Update Euler angles from R_current
            self.euler_current[g] = _rotation_to_euler(self.R_current[g])

    # ------------------------------------------------------------------
    #  Visualization helpers
    # ------------------------------------------------------------------
    def get_von_mises_plastic_strain_field(self):
        ep = self.plastic_strain
        e11, e22, e33 = ep[:, 0], ep[:, 1], ep[:, 2]
        e23, e13, e12 = ep[:, 3] / 2, ep[:, 4] / 2, ep[:, 5] / 2
        vm = np.sqrt(2.0 / 3.0 * ((e11 - e22) ** 2 + (e22 - e33) ** 2 +
                                    (e33 - e11) ** 2 +
                                    6 * (e23 ** 2 + e13 ** 2 + e12 ** 2)))
        return vm.reshape(self.shape)

    def get_total_accumulated_slip_field(self):
        return np.sum(self.accumulated_slip, axis=1).reshape(self.shape)

    def get_slip_resistance_field(self):
        return np.mean(self.slip_resistance, axis=1).reshape(self.shape) / 1e6

    def get_ssd_density_field(self):
        """Mean SSD density per voxel (m⁻²) → (N,N,N)."""
        if self.rho_ssd is not None:
            return np.mean(self.rho_ssd, axis=1).reshape(self.shape)
        return np.zeros(self.shape)

    def get_gnd_density_field(self):
        """Mean GND density per voxel (m⁻²) → (N,N,N)."""
        return np.mean(self.rho_gnd, axis=1).reshape(self.shape)

    def get_backstress_field(self):
        """Mean |χ| per voxel (MPa) → (N,N,N)."""
        if self.chi is not None:
            return np.mean(np.abs(self.chi), axis=1).reshape(self.shape) / 1e6
        return np.zeros(self.shape)

    def get_misorientation_field(self):
        """Misorientation angle from initial orientation (degrees) → (N,N,N)."""
        gids = self.grain_ids.ravel()
        misor = np.zeros(self.n_voxels)
        for g in range(self.n_grains):
            mask = (gids == g)
            dR = self.R_current[g] @ self.R_initial[g].T
            # Misorientation angle from trace:  cos(θ) = (tr(ΔR) − 1)/2
            trace = np.clip(np.trace(dR), -1.0, 3.0)
            theta = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
            misor[mask] = np.degrees(theta)
        return misor.reshape(self.shape)


# ============================================================================
#  6.  Unsymmetrized Green Operator (for displacement gradient H = F − I)
# ============================================================================

def apply_green_operator_unsym(tau_hat, n_field, Ainv):
    """
    Green operator returning the FULL displacement gradient (not symmetrized).

    In finite strain, H = ∇u is non-symmetric (includes rotation).
    Ĥ'_ij = n_j · û_i    where  b_k = n_l τ̂_kl,  û = A⁻¹ b

    Parameters
    ----------
    tau_hat : (..., 3, 3) complex  FT of polarization stress
    n_field : (..., 3)             unit wave direction
    Ainv    : (..., 3, 3)          inverse acoustic tensor

    Returns
    -------
    H_hat : (..., 3, 3) complex  FT of displacement gradient fluctuation
    """
    xp = np  # GPU extension: use _xp(tau_hat) for CuPy
    try:
        from fft_solver import _xp
        xp = _xp(tau_hat)
    except ImportError:
        pass

    b = xp.einsum('...kj,...j->...k', tau_hat, n_field)
    u = xp.einsum('...ik,...k->...i', Ainv, b)
    H_hat = xp.einsum('...i,...j->...ij', u, n_field)  # u_i n_j → NOT symmetrized
    return H_hat


# ============================================================================
#  7.  GND: Nye Tensor from curl(Fp)
# ============================================================================

def compute_nye_tensor(Fp_field, N, derivative_scheme='continuous'):
    """
    Compute the Nye (dislocation density) tensor  α = curl(Fp)
    using spectral differentiation in Fourier space.

        α̂_ij = ε_jkl (iξ_k) F̂p_il

    Fully vectorized — no Python loops over tensor indices.

    Parameters
    ----------
    Fp_field : (N, N, N, 3, 3)  plastic deformation gradient field
    N        : int  grid size
    derivative_scheme : str

    Returns
    -------
    nye : (N, N, N, 3, 3)  Nye tensor
    """
    from fft_solver import _build_freq_vectors
    from numpy.fft import fftn, ifftn

    # Frequency vectors (N,N,N,3)
    xi = _build_freq_vectors(N, derivative_scheme=derivative_scheme, on_gpu=False)

    # Batch FFT: flatten last two dims, FFT each, reshape back
    Fp_flat = Fp_field.reshape(N, N, N, 9)
    Fp_hat_flat = np.empty((N, N, N, 9), dtype=complex)
    for c in range(9):
        Fp_hat_flat[:, :, :, c] = fftn(Fp_flat[:, :, :, c])
    Fp_hat = Fp_hat_flat.reshape(N, N, N, 3, 3)

    # 6 non-zero Levi-Civita entries:  (j, k, l, sign)
    _LC_ENTRIES = [(0,1,2,+1), (1,2,0,+1), (2,0,1,+1),
                   (0,2,1,-1), (2,1,0,-1), (1,0,2,-1)]

    # α̂_ij = Σ_{(j,k,l)} ε_jkl · (iξ_k) · F̂p_il
    nye_hat = np.zeros((N, N, N, 3, 3), dtype=complex)
    for j, k, l, sign in _LC_ENTRIES:
        # xi[:,:,:,k] broadcast with Fp_hat[:,:,:,:,l] along i-axis
        term = (sign * 1j) * xi[:, :, :, k, np.newaxis] * Fp_hat[:, :, :, :, l]
        nye_hat[:, :, :, :, j] += term   # sum over all i at once

    # Batch IFFT
    nye_hat_flat = nye_hat.reshape(N, N, N, 9)
    nye_flat = np.empty((N, N, N, 9))
    for c in range(9):
        nye_flat[:, :, :, c] = np.real(ifftn(nye_hat_flat[:, :, :, c]))
    nye = nye_flat.reshape(N, N, N, 3, 3)

    return nye


def compute_gnd_density(nye, normals_vox, dirs_vox, burgers=2.56e-10):
    """
    Project Nye tensor onto slip systems to get GND density per system.
    Fully vectorized over all 12 slip systems simultaneously.

    Edge:   ρ^α_edge  = (1/b) |α : (s^α ⊗ (s^α × n^α))|
    Screw:  ρ^α_screw = (1/b) |α : (s^α ⊗ s^α)|
    Total:  ρ^α_GND = √(ρ_edge² + ρ_screw²)

    Parameters
    ----------
    nye         : (n_vox, 3, 3)
    normals_vox : (n_vox, 12, 3) slip plane normals
    dirs_vox    : (n_vox, 12, 3) slip directions
    burgers     : float  Burgers vector length (m)

    Returns
    -------
    rho_gnd : (n_vox, 12)  GND density per system (m⁻²)
    """
    # Edge line direction: t = s × n  → (nv, 12, 3)
    t_all = np.cross(dirs_vox, normals_vox)

    # Edge dyad: s ⊗ t  → (nv, 12, 3, 3)
    dyad_edge = np.einsum('vai,vaj->vaij', dirs_vox, t_all)
    rho_edge = np.einsum('vij,vaij->va', nye, dyad_edge) / burgers

    # Screw dyad: s ⊗ s  → (nv, 12, 3, 3)
    dyad_screw = np.einsum('vai,vaj->vaij', dirs_vox, dirs_vox)
    rho_screw = np.einsum('vij,vaij->va', nye, dyad_screw) / burgers

    return np.sqrt(rho_edge ** 2 + rho_screw ** 2)


def update_gnd(state, derivative_scheme='continuous'):
    """
    Compute GND density from curl(Fp) and store in state.

    Parameters
    ----------
    state : FiniteStrainState
    """
    N = state.N
    Fp_field = state.Fp.reshape(N, N, N, 3, 3)
    nye = compute_nye_tensor(Fp_field, N, derivative_scheme=derivative_scheme)
    nye_flat = nye.reshape(-1, 3, 3)
    state.rho_gnd = compute_gnd_density(
        nye_flat, state.normals_vox, state.dirs_vox,
        burgers=state.km.burgers if state.km else 2.56e-10)


# ============================================================================
#  8.  Euler Angle Utilities
# ============================================================================

def _rotation_to_euler(R):
    """
    Convert 3×3 rotation matrix to Bunge Euler angles (φ₁, Φ, φ₂) in radians.
    Inverse of euler_to_rotation (ZXZ convention).
    """
    # From R (ZXZ):
    #   R[2,2] = cos(Φ)
    #   R[2,0] = sin(Φ)sin(φ₁)      R[2,1] = -sin(Φ)cos(φ₁)
    #   R[0,2] = sin(Φ)sin(φ₂)      R[1,2] = sin(Φ)cos(φ₂)
    Phi = np.arccos(np.clip(R[2, 2], -1.0, 1.0))
    if np.abs(np.sin(Phi)) > 1e-6:
        phi1 = np.arctan2(R[2, 0], -R[2, 1])
        phi2 = np.arctan2(R[0, 2], R[1, 2])
    else:
        # Gimbal lock: Φ ≈ 0 or π
        phi1 = np.arctan2(R[0, 1], R[0, 0])
        phi2 = 0.0
    return np.array([phi1 % (2 * np.pi), Phi, phi2 % (2 * np.pi)])


# ============================================================================
#  9.  Finite-Strain EVPFFT Solver
# ============================================================================

def solve_evpfft_finite_strain(state, S_macro_voigt, n_increments=10, dt=0.1,
                                tol_fft=1e-5, tol_stress=1e-4,
                                max_iter_fft=100, max_iter_stress=30,
                                verbose=True, callback=None,
                                derivative_scheme='continuous',
                                ref_medium_mode='mean',
                                implicit_constitutive=True,
                                compute_gnd_every=1):
    """
    Finite-strain elasto-viscoplastic FFT solver.

    Solves:  div(P) = 0   with   P = P(F, state)
    using the Lippmann–Schwinger equation for the displacement gradient.

    Features beyond small-strain EVPFFT:
      - Multiplicative decomposition  F = Fe Fp
      - Mandel stress for resolved shear stress
      - Kocks–Mecking hardening + Armstrong–Frederick back-stress
      - Implicit per-voxel Newton (optional)
      - GND from curl(Fp)
      - Texture evolution

    Parameters
    ----------
    state           : FiniteStrainState
    S_macro_voigt   : (6,) target macroscopic Cauchy stress (Pa)
    n_increments    : int  number of load steps
    dt              : float  time per increment
    tol_fft         : float  equilibrium tolerance
    tol_stress      : float  macroscopic stress tolerance
    max_iter_fft    : max inner FFT iterations
    max_iter_stress : max outer Newton iterations
    derivative_scheme : str  'continuous'|'finite_difference'|'rotated'
    ref_medium_mode : str  'mean'|'contrast_aware'
    implicit_constitutive : bool  use implicit Newton per voxel
    compute_gnd_every : int  GND update frequency (0 = disabled)

    Returns
    -------
    history : list of (eps_field_voigt, sig_field_voigt, frac) per increment
    """
    from fft_solver import (
        build_green_data, fft_3x3, ifft_3x3,
        compute_reference_medium, von_mises_stress,
        strain_v2t, strain_t2v, stress_t2v,
        strain_field_v2t, stress_field_v2t,
        strain_field_t2v, stress_field_t2v,
        apply_stiffness,
        _to_gpu, _to_cpu, HAS_GPU,
    )

    shape = state.shape
    N = shape[0]
    nv = state.n_voxels

    # --- Reference medium ---
    C_field = state.get_stiffness_field()
    C0, lam0, mu0 = compute_reference_medium(C_field, mode=ref_medium_mode)
    C0_6x6 = C0.copy()

    # Green operator data
    use_gpu = HAS_GPU
    n_field_g, Ainv_g = build_green_data(
        N, lam0, mu0, on_gpu=use_gpu, derivative_scheme=derivative_scheme)

    # Compliance for initial strain guess
    S0_inv = np.linalg.inv(C0_6x6)

    # Broadcast reference stiffness
    C0_bc = C0_6x6[np.newaxis, np.newaxis, np.newaxis]

    history = []
    I3 = np.eye(3)

    for inc in range(1, n_increments + 1):
        frac = inc / n_increments
        S_target = S_macro_voigt * frac

        if verbose:
            s_mpa = S_target / 1e6
            print(f"\nFinite-Strain EVPFFT Step {inc}/{n_increments}  "
                  f"σ_target (MPa): [{s_mpa[0]:.0f}, {s_mpa[1]:.0f}, "
                  f"{s_mpa[2]:.0f}]")

        # Initial macro strain guess
        E_target = S0_inv @ S_target
        E_bar = E_target.copy()

        # Macro displacement gradient  H_bar = ε_bar  (symmetric part, no rotation)
        H_bar_t = strain_v2t(E_bar)  # (3,3)

        # Initialize displacement gradient field  H = F - I
        if inc == 1:
            H_field = np.zeros(shape + (3, 3))
            H_field[...] = H_bar_t
        else:
            H_mean = np.mean(H_field.reshape(-1, 3, 3), axis=0)
            H_field += (H_bar_t - H_mean)

        last_dgamma = None

        # Secant compliance for stress control
        S_eff = S0_inv.copy()
        E_bar_prev = None
        sig_mean_prev = None

        # --- Newton loop for stress control ---
        for newton in range(max_iter_stress):
            inner_cap = min(max_iter_fft, 5 if newton < 3 else 15)

            # --- Inner FFT equilibrium loop ---
            for fft_it in range(inner_cap):
                H_old = H_field.copy()

                # F = I + H
                F_field = H_field + I3   # (N,N,N,3,3)
                F_flat = F_field.reshape(-1, 3, 3)

                # Constitutive update:  P = P(F, state)
                if implicit_constitutive:
                    P_flat, last_dgamma = state.constitutive_update_implicit(
                        F_flat, dt=dt)
                else:
                    P_flat, last_dgamma = state.constitutive_update(
                        F_flat, dt=dt)

                P_field = P_flat.reshape(shape + (3, 3))

                # Symmetric part of H for reference stress
                eps_field_t = _sym(H_field)

                # Reference stress  P0 = C0 : sym(H)
                P0_field = apply_stiffness(C0_bc, eps_field_t)

                # Polarization
                tau_field = P_field - P0_field  # (N,N,N,3,3)

                # FFT ——> Green operator ——> IFFT
                if use_gpu:
                    tau_g = _to_gpu(tau_field)
                    tau_hat = fft_3x3(tau_g)
                    H_hat = apply_green_operator_unsym(tau_hat, n_field_g, Ainv_g)
                    H_hat[0, 0, 0, :, :] = 0.0  # zero mean
                    H_fluct = _to_cpu(ifft_3x3(H_hat))
                else:
                    tau_hat = fft_3x3(tau_field)
                    H_hat = apply_green_operator_unsym(tau_hat, n_field_g, Ainv_g)
                    H_hat[0, 0, 0, :, :] = 0.0
                    H_fluct = ifft_3x3(H_hat)

                # Lippmann-Schwinger:  H = H_bar - Γ⁰[τ]
                H_field = H_bar_t - H_fluct

                # Convergence check
                diff = np.sum((H_field - H_old) ** 2)
                ref = np.sum(H_old ** 2)
                err_fft = (diff / max(ref, 1e-30)) ** 0.5
                if err_fft < tol_fft:
                    break

            # --- Macroscopic stress check ---
            # Approximate Cauchy stress:  σ ≈ P · F^T / det(F)
            # For moderate strains, σ ≈ sym(P) ≈ P (since F ≈ I)
            F_field_curr = H_field + I3
            F_flat_curr = F_field_curr.reshape(-1, 3, 3)

            # Proper Cauchy: σ = (1/J) P F^T
            P_flat_curr = P_flat.reshape(-1, 3, 3) if P_flat is not None else P_flat
            J = _batched_det(F_flat_curr)  # (nv,)
            sigma_flat = (P_flat_curr @ F_flat_curr.swapaxes(-2, -1)
                          ) / J[:, None, None]

            # Average Cauchy stress → Voigt
            sigma_mean = np.mean(sigma_flat, axis=0)  # (3,3)
            sig_mean_v = np.array([sigma_mean[0, 0], sigma_mean[1, 1],
                                   sigma_mean[2, 2], sigma_mean[1, 2],
                                   sigma_mean[0, 2], sigma_mean[0, 1]])

            stress_err = norm(sig_mean_v - S_target) / max(norm(S_target), 1e-10)

            if verbose:
                print(f"  Newton {newton}: |ΔS|/|S|={stress_err:.4e}  "
                      f"<σ₁₁>={sig_mean_v[0] / 1e6:.1f} MPa  "
                      f"(FFT: {fft_it + 1} iters, Δε={err_fft:.2e})")

            if stress_err < tol_stress:
                if verbose:
                    print(f"  Converged at Newton iter {newton}")
                break

            # Broyden update
            if E_bar_prev is not None and sig_mean_prev is not None:
                dE = E_bar - E_bar_prev
                dS = sig_mean_v - sig_mean_prev
                dS_n2 = np.dot(dS, dS)
                if dS_n2 > 1e-30:
                    S_eff += np.outer(dE - S_eff @ dS, dS) / dS_n2

            E_bar_prev = E_bar.copy()
            sig_mean_prev = sig_mean_v.copy()

            # Newton correction
            dS_corr = S_target - sig_mean_v
            dE_corr = S_eff @ dS_corr
            E_bar += dE_corr
            H_bar_t = strain_v2t(E_bar)
            H_field += strain_v2t(dE_corr)

        # --- Commit state ---
        if last_dgamma is not None:
            F_commit = (H_field + I3).reshape(-1, 3, 3)
            state.commit_state(last_dgamma, F_field_flat=F_commit)

        # --- GND update ---
        if (state.enable_gnd and compute_gnd_every > 0 and
                inc % compute_gnd_every == 0):
            update_gnd(state, derivative_scheme=derivative_scheme)
            # Re-compute resistance with GND contribution
            if state.km is not None:
                total_rho = state.rho_ssd + state.rho_gnd
                state.slip_resistance = state.km.resistance_from_density(total_rho)

        # --- Store results in small-strain-compatible format ---
        eps_v = strain_field_t2v(_sym(H_field))    # (N,N,N,6)
        sig_v = stress_field_t2v(sigma_flat.reshape(shape + (3, 3)))  # (N,N,N,6)
        history.append((eps_v, sig_v, frac))

        # Snapshots for GUI
        state._history_snapshots.append({
            'ep_eq': state.get_von_mises_plastic_strain_field().copy(),
            'acc_slip': state.get_total_accumulated_slip_field().copy(),
            'slip_res': state.get_slip_resistance_field().copy(),
            'ssd_density': state.get_ssd_density_field().copy(),
            'gnd_density': state.get_gnd_density_field().copy(),
            'backstress': state.get_backstress_field().copy(),
            'misorientation': state.get_misorientation_field().copy(),
        })

        if callback:
            callback(inc, eps_v, sig_v, state)

    return history


# ============================================================================
#  Quick Self-Test
# ============================================================================

if __name__ == '__main__':
    from microstructure import generate_voronoi_microstructure

    print("=" * 60)
    print("  Finite-Strain EVPFFT Self-Test")
    print("=" * 60)

    N = 8
    n_grains = 4
    grain_ids, centers, euler_angles = generate_voronoi_microstructure(
        N, n_grains, seed=42)

    print("\n--- Creating FiniteStrainState (Kocks-Mecking + back-stress) ---")
    state = FiniteStrainState(
        grain_ids, euler_angles,
        hardening_model='kocks_mecking',
        enable_backstress=True,
        enable_gnd=True,
        temperature=1123.0)

    print(f"  Grid: {state.shape}, Grains: {state.n_grains}")
    print(f"  Initial SSD density: {state.rho_ssd[0, 0]:.2e} m⁻²")
    print(f"  Initial slip resistance: {state.slip_resistance[0, 0] / 1e6:.1f} MPa")

    # Test explicit constitutive update
    print("\n--- Testing constitutive_update (explicit) ---")
    I3 = np.eye(3)
    F_test = np.tile(I3, (state.n_voxels, 1, 1))
    F_test[:, 0, 0] = 1.01  # 1% tensile strain

    import time
    # Warm up
    P, dg = state.constitutive_update(F_test, dt=0.1)
    # Benchmark
    t0 = time.perf_counter()
    n_bench = 50
    for _ in range(n_bench):
        P, dg = state.constitutive_update(F_test, dt=0.1)
    t_explicit = (time.perf_counter() - t0) / n_bench
    print(f"  Max |P_11|: {np.max(np.abs(P[:, 0, 0])) / 1e9:.3f} GPa")
    print(f"  Max |Δγ|: {np.max(np.abs(dg)):.6f}")
    print(f"  Explicit update: {t_explicit*1000:.2f} ms/call ({N}³ = {N**3} voxels)")

    # Test implicit constitutive update
    print("\n--- Testing constitutive_update_implicit ---")
    # Warm up
    P_imp, dg_imp = state.constitutive_update_implicit(F_test, dt=0.1)
    t0 = time.perf_counter()
    n_bench_imp = 10
    for _ in range(n_bench_imp):
        P_imp, dg_imp = state.constitutive_update_implicit(F_test, dt=0.1)
    t_implicit = (time.perf_counter() - t0) / n_bench_imp
    print(f"  Max |P_11|: {np.max(np.abs(P_imp[:, 0, 0])) / 1e9:.3f} GPa")
    print(f"  Max |Δγ|: {np.max(np.abs(dg_imp)):.6f}")
    print(f"  Implicit update: {t_implicit*1000:.2f} ms/call ({N}³ = {N**3} voxels)")

    # Test GND
    print("\n--- Testing Nye tensor / GND ---")
    state.Fp[:, 0, 1] = 0.001 * np.random.randn(state.n_voxels)  # perturb Fp
    update_gnd(state, derivative_scheme='continuous')
    print(f"  Max GND density: {np.max(state.rho_gnd):.2e} m⁻²")

    # Test full solver (small)
    print("\n--- Running finite-strain EVPFFT (200 MPa, 3 steps) ---")
    # Reset state
    state = FiniteStrainState(
        grain_ids, euler_angles,
        hardening_model='kocks_mecking',
        enable_backstress=True,
        enable_gnd=True,
        temperature=1123.0)

    S_macro = np.array([200e6, 0, 0, 0, 0, 0])
    history = solve_evpfft_finite_strain(
        state, S_macro, n_increments=3, dt=0.1,
        tol_fft=1e-4, tol_stress=1e-3,
        max_iter_fft=50, max_iter_stress=15,
        verbose=True,
        implicit_constitutive=False,
        compute_gnd_every=1)

    from fft_solver import von_mises_stress
    eps_final, sig_final, _ = history[-1]
    vm = von_mises_stress(sig_final)
    print(f"\n  Max VM stress: {np.max(vm) / 1e6:.1f} MPa")
    ep = state.get_von_mises_plastic_strain_field()
    print(f"  Max eq. plastic strain: {np.max(ep):.6f}")
    print(f"  Mean slip resistance: {np.mean(state.slip_resistance) / 1e6:.1f} MPa")
    print(f"  Max GND density: {np.max(state.rho_gnd):.2e} m⁻²")
    if state.chi is not None:
        print(f"  Max |back-stress|: {np.max(np.abs(state.chi)) / 1e6:.2f} MPa")
    print(f"  Max misorientation: {np.max(state.get_misorientation_field()):.3f}°")
    print("  Done.")
