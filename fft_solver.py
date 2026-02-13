"""
FFT-Based Micromechanical Solver (Tensor Formulation)
======================================================
Full-field solver for heterogeneous elastic and elasto-viscoplastic
problems on periodic RVEs using the spectral (FFT) method.

Uses full (3,3) tensor notation internally to avoid Voigt factor 
issues. Implements both the basic fixed-point iteration (Moulinec &
Suquet 1994/1998) and conjugate gradient acceleration (Zeman et al. 2010).

References:
  - Moulinec & Suquet, C. R. Acad. Sci. Paris, 318(II):1417-1423, 1994
  - Moulinec & Suquet, Comput. Methods Appl. Mech. Eng., 157:69-94, 1998
  - Zeman et al., Int. J. Numer. Meth. Eng., 82:1296-1313, 2010
  - Lebensohn, Acta Mater., 49:2723-2737, 2001
  - Lucarini et al., Modelling Simul. Mater. Sci. Eng., 30:023002, 2022
  - Berbenni et al., Int. J. Solids Struct., 51:4460-4469, 2014
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq


# ============================================================================
# Voigt <-> Tensor conversion utilities
# ============================================================================

_VOIGT_MAP = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]

def strain_v2t(v):
    """(6,) Voigt strain -> (3,3) tensor strain."""
    return np.array([
        [v[0],     v[5]/2, v[4]/2],
        [v[5]/2,   v[1],   v[3]/2],
        [v[4]/2,   v[3]/2, v[2]  ],
    ])

def stress_v2t(v):
    """(6,) Voigt stress -> (3,3) tensor stress."""
    return np.array([
        [v[0], v[5], v[4]],
        [v[5], v[1], v[3]],
        [v[4], v[3], v[2]],
    ])

def strain_t2v(t):
    """(3,3) tensor strain -> (6,) Voigt strain."""
    return np.array([t[0,0], t[1,1], t[2,2], 2*t[1,2], 2*t[0,2], 2*t[0,1]])

def stress_t2v(t):
    """(3,3) tensor stress -> (6,) Voigt stress."""
    return np.array([t[0,0], t[1,1], t[2,2], t[1,2], t[0,2], t[0,1]])

def strain_field_v2t(vf):
    """(..., 6) Voigt strain field -> (..., 3, 3) tensor field."""
    shape = vf.shape[:-1]
    t = np.zeros(shape + (3, 3), dtype=vf.dtype)
    t[..., 0, 0] = vf[..., 0]
    t[..., 1, 1] = vf[..., 1]
    t[..., 2, 2] = vf[..., 2]
    t[..., 1, 2] = t[..., 2, 1] = vf[..., 3] / 2
    t[..., 0, 2] = t[..., 2, 0] = vf[..., 4] / 2
    t[..., 0, 1] = t[..., 1, 0] = vf[..., 5] / 2
    return t

def stress_field_v2t(vf):
    """(..., 6) Voigt stress field -> (..., 3, 3) tensor field."""
    shape = vf.shape[:-1]
    t = np.zeros(shape + (3, 3), dtype=vf.dtype)
    t[..., 0, 0] = vf[..., 0]
    t[..., 1, 1] = vf[..., 1]
    t[..., 2, 2] = vf[..., 2]
    t[..., 1, 2] = t[..., 2, 1] = vf[..., 3]
    t[..., 0, 2] = t[..., 2, 0] = vf[..., 4]
    t[..., 0, 1] = t[..., 1, 0] = vf[..., 5]
    return t

def strain_field_t2v(tf):
    """(..., 3, 3) tensor strain field -> (..., 6) Voigt field."""
    shape = tf.shape[:-2]
    v = np.zeros(shape + (6,), dtype=np.float64)
    v[..., 0] = np.real(tf[..., 0, 0])
    v[..., 1] = np.real(tf[..., 1, 1])
    v[..., 2] = np.real(tf[..., 2, 2])
    v[..., 3] = 2 * np.real(tf[..., 1, 2])
    v[..., 4] = 2 * np.real(tf[..., 0, 2])
    v[..., 5] = 2 * np.real(tf[..., 0, 1])
    return v

def stress_field_t2v(tf):
    """(..., 3, 3) tensor stress field -> (..., 6) Voigt field."""
    shape = tf.shape[:-2]
    v = np.zeros(shape + (6,), dtype=np.float64)
    v[..., 0] = np.real(tf[..., 0, 0])
    v[..., 1] = np.real(tf[..., 1, 1])
    v[..., 2] = np.real(tf[..., 2, 2])
    v[..., 3] = np.real(tf[..., 1, 2])
    v[..., 4] = np.real(tf[..., 0, 2])
    v[..., 5] = np.real(tf[..., 0, 1])
    return v


# ============================================================================
# Constitutive law: stress = C : strain  (Voigt C, tensor fields)
# ============================================================================

def apply_stiffness(C_voigt, eps_tensor):
    """
    Compute stress from Voigt stiffness and tensor strain.

    C_voigt   : (..., 6, 6) Voigt stiffness
    eps_tensor: (..., 3, 3) tensor strain
    Returns   : (..., 3, 3) tensor stress
    """
    eps_v = strain_field_t2v(eps_tensor)
    sig_v = np.einsum('...ij,...j->...i', C_voigt, eps_v)
    return stress_field_v2t(sig_v)


# ============================================================================
# Green's Operator (acoustic tensor approach — no Voigt issues)
# ============================================================================

def build_green_data(N, lam0, mu0):
    """
    Pre-compute data needed for the Green's operator application.

    For isotropic reference medium with Lame constants (lam0, mu0):
        Acoustic tensor: A_ik(n) = (lam0+mu0) n_i n_k + mu0 delta_ik
        Inverse:         A^{-1}_ik(n) = delta_ik/mu0 - alpha * n_i n_k
        where alpha = (lam0+mu0) / (mu0 * (lam0 + 2*mu0))

    Returns:
        n_field  : (N,N,N, 3) unit direction at each frequency
        Ainv     : (N,N,N, 3, 3) inverse acoustic tensor
    """
    freq = fftfreq(N, d=1.0/N)
    xi1, xi2, xi3 = np.meshgrid(freq, freq, freq, indexing='ij')
    xi = np.stack([xi1, xi2, xi3], axis=-1)  # (N,N,N,3)

    xi_sq = np.sum(xi**2, axis=-1)  # (N,N,N)
    xi_sq_safe = np.where(xi_sq < 1e-30, 1.0, xi_sq)
    xi_norm = np.sqrt(xi_sq_safe)

    # Unit direction n = xi / |xi|
    n = xi / xi_norm[..., np.newaxis]  # (N,N,N,3)

    # Inverse acoustic tensor
    alpha = (lam0 + mu0) / (mu0 * (lam0 + 2 * mu0))

    Ainv = np.zeros((N, N, N, 3, 3))
    Ainv[..., 0, 0] = 1.0 / mu0
    Ainv[..., 1, 1] = 1.0 / mu0
    Ainv[..., 2, 2] = 1.0 / mu0
    Ainv -= alpha * np.einsum('...i,...j->...ij', n, n)

    # Zero DC
    Ainv[0, 0, 0, :, :] = 0.0
    n[0, 0, 0, :] = 0.0

    return n, Ainv


def apply_green_operator_tensor(tau_hat, n_field, Ainv):
    """
    Apply the Green's operator to polarization tau_hat in Fourier space.

    Algorithm (acoustic tensor approach):
        b_k = n_j * tau_hat_kj        (project stress onto direction)
        u_i = A^{-1}_ik * b_k         (solve for displacement amplitude)
        eps_hat_ij = sym(n_i * u_j)    (strain from displacement gradient)

    This is: eps_hat_ij = 0.5 * (n_i * u_j + n_j * u_i)
    which exactly reproduces Gamma^0_ijkl : tau_hat_kl without Voigt factors.

    Parameters:
        tau_hat  : (N,N,N, 3,3) complex — FFT of polarization stress
        n_field  : (N,N,N, 3) — unit frequency directions
        Ainv     : (N,N,N, 3,3) — inverse acoustic tensor

    Returns:
        eps_hat  : (N,N,N, 3,3) complex — strain correction in Fourier space
    """
    # b_k = n_j * tau_hat_kj  (contract second index of tau with n)
    b = np.einsum('...kj,...j->...k', tau_hat, n_field)

    # u_i = A^{-1}_ik * b_k
    u = np.einsum('...ik,...k->...i', Ainv, b)

    # eps_hat_ij = (n_i * u_j + n_j * u_i) / 2
    eps_hat = 0.5 * (np.einsum('...i,...j->...ij', n_field, u) +
                     np.einsum('...j,...i->...ij', n_field, u))

    return eps_hat


def fft_3x3(field):
    """FFT of a (N,N,N,3,3) tensor field, component by component."""
    shape = field.shape[:3]
    out = np.zeros(shape + (3,3), dtype=complex)
    for i in range(3):
        for j in range(3):
            out[:,:,:,i,j] = fftn(field[:,:,:,i,j])
    return out


def ifft_3x3(field_hat):
    """Inverse FFT of a (N,N,N,3,3) complex field, returning real."""
    shape = field_hat.shape[:3]
    out = np.zeros(shape + (3,3))
    for i in range(3):
        for j in range(3):
            out[:,:,:,i,j] = np.real(ifftn(field_hat[:,:,:,i,j]))
    return out


# ============================================================================
# Reference Medium
# ============================================================================

def compute_reference_medium(C_field):
    """
    Compute isotropic reference medium from Voigt-averaged stiffness.

    Returns: C0 (6,6), lam0 (float), mu0 (float)
    """
    C0 = np.mean(C_field.reshape(-1, 6, 6), axis=0)

    # Voigt average bulk and shear moduli
    K = (C0[0,0] + C0[1,1] + C0[2,2] + 2*(C0[0,1] + C0[0,2] + C0[1,2])) / 9.0
    mu_V = (C0[0,0] + C0[1,1] + C0[2,2] - C0[0,1] - C0[0,2] - C0[1,2] +
            3*(C0[3,3] + C0[4,4] + C0[5,5])) / 15.0

    lam0 = K - 2*mu_V/3
    return C0, lam0, mu_V


# ============================================================================
# Equilibrium Error (spectral)
# ============================================================================

def _equilibrium_error(sig_tensor, N):
    """
    Normalized equilibrium error: || xi_j sigma_hat_ij || / || sigma_hat ||
    Note: this measures spectral equilibrium and has a discretization floor
    for microstructures with sharp grain boundaries.
    """
    freq = fftfreq(N, d=1.0/N)
    xi1, xi2, xi3 = np.meshgrid(freq, freq, freq, indexing='ij')

    sig_hat = fft_3x3(sig_tensor)

    # div_i = xi_j sigma_hat_ij
    div = np.zeros((N,N,N,3), dtype=complex)
    for i in range(3):
        div[..., i] = xi1 * sig_hat[...,i,0] + xi2 * sig_hat[...,i,1] + xi3 * sig_hat[...,i,2]

    div_norm = np.sqrt(np.sum(np.abs(div)**2))
    sig_norm = np.sqrt(np.sum(np.abs(sig_hat)**2))

    return div_norm / sig_norm if sig_norm > 1e-30 else 0.0


def _strain_change(eps_new, eps_old):
    """Relative norm of strain change between iterations."""
    diff = np.sum((eps_new - eps_old)**2)
    norm = np.sum(eps_old**2)
    return np.sqrt(diff / norm) if norm > 1e-30 else 0.0


# ============================================================================
# Basic Scheme (Moulinec-Suquet 1994/1998)
# ============================================================================

def solve_basic_scheme(C_field, E_macro_voigt, tol=1e-6, max_iter=500,
                        verbose=True, callback=None):
    """
    Solve the Lippmann-Schwinger equation: basic fixed-point iteration.

    Lippmann-Schwinger:  eps = E_bar - Gamma0 * tau
    where tau = (C - C0) : eps  is the polarization.

    Parameters:
        C_field       : (N,N,N, 6,6) local Voigt stiffness
        E_macro_voigt : (6,) macroscopic strain (Voigt)
        tol, max_iter, verbose, callback

    Returns:
        eps_voigt : (N,N,N, 6) converged strain (Voigt)
        sig_voigt : (N,N,N, 6) converged stress (Voigt)
        info      : dict with convergence data
    """
    N = C_field.shape[0]

    C0, lam0, mu0 = compute_reference_medium(C_field)
    if verbose:
        print(f"FFT Basic Scheme | N={N}^3 | mu0={mu0/1e9:.2f} GPa | lam0={lam0/1e9:.2f} GPa")

    n_field, Ainv = build_green_data(N, lam0, mu0)
    E_bar = strain_v2t(E_macro_voigt)
    C0_broadcast = C0[np.newaxis, np.newaxis, np.newaxis]

    # Initialize strain to uniform macro strain
    eps = np.zeros((N,N,N,3,3))
    eps[...] = E_bar

    errors = []
    converged = False

    for it in range(max_iter):
        eps_old = eps.copy()

        # sigma = C : eps
        sig = apply_stiffness(C_field, eps)

        # polarization tau = sigma - C0 : eps = (C - C0) : eps
        sig0 = apply_stiffness(C0_broadcast, eps)
        tau = sig - sig0

        # Fourier space: Gamma0 : tau
        tau_hat = fft_3x3(tau)
        gamma_tau_hat = apply_green_operator_tensor(tau_hat, n_field, Ainv)
        gamma_tau_hat[0, 0, 0, :, :] = 0.0  # DC = 0

        gamma_tau = ifft_3x3(gamma_tau_hat)

        # Lippmann-Schwinger:  eps = E_bar - Gamma0 * tau
        eps = E_bar - gamma_tau

        # Convergence: relative strain change
        err = _strain_change(eps, eps_old)
        errors.append(err)

        if verbose and (it % 10 == 0 or err < tol):
            sig = apply_stiffness(C_field, eps)
            sm = np.mean(sig.reshape(-1,3,3), axis=0)
            eq_err = _equilibrium_error(sig, N)
            print(f"  Iter {it:4d}: Δε={err:.4e}  eq_err={eq_err:.4e}  <σ_11>={sm[0,0]/1e6:.1f} MPa")

        if callback:
            callback(it, strain_field_t2v(eps), stress_field_t2v(sig), err)

        if err < tol:
            converged = True
            if verbose:
                print(f"  Converged at iteration {it} (err = {err:.2e})")
            break

    if not converged and verbose:
        print(f"  Not converged after {max_iter} iters (err = {errors[-1]:.2e})")

    sig = apply_stiffness(C_field, eps)
    info = {
        'converged': converged,
        'iterations': it + 1,
        'errors': np.array(errors),
        'final_error': errors[-1] if errors else float('inf'),
        'C0': C0, 'lam0': lam0, 'mu0': mu0,
    }
    return strain_field_t2v(eps), stress_field_t2v(sig), info


# ============================================================================
# Conjugate Gradient Scheme (Zeman et al. 2010)
# ============================================================================

def solve_conjugate_gradient(C_field, E_macro_voigt, tol=1e-6, max_iter=500,
                              verbose=True, callback=None):
    """
    CG-accelerated solution of the Lippmann-Schwinger equation.

    Decomposes strain: eps = E_bar + eps_tilde  (eps_tilde has zero mean).
    Solves for the fluctuation:
        (I + Gamma0*dC) eps_tilde = -Gamma0 * dC * E_bar
    """
    N = C_field.shape[0]
    Nvox = N**3

    C0, lam0, mu0 = compute_reference_medium(C_field)
    if verbose:
        print(f"FFT CG Scheme | N={N}^3 | mu0={mu0/1e9:.2f} GPa | lam0={lam0/1e9:.2f} GPa")

    n_field, Ainv = build_green_data(N, lam0, mu0)
    E_bar = strain_v2t(E_macro_voigt)
    dC = C_field - C0

    def apply_gamma(tau_tensor):
        """Gamma0 : tau  (real space in, real space out, zero-mean output)."""
        th = fft_3x3(tau_tensor)
        eh = apply_green_operator_tensor(th, n_field, Ainv)
        eh[0,0,0,:,:] = 0.0
        return ifft_3x3(eh)

    def apply_A(x):
        """Operator: A(x) = x + Gamma0 * dC * x."""
        return x + apply_gamma(apply_stiffness(dC, x))

    def inner(a, b):
        return np.sum(a * b)

    # RHS of fluctuation equation: b = -Gamma0 * dC * E_bar
    E_bar_field = np.zeros((N,N,N,3,3))
    E_bar_field[...] = E_bar
    dC_E = apply_stiffness(dC, E_bar_field)
    b = -apply_gamma(dC_E)

    # Initialize fluctuation to zero
    eps_tilde = np.zeros((N,N,N,3,3))

    # Initial residual: r = b - A(eps_tilde) = b  (since eps_tilde=0, A(0)=0)
    r = b.copy()
    d = r.copy()
    rr = inner(r, r)
    bb = inner(b, b)  # for relative residual norm

    errors = []
    converged = False

    for it in range(max_iter):
        if rr < 1e-30:
            converged = True
            break

        Ad = apply_A(d)
        dAd = inner(d, Ad)
        if abs(dAd) < 1e-30:
            break

        alpha_cg = rr / dAd
        eps_tilde = eps_tilde + alpha_cg * d
        r = r - alpha_cg * Ad
        rr_new = inner(r, r)

        # Convergence: relative CG residual
        err = np.sqrt(rr_new / bb) if bb > 1e-30 else 0.0
        errors.append(err)

        if verbose and (it % 5 == 0 or err < tol):
            # Compute full strain/stress for diagnostics
            eps = np.zeros_like(eps_tilde)
            eps[...] = E_bar
            eps += eps_tilde
            sig = apply_stiffness(C_field, eps)
            sm = np.mean(sig.reshape(-1,3,3), axis=0)
            eq_err = _equilibrium_error(sig, N)
            print(f"  CG {it:4d}: res={err:.4e}  eq_err={eq_err:.4e}  <σ_11>={sm[0,0]/1e6:.1f} MPa")

        if callback:
            callback(it, strain_field_t2v(eps), stress_field_t2v(sig), err)

        if err < tol:
            converged = True
            if verbose:
                print(f"  CG converged at iteration {it} (err = {err:.2e})")
            break

        beta_cg = rr_new / rr
        d = r + beta_cg * d
        rr = rr_new

    if not converged and verbose:
        print(f"  CG not converged after {max_iter} iters (err = {errors[-1]:.2e})")

    # Final full strain and stress
    eps = np.zeros_like(eps_tilde)
    eps[...] = E_bar
    eps += eps_tilde
    sig = apply_stiffness(C_field, eps)

    info = {
        'converged': converged,
        'iterations': it + 1 if errors else 0,
        'errors': np.array(errors),
        'final_error': errors[-1] if errors else float('inf'),
        'C0': C0, 'lam0': lam0, 'mu0': mu0,
    }
    return strain_field_t2v(eps), stress_field_t2v(sig), info


# ============================================================================
# Stress-Controlled Solver (Newton iteration on macroscopic strain)
# ============================================================================

def solve_stress_controlled(C_field, S_macro_voigt, tol_fft=1e-6, tol_stress=1e-4,
                             max_iter_fft=500, max_iter_stress=30, solver='cg',
                             verbose=True, callback=None):
    """
    Solve for the macroscopic strain that produces a target average stress.

    Uses Newton-Raphson: iterate on E_macro until <sigma(E_macro)> = S_target.
    The effective compliance is estimated from a finite-difference Jacobian.

    Parameters:
        C_field        : (N,N,N,6,6) local stiffness
        S_macro_voigt  : (6,) target macroscopic stress in Voigt [Pa]
        tol_fft        : FFT solver convergence tolerance
        tol_stress     : relative stress error tolerance
        max_iter_fft   : max FFT iterations per solve
        max_iter_stress: max Newton iterations
        solver         : 'cg' or 'basic'
        verbose, callback

    Returns:
        eps_voigt : (N,N,N,6) converged strain field
        sig_voigt : (N,N,N,6) converged stress field
        info      : dict with convergence details
    """
    solve_fn = solve_conjugate_gradient if solver == 'cg' else solve_basic_scheme
    S_target = S_macro_voigt.copy()
    S_norm = np.linalg.norm(S_target)
    if S_norm < 1e-20:
        # Zero stress → zero strain
        N = C_field.shape[0]
        return np.zeros((N,N,N,6)), np.zeros((N,N,N,6)), {
            'converged': True, 'iterations': 0, 'stress_iterations': 0,
            'errors': np.array([0.0]), 'final_error': 0.0, 'final_stress_error': 0.0,
            'E_macro_converged': np.zeros(6)}

    # Initial strain guess from Voigt-averaged compliance
    C0, _, _ = compute_reference_medium(C_field)
    try:
        S0 = np.linalg.inv(C0)
        E_macro = S0 @ S_target
    except np.linalg.LinAlgError:
        E_macro = S_target / (C0[0,0] if C0[0,0] > 0 else 200e9)

    if verbose:
        print(f"Stress-controlled solver | target σ (MPa): "
              f"[{', '.join(f'{s/1e6:.1f}' for s in S_target)}]")
        print(f"  Initial strain guess: [{', '.join(f'{e:.4e}' for e in E_macro)}]")

    stress_errors = []
    total_fft_iters = 0

    for newton_it in range(max_iter_stress):
        # Solve FFT with current strain guess
        eps_v, sig_v, info_fft = solve_fn(
            C_field, E_macro, tol=tol_fft, max_iter=max_iter_fft, verbose=False)
        total_fft_iters += info_fft['iterations']

        # Volume-averaged stress
        sig_avg = np.mean(sig_v.reshape(-1, 6), axis=0)

        # Stress residual
        dS = S_target - sig_avg
        rel_err = np.linalg.norm(dS) / S_norm
        stress_errors.append(rel_err)

        if verbose:
            print(f"  Newton {newton_it:3d}: |ΔS|/|S|={rel_err:.4e}  "
                  f"<σ₁₁>={sig_avg[0]/1e6:.1f}  <σ₂₂>={sig_avg[1]/1e6:.1f}  "
                  f"<σ₃₃>={sig_avg[2]/1e6:.1f} MPa  (FFT: {info_fft['iterations']} iters)")

        if rel_err < tol_stress:
            if verbose:
                print(f"  Stress-control converged at Newton iter {newton_it} "
                      f"(err = {rel_err:.2e})")
            break

        # Update strain: use Voigt-average compliance as approximate Jacobian
        # dE = S0 @ dS  (Newton correction)
        try:
            dE = S0 @ dS
        except:
            dE = dS / C0[0,0]
        E_macro = E_macro + dE

    converged = rel_err < tol_stress
    if not converged and verbose:
        print(f"  Stress-control not converged after {max_iter_stress} Newton iters "
              f"(err = {stress_errors[-1]:.2e})")

    info = {
        'converged': converged,
        'iterations': total_fft_iters,
        'stress_iterations': newton_it + 1,
        'errors': np.array(stress_errors),
        'final_error': info_fft['final_error'],
        'final_stress_error': stress_errors[-1] if stress_errors else float('inf'),
        'E_macro_converged': E_macro,
        'C0': info_fft.get('C0'), 'lam0': info_fft.get('lam0'), 'mu0': info_fft.get('mu0'),
    }
    return eps_v, sig_v, info


# ============================================================================
# Effective (homogenized) stiffness
# ============================================================================

def compute_effective_stiffness(C_field, solver='cg', tol=1e-6, max_iter=200, verbose=False):
    """Compute effective stiffness C_eff by solving 6 loading cases."""
    C_eff = np.zeros((6, 6))
    solve_fn = solve_conjugate_gradient if solver == 'cg' else solve_basic_scheme

    for J in range(6):
        E_macro = np.zeros(6)
        E_macro[J] = 1.0
        if verbose:
            print(f"\n--- Loading case {J+1}/6: E[{J}] = 1 ---")
        _, sig, _ = solve_fn(C_field, E_macro, tol=tol, max_iter=max_iter, verbose=verbose)
        C_eff[:, J] = np.mean(sig.reshape(-1, 6), axis=0)

    return C_eff


# ============================================================================
# Post-processing
# ============================================================================

def von_mises_stress(sig_voigt):
    """Von Mises equivalent stress from Voigt [s11,s22,s33,s23,s13,s12]."""
    s = sig_voigt
    return np.sqrt(0.5 * ((s[...,0]-s[...,1])**2 + (s[...,1]-s[...,2])**2 +
                           (s[...,2]-s[...,0])**2 +
                           6*(s[...,3]**2 + s[...,4]**2 + s[...,5]**2)))


def von_mises_strain(eps_voigt):
    """Von Mises equivalent strain from Voigt [e11,e22,e33,2e23,2e13,2e12]."""
    e = eps_voigt
    g23 = e[...,3]/2; g13 = e[...,4]/2; g12 = e[...,5]/2
    return np.sqrt(2.0/3.0 * ((e[...,0]-e[...,1])**2 + (e[...,1]-e[...,2])**2 +
                                (e[...,2]-e[...,0])**2 +
                                6*(g23**2 + g13**2 + g12**2)))


# ============================================================================
# Quick test
# ============================================================================

if __name__ == '__main__':
    from microstructure import generate_voronoi_microstructure, build_local_stiffness_field_fast
    from microstructure import cubic_stiffness_tensor, rotate_stiffness_voigt

    print("=" * 60)
    print("  FFT Solver Test - Tensor Formulation")
    print("=" * 60)

    # --- Test 1: Homogeneous medium (validation) ---
    print("\n--- Test 1: Homogeneous Medium (should give ~0 error) ---")
    N = 8
    C11, C12, C44 = 168.4e9, 121.4e9, 75.4e9
    C_cubic = cubic_stiffness_tensor(C11, C12, C44)
    C_homo = np.tile(C_cubic, (N,N,N,1,1))
    E_macro = np.array([0.01, 0, 0, 0, 0, 0])

    eps_h, sig_h, info_h = solve_basic_scheme(C_homo, E_macro, tol=1e-10, max_iter=5, verbose=True)
    print(f"  Homogeneous error: {info_h['final_error']:.2e} (should be ~1e-16)")

    # --- Test 2: Polycrystalline copper ---
    print("\n--- Test 2: Polycrystalline Copper ---")
    N = 16
    n_grains = 8
    grain_ids, centers, euler_angles = generate_voronoi_microstructure(N, n_grains, seed=42)
    C_field = build_local_stiffness_field_fast(grain_ids, euler_angles, C11, C12, C44)

    print("\n  Basic Scheme:")
    eps_b, sig_b, info_b = solve_basic_scheme(C_field, E_macro, tol=1e-5, max_iter=200)
    print(f"  Mean sigma (GPa): {np.mean(sig_b.reshape(-1,6), axis=0) / 1e9}")
    print(f"  Max VM stress: {np.max(von_mises_stress(sig_b))/1e9:.3f} GPa")

    print("\n  CG Scheme:")
    eps_cg, sig_cg, info_cg = solve_conjugate_gradient(C_field, E_macro, tol=1e-5, max_iter=200)
    print(f"  Mean sigma (GPa): {np.mean(sig_cg.reshape(-1,6), axis=0) / 1e9}")
    print(f"  Max VM stress: {np.max(von_mises_stress(sig_cg))/1e9:.3f} GPa")

    print(f"\n  Basic: {info_b['iterations']} iters | CG: {info_cg['iterations']} iters")
