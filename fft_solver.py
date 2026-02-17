"""
FFT-Based Micromechanical Solver (Tensor Formulation) — v2
============================================================
Full-field solver for heterogeneous elastic and elasto-viscoplastic
problems on periodic RVEs using the spectral (FFT) method.

v2 features:
  - Discrete derivative schemes: continuous / rotated / finite_difference
  - Anderson acceleration for fixed-point (Moulinec-Suquet) iteration
  - Reference medium: mean / bulk_shear / contrast_aware
  - Stress-control with Broyden secant update
  - Batched tensor FFT (GPU), optional real FFT (CPU)
  - Per-iteration profiling via solver_utils.SolverProfiler
  - Physics-invariant debug checks

References:
  - Moulinec & Suquet, CMAME 157:69-94, 1998
  - Zeman et al., IJNME 82:1296-1313, 2010
  - Willot, CMAME 294:313-344, 2015  (discrete / rotated derivatives)
  - Lebensohn, Acta Mater. 49:2723-2737, 2001
  - Lucarini et al., MSME 30:023002, 2022
"""

import numpy as np
from numpy.fft import fftn, ifftn, rfftn, irfftn, fftfreq

# --------------- GPU acceleration (CuPy, optional) ---------------
try:
    # Ensure pip-installed NVIDIA DLLs are discoverable before CuPy import
    import os as _os, sys as _sys
    if _sys.platform == 'win32':
        # CuPy needs nvrtc, cublas, cusolver, cusparse, cufft, curand, nvjitlink
        _nvidia_libs = [
            'nvidia.cuda_nvrtc', 'nvidia.cuda_runtime', 'nvidia.cublas',
            'nvidia.cusolver', 'nvidia.cusparse', 'nvidia.cufft',
            'nvidia.curand', 'nvidia.nvjitlink',
        ]
        import importlib
        for _mod_name in _nvidia_libs:
            try:
                _mod = importlib.import_module(_mod_name)
                _bin = _os.path.join(_mod.__path__[0], 'bin')
                if _os.path.isdir(_bin):
                    if hasattr(_os, 'add_dll_directory'):
                        _os.add_dll_directory(_bin)
                    _os.environ['PATH'] = _bin + _os.pathsep + _os.environ.get('PATH', '')
            except ImportError:
                pass

    import cupy as cp
    import cupy.fft as cpfft
    HAS_GPU = True
    print("[FFT Solver] GPU acceleration enabled (CuPy + CUDA)")
except (ImportError, Exception) as _e:
    HAS_GPU = False
    print(f"[FFT Solver] GPU not available, falling back to CPU: {_e}")

def _xp(*arrays):
    """Return cupy if any input lives on GPU, else numpy."""
    if HAS_GPU:
        for a in arrays:
            if isinstance(a, cp.ndarray):
                return cp
    return np

def _to_gpu(arr):
    """Transfer ndarray to GPU if CuPy available."""
    return cp.asarray(arr) if HAS_GPU else arr

def _to_cpu(arr):
    """Transfer array to CPU (always returns numpy ndarray)."""
    if HAS_GPU and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


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
    xp = _xp(vf)
    shape = vf.shape[:-1]
    t = xp.zeros(shape + (3, 3), dtype=vf.dtype)
    t[..., 0, 0] = vf[..., 0]
    t[..., 1, 1] = vf[..., 1]
    t[..., 2, 2] = vf[..., 2]
    t[..., 1, 2] = t[..., 2, 1] = vf[..., 3] / 2
    t[..., 0, 2] = t[..., 2, 0] = vf[..., 4] / 2
    t[..., 0, 1] = t[..., 1, 0] = vf[..., 5] / 2
    return t

def stress_field_v2t(vf):
    """(..., 6) Voigt stress field -> (..., 3, 3) tensor field."""
    xp = _xp(vf)
    shape = vf.shape[:-1]
    t = xp.zeros(shape + (3, 3), dtype=vf.dtype)
    t[..., 0, 0] = vf[..., 0]
    t[..., 1, 1] = vf[..., 1]
    t[..., 2, 2] = vf[..., 2]
    t[..., 1, 2] = t[..., 2, 1] = vf[..., 3]
    t[..., 0, 2] = t[..., 2, 0] = vf[..., 4]
    t[..., 0, 1] = t[..., 1, 0] = vf[..., 5]
    return t

def strain_field_t2v(tf):
    """(..., 3, 3) tensor strain field -> (..., 6) Voigt field."""
    xp = _xp(tf)
    shape = tf.shape[:-2]
    v = xp.zeros(shape + (6,), dtype=np.float64)
    v[..., 0] = xp.real(tf[..., 0, 0])
    v[..., 1] = xp.real(tf[..., 1, 1])
    v[..., 2] = xp.real(tf[..., 2, 2])
    v[..., 3] = 2 * xp.real(tf[..., 1, 2])
    v[..., 4] = 2 * xp.real(tf[..., 0, 2])
    v[..., 5] = 2 * xp.real(tf[..., 0, 1])
    return v

def stress_field_t2v(tf):
    """(..., 3, 3) tensor stress field -> (..., 6) Voigt field."""
    xp = _xp(tf)
    shape = tf.shape[:-2]
    v = xp.zeros(shape + (6,), dtype=np.float64)
    v[..., 0] = xp.real(tf[..., 0, 0])
    v[..., 1] = xp.real(tf[..., 1, 1])
    v[..., 2] = xp.real(tf[..., 2, 2])
    v[..., 3] = xp.real(tf[..., 1, 2])
    v[..., 4] = xp.real(tf[..., 0, 2])
    v[..., 5] = xp.real(tf[..., 0, 1])
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
    xp = _xp(eps_tensor)
    eps_v = strain_field_t2v(eps_tensor)
    sig_v = xp.einsum('...ij,...j->...i', C_voigt, eps_v)
    return stress_field_v2t(sig_v)


# ============================================================================
# Discrete-Derivative Frequency Vectors
# ============================================================================

def _build_freq_vectors(N, derivative_scheme="continuous", on_gpu=False):
    """
    Modified wave-number vectors for Green operator.

    "continuous"        : k_i = ξ_i  (standard spectral)
    "finite_difference" : k_i = (N/π)sin(π ξ_i/N)  (Willot FD-compatible)
    "rotated"           : k_i = sin(2π ξ_i/N)/h    (rotated scheme)
    """
    xp = cp if (on_gpu and HAS_GPU) else np
    _fftfreq_fn = (cpfft.fftfreq if (on_gpu and HAS_GPU) else fftfreq)
    freq = _fftfreq_fn(N, d=1.0 / N)  # integer frequencies

    if derivative_scheme == "continuous":
        xi1, xi2, xi3 = xp.meshgrid(freq, freq, freq, indexing='ij')
        return xp.stack([xi1, xi2, xi3], axis=-1)

    elif derivative_scheme == "finite_difference":
        kmod = xp.sin(xp.pi * freq / N) / (xp.pi / N)
        xi1, xi2, xi3 = xp.meshgrid(kmod, kmod, kmod, indexing='ij')
        return xp.stack([xi1, xi2, xi3], axis=-1)

    elif derivative_scheme == "rotated":
        h = 1.0 / N
        kmod = xp.sin(2.0 * xp.pi * freq / N) / h
        xi1, xi2, xi3 = xp.meshgrid(kmod, kmod, kmod, indexing='ij')
        return xp.stack([xi1, xi2, xi3], axis=-1)

    else:
        raise ValueError(f"Unknown derivative_scheme: {derivative_scheme!r}")


# ============================================================================
# Green's Operator (acoustic tensor approach — no Voigt issues)
# ============================================================================

def build_green_data(N, lam0, mu0, on_gpu=None, derivative_scheme="continuous"):
    """
    Pre-compute data for the Green's operator.  Built on GPU when available.

    Parameters
    ----------
    derivative_scheme : str  "continuous" | "finite_difference" | "rotated"
    """
    if on_gpu is None:
        on_gpu = HAS_GPU
    xp = cp if on_gpu else np

    xi = _build_freq_vectors(N, derivative_scheme=derivative_scheme, on_gpu=on_gpu)

    xi_sq = xp.sum(xi**2, axis=-1)
    xi_sq_safe = xp.where(xi_sq < 1e-30, 1.0, xi_sq)
    xi_norm = xp.sqrt(xi_sq_safe)

    n = xi / xi_norm[..., None]

    alpha = (lam0 + mu0) / (mu0 * (lam0 + 2 * mu0))

    Ainv = xp.zeros((N, N, N, 3, 3))
    Ainv[..., 0, 0] = 1.0 / mu0
    Ainv[..., 1, 1] = 1.0 / mu0
    Ainv[..., 2, 2] = 1.0 / mu0
    Ainv -= alpha * xp.einsum('...i,...j->...ij', n, n)

    Ainv[0, 0, 0, :, :] = 0.0
    n[0, 0, 0, :] = 0.0

    return n, Ainv


def apply_green_operator_tensor(tau_hat, n_field, Ainv):
    """Apply the Green's operator in Fourier space (GPU-aware)."""
    xp = _xp(tau_hat)
    b = xp.einsum('...kj,...j->...k', tau_hat, n_field)
    u = xp.einsum('...ik,...k->...i', Ainv, b)
    eps_hat = 0.5 * (xp.einsum('...i,...j->...ij', n_field, u) +
                     xp.einsum('...j,...i->...ij', n_field, u))
    return eps_hat


def fft_3x3(field, use_rfft=False):
    """FFT of a (N,N,N,3,3) tensor field. Batched on GPU, component-loop on CPU."""
    if HAS_GPU and isinstance(field, cp.ndarray):
        N = field.shape[0]
        # Batch as (9, N, N, N) for faster GPU FFT
        f9 = field.reshape(N, N, N, 9).transpose(3, 0, 1, 2).copy()
        out9 = cp.empty_like(f9, dtype=cp.complex128)
        for c in range(9):
            out9[c] = cpfft.fftn(f9[c])
        return out9.transpose(1, 2, 3, 0).reshape(N, N, N, 3, 3)
    else:
        N = field.shape[0]
        if use_rfft:
            Nh = N // 2 + 1
            out = np.zeros((N, N, Nh, 3, 3), dtype=complex)
            for i in range(3):
                for j in range(3):
                    out[:, :, :, i, j] = rfftn(field[:, :, :, i, j])
            return out
        else:
            out = np.zeros_like(field, dtype=complex)
            for i in range(3):
                for j in range(3):
                    out[:, :, :, i, j] = fftn(field[:, :, :, i, j])
            return out


def ifft_3x3(field_hat, use_rfft=False, N_full=None):
    """Inverse FFT of a (N,N,N,3,3) complex field. Batched on GPU."""
    if HAS_GPU and isinstance(field_hat, cp.ndarray):
        N = field_hat.shape[0]
        f9 = field_hat.reshape(N, N, N, 9).transpose(3, 0, 1, 2).copy()
        out9 = cp.empty((9, N, N, N), dtype=cp.float64)
        for c in range(9):
            out9[c] = cp.real(cpfft.ifftn(f9[c]))
        return out9.transpose(1, 2, 3, 0).reshape(N, N, N, 3, 3)
    else:
        if use_rfft:
            N = N_full or field_hat.shape[0]
            out = np.zeros((N, N, N, 3, 3))
            for i in range(3):
                for j in range(3):
                    out[:, :, :, i, j] = irfftn(field_hat[:, :, :, i, j], s=(N,N,N))
            return out
        else:
            shape = field_hat.shape[:3]
            out = np.zeros(shape + (3, 3))
            for i in range(3):
                for j in range(3):
                    out[:, :, :, i, j] = np.real(ifftn(field_hat[:, :, :, i, j]))
            return out


# ============================================================================
# Reference Medium
# ============================================================================

def _isotropic_stiffness(lam, mu):
    """Return (6,6) isotropic stiffness from Lamé constants."""
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def _extract_K_mu(C66):
    """Extract isotropic bulk K and shear mu from a (6,6) stiffness."""
    K = (C66[0,0] + C66[1,1] + C66[2,2] + 2*(C66[0,1] + C66[0,2] + C66[1,2])) / 9.0
    mu = (C66[0,0] + C66[1,1] + C66[2,2] - C66[0,1] - C66[0,2] - C66[1,2] +
          3*(C66[3,3] + C66[4,4] + C66[5,5])) / 15.0
    return K, mu


def compute_reference_medium(C_field, mode='mean'):
    """
    Compute isotropic reference medium from a heterogeneous stiffness field.

    Parameters:
        C_field : (N,N,N,6,6) local stiffness tensor field
        mode    : 'mean'            — Voigt (arithmetic) average  (default, safe)
                  'bulk_shear'      — same as mean (synonym)
                  'contrast_aware'  — geometric mean of per-voxel K and mu
                                      (better for high-contrast composites)

    Returns: C0 (6,6), lam0 (float), mu0 (float)
    """
    C_flat = C_field.reshape(-1, 6, 6)

    if mode in ('mean', 'bulk_shear'):
        C0 = np.mean(C_flat, axis=0)
        K, mu0 = _extract_K_mu(C0)
    elif mode == 'contrast_aware':
        # Per-voxel K, mu → geometric mean
        Nvox = C_flat.shape[0]
        K_all = np.zeros(Nvox)
        mu_all = np.zeros(Nvox)
        for v in range(Nvox):
            K_all[v], mu_all[v] = _extract_K_mu(C_flat[v])
        K = np.exp(np.mean(np.log(np.clip(K_all, 1e-30, None))))
        mu0 = np.exp(np.mean(np.log(np.clip(mu_all, 1e-30, None))))
    else:
        raise ValueError(f"Unknown reference medium mode: {mode!r}")

    lam0 = K - 2 * mu0 / 3
    C0 = _isotropic_stiffness(lam0, mu0)
    return C0, lam0, mu0


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
    """Relative norm of strain change between iterations (GPU-aware)."""
    xp = _xp(eps_new)
    diff = float(xp.sum((eps_new - eps_old)**2))
    norm = float(xp.sum(eps_old**2))
    return (diff / norm) ** 0.5 if norm > 1e-30 else 0.0


# ============================================================================
# Basic Scheme (Moulinec-Suquet 1994/1998)
# ============================================================================

def solve_basic_scheme(C_field, E_macro_voigt, tol=1e-6, max_iter=500,
                        verbose=True, callback=None,
                        derivative_scheme='continuous', ref_medium_mode='mean',
                        anderson_m=0, use_rfft=False, debug_checks=False,
                        profiler=None):
    """
    Solve the Lippmann-Schwinger equation: basic fixed-point iteration.
    GPU-accelerated when CuPy is available.

    New v2 parameters:
        derivative_scheme : 'continuous' | 'finite_difference' | 'rotated'
        ref_medium_mode   : 'mean' | 'contrast_aware'
        anderson_m        : Anderson mixing window (0 = off)
        use_rfft          : use real FFT on CPU (ignored on GPU)
        debug_checks      : run physics invariant checks each step
        profiler          : solver_utils.SolverProfiler instance (or None)
    """
    from solver_utils import AndersonAccelerator, run_all_checks, SolverProfiler

    if profiler is None:
        profiler = SolverProfiler()
    prof = profiler

    N = C_field.shape[0]

    # Reference medium (always CPU – small output)
    with prof.phase('ref_medium'):
        C0_cpu, lam0, mu0 = compute_reference_medium(C_field, mode=ref_medium_mode)
    if verbose:
        gpu_tag = " [GPU]" if HAS_GPU else ""
        aa_tag = f" AA(m={anderson_m})" if anderson_m > 0 else ""
        print(f"FFT Basic{gpu_tag}{aa_tag} | N={N}^3 | mu0={mu0/1e9:.2f} GPa | "
              f"deriv={derivative_scheme} | ref={ref_medium_mode}")

    # ---- transfer to GPU ----
    C_field = _to_gpu(C_field)
    xp = _xp(C_field)
    on_gpu = HAS_GPU and isinstance(C_field, cp.ndarray)

    with prof.phase('build_green'):
        n_field, Ainv = build_green_data(N, lam0, mu0, on_gpu=on_gpu,
                                          derivative_scheme=derivative_scheme)
    E_bar = strain_v2t(E_macro_voigt)                         # (3,3) numpy
    C0_broadcast = _to_gpu(C0_cpu[np.newaxis, np.newaxis, np.newaxis])

    # Initialize strain to uniform macro strain
    eps = xp.zeros((N,N,N,3,3))
    eps[...] = xp.asarray(E_bar)

    # Anderson accelerator
    anderson = AndersonAccelerator(m=anderson_m) if anderson_m > 0 else None

    errors = []
    converged = False

    for it in range(max_iter):
        eps_old = eps.copy()

        with prof.phase('constitutive'):
            sig = apply_stiffness(C_field, eps)
            sig0 = apply_stiffness(C0_broadcast, eps)
            tau = sig - sig0

        with prof.phase('fft'):
            tau_hat = fft_3x3(tau, use_rfft=use_rfft)

        with prof.phase('green_op'):
            gamma_tau_hat = apply_green_operator_tensor(tau_hat, n_field, Ainv)
            gamma_tau_hat[0, 0, 0, :, :] = 0.0  # DC = 0

        with prof.phase('ifft'):
            gamma_tau = ifft_3x3(gamma_tau_hat, use_rfft=use_rfft, N_full=N)

        # Lippmann-Schwinger:  eps = E_bar - Gamma0 * tau
        eps_new = xp.asarray(E_bar) - gamma_tau

        # Anderson acceleration
        if anderson is not None:
            with prof.phase('anderson'):
                flat_old = _to_cpu(eps_old).ravel()
                flat_new = _to_cpu(eps_new).ravel()
                flat_acc = anderson.step(flat_old, flat_new)
                eps = xp.asarray(flat_acc.reshape(N, N, N, 3, 3))
        else:
            eps = eps_new

        # Convergence: relative strain change
        err = _strain_change(eps, eps_old)
        errors.append(err)

        if verbose and (it % 10 == 0 or err < tol):
            sig = apply_stiffness(C_field, eps)
            sm = _to_cpu(xp.mean(sig.reshape(-1,3,3), axis=0))
            eq_err = _equilibrium_error(_to_cpu(sig), N)
            print(f"  Iter {it:4d}: Δε={err:.4e}  eq_err={eq_err:.4e}  <σ_11>={sm[0,0]/1e6:.1f} MPa")

        if callback:
            callback(it, strain_field_t2v(_to_cpu(eps)),
                     stress_field_t2v(_to_cpu(sig)), err)

        if err < tol:
            converged = True
            if verbose:
                print(f"  Converged at iteration {it} (err = {err:.2e})")
            break

    if not converged and verbose:
        print(f"  Not converged after {max_iter} iters (err = {errors[-1]:.2e})")

    sig = apply_stiffness(C_field, eps)

    # Debug checks
    if debug_checks:
        checks = run_all_checks(
            _to_cpu(strain_field_t2v(eps)),
            _to_cpu(stress_field_t2v(sig)),
            E_macro_voigt, None)
        if verbose:
            for name, val in checks.items():
                print(f"  [CHECK] {name}: {val:.4e}")

    if verbose:
        print(prof.summary("Basic scheme"))

    info = {
        'converged': converged,
        'iterations': it + 1,
        'errors': np.array(errors),
        'final_error': errors[-1] if errors else float('inf'),
        'C0': C0_cpu, 'lam0': lam0, 'mu0': mu0,
        'profile': prof.as_dict(),
    }
    return strain_field_t2v(_to_cpu(eps)), stress_field_t2v(_to_cpu(sig)), info


# ============================================================================
# Conjugate Gradient Scheme (Zeman et al. 2010)
# ============================================================================

def solve_conjugate_gradient(C_field, E_macro_voigt, tol=1e-6, max_iter=500,
                              verbose=True, callback=None,
                              derivative_scheme='continuous', ref_medium_mode='mean',
                              use_rfft=False, debug_checks=False, profiler=None):
    """
    CG-accelerated solution of the Lippmann-Schwinger equation.
    GPU-accelerated when CuPy is available.

    New v2 parameters:
        derivative_scheme : 'continuous' | 'finite_difference' | 'rotated'
        ref_medium_mode   : 'mean' | 'contrast_aware'
        use_rfft          : use real FFT on CPU
        debug_checks      : run physics invariant checks
        profiler          : solver_utils.SolverProfiler instance (or None)
    """
    from solver_utils import SolverProfiler, run_all_checks

    if profiler is None:
        profiler = SolverProfiler()
    prof = profiler

    N = C_field.shape[0]
    Nvox = N**3

    # Reference medium (CPU)
    with prof.phase('ref_medium'):
        C0_cpu, lam0, mu0 = compute_reference_medium(C_field, mode=ref_medium_mode)
    if verbose:
        gpu_tag = " [GPU]" if HAS_GPU else ""
        print(f"FFT CG{gpu_tag} | N={N}^3 | mu0={mu0/1e9:.2f} GPa | "
              f"deriv={derivative_scheme} | ref={ref_medium_mode}")

    # ---- transfer to GPU ----
    C_field = _to_gpu(C_field)
    xp = _xp(C_field)
    on_gpu = HAS_GPU and isinstance(C_field, cp.ndarray)

    with prof.phase('build_green'):
        n_field, Ainv = build_green_data(N, lam0, mu0, on_gpu=on_gpu,
                                          derivative_scheme=derivative_scheme)
    E_bar = strain_v2t(E_macro_voigt)           # (3,3) numpy
    dC = C_field - _to_gpu(C0_cpu)

    def apply_gamma(tau_tensor):
        """Gamma0 : tau  (real space in, real space out, zero-mean output)."""
        with prof.phase('fft'):
            th = fft_3x3(tau_tensor, use_rfft=use_rfft)
        with prof.phase('green_op'):
            eh = apply_green_operator_tensor(th, n_field, Ainv)
            eh[0,0,0,:,:] = 0.0
        with prof.phase('ifft'):
            return ifft_3x3(eh, use_rfft=use_rfft, N_full=N)

    def apply_A(x):
        """Operator: A(x) = x + Gamma0 * dC * x."""
        with prof.phase('constitutive'):
            dCx = apply_stiffness(dC, x)
        return x + apply_gamma(dCx)

    def inner(a, b):
        return float(xp.sum(a * b))

    # RHS of fluctuation equation: b = -Gamma0 * dC * E_bar
    E_bar_field = xp.zeros((N,N,N,3,3))
    E_bar_field[...] = xp.asarray(E_bar)
    with prof.phase('constitutive'):
        dC_E = apply_stiffness(dC, E_bar_field)
    b = -apply_gamma(dC_E)

    # Initialize fluctuation to zero
    eps_tilde = xp.zeros((N,N,N,3,3))

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
        err = (rr_new / bb) ** 0.5 if bb > 1e-30 else 0.0
        errors.append(err)

        if verbose and (it % 5 == 0 or err < tol):
            eps = xp.zeros_like(eps_tilde)
            eps[...] = xp.asarray(E_bar)
            eps += eps_tilde
            sig = apply_stiffness(C_field, eps)
            sm = _to_cpu(xp.mean(sig.reshape(-1,3,3), axis=0))
            eq_err = _equilibrium_error(_to_cpu(sig), N)
            print(f"  CG {it:4d}: res={err:.4e}  eq_err={eq_err:.4e}  <σ_11>={sm[0,0]/1e6:.1f} MPa")

        if callback:
            eps_cb = xp.zeros_like(eps_tilde)
            eps_cb[...] = xp.asarray(E_bar)
            eps_cb += eps_tilde
            sig_cb = apply_stiffness(C_field, eps_cb)
            callback(it, strain_field_t2v(_to_cpu(eps_cb)),
                     stress_field_t2v(_to_cpu(sig_cb)), err)

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
    eps = xp.zeros_like(eps_tilde)
    eps[...] = xp.asarray(E_bar)
    eps += eps_tilde
    sig = apply_stiffness(C_field, eps)

    # Debug checks
    if debug_checks:
        checks = run_all_checks(
            _to_cpu(strain_field_t2v(eps)),
            _to_cpu(stress_field_t2v(sig)),
            E_macro_voigt, None)
        if verbose:
            for name, val in checks.items():
                print(f"  [CHECK] {name}: {val:.4e}")

    if verbose:
        print(prof.summary("CG scheme"))

    info = {
        'converged': converged,
        'iterations': it + 1 if errors else 0,
        'errors': np.array(errors),
        'final_error': errors[-1] if errors else float('inf'),
        'C0': C0_cpu, 'lam0': lam0, 'mu0': mu0,
        'profile': prof.as_dict(),
    }
    return strain_field_t2v(_to_cpu(eps)), stress_field_t2v(_to_cpu(sig)), info


# ============================================================================
# Stress-Controlled Solver (Newton iteration on macroscopic strain)
# ============================================================================

def solve_stress_controlled(C_field, S_macro_voigt, tol_fft=1e-6, tol_stress=1e-4,
                             max_iter_fft=500, max_iter_stress=30, solver='cg',
                             verbose=True, callback=None,
                             derivative_scheme='continuous', ref_medium_mode='mean',
                             anderson_m=0, use_rfft=False, debug_checks=False):
    """
    Solve for the macroscopic strain that produces a target average stress.

    Uses Newton-Raphson with Broyden rank-1 secant update for the Jacobian.

    v2 additions:
        derivative_scheme, ref_medium_mode, anderson_m — forwarded to inner solver
        Broyden secant update replaces fixed Voigt compliance after first iteration

    Parameters:
        C_field        : (N,N,N,6,6) local stiffness
        S_macro_voigt  : (6,) target macroscopic stress in Voigt [Pa]
        tol_fft        : FFT solver convergence tolerance
        tol_stress     : relative stress error tolerance
        max_iter_fft   : max FFT iterations per solve
        max_iter_stress: max Newton iterations
        solver         : 'cg' or 'basic'
        verbose, callback
    """
    solve_fn = solve_conjugate_gradient if solver == 'cg' else solve_basic_scheme
    S_target = S_macro_voigt.copy()
    S_norm = np.linalg.norm(S_target)
    if S_norm < 1e-20:
        N = C_field.shape[0]
        return np.zeros((N,N,N,6)), np.zeros((N,N,N,6)), {
            'converged': True, 'iterations': 0, 'stress_iterations': 0,
            'errors': np.array([0.0]), 'final_error': 0.0, 'final_stress_error': 0.0,
            'E_macro_converged': np.zeros(6)}

    # Common kwargs for inner solver
    inner_kw = dict(tol=tol_fft, max_iter=max_iter_fft, verbose=False,
                    derivative_scheme=derivative_scheme, ref_medium_mode=ref_medium_mode,
                    use_rfft=use_rfft, debug_checks=False)
    if solver == 'basic':
        inner_kw['anderson_m'] = anderson_m

    # Initial strain guess from Voigt-averaged compliance
    C0, _, _ = compute_reference_medium(C_field, mode=ref_medium_mode)
    try:
        S0 = np.linalg.inv(C0)
        E_macro = S0 @ S_target
    except np.linalg.LinAlgError:
        E_macro = S_target / (C0[0,0] if C0[0,0] > 0 else 200e9)

    if verbose:
        print(f"Stress-controlled solver | target σ (MPa): "
              f"[{', '.join(f'{s/1e6:.1f}' for s in S_target)}]")
        print(f"  Initial strain guess: [{', '.join(f'{e:.4e}' for e in E_macro)}]")

    # Broyden: initialise secant compliance to Voigt compliance
    try:
        S_eff = np.linalg.inv(C0).copy()
    except np.linalg.LinAlgError:
        S_eff = np.eye(6) / (C0[0,0] if C0[0,0] > 0 else 200e9)

    stress_errors = []
    total_fft_iters = 0
    E_bar_prev = None
    sig_mean_prev = None

    for newton_it in range(max_iter_stress):
        # Solve FFT with current strain guess
        eps_v, sig_v, info_fft = solve_fn(C_field, E_macro, **inner_kw)
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

        # Broyden secant update of compliance (same pattern as EVPFFT solver)
        if E_bar_prev is not None and sig_mean_prev is not None:
            dE_actual = E_macro - E_bar_prev
            dS_actual = sig_avg - sig_mean_prev
            dS_norm2 = np.dot(dS_actual, dS_actual)
            if dS_norm2 > 1e-30:
                S_eff += np.outer(dE_actual - S_eff @ dS_actual, dS_actual) / dS_norm2

        E_bar_prev = E_macro.copy()
        sig_mean_prev = sig_avg.copy()

        # Newton correction:  dE = S_eff @ (S_target - sig_avg)
        dE = S_eff @ dS
        E_macro = E_macro + dE

    converged = rel_err < tol_stress
    if not converged and verbose:
        print(f"  Stress-control not converged after {max_iter_stress} Newton iters "
              f"(err = {stress_errors[-1]:.2e})")

    # Debug checks on final result
    if debug_checks:
        from solver_utils import run_all_checks
        checks = run_all_checks(eps_v, sig_v, E_macro, S_target)
        if verbose:
            for name, val in checks.items():
                print(f"  [CHECK] {name}: {val:.4e}")

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

def compute_effective_stiffness(C_field, solver='cg', tol=1e-6, max_iter=200, verbose=False,
                                 derivative_scheme='continuous', ref_medium_mode='mean'):
    """Compute effective stiffness C_eff by solving 6 loading cases."""
    C_eff = np.zeros((6, 6))
    solve_fn = solve_conjugate_gradient if solver == 'cg' else solve_basic_scheme

    for J in range(6):
        E_macro = np.zeros(6)
        E_macro[J] = 1.0
        if verbose:
            print(f"\n--- Loading case {J+1}/6: E[{J}] = 1 ---")
        _, sig, _ = solve_fn(C_field, E_macro, tol=tol, max_iter=max_iter, verbose=verbose,
                              derivative_scheme=derivative_scheme, ref_medium_mode=ref_medium_mode)
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


def compute_displacement_field(eps_voigt):
    """
    Compute the displacement field from a Voigt strain field using FFT integration.

    Uses the kinematic relation in Fourier space:
        û_i(ξ) = −i / |ξ|²  [ ξ_j ε̂_ij − ξ_i tr(ε̂) / 2 ]
    The mean (affine) part u_macro = ε_mean · x is added back.

    Parameters:
        eps_voigt : (N,N,N,6) strain field in Voigt notation
    Returns:
        u     : (N,N,N,3) displacement vector field
        u_mag : (N,N,N) displacement magnitude
    """
    N = eps_voigt.shape[0]
    eps_tensor = strain_field_v2t(eps_voigt)

    # Separate mean and fluctuation
    eps_mean = np.mean(eps_tensor.reshape(-1, 3, 3), axis=0)
    eps_fluct = eps_tensor - eps_mean

    # FFT of fluctuation strain
    eps_hat = fft_3x3(eps_fluct)

    # Frequency grid
    freq = fftfreq(N, d=1.0 / N)
    xi1, xi2, xi3 = np.meshgrid(freq, freq, freq, indexing='ij')
    xi = np.stack([xi1, xi2, xi3], axis=-1)  # (N,N,N,3)
    xi_sq = np.sum(xi ** 2, axis=-1)
    xi_sq_safe = np.where(xi_sq < 1e-30, 1.0, xi_sq)

    # Trace of eps_hat
    tr_eps = eps_hat[..., 0, 0] + eps_hat[..., 1, 1] + eps_hat[..., 2, 2]

    # û_i = −i / |ξ|² [ ξ_j ε̂_ij − ξ_i tr(ε̂)/2 ]
    u_hat = np.zeros((N, N, N, 3), dtype=complex)
    for i in range(3):
        b_i = np.zeros((N, N, N), dtype=complex)
        for j in range(3):
            b_i += xi[..., j] * eps_hat[..., i, j]
        b_i -= xi[..., i] * tr_eps / 2.0
        u_hat[..., i] = -1j * b_i / xi_sq_safe
    u_hat[0, 0, 0, :] = 0.0

    # IFFT → real-space fluctuation displacement
    u = np.zeros((N, N, N, 3))
    for i in range(3):
        u[..., i] = np.real(ifftn(u_hat[..., i]))

    # Add mean (affine) displacement: u_macro = ε_mean · x
    coords = (np.arange(N) + 0.5) / N
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
    x = np.stack([xx, yy, zz], axis=-1)
    u += np.einsum('ij,...j->...i', eps_mean, x)

    u_mag = np.sqrt(np.sum(u ** 2, axis=-1))
    return u, u_mag


# ============================================================================
# Quick test
# ============================================================================

if __name__ == '__main__':
    import time
    from microstructure import generate_voronoi_microstructure, build_local_stiffness_field_fast
    from microstructure import cubic_stiffness_tensor, rotate_stiffness_voigt
    from solver_utils import SolverProfiler

    print("=" * 60)
    print("  FFT Solver v2 Test — Tensor Formulation")
    print("=" * 60)

    C11, C12, C44 = 168.4e9, 121.4e9, 75.4e9
    C_cubic = cubic_stiffness_tensor(C11, C12, C44)
    E_macro = np.array([0.01, 0, 0, 0, 0, 0])

    # --- Test 1: Homogeneous medium ---
    print("\n--- Test 1: Homogeneous Medium (should give ~0 error) ---")
    N = 8
    C_homo = np.tile(C_cubic, (N,N,N,1,1))
    eps_h, sig_h, info_h = solve_basic_scheme(C_homo, E_macro, tol=1e-10, max_iter=5,
                                                verbose=True, debug_checks=True)
    print(f"  Homogeneous error: {info_h['final_error']:.2e} (should be ~1e-16)")

    # --- Test 2: Polycrystal — compare derivative schemes ---
    print("\n--- Test 2: Polycrystalline Copper (derivative scheme comparison) ---")
    N = 16
    n_grains = 8
    grain_ids, centers, euler_angles = generate_voronoi_microstructure(N, n_grains, seed=42)
    C_field = build_local_stiffness_field_fast(grain_ids, euler_angles, C11, C12, C44)

    for scheme in ['continuous', 'finite_difference', 'rotated']:
        prof = SolverProfiler()
        t0 = time.perf_counter()
        eps_s, sig_s, info_s = solve_conjugate_gradient(
            C_field, E_macro, tol=1e-5, max_iter=200,
            derivative_scheme=scheme, profiler=prof, verbose=False)
        dt = time.perf_counter() - t0
        vm = np.max(von_mises_stress(sig_s)) / 1e9
        print(f"  {scheme:20s}: {info_s['iterations']:3d} CG iters | "
              f"VM_max={vm:.3f} GPa | {dt:.3f}s")

    # --- Test 3: Anderson acceleration on basic scheme ---
    print("\n--- Test 3: Anderson Acceleration (basic scheme) ---")
    for aa_m in [0, 3, 5]:
        t0 = time.perf_counter()
        eps_a, sig_a, info_a = solve_basic_scheme(
            C_field, E_macro, tol=1e-5, max_iter=300,
            anderson_m=aa_m, verbose=False)
        dt = time.perf_counter() - t0
        print(f"  AA(m={aa_m}): {info_a['iterations']:3d} iters | {dt:.3f}s | "
              f"final_err={info_a['final_error']:.2e}")

    # --- Test 4: CG vs Basic iteration count ---
    print("\n--- Test 4: Basic vs CG ---")
    t0 = time.perf_counter()
    _, _, ib = solve_basic_scheme(C_field, E_macro, tol=1e-5, max_iter=200, verbose=False)
    dt_b = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, _, ic = solve_conjugate_gradient(C_field, E_macro, tol=1e-5, max_iter=200, verbose=False)
    dt_c = time.perf_counter() - t0
    print(f"  Basic: {ib['iterations']} iters ({dt_b:.3f}s) | CG: {ic['iterations']} iters ({dt_c:.3f}s)")
