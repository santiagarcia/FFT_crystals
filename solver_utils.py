"""
Solver Utilities for FFT-based Micromechanics
==============================================
Small, well-commented helper functions:
  - Anderson acceleration (mixing)
  - Lightweight per-iteration profiler
  - Physics-invariant debug checks
"""

import numpy as np
import time


# ============================================================================
# Anderson Acceleration (Type-I, window = m)
# ============================================================================

class AndersonAccelerator:
    """
    Anderson mixing for fixed-point iteration  x = G(x).

    Stores the last *m* residuals  F_k = G(x_k) - x_k  and iterates
    x_{k+1} = x_k + ... to minimise ||F||^2 in the affine span.

    Parameters
    ----------
    m       : int   mixing depth (5-10 typical)
    beta    : float damping / relaxation (1.0 = full Anderson, <1 = damped)
    restart : int   restart every *restart* steps for numerical hygiene
    """

    def __init__(self, m=5, beta=1.0, restart=50):
        self.m = m
        self.beta = beta
        self.restart = restart
        self._Xs = []       # past iterates  (flattened)
        self._Fs = []       # past residuals (flattened)
        self._k = 0

    def reset(self):
        self._Xs.clear()
        self._Fs.clear()
        self._k = 0

    def step(self, x_k, Gx_k):
        """
        Given current iterate x_k and G(x_k), return the accelerated
        next iterate x_{k+1}.

        Parameters
        ----------
        x_k   : ndarray (any shape, will be flattened internally)
        Gx_k  : ndarray  same shape as x_k — the fixed-point map output

        Returns
        -------
        x_new : ndarray  same shape as x_k
        """
        shape = x_k.shape
        xf = x_k.ravel().copy()
        gf = Gx_k.ravel().copy()
        fk = gf - xf                  # residual

        self._Xs.append(xf)
        self._Fs.append(fk)
        self._k += 1

        # Periodic restart for numerical stability
        if self._k % self.restart == 0:
            self._Xs = self._Xs[-1:]
            self._Fs = self._Fs[-1:]

        mk = min(len(self._Fs) - 1, self.m)

        if mk == 0:
            # No history yet → simple mixing (damped fixed-point)
            x_new = xf + self.beta * fk
            return x_new.reshape(shape)

        # Build ΔF matrix:  ΔF[:,j] = F_{k} - F_{k-j-1}
        dF = np.column_stack([
            self._Fs[-1] - self._Fs[-1 - j - 1]
            for j in range(mk)
        ])

        # Solve  min ||F_k - ΔF θ||^2  →  θ = (ΔF^T ΔF)^{-1} ΔF^T F_k
        try:
            gram = dF.T @ dF
            # Tikhonov regularisation for ill-conditioning
            gram += 1e-12 * np.eye(mk)
            theta = np.linalg.solve(gram, dF.T @ fk)
        except np.linalg.LinAlgError:
            # Fall back to plain damped fixed-point
            x_new = xf + self.beta * fk
            return x_new.reshape(shape)

        # Build ΔX matrix
        dX = np.column_stack([
            self._Xs[-1] - self._Xs[-1 - j - 1]
            for j in range(mk)
        ])

        # Accelerated iterate
        x_new = (xf + self.beta * fk) - (dX + self.beta * dF) @ theta

        # Trim history to window
        if len(self._Fs) > self.m + 2:
            self._Xs = self._Xs[-(self.m + 1):]
            self._Fs = self._Fs[-(self.m + 1):]

        return x_new.reshape(shape)


# ============================================================================
# Lightweight Per-Iteration Profiler
# ============================================================================

class SolverProfiler:
    """
    Records cumulative wall-clock time for named phases within the
    iterative solver (FFT, constitutive, residual, etc.).

    Usage
    -----
    >>> prof = SolverProfiler()
    >>> with prof.phase('fft'):
    ...     tau_hat = fftn(tau)
    >>> prof.summary()
    """

    def __init__(self):
        self._times = {}       # phase -> accumulated seconds
        self._counts = {}      # phase -> call count
        self._t0 = None
        self._current = None
        self._total_start = time.perf_counter()

    # Context-manager interface
    class _Phase:
        def __init__(self, profiler, name):
            self.profiler = profiler
            self.name = name
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *args):
            dt = time.perf_counter() - self.t0
            p = self.profiler
            p._times[self.name] = p._times.get(self.name, 0.0) + dt
            p._counts[self.name] = p._counts.get(self.name, 0) + 1

    def phase(self, name):
        """Return a context manager that times the enclosed block."""
        return self._Phase(self, name)

    def tick(self, name):
        """Start timing phase *name* (manual start/stop API)."""
        self._current = name
        self._t0 = time.perf_counter()

    def tock(self):
        """Stop timing the current phase."""
        if self._t0 is not None and self._current is not None:
            dt = time.perf_counter() - self._t0
            self._times[self._current] = self._times.get(self._current, 0.0) + dt
            self._counts[self._current] = self._counts.get(self._current, 0) + 1
        self._t0 = None
        self._current = None

    @property
    def total_elapsed(self):
        return time.perf_counter() - self._total_start

    def as_dict(self):
        """Return profiling results as a plain dict."""
        d = {k: round(v, 4) for k, v in self._times.items()}
        d['total'] = round(self.total_elapsed, 4)
        d['counts'] = dict(self._counts)
        return d

    def summary(self, label="Solver profile"):
        """Print a short summary table."""
        total = self.total_elapsed
        lines = [f"  {label} ({total:.3f}s total)"]
        for k in sorted(self._times, key=self._times.get, reverse=True):
            t = self._times[k]
            n = self._counts.get(k, 0)
            pct = 100.0 * t / total if total > 0 else 0
            lines.append(f"    {k:20s}  {t:8.3f}s  ({pct:5.1f}%)  [{n} calls]")
        accounted = sum(self._times.values())
        other = total - accounted
        if other > 0.001:
            lines.append(f"    {'(other)':20s}  {other:8.3f}s  ({100*other/total:5.1f}%)")
        return "\n".join(lines)


# ============================================================================
# Physics Invariant Checks (debug mode)
# ============================================================================

def check_symmetry(tensor_field, name="field", tol=1e-10):
    """
    Check that a (N,N,N,3,3) tensor field is symmetric.
    Returns max asymmetry norm.
    """
    asym = tensor_field - np.swapaxes(tensor_field, -2, -1)
    max_asym = float(np.max(np.abs(asym)))
    if max_asym > tol:
        print(f"  [DEBUG] {name} asymmetry: max|A-A^T| = {max_asym:.2e}  (> tol {tol:.0e})")
    return max_asym


def check_macro_strain(eps_field_voigt, E_macro_voigt, name="strain", tol=1e-6):
    """
    Check that <ε> matches the imposed macroscopic strain.
    """
    eps_avg = np.mean(eps_field_voigt.reshape(-1, 6), axis=0)
    err = np.linalg.norm(eps_avg - E_macro_voigt)
    ref = max(np.linalg.norm(E_macro_voigt), 1e-20)
    rel = err / ref
    if rel > tol:
        print(f"  [DEBUG] {name} macro mismatch: |<ε>-Ē|/|Ē| = {rel:.2e}")
    return rel


def check_macro_stress(sig_field_voigt, S_target_voigt, name="stress", tol=1e-4):
    """
    Check that <σ> matches the target macroscopic stress.
    """
    sig_avg = np.mean(sig_field_voigt.reshape(-1, 6), axis=0)
    err = np.linalg.norm(sig_avg - S_target_voigt)
    ref = max(np.linalg.norm(S_target_voigt), 1e-20)
    rel = err / ref
    if rel > tol:
        print(f"  [DEBUG] {name} macro mismatch: |<σ>-S̄|/|S̄| = {rel:.2e}")
    return rel


def check_energy_positivity(eps_voigt, sig_voigt, name="energy"):
    """
    Check that the elastic energy density  W = ½ σ:ε ≥ 0  everywhere.
    Returns the minimum energy density found.
    """
    # W = σ_ij ε_ij  with Voigt weights
    weights = np.array([1, 1, 1, 2, 2, 2])
    w = 0.5 * np.sum(sig_voigt * eps_voigt * weights, axis=-1)
    w_min = float(np.min(w))
    if w_min < -1e-10:
        neg_frac = float(np.mean(w < 0)) * 100
        print(f"  [DEBUG] {name}: min W = {w_min:.4e} ({neg_frac:.1f}% negative)")
    return w_min


def run_all_checks(eps_voigt, sig_voigt, E_macro=None, S_target=None,
                   tol_sym=1e-10, tol_macro=1e-6):
    """Run all physics invariant checks. Returns dict of results."""
    from fft_solver import strain_field_v2t, stress_field_v2t
    results = {}
    results['strain_symmetry'] = check_symmetry(
        strain_field_v2t(eps_voigt), "strain", tol_sym)
    results['stress_symmetry'] = check_symmetry(
        stress_field_v2t(sig_voigt), "stress", tol_sym)
    if E_macro is not None:
        results['macro_strain_err'] = check_macro_strain(
            eps_voigt, E_macro, tol=tol_macro)
    if S_target is not None:
        results['macro_stress_err'] = check_macro_stress(
            sig_voigt, S_target, tol=tol_macro)
    results['min_energy'] = check_energy_positivity(eps_voigt, sig_voigt)
    return results
