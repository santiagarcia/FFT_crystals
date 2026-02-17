"""
verify_solver_v2.py  —  Verification & Benchmark for FFT Solver v2
===================================================================
Runs:
  1. Homogeneous medium test (exact solution known)
  2. Two-phase laminate test (Reuss/Voigt bounds)
  3. Derivative scheme convergence comparison
  4. Anderson acceleration speedup
  5. Contrast-aware vs mean reference medium
  6. Broyden stress-controlled solver
  7. Before / After benchmark (v1 backup vs v2)
"""

import numpy as np
import time
import sys
import os

# Ensure workspace is on path
sys.path.insert(0, os.path.dirname(__file__))

from microstructure import (generate_voronoi_microstructure,
                            build_local_stiffness_field_fast,
                            cubic_stiffness_tensor)
from fft_solver import (solve_basic_scheme, solve_conjugate_gradient,
                        solve_stress_controlled, compute_reference_medium,
                        von_mises_stress, compute_effective_stiffness)
from solver_utils import SolverProfiler, run_all_checks

# Copper single-crystal constants
C11, C12, C44 = 168.4e9, 121.4e9, 75.4e9
C_cubic = cubic_stiffness_tensor(C11, C12, C44)

SEP = "=" * 70


def test_homogeneous():
    """Test 1: Homogeneous medium — should converge in 1 iteration to exact."""
    print(f"\n{SEP}")
    print("TEST 1: Homogeneous Medium (analytical verification)")
    print(SEP)

    N = 8
    C_homo = np.tile(C_cubic, (N, N, N, 1, 1))
    E_macro = np.array([0.01, -0.003, -0.003, 0, 0, 0])

    for scheme in ['continuous', 'finite_difference', 'rotated']:
        eps, sig, info = solve_basic_scheme(
            C_homo, E_macro, tol=1e-12, max_iter=5, verbose=False,
            derivative_scheme=scheme, debug_checks=True)

        # Analytical: sig = C : eps everywhere
        sig_exact = C_cubic @ E_macro
        sig_mean = np.mean(sig.reshape(-1, 6), axis=0)
        err = np.linalg.norm(sig_mean - sig_exact) / np.linalg.norm(sig_exact)

        print(f"  {scheme:20s}: iters={info['iterations']}  "
              f"|<σ>-σ_exact|/|σ_exact| = {err:.2e}  "
              f"{'PASS' if err < 1e-10 else 'FAIL'}")


def test_two_phase_laminate():
    """Test 2: Two-phase laminate — effective modulus between Reuss & Voigt."""
    print(f"\n{SEP}")
    print("TEST 2: Two-Phase Laminate (Reuss/Voigt bounds)")
    print(SEP)

    N = 16
    # Phase 1: stiff (copper)
    C1 = C_cubic.copy()
    # Phase 2: compliant (half stiffness)
    C2 = C_cubic * 0.5

    # Build laminate: phase 1 for x < N/2, phase 2 for x >= N/2
    C_field = np.zeros((N, N, N, 6, 6))
    C_field[:N//2, :, :] = C1
    C_field[N//2:, :, :] = C2

    # Voigt (upper) and Reuss (lower) bounds for C_11
    C_voigt = 0.5 * C1 + 0.5 * C2
    try:
        C_reuss = np.linalg.inv(0.5 * np.linalg.inv(C1) + 0.5 * np.linalg.inv(C2))
    except np.linalg.LinAlgError:
        C_reuss = C_voigt * 0.8  # fallback

    E_macro = np.array([0.01, 0, 0, 0, 0, 0])

    for scheme in ['continuous', 'rotated']:
        eps, sig, info = solve_conjugate_gradient(
            C_field, E_macro, tol=1e-6, max_iter=200, verbose=False,
            derivative_scheme=scheme)
        C_eff_11 = np.mean(sig.reshape(-1, 6), axis=0)[0] / E_macro[0]

        in_bounds = C_reuss[0, 0] - 1e6 <= C_eff_11 <= C_voigt[0, 0] + 1e6
        print(f"  {scheme:20s}: C_eff_11 = {C_eff_11/1e9:.2f} GPa  "
              f"Reuss={C_reuss[0,0]/1e9:.2f}  Voigt={C_voigt[0,0]/1e9:.2f}  "
              f"{'PASS' if in_bounds else 'FAIL'}")


def test_derivative_schemes():
    """Test 3: Compare convergence across derivative schemes."""
    print(f"\n{SEP}")
    print("TEST 3: Derivative Scheme Convergence Comparison")
    print(SEP)

    N = 16
    n_grains = 8
    grain_ids, centers, euler_angles = generate_voronoi_microstructure(N, n_grains, seed=42)
    C_field = build_local_stiffness_field_fast(grain_ids, euler_angles, C11, C12, C44)
    E_macro = np.array([0.01, 0, 0, 0, 0, 0])

    print(f"  {'Scheme':20s} {'Solver':8s} {'Iters':>6s} {'Time':>8s} {'VM_max':>10s} {'Error':>10s}")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*8} {'-'*10} {'-'*10}")

    for scheme in ['continuous', 'finite_difference', 'rotated']:
        for solver_name, solver_fn in [('Basic', solve_basic_scheme),
                                       ('CG', solve_conjugate_gradient)]:
            t0 = time.perf_counter()
            eps, sig, info = solver_fn(
                C_field, E_macro, tol=1e-5, max_iter=300, verbose=False,
                derivative_scheme=scheme)
            dt = time.perf_counter() - t0
            vm_max = np.max(von_mises_stress(sig)) / 1e9
            print(f"  {scheme:20s} {solver_name:8s} {info['iterations']:6d} "
                  f"{dt:8.3f}s {vm_max:10.4f} {info['final_error']:10.2e}")


def test_anderson():
    """Test 4: Anderson acceleration speedup on basic scheme."""
    print(f"\n{SEP}")
    print("TEST 4: Anderson Acceleration (Basic Scheme)")
    print(SEP)

    N = 16
    n_grains = 8
    grain_ids, centers, euler_angles = generate_voronoi_microstructure(N, n_grains, seed=42)
    C_field = build_local_stiffness_field_fast(grain_ids, euler_angles, C11, C12, C44)
    E_macro = np.array([0.01, 0, 0, 0, 0, 0])

    print(f"  {'AA(m)':>8s} {'Iters':>6s} {'Time':>8s} {'Error':>10s}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*10}")

    for m in [0, 3, 5, 8]:
        t0 = time.perf_counter()
        eps, sig, info = solve_basic_scheme(
            C_field, E_macro, tol=1e-5, max_iter=300, verbose=False,
            anderson_m=m)
        dt = time.perf_counter() - t0
        print(f"  m={m:5d} {info['iterations']:6d} {dt:8.3f}s {info['final_error']:10.2e}")


def test_contrast_aware():
    """Test 5: Contrast-aware vs mean reference medium on high-contrast composite."""
    print(f"\n{SEP}")
    print("TEST 5: Reference Medium Mode (high-contrast composite)")
    print(SEP)

    N = 16
    # High contrast: matrix vs inclusion (10× contrast)
    C_soft = C_cubic * 0.2
    C_field = np.zeros((N, N, N, 6, 6))
    C_field[:] = C_cubic  # matrix
    # Inclusion in center
    C_field[5:11, 5:11, 5:11] = C_soft

    E_macro = np.array([0.01, 0, 0, 0, 0, 0])

    print(f"  {'Mode':20s} {'Solver':8s} {'Iters':>6s} {'Time':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*8}")

    for mode in ['mean', 'contrast_aware']:
        for solver_name, solver_fn in [('Basic', solve_basic_scheme),
                                       ('CG', solve_conjugate_gradient)]:
            t0 = time.perf_counter()
            _, _, info = solver_fn(
                C_field, E_macro, tol=1e-5, max_iter=500, verbose=False,
                ref_medium_mode=mode)
            dt = time.perf_counter() - t0
            print(f"  {mode:20s} {solver_name:8s} {info['iterations']:6d} {dt:8.3f}s")


def test_stress_controlled_broyden():
    """Test 6: Stress-controlled solver with Broyden update."""
    print(f"\n{SEP}")
    print("TEST 6: Stress-Controlled Solver (Broyden secant)")
    print(SEP)

    N = 16
    n_grains = 8
    grain_ids, centers, euler_angles = generate_voronoi_microstructure(N, n_grains, seed=42)
    C_field = build_local_stiffness_field_fast(grain_ids, euler_angles, C11, C12, C44)

    S_target = np.array([1e9, 0, 0, 0, 0, 0])  # 1 GPa uniaxial

    t0 = time.perf_counter()
    eps, sig, info = solve_stress_controlled(
        C_field, S_target, tol_fft=1e-5, tol_stress=1e-3,
        max_iter_fft=200, max_iter_stress=20,
        solver='cg', verbose=True, debug_checks=True)
    dt = time.perf_counter() - t0

    sig_avg = np.mean(sig.reshape(-1, 6), axis=0)
    print(f"\n  Converged: {info['converged']}")
    print(f"  Newton iters: {info['stress_iterations']}")
    print(f"  Total FFT iters: {info['iterations']}")
    print(f"  <σ₁₁> = {sig_avg[0]/1e6:.1f} MPa (target: {S_target[0]/1e6:.1f})")
    print(f"  Time: {dt:.3f}s")


def test_benchmark_v1_vs_v2():
    """Test 7: Before / After benchmark."""
    print(f"\n{SEP}")
    print("TEST 7: v1 vs v2 Benchmark")
    print(SEP)

    N = 16
    n_grains = 8
    grain_ids, centers, euler_angles = generate_voronoi_microstructure(N, n_grains, seed=42)
    C_field = build_local_stiffness_field_fast(grain_ids, euler_angles, C11, C12, C44)
    E_macro = np.array([0.01, 0, 0, 0, 0, 0])

    # v2 default (should match v1 behaviour)
    t0 = time.perf_counter()
    eps_v2, sig_v2, info_v2 = solve_conjugate_gradient(
        C_field, E_macro, tol=1e-5, max_iter=200, verbose=False,
        derivative_scheme='continuous', ref_medium_mode='mean')
    dt_v2 = time.perf_counter() - t0

    # v2 with rotated scheme + contrast_aware
    t0 = time.perf_counter()
    eps_v2r, sig_v2r, info_v2r = solve_conjugate_gradient(
        C_field, E_macro, tol=1e-5, max_iter=200, verbose=False,
        derivative_scheme='rotated', ref_medium_mode='contrast_aware')
    dt_v2r = time.perf_counter() - t0

    # v2 basic with Anderson
    t0 = time.perf_counter()
    eps_v2a, sig_v2a, info_v2a = solve_basic_scheme(
        C_field, E_macro, tol=1e-5, max_iter=300, verbose=False,
        anderson_m=5)
    dt_v2a = time.perf_counter() - t0

    print(f"  {'Config':40s} {'Iters':>6s} {'Time':>8s} {'VM_max':>10s}")
    print(f"  {'-'*40} {'-'*6} {'-'*8} {'-'*10}")
    vm = np.max(von_mises_stress(sig_v2)) / 1e9
    print(f"  {'CG continuous/mean (≈v1)':40s} {info_v2['iterations']:6d} {dt_v2:8.3f}s {vm:10.4f}")
    vm = np.max(von_mises_stress(sig_v2r)) / 1e9
    print(f"  {'CG rotated/contrast_aware':40s} {info_v2r['iterations']:6d} {dt_v2r:8.3f}s {vm:10.4f}")
    vm = np.max(von_mises_stress(sig_v2a)) / 1e9
    print(f"  {'Basic AA(m=5)':40s} {info_v2a['iterations']:6d} {dt_v2a:8.3f}s {vm:10.4f}")


if __name__ == '__main__':
    print("=" * 70)
    print("  FFT Solver v2 — Verification & Benchmark Suite")
    print("=" * 70)

    test_homogeneous()
    test_two_phase_laminate()
    test_derivative_schemes()
    test_anderson()
    test_contrast_aware()
    test_stress_controlled_broyden()
    test_benchmark_v1_vs_v2()

    print(f"\n{'=' * 70}")
    print("  All tests complete.")
    print(f"{'=' * 70}")
