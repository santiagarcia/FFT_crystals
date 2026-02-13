"""
Post-Processing & Visualization for FFT Crystal Plasticity
============================================================
Field visualization, pole figures, stress-strain curves, and
statistical analysis of polycrystalline simulation results.

Provides standalone plotting utilities that work with both the
GUI and command-line/notebook workflows.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec


# ============================================================================
# Custom colormaps for crystal plasticity
# ============================================================================

CRYSTAL_CMAP = LinearSegmentedColormap.from_list(
    'crystal', ['#0d1b2a', '#1b263b', '#415a77', '#778da9', '#e0e1dd', 
                '#ffd166', '#ef476f', '#ff006e'], N=256
)

STRESS_CMAP = LinearSegmentedColormap.from_list(
    'stress', ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6',
               '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226'], N=256
)

GRAIN_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9a6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
    '#e6beff', '#1abc9c', '#e74c3c', '#3498db', '#2ecc71',
]


# ============================================================================
# Field slice visualization
# ============================================================================

def plot_field_slice(field, title='Field', slice_axis=2, slice_idx=None,
                     cmap=None, vmin=None, vmax=None, units='',
                     ax=None, colorbar=True, grain_boundaries=None):
    """
    Plot a 2D slice of a 3D scalar field.
    
    Parameters:
        field       : (N,N,N) array
        title       : str
        slice_axis  : 0, 1, or 2 (x, y, z)
        slice_idx   : int (default: middle slice)
        cmap        : colormap
        grain_boundaries : (N,N,N) grain_ids for overlay (optional)
    """
    if cmap is None:
        cmap = STRESS_CMAP
    
    N = field.shape[0]
    if slice_idx is None:
        slice_idx = N // 2
    
    if slice_axis == 0:
        data = field[slice_idx, :, :]
        xlabel, ylabel = 'Y', 'Z'
        gb_slice = grain_boundaries[slice_idx, :, :] if grain_boundaries is not None else None
    elif slice_axis == 1:
        data = field[:, slice_idx, :]
        xlabel, ylabel = 'X', 'Z'
        gb_slice = grain_boundaries[:, slice_idx, :] if grain_boundaries is not None else None
    else:
        data = field[:, :, slice_idx]
        xlabel, ylabel = 'X', 'Y'
        gb_slice = grain_boundaries[:, :, slice_idx] if grain_boundaries is not None else None
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    im = ax.imshow(data.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                    interpolation='nearest', extent=[0, 1, 0, 1])
    
    # Overlay grain boundaries
    if gb_slice is not None:
        _draw_grain_boundaries(ax, gb_slice)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal')
    
    if colorbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if units:
            cb.set_label(units)
    
    return ax


def _draw_grain_boundaries(ax, grain_slice):
    """Draw grain boundaries on a 2D slice by finding neighboring voxels with different grain IDs."""
    N = grain_slice.shape[0]
    
    # Detect boundaries: where neighboring voxels have different grain IDs
    bx = np.zeros((N, N), dtype=bool)
    by = np.zeros((N, N), dtype=bool)
    
    bx[:-1, :] = grain_slice[:-1, :] != grain_slice[1:, :]
    by[:, :-1] = grain_slice[:, :-1] != grain_slice[:, 1:]
    
    boundary = bx | by
    
    # Plot boundary pixels
    bdy, bdx = np.where(boundary.T)
    ax.scatter(bdx / N + 0.5/N, bdy / N + 0.5/N, s=0.3, c='black', alpha=0.6, marker='s')


def plot_grain_map(grain_ids, slice_axis=2, slice_idx=None, ax=None):
    """Plot grain ID map as colored regions."""
    N = grain_ids.shape[0]
    if slice_idx is None:
        slice_idx = N // 2
    
    if slice_axis == 2:
        data = grain_ids[:, :, slice_idx]
    elif slice_axis == 1:
        data = grain_ids[:, slice_idx, :]
    else:
        data = grain_ids[slice_idx, :, :]
    
    n_grains = len(np.unique(grain_ids))
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    cmap = plt.cm.get_cmap('tab20', n_grains)
    im = ax.imshow(data.T, origin='lower', cmap=cmap, interpolation='nearest',
                    extent=[0, 1, 0, 1])
    ax.set_title(f'Grain Map (slice at {"XYZ"[slice_axis]}={slice_idx})')
    ax.set_xlabel('X' if slice_axis != 0 else 'Y')
    ax.set_ylabel('Y' if slice_axis == 2 else 'Z')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Grain ID')
    
    return ax


# ============================================================================
# Multi-panel result visualization
# ============================================================================

def plot_simulation_results(eps_field, sig_field, grain_ids, 
                             slice_axis=2, slice_idx=None, save_path=None):
    """
    Create a multi-panel figure showing key simulation results.
    
    Panels:
      1. Grain map
      2. Von Mises stress
      3. ε₁₁ strain
      4. σ₁₁ stress
      5. Hydrostatic pressure
      6. Strain energy density
    """
    from fft_solver import von_mises_stress, von_mises_strain
    
    N = grain_ids.shape[0]
    if slice_idx is None:
        slice_idx = N // 2
    
    fig = plt.figure(figsize=(18, 11), facecolor='#0f0f23')
    fig.suptitle('FFT Crystal Plasticity — Polycrystal Simulation Results',
                 color='white', fontsize=16, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.35)
    
    # 1. Grain map
    ax1 = fig.add_subplot(gs[0, 0])
    _style_axis(ax1)
    plot_grain_map(grain_ids, slice_axis=slice_axis, slice_idx=slice_idx, ax=ax1)
    
    # 2. Von Mises stress
    vm_stress = von_mises_stress(sig_field)
    ax2 = fig.add_subplot(gs[0, 1])
    _style_axis(ax2)
    plot_field_slice(vm_stress / 1e6, title='Von Mises Stress',
                     cmap=STRESS_CMAP, units='MPa',
                     slice_axis=slice_axis, slice_idx=slice_idx, ax=ax2,
                     grain_boundaries=grain_ids)
    
    # 3. ε₁₁
    if len(eps_field.shape) == 4:  # 3D
        eps_11 = eps_field[:, :, :, 0]
    else:
        eps_11 = eps_field[:, :, 0]
    ax3 = fig.add_subplot(gs[0, 2])
    _style_axis(ax3)
    plot_field_slice(eps_11 * 100, title='ε₁₁ Strain',
                     cmap='RdBu_r', units='%',
                     slice_axis=slice_axis, slice_idx=slice_idx, ax=ax3,
                     grain_boundaries=grain_ids)
    
    # 4. σ₁₁
    if len(sig_field.shape) == 4:
        sig_11 = sig_field[:, :, :, 0]
    else:
        sig_11 = sig_field[:, :, 0]
    ax4 = fig.add_subplot(gs[1, 0])
    _style_axis(ax4)
    plot_field_slice(sig_11 / 1e6, title='σ₁₁ Stress',
                     cmap='RdBu_r', units='MPa',
                     slice_axis=slice_axis, slice_idx=slice_idx, ax=ax4,
                     grain_boundaries=grain_ids)
    
    # 5. Hydrostatic pressure
    if len(sig_field.shape) == 4:
        p = -(sig_field[:,:,:,0] + sig_field[:,:,:,1] + sig_field[:,:,:,2]) / 3
    else:
        p = -(sig_field[:,:,0] + sig_field[:,:,1] + sig_field[:,:,2]) / 3
    ax5 = fig.add_subplot(gs[1, 1])
    _style_axis(ax5)
    plot_field_slice(p / 1e6, title='Hydrostatic Pressure',
                     cmap='coolwarm', units='MPa',
                     slice_axis=slice_axis, slice_idx=slice_idx, ax=ax5,
                     grain_boundaries=grain_ids)
    
    # 6. Strain energy density: w = 0.5 * σ : ε
    if len(sig_field.shape) == 4:
        # Voigt: w = σ₁ε₁ + σ₂ε₂ + σ₃ε₃ + σ₄*2ε₄ + ... (with engineering shear)
        w = 0.5 * np.sum(sig_field * eps_field, axis=-1)
    else:
        w = 0.5 * np.sum(sig_field * eps_field, axis=-1)
    ax6 = fig.add_subplot(gs[1, 2])
    _style_axis(ax6)
    plot_field_slice(w / 1e3, title='Strain Energy Density',
                     cmap=CRYSTAL_CMAP, units='kJ/m³',
                     slice_axis=slice_axis, slice_idx=slice_idx, ax=ax6,
                     grain_boundaries=grain_ids)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    
    return fig


def _style_axis(ax):
    """Apply dark theme to an axis."""
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white', labelsize=8)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('#334466')


# ============================================================================
# Stress-strain curves
# ============================================================================

def plot_stress_strain_history(history, component=0, ax=None, label=None):
    """
    Plot macroscopic stress-strain curve from EVP-FFT history.
    
    Parameters:
        history   : list of dicts from solve_evpfft
        component : Voigt index (0=11, 1=22, 2=33, 3=23, 4=13, 5=12)
    """
    voigt_labels = ['11', '22', '33', '23', '13', '12']
    
    strains = [h['macro_strain'][component] for h in history]
    stresses = [h['macro_stress'][component] / 1e6 for h in history]
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    lbl = label or f'σ_{voigt_labels[component]} vs ε_{voigt_labels[component]}'
    ax.plot(strains, stresses, 'o-', linewidth=2, markersize=5, label=lbl)
    ax.set_xlabel(f'ε_{voigt_labels[component]}')
    ax.set_ylabel(f'σ_{voigt_labels[component]} (MPa)')
    ax.set_title('Macroscopic Stress-Strain Response')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


# ============================================================================
# Convergence history plot
# ============================================================================

def plot_convergence(info_or_history, ax=None):
    """Plot convergence history (errors vs iteration)."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    if isinstance(info_or_history, dict):
        # Single solve info
        errors = info_or_history.get('errors', [])
        ax.semilogy(errors, 'b-o', markersize=3, linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Equilibrium Error')
        ax.set_title('FFT Solver Convergence')
    elif isinstance(info_or_history, list):
        # EVP-FFT history with per-increment convergence
        for i, h in enumerate(info_or_history):
            ax.bar(i, h.get('fft_iterations', 0), color=GRAIN_COLORS[i % len(GRAIN_COLORS)])
        ax.set_xlabel('Load Increment')
        ax.set_ylabel('FFT Iterations to Converge')
        ax.set_title('Iterations per Load Increment')
    
    ax.grid(True, alpha=0.3)
    return ax


# ============================================================================
# Pole figure (basic)
# ============================================================================

def plot_pole_figure(euler_angles, pole=[1, 0, 0], ax=None, title=None):
    """
    Plot a basic equal-area pole figure for the crystallographic texture.
    
    Parameters:
        euler_angles : (n_grains, 3) Euler angles in radians
        pole         : [h, k, l] crystal direction to plot
    """
    from microstructure import euler_to_rotation
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={'projection': 'polar'})
    
    pole = np.array(pole, dtype=float)
    pole /= np.linalg.norm(pole)
    
    n_grains = euler_angles.shape[0]
    azimuths = []
    radii = []
    
    for g in range(n_grains):
        R = euler_to_rotation(*euler_angles[g])
        # Transform pole to sample frame
        p_sample = R.T @ pole  # inverse rotation
        
        # Stereographic projection
        if p_sample[2] < 0:
            p_sample = -p_sample  # upper hemisphere
        
        theta = np.arctan2(p_sample[1], p_sample[0])
        r = np.sqrt(p_sample[0]**2 + p_sample[1]**2)
        r_proj = r / (1 + p_sample[2])  # equal-area: r = sqrt(2) * sin(θ/2)
        
        azimuths.append(theta)
        radii.append(r_proj)
    
    ax.scatter(azimuths, radii, s=30, c='#ff6b6b', edgecolors='white',
               linewidths=0.5, alpha=0.8, zorder=5)
    ax.set_rmax(1.0)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', ''])
    
    if title is None:
        title = f'({pole[0]:.0f}{pole[1]:.0f}{pole[2]:.0f}) Pole Figure'
    ax.set_title(title, pad=15)
    
    return ax


# ============================================================================
# 3D grain visualization
# ============================================================================

def plot_3d_grain_structure(grain_ids, alpha=0.3, ax=None):
    """
    3D voxelized visualization of grain structure.
    Only plots grain boundaries for efficiency.
    """
    N = grain_ids.shape[0]
    n_grains = len(np.unique(grain_ids))
    
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Find boundary voxels
    boundary = np.zeros_like(grain_ids, dtype=bool)
    boundary[:-1, :, :] |= grain_ids[:-1, :, :] != grain_ids[1:, :, :]
    boundary[:, :-1, :] |= grain_ids[:, :-1, :] != grain_ids[:, 1:, :]
    boundary[:, :, :-1] |= grain_ids[:, :, :-1] != grain_ids[:, :, 1:]
    
    # Plot boundary voxels colored by grain
    bi, bj, bk = np.where(boundary)
    colors = [GRAIN_COLORS[grain_ids[i, j, k] % len(GRAIN_COLORS)] 
              for i, j, k in zip(bi, bj, bk)]
    
    ax.scatter(bi/N, bj/N, bk/N, c=colors, s=2, alpha=alpha, depthshade=True)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Grain Boundaries ({n_grains} grains, {N}³ grid)')
    
    return ax


# ============================================================================
# Statistics summary
# ============================================================================

def print_simulation_summary(eps_field, sig_field, grain_ids, info=None):
    """Print a summary of simulation results."""
    from fft_solver import von_mises_stress, von_mises_strain
    
    vm_sig = von_mises_stress(sig_field)
    vm_eps = von_mises_strain(eps_field)
    
    sig_mean = np.mean(sig_field.reshape(-1, 6), axis=0)
    eps_mean = np.mean(eps_field.reshape(-1, 6), axis=0)
    
    n_grains = len(np.unique(grain_ids))
    N = grain_ids.shape[0]
    
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"  Grid resolution:   {N}³ = {N**3} voxels")
    print(f"  Number of grains:  {n_grains}")
    
    if info:
        print(f"  Solver converged:  {info.get('converged', '?')}")
        print(f"  Iterations:        {info.get('iterations', '?')}")
        print(f"  Final error:       {info.get('final_error', '?'):.2e}")
    
    print(f"\n  Mean strain (Voigt): [{', '.join(f'{e:.4e}' for e in eps_mean)}]")
    print(f"  Mean stress (MPa):   [{', '.join(f'{s/1e6:.1f}' for s in sig_mean)}]")
    
    print(f"\n  Von Mises stress:")
    print(f"    Mean: {np.mean(vm_sig)/1e6:.1f} MPa")
    print(f"    Max:  {np.max(vm_sig)/1e6:.1f} MPa")
    print(f"    Min:  {np.min(vm_sig)/1e6:.1f} MPa")
    print(f"    Std:  {np.std(vm_sig)/1e6:.1f} MPa")
    
    print(f"\n  Von Mises strain:")
    print(f"    Mean: {np.mean(vm_eps):.4e}")
    print(f"    Max:  {np.max(vm_eps):.4e}")
    
    # Per-grain statistics
    print(f"\n  Per-grain mean von Mises stress (MPa):")
    for g in range(min(n_grains, 10)):
        mask = grain_ids == g
        mean_vm = np.mean(vm_sig[mask]) / 1e6
        print(f"    Grain {g:2d}: {mean_vm:.1f}")
    if n_grains > 10:
        print(f"    ... ({n_grains - 10} more grains)")
    print("=" * 60)
