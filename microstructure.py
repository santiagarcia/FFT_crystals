"""
Polycrystalline Microstructure Generator
=========================================
Generates Representative Volume Elements (RVEs) for FFT-based
crystal plasticity simulations.

Methods:
  - Voronoi tessellation (periodic) for equiaxed grains
  - Random crystallographic orientation assignment (Euler angles)
  - Grain statistics and visualization utilities

References:
  Moulinec & Suquet (1994, 1998) — FFT-based homogenization
  Lebensohn (2001, 2012) — VPFFT for polycrystals
  Lucarini et al. (2022) — FFT fundamentals review
"""

import numpy as np
from scipy.spatial import cKDTree


# ============================================================================
# Orientation utilities (Bunge Euler angles)
# ============================================================================

def euler_to_rotation(phi1, Phi, phi2):
    """
    Convert Bunge Euler angles (φ₁, Φ, φ₂) in radians to a 3×3 rotation matrix.
    Convention: ZXZ rotations.
    """
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)

    R = np.array([
        [c1*c2 - s1*s2*c,   s1*c2 + c1*s2*c,   s2*s],
        [-c1*s2 - s1*c2*c, -s1*s2 + c1*c2*c,   c2*s],
        [s1*s,             -c1*s,               c    ],
    ])
    return R


def random_orientations(n_grains, seed=None):
    """
    Generate uniformly distributed random orientations as Euler angles [φ₁, Φ, φ₂].
    Uses the proper measure for SO(3): Φ sampled from arccos(uniform).
    
    Returns: (n_grains, 3) array of Euler angles in radians.
    """
    rng = np.random.default_rng(seed)
    phi1 = rng.uniform(0, 2*np.pi, n_grains)
    Phi = np.arccos(rng.uniform(-1, 1, n_grains))  # proper SO(3) measure
    phi2 = rng.uniform(0, 2*np.pi, n_grains)
    return np.column_stack([phi1, Phi, phi2])


def rotation_matrices_from_euler(euler_angles):
    """
    Vectorized conversion of Euler angles to rotation matrices.
    
    Parameters:
        euler_angles: (n_grains, 3) array — [φ₁, Φ, φ₂] in radians
    Returns:
        (n_grains, 3, 3) array of rotation matrices
    """
    n = euler_angles.shape[0]
    R = np.zeros((n, 3, 3))
    for i in range(n):
        R[i] = euler_to_rotation(*euler_angles[i])
    return R


# ============================================================================
# Voronoi Tessellation (Periodic)
# ============================================================================

def generate_voronoi_microstructure(N, n_grains, seed=None, dimensions=3):
    """
    Generate a periodic Voronoi tessellation on an N^dim grid.
    
    Uses periodic replication of seed points (3^dim copies) and nearest-neighbor
    assignment to produce a grain map with periodic boundary conditions.
    
    Parameters:
        N        : int — grid resolution (N voxels per side)
        n_grains : int — number of grains
        seed     : int — random seed for reproducibility
        dimensions : 2 or 3 — spatial dimension
    
    Returns:
        grain_ids : (N,N) or (N,N,N) int array — grain ID at each voxel
        centers   : (n_grains, dim) array — grain center positions [0,1)
        euler_angles : (n_grains, 3) array — orientation Euler angles
    """
    rng = np.random.default_rng(seed)
    
    # Random seed points in [0, 1)^dim
    centers = rng.random((n_grains, dimensions))
    
    # Periodic replication: tile seed points to 3^dim neighbors
    offsets = _periodic_offsets(dimensions)
    tiled_centers = []
    tiled_ids = []
    for off in offsets:
        tiled_centers.append(centers + off)
        tiled_ids.append(np.arange(n_grains))
    tiled_centers = np.vstack(tiled_centers)
    tiled_ids = np.concatenate(tiled_ids)
    
    # Build KD-tree for efficient nearest neighbor
    tree = cKDTree(tiled_centers)
    
    # Grid of voxel center coordinates in [0, 1)
    coords_1d = (np.arange(N) + 0.5) / N
    if dimensions == 3:
        xx, yy, zz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
        query_pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    else:
        xx, yy = np.meshgrid(coords_1d, coords_1d, indexing='ij')
        query_pts = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Assign each voxel to nearest grain center
    _, indices = tree.query(query_pts)
    grain_ids = tiled_ids[indices]
    
    if dimensions == 3:
        grain_ids = grain_ids.reshape(N, N, N)
    else:
        grain_ids = grain_ids.reshape(N, N)
    
    # Generate random orientations
    euler_angles = random_orientations(n_grains, seed=seed)
    
    return grain_ids, centers, euler_angles


def _periodic_offsets(dim):
    """Generate offset vectors for periodic replication: [-1, 0, +1]^dim."""
    if dim == 2:
        offsets = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                offsets.append(np.array([dx, dy], dtype=float))
        return offsets
    else:
        offsets = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    offsets.append(np.array([dx, dy, dz], dtype=float))
        return offsets


# ============================================================================
# Stiffness tensor utilities (FCC cubic elasticity)
# ============================================================================

def cubic_stiffness_tensor(C11, C12, C44):
    """
    Build the 6×6 Voigt stiffness matrix for cubic symmetry.
    
    Parameters:
        C11, C12, C44 : float — elastic constants (Pa)
    Returns:
        C : (6, 6) array — Voigt stiffness matrix
    """
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C[2, 2] = C11
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = C12
    C[3, 3] = C[4, 4] = C[5, 5] = C44
    return C


def rotate_stiffness_voigt(C_voigt, R):
    """
    Rotate a 6×6 Voigt stiffness matrix by rotation matrix R.
    
    Uses the full 4th-order tensor transformation:
        C'_ijkl = R_ip R_jq R_kr R_ls C_pqrs
    then converts back to Voigt notation.
    
    Parameters:
        C_voigt : (6, 6) array — Voigt stiffness
        R       : (3, 3) array — rotation matrix
    Returns:
        C_rot   : (6, 6) array — rotated Voigt stiffness
    """
    # Expand Voigt to full tensor
    C_full = voigt_to_tensor(C_voigt)
    
    # Rotate: C'_ijkl = R_ip R_jq R_kr R_ls C_pqrs
    C_rot_full = np.einsum('ip,jq,kr,ls,pqrs->ijkl', R, R, R, R, C_full)
    
    # Compress back to Voigt
    return tensor_to_voigt(C_rot_full)


def voigt_to_tensor(C_voigt):
    """Convert 6×6 Voigt matrix to 3×3×3×3 tensor."""
    voigt_map = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]
    C = np.zeros((3, 3, 3, 3))
    for I in range(6):
        i, j = voigt_map[I]
        for J in range(6):
            k, l = voigt_map[J]
            C[i,j,k,l] = C_voigt[I, J]
            C[j,i,k,l] = C_voigt[I, J]
            C[i,j,l,k] = C_voigt[I, J]
            C[j,i,l,k] = C_voigt[I, J]
    return C


def tensor_to_voigt(C_tensor):
    """Convert 3×3×3×3 tensor to 6×6 Voigt matrix."""
    voigt_map = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]
    C = np.zeros((6, 6))
    for I in range(6):
        i, j = voigt_map[I]
        for J in range(6):
            k, l = voigt_map[J]
            C[I, J] = C_tensor[i, j, k, l]
    return C


def build_local_stiffness_field(grain_ids, euler_angles, C11, C12, C44):
    """
    Construct the spatially-varying stiffness tensor field C(x).
    
    Parameters:
        grain_ids    : (N, N, N) int array — grain ID per voxel
        euler_angles : (n_grains, 3) array — Euler angles
        C11, C12, C44: elastic constants
    Returns:
        C_field : (N, N, N, 6, 6) array — Voigt stiffness at each voxel
    """
    shape = grain_ids.shape
    N = shape[0]
    n_grains = euler_angles.shape[0]
    
    # Pre-compute rotated stiffness for each grain
    C_base = cubic_stiffness_tensor(C11, C12, C44)
    C_grains = np.zeros((n_grains, 6, 6))
    for g in range(n_grains):
        R = euler_to_rotation(*euler_angles[g])
        C_grains[g] = rotate_stiffness_voigt(C_base, R)
    
    # Map to voxels
    if len(shape) == 3:
        C_field = np.zeros((N, N, N, 6, 6))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    C_field[i, j, k] = C_grains[grain_ids[i, j, k]]
    else:
        C_field = np.zeros((N, N, 6, 6))
        for i in range(N):
            for j in range(N):
                C_field[i, j] = C_grains[grain_ids[i, j]]
    
    return C_field


def build_local_stiffness_field_fast(grain_ids, euler_angles, C11, C12, C44):
    """
    Vectorized version of build_local_stiffness_field.
    Uses array indexing instead of nested loops.
    """
    n_grains = euler_angles.shape[0]
    
    C_base = cubic_stiffness_tensor(C11, C12, C44)
    C_grains = np.zeros((n_grains, 6, 6))
    for g in range(n_grains):
        R = euler_to_rotation(*euler_angles[g])
        C_grains[g] = rotate_stiffness_voigt(C_base, R)
    
    # Vectorized lookup
    C_field = C_grains[grain_ids]
    return C_field


# ============================================================================
# Grain statistics
# ============================================================================

def grain_statistics(grain_ids):
    """
    Compute basic statistics for the microstructure.
    
    Returns dict with:
        n_grains, voxel_counts, volume_fractions, mean_size, std_size
    """
    unique, counts = np.unique(grain_ids, return_counts=True)
    total = grain_ids.size
    vf = counts / total
    
    return {
        'n_grains': len(unique),
        'grain_ids': unique,
        'voxel_counts': counts,
        'volume_fractions': vf,
        'mean_volume_fraction': np.mean(vf),
        'std_volume_fraction': np.std(vf),
        'total_voxels': total,
    }


# ============================================================================
# Quick test
# ============================================================================
if __name__ == '__main__':
    print("Generating 3D Voronoi microstructure...")
    N = 32
    n_grains = 20
    grain_ids, centers, euler_angles = generate_voronoi_microstructure(N, n_grains, seed=42)
    
    stats = grain_statistics(grain_ids)
    print(f"  Grid: {N}³ = {stats['total_voxels']} voxels")
    print(f"  Grains: {stats['n_grains']}")
    print(f"  Mean vol. fraction: {stats['mean_volume_fraction']:.4f}")
    print(f"  Std vol. fraction:  {stats['std_volume_fraction']:.4f}")
    
    print("\nBuilding stiffness field (Copper: C11=168.4, C12=121.4, C44=75.4 GPa)...")
    C_field = build_local_stiffness_field_fast(
        grain_ids, euler_angles,
        C11=168.4e9, C12=121.4e9, C44=75.4e9
    )
    print(f"  C_field shape: {C_field.shape}")
    print("  Done.")
