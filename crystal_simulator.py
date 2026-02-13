"""
Crystal Plasticity Flow Rule Simulator — PyQt5 + Matplotlib 3D Animation
=========================================================================
Interactive desktop app for:
  1. Single-crystal slip deformation visualization (OTIS engine)
  2. Polycrystalline FFT-based micromechanical simulations

Controls on the left, animated 3D crystal / field plots on the right.
Press "Run & Animate" to watch the crystal deform in real time.
"""

import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

import physics_engine
import microstructure as micro
import fft_solver
import constitutive
import postprocessing as pp


# ============================================================================
# Crystal geometry constants (FCC, 12 slip systems)
# ============================================================================
PLANE_LABELS = [
    '(111)','(111)','(111)',
    '(-111)','(-111)','(-111)',
    '(1-11)','(1-11)','(1-11)',
    '(11-1)','(11-1)','(11-1)',
]

DIR_LABELS = [
    '[01-1]','[-101]','[1-10]',
    '[01-1]','[101]','[-1-10]',
    '[011]','[-101]','[110]',
    '[011]','[101]','[-110]',
]

SYSTEM_LABELS = [f"S{i+1}" for i in range(12)]

SLIP_NORMALS = np.array([
    [ 1, 1, 1],[ 1, 1, 1],[ 1, 1, 1],
    [-1, 1, 1],[-1, 1, 1],[-1, 1, 1],
    [ 1,-1, 1],[ 1,-1, 1],[ 1,-1, 1],
    [ 1, 1,-1],[ 1, 1,-1],[ 1, 1,-1],
], dtype=float) / np.sqrt(3)

SLIP_DIRS = np.array([
    [ 0, 1,-1],[-1, 0, 1],[ 1,-1, 0],
    [ 0, 1,-1],[ 1, 0, 1],[-1,-1, 0],
    [ 0, 1, 1],[-1, 0, 1],[ 1, 1, 0],
    [ 0, 1, 1],[ 1, 0, 1],[-1, 1, 0],
], dtype=float) / np.sqrt(2)

# Unit cube corners
CUBE_CORNERS = np.array([
    [-0.5,-0.5,-0.5],[ 0.5,-0.5,-0.5],
    [ 0.5, 0.5,-0.5],[-0.5, 0.5,-0.5],
    [-0.5,-0.5, 0.5],[ 0.5,-0.5, 0.5],
    [ 0.5, 0.5, 0.5],[-0.5, 0.5, 0.5],
])

# 6 cube faces (each a quad of 4 vertex indices)
CUBE_FACES = [
    [0,1,2,3],[4,5,6,7],  # bottom / top
    [0,1,5,4],[2,3,7,6],  # front / back
    [0,3,7,4],[1,2,6,5],  # left / right
]

# 12 cube edges
CUBE_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

# FCC atoms = 8 corners + 6 face centres
FACE_CENTRES = np.array([
    [ 0, 0,-0.5],[ 0, 0, 0.5],
    [ 0,-0.5, 0],[ 0, 0.5, 0],
    [-0.5, 0, 0],[ 0.5, 0, 0],
])
ATOM_POS = np.vstack([CUBE_CORNERS, FACE_CENTRES])


def calculate_rss(stress_tensor):
    """Resolved shear stress on each of the 12 FCC slip systems."""
    rss = np.zeros(12)
    for i in range(12):
        rss[i] = np.dot(SLIP_NORMALS[i], stress_tensor @ SLIP_DIRS[i])
    return rss


def deformation_gradient(slip_vals, scale):
    """F = I + scale * Σ γ_α (s_α ⊗ n_α)"""
    F = np.eye(3)
    for i in range(12):
        F += slip_vals[i] * scale * np.outer(SLIP_DIRS[i], SLIP_NORMALS[i])
    return F


def compute_auto_scale(slip_vals, target_max_disp=0.3):
    """
    Compute the magnification factor so the maximum corner displacement
    equals target_max_disp (in unit-cube coordinates).
    This ensures deformation is always clearly visible.
    """
    # Compute dF/d(scale) = Σ γ_α (s_α ⊗ n_α)
    dF = np.zeros((3, 3))
    for i in range(12):
        dF += slip_vals[i] * np.outer(SLIP_DIRS[i], SLIP_NORMALS[i])
    # Max displacement = max over corners of |dF @ corner|
    max_disp = 0.0
    for c in CUBE_CORNERS:
        disp = np.linalg.norm(dF @ c)
        if disp > max_disp:
            max_disp = disp
    if max_disp < 1e-30:
        return 1.0  # no slip at all
    return target_max_disp / max_disp


# ============================================================================
# 3D Canvas (Matplotlib embedded in Qt)
# ============================================================================
class CrystalCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 7), facecolor='#1a1a2e')
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='#16213e')
        self._setup_axes()
        self.anim = None
        self.slip_vals = np.zeros(12)
        self.gamma_scale = 500.0
        self.n_frames = 60
        self._draw_initial()

    def _setup_axes(self):
        ax = self.ax
        lim = 1.0
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel('X [100]', color='white', fontsize=10)
        ax.set_ylabel('Y [010]', color='white', fontsize=10)
        ax.set_zlabel('Z [001]', color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#334466')
        ax.yaxis.pane.set_edgecolor('#334466')
        ax.zaxis.pane.set_edgecolor('#334466')
        ax.grid(True, color='#334466', linewidth=0.4)

    def _draw_initial(self):
        """Draw the undeformed crystal."""
        self.ax.cla()
        self._setup_axes()
        self._draw_crystal(np.eye(3), alpha_faces=0.25, edge_color='#5599ff',
                           atom_color='gold', face_color='#4488cc', title='Undeformed Crystal')
        self.draw_idle()

    def _draw_crystal(self, F, alpha_faces=0.3, edge_color='#2255aa',
                      atom_color='gold', face_color='#4488cc', title=''):
        ax = self.ax
        # Deform
        pts = (F @ CUBE_CORNERS.T).T
        atoms = (F @ ATOM_POS.T).T

        # Faces
        face_verts = [[pts[idx] for idx in face] for face in CUBE_FACES]
        faces_coll = Poly3DCollection(face_verts, alpha=alpha_faces,
                                      facecolor=face_color, edgecolor=edge_color, linewidth=0.6)
        ax.add_collection3d(faces_coll)

        # Edges (thicker)
        edge_lines = [[pts[e[0]], pts[e[1]]] for e in CUBE_EDGES]
        edge_coll = Line3DCollection(edge_lines, colors=edge_color, linewidths=2.0)
        ax.add_collection3d(edge_coll)

        # Atoms
        ax.scatter(atoms[:,0], atoms[:,1], atoms[:,2],
                   s=90, c=atom_color, edgecolors='black', linewidths=0.8,
                   zorder=10, depthshade=True)

        if title:
            ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=10)

    def start_animation(self, slip_vals, gamma_scale, n_frames=60):
        """Start the deformation animation from identity to final F."""
        self.slip_vals = slip_vals
        self.gamma_scale = gamma_scale
        self.n_frames = n_frames

        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except (AttributeError, TypeError):
                pass
            self.anim = None

        self.anim = animation.FuncAnimation(
            self.fig, self._update_frame,
            frames=n_frames + 1,
            interval=40,   # ~25 fps
            repeat=False,
            blit=False,
        )
        self.draw_idle()

    def _update_frame(self, frame_idx):
        t = frame_idx / self.n_frames
        self.ax.cla()
        self._setup_axes()

        # Ghost crystal (undeformed)
        ghost_verts = [[CUBE_CORNERS[idx] for idx in face] for face in CUBE_FACES]
        ghost = Poly3DCollection(ghost_verts, alpha=0.08,
                                 facecolor='#8888bb', edgecolor='#5566aa', linewidth=0.5,
                                 linestyle='--')
        self.ax.add_collection3d(ghost)
        ghost_edges = [[CUBE_CORNERS[e[0]], CUBE_CORNERS[e[1]]] for e in CUBE_EDGES]
        self.ax.add_collection3d(
            Line3DCollection(ghost_edges, colors='#5566aa', linewidths=0.7, linestyles='dashed'))

        # Deformed crystal
        F = deformation_gradient(self.slip_vals, self.gamma_scale * t)

        # Color ramps from blue → red based on t
        r_val = int(60 + 195 * t)
        g_val = int(130 - 80 * t)
        b_val = int(220 - 160 * t)
        face_color = f'#{r_val:02x}{g_val:02x}{b_val:02x}'
        edge_color = f'#{min(255,r_val+40):02x}{max(0,g_val-20):02x}{max(0,b_val-40):02x}'

        self._draw_crystal(F, alpha_faces=0.35, edge_color=edge_color,
                           atom_color='#ffcc00', face_color=face_color, title='')

        # Draw slip direction arrows for top 3 active systems
        ranked = np.argsort(-np.abs(self.slip_vals))
        arrow_colors = ['#ff4444', '#44ff44', '#ff44ff']
        for rank_i, color in zip(ranked[:3], arrow_colors):
            if abs(self.slip_vals[rank_i]) < 1e-30:
                continue
            s = SLIP_DIRS[rank_i] * np.sign(self.slip_vals[rank_i])
            length = 0.5 * t  # grow with time
            self.ax.quiver(0, 0, 0, s[0]*length, s[1]*length, s[2]*length,
                           color=color, arrow_length_ratio=0.2, linewidth=2.5)

        pct = int(t * 100)
        self.ax.set_title(f'Crystal Deformation  —  {pct}%',
                          color='white', fontsize=13, fontweight='bold', pad=10)


# ============================================================================
# Polycrystal Canvas (FFT results: 3D + 2D field plots)
# ============================================================================
class PolycrystalCanvas(FigureCanvas):
    """Canvas for displaying polycrystal FFT simulation results with 3D views."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 8), facecolor='#1a1a2e')
        super().__init__(self.fig)
        self.setParent(parent)
        self._draw_placeholder()

    def _style_ax(self, ax):
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white', labelsize=7)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for sp in ax.spines.values():
            sp.set_color('#334466')

    def _style_ax3d(self, ax):
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white', labelsize=6)
        ax.set_xlabel('X', color='white', fontsize=8)
        ax.set_ylabel('Y', color='white', fontsize=8)
        ax.set_zlabel('Z', color='white', fontsize=8)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#334466')
        ax.yaxis.pane.set_edgecolor('#334466')
        ax.zaxis.pane.set_edgecolor('#334466')
        ax.grid(True, color='#334466', linewidth=0.3)

    def _draw_placeholder(self):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#16213e')
        ax.text(0.5, 0.5, 'Polycrystal FFT Simulator\n\nConfigure and press Run',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=16, color='#5599ff', fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        self.draw_idle()

    def _draw_stress_arrows_3d(self, ax, stress_voigt, scale=0.3):
        """
        Draw stress arrows on the faces of the unit cube showing the applied
        boundary traction. Normal stresses shown as arrows perpendicular to faces;
        shear stresses shown as in-plane arrows on faces.
        """
        s11, s22, s33, s23, s13, s12 = stress_voigt / max(np.max(np.abs(stress_voigt)), 1e-10)

        # Face definitions: (center, normal_dir, tangent1, tangent2)
        faces = [
            # +X / -X faces
            (np.array([1, 0.5, 0.5]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])),
            (np.array([0, 0.5, 0.5]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])),
            # +Y / -Y faces
            (np.array([0.5, 1, 0.5]), np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([0, 0, 1])),
            (np.array([0.5, 0, 0.5]), np.array([0, -1, 0]), np.array([1, 0, 0]), np.array([0, 0, 1])),
            # +Z / -Z faces
            (np.array([0.5, 0.5, 1]), np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0])),
            (np.array([0.5, 0.5, 0]), np.array([0, 0, -1]), np.array([1, 0, 0]), np.array([0, 1, 0])),
        ]

        # Traction on each face: t_i = sigma_ij * n_j
        # +X face (n=[1,0,0]): t = [s11, s12, s13]
        # +Y face (n=[0,1,0]): t = [s12, s22, s23]
        # +Z face (n=[0,0,1]): t = [s13, s23, s33]
        tractions = [
            np.array([s11, s12, s13]),   # +X
            np.array([-s11, -s12, -s13]),  # -X
            np.array([s12, s22, s23]),    # +Y
            np.array([-s12, -s22, -s23]),  # -Y
            np.array([s13, s23, s33]),    # +Z
            np.array([-s13, -s23, -s33]),  # -Z
        ]

        face_labels = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
        for (center, normal, t1, t2), traction, label in zip(faces, tractions, face_labels):
            mag = np.linalg.norm(traction)
            if mag < 0.02:
                continue
            arrow = traction * scale
            color = '#ff4444' if np.dot(traction, normal) > 0 else '#44aaff'  # tension=red, compression=blue
            ax.quiver(center[0], center[1], center[2],
                      arrow[0], arrow[1], arrow[2],
                      color=color, arrow_length_ratio=0.25, linewidth=2.0, alpha=0.9)

    def _draw_rve_wireframe(self, ax):
        """Draw a wireframe cube representing the RVE."""
        r = [[0, 1], [0, 1], [0, 1]]
        for s, e in [(0, 1)]:
            pass
        # Draw 12 edges of unit cube
        edges = [
            ([0,1],[0,0],[0,0]), ([0,1],[1,1],[0,0]), ([0,1],[0,0],[1,1]), ([0,1],[1,1],[1,1]),
            ([0,0],[0,1],[0,0]), ([1,1],[0,1],[0,0]), ([0,0],[0,1],[1,1]), ([1,1],[0,1],[1,1]),
            ([0,0],[0,0],[0,1]), ([1,1],[0,0],[0,1]), ([0,0],[1,1],[0,1]), ([1,1],[1,1],[0,1]),
        ]
        for xs, ys, zs in edges:
            ax.plot(xs, ys, zs, color='#5599ff', linewidth=1.0, alpha=0.6)

    def show_loading_diagram(self, stress_voigt):
        """Show a 3D diagram of the RVE with boundary stress arrows."""
        self.fig.clf()
        ax = self.fig.add_subplot(111, projection='3d', facecolor='#16213e')
        self._style_ax3d(ax)

        self._draw_rve_wireframe(ax)
        self._draw_stress_arrows_3d(ax, stress_voigt, scale=0.35)

        # Label stress components
        s = stress_voigt / 1e6
        info_text = (f'Applied Stress (MPa)\n'
                     f'σ₁₁={s[0]:.0f}  σ₂₂={s[1]:.0f}  σ₃₃={s[2]:.0f}\n'
                     f'σ₂₃={s[3]:.0f}  σ₁₃={s[4]:.0f}  σ₁₂={s[5]:.0f}')
        ax.text2D(0.02, 0.95, info_text, transform=ax.transAxes,
                  color='white', fontsize=9, verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a3e',
                            edgecolor='#334466', alpha=0.9))

        # Legend for arrow colors
        ax.text2D(0.02, 0.05, '■ Tension    ■ Compression', transform=ax.transAxes,
                  color='white', fontsize=8)
        ax.text2D(0.02, 0.05, '■', transform=ax.transAxes, color='#ff4444', fontsize=8)
        ax.text2D(0.095, 0.05, '■', transform=ax.transAxes, color='#44aaff', fontsize=8)

        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.3, 1.3)
        ax.set_zlim(-0.3, 1.3)
        ax.set_title('Boundary Stress Loading on RVE', color='white', fontsize=13,
                      fontweight='bold', pad=10)
        ax.view_init(elev=25, azim=-60)
        self.draw_idle()

    def show_grain_map(self, grain_ids, slice_idx=None):
        """Show the grain microstructure with 3D boundary view + 2D slices."""
        self.fig.clf()
        N = grain_ids.shape[0]
        if slice_idx is None:
            slice_idx = N // 2

        gs = gridspec.GridSpec(1, 2, figure=self.fig, wspace=0.05,
                               width_ratios=[1.2, 1])

        # Left: 3D grain boundary scatter
        ax3d = self.fig.add_subplot(gs[0, 0], projection='3d')
        self._style_ax3d(ax3d)

        n_grains = len(np.unique(grain_ids))
        boundary = np.zeros_like(grain_ids, dtype=bool)
        boundary[:-1, :, :] |= grain_ids[:-1, :, :] != grain_ids[1:, :, :]
        boundary[:, :-1, :] |= grain_ids[:, :-1, :] != grain_ids[:, 1:, :]
        boundary[:, :, :-1] |= grain_ids[:, :, :-1] != grain_ids[:, :, 1:]

        bi, bj, bk = np.where(boundary)
        # Subsample if too many points
        max_pts = 8000
        if len(bi) > max_pts:
            idx = np.random.choice(len(bi), max_pts, replace=False)
            bi, bj, bk = bi[idx], bj[idx], bk[idx]

        colors = [pp.GRAIN_COLORS[grain_ids[i, j, k] % len(pp.GRAIN_COLORS)]
                  for i, j, k in zip(bi, bj, bk)]
        ax3d.scatter(bi/N, bj/N, bk/N, c=colors, s=3, alpha=0.5, depthshade=True)
        ax3d.set_xlim(0, 1); ax3d.set_ylim(0, 1); ax3d.set_zlim(0, 1)
        ax3d.set_title(f'3D Grain Boundaries', color='white', fontsize=10)
        ax3d.view_init(elev=25, azim=-60)

        # Right: 3 orthogonal 2D slices stacked
        gs_right = gs[0, 1].subgridspec(3, 1, hspace=0.35)
        for panel, (axis, label) in enumerate([(2, 'Z'), (1, 'Y'), (0, 'X')]):
            ax = self.fig.add_subplot(gs_right[panel])
            self._style_ax(ax)
            if axis == 2:
                data = grain_ids[:, :, slice_idx]
            elif axis == 1:
                data = grain_ids[:, slice_idx, :]
            else:
                data = grain_ids[slice_idx, :, :]

            cmap = matplotlib.colormaps.get_cmap('tab20').resampled(n_grains)
            ax.imshow(data.T, origin='lower', cmap=cmap,
                      interpolation='nearest', extent=[0, 1, 0, 1])
            ax.set_title(f'{label}-slice', color='white', fontsize=9)
            ax.set_aspect('equal')

        self.fig.suptitle(f'Grain Microstructure  —  {n_grains} grains, {N}³ grid',
                          color='white', fontsize=14, fontweight='bold', y=0.98)
        self.draw_idle()

    def show_results(self, eps_field, sig_field, grain_ids, stress_applied=None, slice_idx=None):
        """Show results: 3D VM stress + loading arrows, plus 2D result panels."""
        self.fig.clf()
        N = grain_ids.shape[0]
        if slice_idx is None:
            slice_idx = N // 2

        gs = gridspec.GridSpec(2, 3, figure=self.fig, hspace=0.35, wspace=0.35)

        # 1 — 3D Von Mises stress on grain boundaries with loading arrows
        vm = fft_solver.von_mises_stress(sig_field)
        ax3d = self.fig.add_subplot(gs[0, 0], projection='3d')
        self._style_ax3d(ax3d)

        # Grain boundary voxels colored by VM stress
        boundary = np.zeros_like(grain_ids, dtype=bool)
        boundary[:-1, :, :] |= grain_ids[:-1, :, :] != grain_ids[1:, :, :]
        boundary[:, :-1, :] |= grain_ids[:, :-1, :] != grain_ids[:, 1:, :]
        boundary[:, :, :-1] |= grain_ids[:, :, :-1] != grain_ids[:, :, 1:]

        bi, bj, bk = np.where(boundary)
        max_pts = 6000
        if len(bi) > max_pts:
            idx = np.random.choice(len(bi), max_pts, replace=False)
            bi, bj, bk = bi[idx], bj[idx], bk[idx]

        vm_vals = vm[bi, bj, bk] / 1e6
        vm_min, vm_max = np.min(vm) / 1e6, np.max(vm) / 1e6
        norm_vals = (vm_vals - vm_min) / (vm_max - vm_min + 1e-10)
        cmap_obj = matplotlib.colormaps.get_cmap('hot')
        colors_3d = cmap_obj(norm_vals)

        sc = ax3d.scatter(bi/N, bj/N, bk/N, c=vm_vals, cmap='hot', s=3,
                          alpha=0.6, depthshade=True, vmin=vm_min, vmax=vm_max)
        self.fig.colorbar(sc, ax=ax3d, fraction=0.03, pad=0.1, label='MPa',
                          shrink=0.6)

        # Draw loading arrows if stress is given
        if stress_applied is not None:
            self._draw_stress_arrows_3d(ax3d, stress_applied, scale=0.25)
        self._draw_rve_wireframe(ax3d)

        ax3d.set_xlim(0, 1); ax3d.set_ylim(0, 1); ax3d.set_zlim(0, 1)
        ax3d.set_title('3D Von Mises + Loading', color='white', fontsize=9)
        ax3d.view_init(elev=25, azim=-60)

        # 2 — Von Mises stress 2D slice
        ax2 = self.fig.add_subplot(gs[0, 1])
        self._style_ax(ax2)
        im2 = ax2.imshow(vm[:, :, slice_idx].T / 1e6, origin='lower',
                         cmap=pp.STRESS_CMAP, interpolation='nearest', extent=[0, 1, 0, 1])
        ax2.set_title('Von Mises Stress (Z-slice)', color='white', fontsize=9)
        ax2.set_aspect('equal')
        self.fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='MPa')

        # 3 — Grain map 2D
        ax3 = self.fig.add_subplot(gs[0, 2])
        self._style_ax(ax3)
        data = grain_ids[:, :, slice_idx]
        n_grains = len(np.unique(grain_ids))
        cmap_g = matplotlib.colormaps.get_cmap('tab20').resampled(n_grains)
        ax3.imshow(data.T, origin='lower', cmap=cmap_g,
                   interpolation='nearest', extent=[0, 1, 0, 1])
        ax3.set_title('Grain Map', color='white', fontsize=9)
        ax3.set_aspect('equal')

        # 4 — σ₁₁
        ax4 = self.fig.add_subplot(gs[1, 0])
        self._style_ax(ax4)
        s11 = sig_field[:, :, slice_idx, 0] if len(sig_field.shape) == 4 else sig_field[:, :, 0]
        im4 = ax4.imshow(s11.T / 1e6, origin='lower', cmap='RdBu_r',
                         interpolation='nearest', extent=[0, 1, 0, 1])
        ax4.set_title('σ₁₁ Stress', color='white', fontsize=9)
        ax4.set_aspect('equal')
        self.fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='MPa')

        # 5 — ε₁₁
        ax5 = self.fig.add_subplot(gs[1, 1])
        self._style_ax(ax5)
        e11 = eps_field[:, :, slice_idx, 0] if len(eps_field.shape) == 4 else eps_field[:, :, 0]
        im5 = ax5.imshow(e11.T * 100, origin='lower', cmap='RdBu_r',
                         interpolation='nearest', extent=[0, 1, 0, 1])
        ax5.set_title('ε₁₁ Strain', color='white', fontsize=9)
        ax5.set_aspect('equal')
        self.fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label='%')

        # 6 — Hydrostatic pressure
        ax6 = self.fig.add_subplot(gs[1, 2])
        self._style_ax(ax6)
        p = -(sig_field[:, :, slice_idx, 0] + sig_field[:, :, slice_idx, 1] +
              sig_field[:, :, slice_idx, 2]) / 3
        im6 = ax6.imshow(p.T / 1e6, origin='lower', cmap='coolwarm',
                         interpolation='nearest', extent=[0, 1, 0, 1])
        ax6.set_title('Hydrostatic Pressure', color='white', fontsize=9)
        ax6.set_aspect('equal')
        self.fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04, label='MPa')

        self.fig.suptitle('FFT Crystal Plasticity  —  Polycrystal Results',
                          color='white', fontsize=13, fontweight='bold', y=0.99)
        self.draw_idle()

    def show_convergence(self, info):
        """Show convergence plot."""
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        self._style_ax(ax)
        errors = info.get('errors', [])
        ax.semilogy(errors, 'o-', color='#5599ff', markersize=4, linewidth=1.5)
        ax.set_xlabel('Iteration', color='white')
        ax.set_ylabel('Stress Error (Newton)' if 'stress_iterations' in info else 'FFT Error',
                       color='white')
        ax.set_title('Solver Convergence', color='white', fontsize=13, fontweight='bold')
        ax.grid(True, color='#334466', linewidth=0.4)
        ax.axhline(y=info.get('final_stress_error', info.get('final_error', 0)),
                   color='#ff4444', linestyle='--', linewidth=1, alpha=0.7,
                   label=f"Final: {info.get('final_stress_error', info.get('final_error', 0)):.2e}")
        ax.legend(facecolor='#1a1a3e', edgecolor='#334466', labelcolor='white')
        self.draw_idle()


# ============================================================================
# Control Panel
# ============================================================================
class ControlPanel(QtWidgets.QWidget):
    run_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)
        self.setStyleSheet("""
            QWidget { background-color: #0f0f23; color: #ccccdd; font-size: 13px; }
            QGroupBox { border: 1px solid #334466; border-radius: 6px; margin-top: 10px; 
                        padding: 12px 8px 8px 8px; font-weight: bold; color: #88aadd; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
            QLabel { color: #aabbcc; }
            QDoubleSpinBox, QSpinBox { background: #1a1a3e; border: 1px solid #334466;
                                       border-radius: 4px; padding: 4px; color: white; }
            QSlider::groove:horizontal { background: #334466; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #5599ff; width: 16px; margin: -5px 0;
                                          border-radius: 8px; }
            QPushButton { background: #2255aa; color: white; border: none; border-radius: 6px;
                          padding: 10px; font-size: 14px; font-weight: bold; }
            QPushButton:hover { background: #3366cc; }
            QPushButton:pressed { background: #1144aa; }
            QComboBox { background: #1a1a3e; border: 1px solid #334466; border-radius: 4px;
                        padding: 4px; color: white; }
            QComboBox QAbstractItemView { background: #1a1a3e; color: white;
                                          selection-background-color: #2255aa; }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QtWidgets.QLabel("🔬  Crystal Plasticity\n      Simulator")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #5599ff; padding: 8px 0;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        # --- Presets ---
        preset_group = QtWidgets.QGroupBox("Quick Presets")
        pg_layout = QtWidgets.QVBoxLayout(preset_group)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems([
            "Uniaxial [100]  (300 MPa)",
            "Uniaxial [111]  (300 MPa)",
            "Pure Shear XY  (150 MPa)",
            "Biaxial XX+YY  (300 MPa)",
            "Custom",
        ])
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        pg_layout.addWidget(self.preset_combo)
        layout.addWidget(preset_group)

        # --- Stress ---
        stress_group = QtWidgets.QGroupBox("Stress Tensor σ (MPa)")
        sg_layout = QtWidgets.QGridLayout(stress_group)
        labels = ['σ₁₁', 'σ₂₂', 'σ₃₃', 'σ₁₂', 'σ₂₃', 'σ₁₃']
        self.stress_spins = {}
        for i, name in enumerate(labels):
            lbl = QtWidgets.QLabel(name)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-5000, 5000)
            spin.setDecimals(1)
            spin.setSingleStep(10)
            spin.setSuffix(" MPa")
            self.stress_spins[name] = spin
            row, col = divmod(i, 2)
            sg_layout.addWidget(lbl, row, col * 2)
            sg_layout.addWidget(spin, row, col * 2 + 1)
        layout.addWidget(stress_group)

        # --- Temperature ---
        temp_group = QtWidgets.QGroupBox("Temperature")
        tg_layout = QtWidgets.QHBoxLayout(temp_group)
        self.temp_spin = QtWidgets.QDoubleSpinBox()
        self.temp_spin.setRange(200, 2500)
        self.temp_spin.setValue(1123)
        self.temp_spin.setSingleStep(10)
        self.temp_spin.setSuffix(" K")
        tg_layout.addWidget(QtWidgets.QLabel("T"))
        tg_layout.addWidget(self.temp_spin)
        layout.addWidget(temp_group)

        # --- Deformation Intensity ---
        mag_group = QtWidgets.QGroupBox("Deformation Intensity")
        mg_layout = QtWidgets.QVBoxLayout(mag_group)
        self.mag_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.mag_slider.setRange(5, 100)
        self.mag_slider.setValue(50)
        self.mag_label = QtWidgets.QLabel("50%")
        self.mag_label.setAlignment(QtCore.Qt.AlignCenter)
        self.mag_slider.valueChanged.connect(
            lambda v: self.mag_label.setText(f"{v}%"))
        mg_layout.addWidget(self.mag_slider)
        mg_layout.addWidget(self.mag_label)
        layout.addWidget(mag_group)

        # --- Animation speed ---
        speed_group = QtWidgets.QGroupBox("Animation Speed")
        spg_layout = QtWidgets.QHBoxLayout(speed_group)
        self.speed_spin = QtWidgets.QSpinBox()
        self.speed_spin.setRange(20, 120)
        self.speed_spin.setValue(60)
        self.speed_spin.setSuffix(" frames")
        spg_layout.addWidget(QtWidgets.QLabel("Frames"))
        spg_layout.addWidget(self.speed_spin)
        layout.addWidget(speed_group)

        # --- Run button ---
        self.run_btn = QtWidgets.QPushButton("▶   Run  &&  Animate")
        self.run_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.run_requested.emit)
        layout.addWidget(self.run_btn)

        # --- Results text ---
        self.result_text = QtWidgets.QLabel("")
        self.result_text.setStyleSheet("font-size: 11px; color: #88cc88; padding-top: 6px;")
        self.result_text.setWordWrap(True)
        layout.addWidget(self.result_text)

        layout.addStretch()

        # Apply preset defaults
        self._apply_preset(0)

    def _apply_preset(self, idx):
        presets = [
            dict(s11=300, s22=0, s33=0, s12=0, s23=0, s13=0),
            dict(s11=100, s22=100, s33=100, s12=100, s23=100, s13=100),
            dict(s11=0, s22=0, s33=0, s12=150, s23=0, s13=0),
            dict(s11=300, s22=300, s33=0, s12=0, s23=0, s13=0),
            None,  # Custom → don't touch
        ]
        if idx < len(presets) and presets[idx] is not None:
            mapping = {'σ₁₁':'s11','σ₂₂':'s22','σ₃₃':'s33',
                       'σ₁₂':'s12','σ₂₃':'s23','σ₁₃':'s13'}
            for lbl, key in mapping.items():
                self.stress_spins[lbl].setValue(presets[idx][key])

    def get_stress_tensor_pa(self):
        s = {k: v.value() * 1e6 for k, v in self.stress_spins.items()}
        return np.array([
            [s['σ₁₁'], s['σ₁₂'], s['σ₁₃']],
            [s['σ₁₂'], s['σ₂₂'], s['σ₂₃']],
            [s['σ₁₃'], s['σ₂₃'], s['σ₃₃']],
        ])


# ============================================================================
# Polycrystal Control Panel
# ============================================================================
class PolycrystalControlPanel(QtWidgets.QWidget):
    run_requested = QtCore.pyqtSignal()
    generate_requested = QtCore.pyqtSignal()
    preview_loading = QtCore.pyqtSignal()

    PANEL_STYLE = """
        QWidget { background-color: #0f0f23; color: #ccccdd; font-size: 13px; }
        QGroupBox { border: 1px solid #334466; border-radius: 6px; margin-top: 10px;
                    padding: 12px 8px 8px 8px; font-weight: bold; color: #88aadd; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; }
        QLabel { color: #aabbcc; }
        QDoubleSpinBox, QSpinBox { background: #1a1a3e; border: 1px solid #334466;
                                   border-radius: 4px; padding: 4px; color: white; }
        QSlider::groove:horizontal { background: #334466; height: 6px; border-radius: 3px; }
        QSlider::handle:horizontal { background: #5599ff; width: 16px; margin: -5px 0;
                                      border-radius: 8px; }
        QPushButton { background: #2255aa; color: white; border: none; border-radius: 6px;
                      padding: 10px; font-size: 14px; font-weight: bold; }
        QPushButton:hover { background: #3366cc; }
        QPushButton:pressed { background: #1144aa; }
        QComboBox { background: #1a1a3e; border: 1px solid #334466; border-radius: 4px;
                    padding: 4px; color: white; }
        QComboBox QAbstractItemView { background: #1a1a3e; color: white;
                                      selection-background-color: #2255aa; }
        QProgressBar { background: #1a1a3e; border: 1px solid #334466; border-radius: 4px;
                       text-align: center; color: white; }
        QProgressBar::chunk { background: #2255aa; border-radius: 3px; }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(340)
        self.setStyleSheet(self.PANEL_STYLE)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QtWidgets.QLabel("Polycrystal FFT\n      Simulator")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #5599ff; padding: 8px 0;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        # --- Microstructure ---
        micro_group = QtWidgets.QGroupBox("Microstructure")
        mg_layout = QtWidgets.QGridLayout(micro_group)

        mg_layout.addWidget(QtWidgets.QLabel("Grid N"), 0, 0)
        self.grid_spin = QtWidgets.QSpinBox()
        self.grid_spin.setRange(8, 128)
        self.grid_spin.setValue(32)
        self.grid_spin.setSingleStep(8)
        mg_layout.addWidget(self.grid_spin, 0, 1)

        mg_layout.addWidget(QtWidgets.QLabel("Grains"), 1, 0)
        self.grains_spin = QtWidgets.QSpinBox()
        self.grains_spin.setRange(2, 500)
        self.grains_spin.setValue(20)
        mg_layout.addWidget(self.grains_spin, 1, 1)

        mg_layout.addWidget(QtWidgets.QLabel("Seed"), 2, 0)
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        mg_layout.addWidget(self.seed_spin, 2, 1)

        self.gen_btn = QtWidgets.QPushButton("Generate Microstructure")
        self.gen_btn.setStyleSheet(
            "background: #1a6b3f; font-size: 12px; padding: 8px;")
        self.gen_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.gen_btn.clicked.connect(self.generate_requested.emit)
        mg_layout.addWidget(self.gen_btn, 3, 0, 1, 2)

        layout.addWidget(micro_group)

        # --- Material ---
        mat_group = QtWidgets.QGroupBox("Material (Cubic)")
        matl = QtWidgets.QGridLayout(mat_group)

        self.mat_combo = QtWidgets.QComboBox()
        self.mat_combo.addItems(['Copper', 'Aluminium', 'Nickel', 'Iron (BCC approx)', 'Custom'])
        self.mat_combo.currentIndexChanged.connect(self._apply_material)
        matl.addWidget(self.mat_combo, 0, 0, 1, 2)

        for row, (lbl, attr, default) in enumerate([
            ('C₁₁ (GPa)', 'c11_spin', 168.4),
            ('C₁₂ (GPa)', 'c12_spin', 121.4),
            ('C₄₄ (GPa)', 'c44_spin', 75.4),
        ], start=1):
            matl.addWidget(QtWidgets.QLabel(lbl), row, 0)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0, 2000)
            spin.setValue(default)
            spin.setDecimals(1)
            spin.setSuffix(' GPa')
            setattr(self, attr, spin)
            matl.addWidget(spin, row, 1)

        layout.addWidget(mat_group)

        # --- Boundary Stress Loading ---
        load_group = QtWidgets.QGroupBox("Boundary Stress (MPa)")
        ll = QtWidgets.QVBoxLayout(load_group)

        # Loading preset selector
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(QtWidgets.QLabel("Preset:"))
        self.load_preset = QtWidgets.QComboBox()
        self.load_preset.addItems([
            'Uniaxial Tension X  (500 MPa)',
            'Uniaxial Tension Y  (500 MPa)',
            'Uniaxial Tension Z  (500 MPa)',
            'Biaxial XY  (500 MPa)',
            'Hydrostatic  (500 MPa)',
            'Pure Shear XY  (250 MPa)',
            'Pure Shear XZ  (250 MPa)',
            'Custom',
        ])
        self.load_preset.currentIndexChanged.connect(self._apply_load_preset)
        preset_row.addWidget(self.load_preset)
        ll.addLayout(preset_row)

        # Stress tensor input: 3x3 symmetric grid layout
        tensor_grid = QtWidgets.QGridLayout()

        # Header: column labels
        for col, lbl in enumerate(['X', 'Y', 'Z']):
            h = QtWidgets.QLabel(lbl)
            h.setAlignment(QtCore.Qt.AlignCenter)
            h.setStyleSheet("color: #5599ff; font-weight: bold;")
            tensor_grid.addWidget(h, 0, col + 1)
        # Row labels
        for row, lbl in enumerate(['X', 'Y', 'Z']):
            h = QtWidgets.QLabel(lbl)
            h.setAlignment(QtCore.Qt.AlignCenter)
            h.setStyleSheet("color: #5599ff; font-weight: bold;")
            tensor_grid.addWidget(h, row + 1, 0)

        # Stress spin boxes in matrix form
        self.stress_matrix_spins = {}
        stress_map = {
            (0, 0): 'σ₁₁', (1, 1): 'σ₂₂', (2, 2): 'σ₃₃',
            (0, 1): 'σ₁₂', (1, 0): 'σ₁₂',
            (0, 2): 'σ₁₃', (2, 0): 'σ₁₃',
            (1, 2): 'σ₂₃', (2, 1): 'σ₂₃',
        }
        # Create unique spins for upper triangle
        voigt_labels = ['σ₁₁', 'σ₂₂', 'σ₃₃', 'σ₂₃', 'σ₁₃', 'σ₁₂']
        self.stress_spins = {}
        for name in voigt_labels:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-5000, 5000)
            spin.setDecimals(1)
            spin.setSingleStep(10)
            spin.setSuffix(" MPa")
            spin.setFixedWidth(110)
            self.stress_spins[name] = spin

        # Place in matrix grid
        positions = {
            'σ₁₁': (1, 1), 'σ₁₂': (1, 2), 'σ₁₃': (1, 3),
            'σ₂₂': (2, 2), 'σ₂₃': (2, 3),
            'σ₃₃': (3, 3),
        }
        for name, (r, c) in positions.items():
            tensor_grid.addWidget(self.stress_spins[name], r, c)

        # Mirror labels for lower triangle
        for name, (r, c) in [('σ₁₂', (2, 1)), ('σ₁₃', (3, 1)), ('σ₂₃', (3, 2))]:
            mirror_lbl = QtWidgets.QLabel("= " + name)
            mirror_lbl.setStyleSheet("color: #667788; font-size: 11px;")
            mirror_lbl.setAlignment(QtCore.Qt.AlignCenter)
            tensor_grid.addWidget(mirror_lbl, r, c)

        ll.addLayout(tensor_grid)

        # Loading description label
        self.load_desc = QtWidgets.QLabel("")
        self.load_desc.setStyleSheet("color: #88aadd; font-size: 11px; padding: 4px;")
        self.load_desc.setWordWrap(True)
        ll.addWidget(self.load_desc)

        # Preview loading button
        self.preview_btn = QtWidgets.QPushButton("Preview Loading on RVE")
        self.preview_btn.setStyleSheet(
            "background: #664400; font-size: 11px; padding: 6px;")
        self.preview_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.preview_btn.clicked.connect(self.preview_loading.emit)
        ll.addWidget(self.preview_btn)

        layout.addWidget(load_group)

        # --- Solver ---
        solver_group = QtWidgets.QGroupBox("Solver")
        sl = QtWidgets.QGridLayout(solver_group)

        sl.addWidget(QtWidgets.QLabel("Method"), 0, 0)
        self.solver_combo = QtWidgets.QComboBox()
        self.solver_combo.addItems(['Conjugate Gradient (Zeman 2010)', 'Basic Scheme (Moulinec-Suquet)'])
        sl.addWidget(self.solver_combo, 0, 1)

        sl.addWidget(QtWidgets.QLabel("FFT Tol"), 1, 0)
        self.tol_combo = QtWidgets.QComboBox()
        self.tol_combo.addItems(['1e-4', '1e-5', '1e-6', '1e-7'])
        self.tol_combo.setCurrentIndex(1)
        sl.addWidget(self.tol_combo, 1, 1)

        sl.addWidget(QtWidgets.QLabel("Stress Tol"), 2, 0)
        self.stress_tol_combo = QtWidgets.QComboBox()
        self.stress_tol_combo.addItems(['1e-3', '1e-4', '1e-5'])
        self.stress_tol_combo.setCurrentIndex(1)
        sl.addWidget(self.stress_tol_combo, 2, 1)

        sl.addWidget(QtWidgets.QLabel("Max Iter"), 3, 0)
        self.maxiter_spin = QtWidgets.QSpinBox()
        self.maxiter_spin.setRange(10, 5000)
        self.maxiter_spin.setValue(500)
        sl.addWidget(self.maxiter_spin, 3, 1)

        layout.addWidget(solver_group)

        # --- Run ---
        self.run_btn = QtWidgets.QPushButton("Run FFT Simulation")
        self.run_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.run_requested.emit)
        layout.addWidget(self.run_btn)

        # Progress bar
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Results
        self.result_text = QtWidgets.QLabel("")
        self.result_text.setStyleSheet("font-size: 11px; color: #88cc88; padding-top: 6px;")
        self.result_text.setWordWrap(True)
        layout.addWidget(self.result_text)

        layout.addStretch()
        self._apply_material(0)
        self._apply_load_preset(0)

    def _apply_material(self, idx):
        materials = [
            (168.4, 121.4, 75.4),   # Copper
            (108.2, 61.3, 28.5),    # Aluminium
            (246.5, 147.3, 124.7),  # Nickel
            (231.4, 134.7, 116.4),  # Iron
            None,
        ]
        if idx < len(materials) and materials[idx] is not None:
            c11, c12, c44 = materials[idx]
            self.c11_spin.setValue(c11)
            self.c12_spin.setValue(c12)
            self.c44_spin.setValue(c44)

    def _apply_load_preset(self, idx):
        presets = [
            # (σ₁₁, σ₂₂, σ₃₃, σ₂₃, σ₁₃, σ₁₂, description)
            (500,   0,   0,   0,   0,   0,  "Uniaxial tension along X:\nσ₁₁ applied on ±X faces"),
            (  0, 500,   0,   0,   0,   0,  "Uniaxial tension along Y:\nσ₂₂ applied on ±Y faces"),
            (  0,   0, 500,   0,   0,   0,  "Uniaxial tension along Z:\nσ₃₃ applied on ±Z faces"),
            (500, 500,   0,   0,   0,   0,  "Biaxial tension in XY plane:\nσ₁₁ on ±X, σ₂₂ on ±Y faces"),
            (500, 500, 500,   0,   0,   0,  "Hydrostatic stress:\nEqual normal stress on all faces"),
            (  0,   0,   0,   0,   0, 250,  "Pure shear in XY plane:\nσ₁₂ shear on X and Y faces"),
            (  0,   0,   0,   0, 250,   0,  "Pure shear in XZ plane:\nσ₁₃ shear on X and Z faces"),
            None,   # Custom
        ]
        if idx < len(presets) and presets[idx] is not None:
            s11, s22, s33, s23, s13, s12, desc = presets[idx]
            self.stress_spins['σ₁₁'].setValue(s11)
            self.stress_spins['σ₂₂'].setValue(s22)
            self.stress_spins['σ₃₃'].setValue(s33)
            self.stress_spins['σ₂₃'].setValue(s23)
            self.stress_spins['σ₁₃'].setValue(s13)
            self.stress_spins['σ₁₂'].setValue(s12)
            self.load_desc.setText(desc)
        else:
            self.load_desc.setText("Custom stress state — set tensor components manually")

    def get_stress_voigt_pa(self):
        """Return macroscopic stress as Voigt vector in Pa."""
        keys = ['σ₁₁', 'σ₂₂', 'σ₃₃', 'σ₂₃', 'σ₁₃', 'σ₁₂']
        return np.array([self.stress_spins[k].value() * 1e6 for k in keys])


# ============================================================================
# Main Window (tabbed: Single Crystal + Polycrystal)
# ============================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crystal Plasticity Simulator  —  Single Crystal + FFT Polycrystal")
        self.setMinimumSize(1300, 800)
        self.setStyleSheet("background-color: #0f0f23;")

        # Tab widget
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #334466; background: #0f0f23; }
            QTabBar::tab { background: #1a1a3e; color: #aabbcc; padding: 10px 20px;
                          border: 1px solid #334466; border-bottom: none; border-radius: 6px 6px 0 0;
                          font-size: 13px; font-weight: bold; margin-right: 2px; }
            QTabBar::tab:selected { background: #2255aa; color: white; }
            QTabBar::tab:hover { background: #334466; }
        """)
        self.setCentralWidget(self.tabs)

        # --- Tab 1: Single Crystal ---
        tab1 = QtWidgets.QWidget()
        t1_layout = QtWidgets.QHBoxLayout(tab1)
        t1_layout.setContentsMargins(0, 0, 0, 0)

        self.sc_controls = ControlPanel()
        self.sc_controls.run_requested.connect(self._run_single_crystal)
        t1_layout.addWidget(self.sc_controls)

        self.sc_canvas = CrystalCanvas()
        t1_layout.addWidget(self.sc_canvas, stretch=1)

        self.tabs.addTab(tab1, "🔷  Single Crystal")

        # --- Tab 2: Polycrystal FFT ---
        tab2 = QtWidgets.QWidget()
        t2_layout = QtWidgets.QHBoxLayout(tab2)
        t2_layout.setContentsMargins(0, 0, 0, 0)

        self.pc_controls = PolycrystalControlPanel()
        self.pc_controls.generate_requested.connect(self._generate_microstructure)
        self.pc_controls.run_requested.connect(self._run_polycrystal)
        self.pc_controls.preview_loading.connect(self._preview_loading)
        t2_layout.addWidget(self.pc_controls)

        self.pc_canvas = PolycrystalCanvas()
        t2_layout.addWidget(self.pc_canvas, stretch=1)

        self.tabs.addTab(tab2, "🔶  Polycrystal FFT")

        # State
        self.grain_ids = None
        self.euler_angles = None
        self.pc_eps_field = None
        self.pc_sig_field = None

    # ---- Single crystal simulation (unchanged logic) ----
    def _run_single_crystal(self):
        stress_pa = self.sc_controls.get_stress_tensor_pa()
        temperature = self.sc_controls.temp_spin.value()
        intensity_pct = float(self.sc_controls.mag_slider.value()) / 100.0
        n_frames = self.sc_controls.speed_spin.value()

        rss = calculate_rss(stress_pa)

        try:
            results = physics_engine.compute_flow_rule(rss, temperature)
        except Exception as e:
            self.sc_controls.result_text.setText(f"❌ Error: {e}")
            self.sc_controls.result_text.setStyleSheet("font-size: 11px; color: #ff6666;")
            return

        slip_vals = results['Increment_Slip_Plastic']
        total_slip = np.sum(np.abs(slip_vals))

        auto_scale = compute_auto_scale(slip_vals, target_max_disp=0.35)
        gamma_scale = auto_scale * intensity_pct

        top3 = np.argsort(-np.abs(slip_vals))[:3]
        summary_lines = [f"✅ Simulation complete  (Total |Δγ| = {total_slip:.3e})"]
        summary_lines.append(f"   Auto-magnification: ×{auto_scale:.1e}")
        for rank, idx in enumerate(top3):
            summary_lines.append(
                f"  #{rank+1}  {SYSTEM_LABELS[idx]} {PLANE_LABELS[idx]}{DIR_LABELS[idx]}  "
                f"Δγ = {slip_vals[idx]:.3e}")
        self.sc_controls.result_text.setText("\n".join(summary_lines))
        self.sc_controls.result_text.setStyleSheet("font-size: 11px; color: #88cc88;")

        self.sc_canvas.start_animation(slip_vals, gamma_scale, n_frames)

    # ---- Polycrystal: preview loading diagram ----
    def _preview_loading(self):
        stress_pa = self.pc_controls.get_stress_voigt_pa()
        self.pc_canvas.show_loading_diagram(stress_pa)

    # ---- Polycrystal: generate microstructure ----
    def _generate_microstructure(self):
        N = self.pc_controls.grid_spin.value()
        n_grains = self.pc_controls.grains_spin.value()
        seed = self.pc_controls.seed_spin.value()

        self.pc_controls.result_text.setText("Generating microstructure...")
        self.pc_controls.result_text.setStyleSheet("font-size: 11px; color: #ffcc00;")
        QtWidgets.QApplication.processEvents()

        self.grain_ids, self.centers, self.euler_angles = \
            micro.generate_voronoi_microstructure(N, n_grains, seed=seed)

        stats = micro.grain_statistics(self.grain_ids)
        self.pc_canvas.show_grain_map(self.grain_ids)

        self.pc_controls.result_text.setText(
            f"✅ Microstructure generated\n"
            f"   {N}³ grid = {stats['total_voxels']} voxels\n"
            f"   {stats['n_grains']} grains\n"
            f"   Mean vol. frac: {stats['mean_volume_fraction']:.4f}\n"
            f"   Std vol. frac:  {stats['std_volume_fraction']:.4f}")
        self.pc_controls.result_text.setStyleSheet("font-size: 11px; color: #88cc88;")

    # ---- Polycrystal: run FFT simulation ----
    def _run_polycrystal(self):
        if self.grain_ids is None:
            self.pc_controls.result_text.setText("Generate microstructure first!")
            self.pc_controls.result_text.setStyleSheet("font-size: 11px; color: #ff6666;")
            return

        self.pc_controls.run_btn.setEnabled(False)
        self.pc_controls.progress.setVisible(True)
        self.pc_controls.progress.setValue(10)
        self.pc_controls.result_text.setText("Building stiffness field...")
        self.pc_controls.result_text.setStyleSheet("font-size: 11px; color: #ffcc00;")
        QtWidgets.QApplication.processEvents()

        C11 = self.pc_controls.c11_spin.value() * 1e9
        C12 = self.pc_controls.c12_spin.value() * 1e9
        C44 = self.pc_controls.c44_spin.value() * 1e9

        C_field = micro.build_local_stiffness_field_fast(
            self.grain_ids, self.euler_angles, C11, C12, C44)

        S_macro = self.pc_controls.get_stress_voigt_pa()
        tol_fft = float(self.pc_controls.tol_combo.currentText())
        tol_stress = float(self.pc_controls.stress_tol_combo.currentText())
        max_iter = self.pc_controls.maxiter_spin.value()
        use_cg = self.pc_controls.solver_combo.currentIndex() == 0
        solver_name = 'CG' if use_cg else 'Basic'

        self.pc_controls.progress.setValue(30)
        self.pc_controls.result_text.setText(
            f"Running stress-controlled {solver_name} solver...")
        QtWidgets.QApplication.processEvents()

        try:
            eps_field, sig_field, info = fft_solver.solve_stress_controlled(
                C_field, S_macro,
                tol_fft=tol_fft, tol_stress=tol_stress,
                max_iter_fft=max_iter, max_iter_stress=30,
                solver='cg' if use_cg else 'basic',
                verbose=True)
        except Exception as e:
            self.pc_controls.result_text.setText(f"Error: {e}")
            self.pc_controls.result_text.setStyleSheet("font-size: 11px; color: #ff6666;")
            self.pc_controls.run_btn.setEnabled(True)
            self.pc_controls.progress.setVisible(False)
            return

        self.pc_eps_field = eps_field
        self.pc_sig_field = sig_field

        self.pc_controls.progress.setValue(90)
        QtWidgets.QApplication.processEvents()

        # Display results with loading arrows
        self.pc_canvas.show_results(eps_field, sig_field, self.grain_ids,
                                    stress_applied=S_macro)

        self.pc_controls.progress.setValue(100)

        # Summary
        vm = fft_solver.von_mises_stress(sig_field)
        sig_mean = np.mean(sig_field.reshape(-1, 6), axis=0)
        eps_mean = np.mean(eps_field.reshape(-1, 6), axis=0)
        N = self.grain_ids.shape[0]

        lines = [
            f"FFT simulation complete ({solver_name}, stress-controlled)",
            f"   Grid: {N}³,  {info['iterations']} FFT iters,  "
            f"{info.get('stress_iterations', 0)} Newton iters",
            f"   Stress err: {info.get('final_stress_error', 0):.2e}",
            f"   <σ₁₁>={sig_mean[0]/1e6:.1f}  <σ₂₂>={sig_mean[1]/1e6:.1f}  "
            f"<σ₃₃>={sig_mean[2]/1e6:.1f} MPa",
            f"   <ε₁₁>={eps_mean[0]*100:.3f}%  <ε₂₂>={eps_mean[1]*100:.3f}%  "
            f"<ε₃₃>={eps_mean[2]*100:.3f}%",
            f"   VM: mean={np.mean(vm)/1e6:.1f}, max={np.max(vm)/1e6:.1f} MPa",
        ]
        self.pc_controls.result_text.setText("\n".join(lines))
        self.pc_controls.result_text.setStyleSheet("font-size: 11px; color: #88cc88;")

        self.pc_controls.run_btn.setEnabled(True)
        self.pc_controls.progress.setVisible(False)


# ============================================================================
# Entry point
# ============================================================================
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')

    # Dark palette
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor('#0f0f23'))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor('#ccccdd'))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor('#1a1a3e'))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor('#ffffff'))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor('#2255aa'))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor('#ffffff'))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor('#3366cc'))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
