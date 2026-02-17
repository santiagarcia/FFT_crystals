"""
Crystal Plasticity FFT Simulator — Polycrystal Micromechanics
==============================================================
GPU-accelerated FFT-based crystal plasticity simulator with an
Abaqus-style model tree interface.

Left panel: collapsible model tree for Microstructure, Material,
Boundary Conditions, Solver, and Job settings.
Right panel: GPU-rendered 3D results with deformation animation.
"""

import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

import microstructure as micro
import fft_solver
import postprocessing as pp
import constitutive

# GPU-accelerated 3D viewer (PyVista / VTK)
try:
    import pyvista as pv
    import pyvistaqt
    HAS_PYVISTA = True
    pv.global_theme.background = '#1a1a2e'
    print("[Viewer] GPU-accelerated 3D rendering enabled (PyVista/VTK)")
except ImportError:
    HAS_PYVISTA = False
    print("[Viewer] PyVista not found — install pyvista + pyvistaqt for GPU rendering")



# ============================================================================
# GPU-Accelerated Polycrystal Viewer (PyVista / VTK)
# ============================================================================

class PolycrystalViewer(QtWidgets.QWidget):
    """GPU-accelerated 3D viewer for polycrystal FFT results using PyVista/VTK.
    Uses a single QtInteractor to avoid OpenGL context conflicts on Windows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = pyvistaqt.QtInteractor(self)
        self.plotter.set_background('#1a1a2e')
        layout.addWidget(self.plotter, stretch=1)

        # Animation controls bar (hidden until results exist)
        self._anim_bar = QtWidgets.QWidget()
        ab_layout = QtWidgets.QVBoxLayout(self._anim_bar)
        ab_layout.setContentsMargins(8, 4, 8, 4)
        ab_layout.setSpacing(4)
        self._anim_bar.setStyleSheet(
            "background: #121230; border-top: 1px solid #334466;")

        # Row 1: play button + magnification
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(8)

        self._play_btn = QtWidgets.QPushButton("▶  Play")
        self._play_btn.setStyleSheet(
            "background: #2255aa; color: white; border: none; border-radius: 4px;"
            " padding: 6px 14px; font-size: 12px; font-weight: bold;")
        self._play_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._play_btn.clicked.connect(self._toggle_animation)
        row1.addWidget(self._play_btn)

        row1.addWidget(QtWidgets.QLabel("Magnification:"))

        self._scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._scale_slider.setRange(1, 500)
        self._scale_slider.setValue(100)
        self._scale_slider.setFixedWidth(140)
        self._scale_slider.valueChanged.connect(self._on_scale_changed)
        row1.addWidget(self._scale_slider)

        self._scale_val = QtWidgets.QLabel("100×")
        self._scale_val.setStyleSheet("color: #5599ff; font-size: 12px; min-width: 40px;")
        row1.addWidget(self._scale_val)

        self._frame_label = QtWidgets.QLabel("")
        self._frame_label.setStyleSheet("color: #88aadd; font-size: 11px;")
        row1.addStretch()
        row1.addWidget(self._frame_label)
        ab_layout.addLayout(row1)

        # Row 2: frame scrub slider
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(6)
        row2.addWidget(QtWidgets.QLabel("Frame:"))

        self._frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._frame_slider.setRange(0, 60)
        self._frame_slider.setValue(0)
        self._frame_slider.setTracking(False)  # fire valueChanged only on release
        self._frame_slider.valueChanged.connect(self._on_frame_scrub)
        self._frame_slider.sliderMoved.connect(self._on_frame_preview)
        row2.addWidget(self._frame_slider, stretch=1)

        self._frame_num_label = QtWidgets.QLabel("0 / 60")
        self._frame_num_label.setStyleSheet(
            "color: #5599ff; font-size: 12px; min-width: 50px;")
        row2.addWidget(self._frame_num_label)
        ab_layout.addLayout(row2)

        # Row 3: field selector
        row3 = QtWidgets.QHBoxLayout()
        row3.setSpacing(6)
        row3.addWidget(QtWidgets.QLabel("Display Field:"))
        self._field_combo = QtWidgets.QComboBox()
        self._field_combo.addItems([
            'VM Stress (MPa)',
            '\u03c3\u2081\u2081 (MPa)', '\u03c3\u2082\u2082 (MPa)',
            '\u03c3\u2083\u2083 (MPa)', '\u03c3\u2082\u2083 (MPa)',
            '\u03c3\u2081\u2083 (MPa)', '\u03c3\u2081\u2082 (MPa)',
            '\u03b5\u2081\u2081', '\u03b5\u2082\u2082', '\u03b5\u2083\u2083',
            '\u03b5\u2082\u2083', '\u03b5\u2081\u2083', '\u03b5\u2081\u2082',
            '|u| (mm)',
            '\u03b5\u1d56 eq (plastic)',
            'Accum. Slip',
            'Slip Resistance (MPa)',
            'SSD Density (m\u207b\u00b2)',
            'GND Density (m\u207b\u00b2)',
            'Back-stress (MPa)',
            'Misorientation (\u00b0)',
        ])
        self._field_combo.setStyleSheet(
            "background: #1a1a3e; border: 1px solid #334466; border-radius: 3px;"
            " padding: 3px 8px; color: white; font-size: 12px; min-width: 130px;")
        self._field_combo.currentIndexChanged.connect(self._on_field_changed)
        row3.addWidget(self._field_combo)
        row3.addStretch()
        ab_layout.addLayout(row3)

        layout.addWidget(self._anim_bar)
        self._anim_bar.setVisible(False)

        # Animation state
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.timeout.connect(self._anim_step)
        self._anim_running = False
        self._anim_frame = 0
        self._anim_n_frames = 60
        self._anim_forward = True
        self._anim_data = None   # dict with mesh data for animation
        self._anim_mesh = None   # live PyVista mesh (update points in-place)
        self._anim_actor = None  # VTK actor reference
        self._scrubbing = False  # avoid recursion from slider updates

        self._draw_placeholder()

    def _draw_placeholder(self):
        self.plotter.clear()
        self._anim_data = None
        self.plotter.add_text(
            'Polycrystal FFT Simulator\n\nConfigure and press Run',
            position='upper_left', color='#5599ff', font_size=14)

    # ---- mesh utilities ----

    @staticmethod
    def _extract_boundary_quads(grain_ids, max_faces=100000):
        """
        Extract quad faces between voxels with different grain IDs.
        Higher face limit than matplotlib version — VTK handles it easily.
        """
        N = grain_ids.shape[0]
        h = 1.0 / N
        all_quads, all_ga, all_gb = [], [], []
        all_ci, all_cj, all_ck = [], [], []

        # X-direction boundaries
        mask = grain_ids[:-1, :, :] != grain_ids[1:, :, :]
        ix, iy, iz = np.where(mask)
        n = len(ix)
        if n > 0:
            x = (ix + 1) * h
            y0, y1 = iy * h, (iy + 1) * h
            z0, z1 = iz * h, (iz + 1) * h
            q = np.zeros((n, 4, 3))
            q[:, 0] = np.column_stack([x, y0, z0])
            q[:, 1] = np.column_stack([x, y1, z0])
            q[:, 2] = np.column_stack([x, y1, z1])
            q[:, 3] = np.column_stack([x, y0, z1])
            all_quads.append(q)
            all_ga.append(grain_ids[ix, iy, iz])
            all_gb.append(grain_ids[ix + 1, iy, iz])
            all_ci.append(ix); all_cj.append(iy); all_ck.append(iz)

        # Y-direction boundaries
        mask = grain_ids[:, :-1, :] != grain_ids[:, 1:, :]
        ix, iy, iz = np.where(mask)
        n = len(ix)
        if n > 0:
            y = (iy + 1) * h
            x0, x1 = ix * h, (ix + 1) * h
            z0, z1 = iz * h, (iz + 1) * h
            q = np.zeros((n, 4, 3))
            q[:, 0] = np.column_stack([x0, y, z0])
            q[:, 1] = np.column_stack([x1, y, z0])
            q[:, 2] = np.column_stack([x1, y, z1])
            q[:, 3] = np.column_stack([x0, y, z1])
            all_quads.append(q)
            all_ga.append(grain_ids[ix, iy, iz])
            all_gb.append(grain_ids[ix, iy + 1, iz])
            all_ci.append(ix); all_cj.append(iy); all_ck.append(iz)

        # Z-direction boundaries
        mask = grain_ids[:, :, :-1] != grain_ids[:, :, 1:]
        ix, iy, iz = np.where(mask)
        n = len(ix)
        if n > 0:
            z = (iz + 1) * h
            x0, x1 = ix * h, (ix + 1) * h
            y0, y1 = iy * h, (iy + 1) * h
            q = np.zeros((n, 4, 3))
            q[:, 0] = np.column_stack([x0, y0, z])
            q[:, 1] = np.column_stack([x1, y0, z])
            q[:, 2] = np.column_stack([x1, y1, z])
            q[:, 3] = np.column_stack([x0, y1, z])
            all_quads.append(q)
            all_ga.append(grain_ids[ix, iy, iz])
            all_gb.append(grain_ids[ix, iy, iz + 1])
            all_ci.append(ix); all_cj.append(iy); all_ck.append(iz)

        if not all_quads:
            empty = np.zeros((0,), dtype=int)
            return np.zeros((0, 4, 3)), empty, empty, empty, empty, empty

        quads = np.concatenate(all_quads)
        ga = np.concatenate(all_ga)
        gb = np.concatenate(all_gb)
        ci = np.concatenate(all_ci)
        cj = np.concatenate(all_cj)
        ck = np.concatenate(all_ck)

        if len(quads) > max_faces:
            idx = np.random.choice(len(quads), max_faces, replace=False)
            quads, ga, gb = quads[idx], ga[idx], gb[idx]
            ci, cj, ck = ci[idx], cj[idx], ck[idx]

        return quads, ga, gb, ci, cj, ck

    @staticmethod
    def _quads_to_mesh(quads, scalars=None, scalar_name='data'):
        """Convert (n_faces, 4, 3) boundary quads to a PyVista PolyData mesh."""
        n_faces = quads.shape[0]
        points = quads.reshape(-1, 3).astype(np.float64)
        # VTK face format: [n_verts, v0, v1, v2, v3, ...]
        faces = np.empty(n_faces * 5, dtype=np.int64)
        faces[0::5] = 4
        for k in range(4):
            faces[(k + 1)::5] = np.arange(n_faces) * 4 + k
        mesh = pv.PolyData(points, faces)
        if scalars is not None:
            mesh.cell_data[scalar_name] = scalars
        return mesh

    def _add_rve_wireframe(self, plotter):
        """Add unit-cube wireframe to a plotter."""
        cube = pv.Box(bounds=[0, 1, 0, 1, 0, 1])
        plotter.add_mesh(cube, style='wireframe', color='#5599ff',
                         line_width=2, render_lines_as_tubes=False)

    def _add_stress_arrows(self, plotter, stress_voigt, scale=0.35):
        """Draw boundary traction arrows on RVE faces."""
        s_norm = stress_voigt / max(np.max(np.abs(stress_voigt)), 1e-10)

        face_defs = [
            ([1, 0.5, 0.5],   [ 1, 0, 0]),
            ([0, 0.5, 0.5],   [-1, 0, 0]),
            ([0.5, 1, 0.5],   [ 0, 1, 0]),
            ([0.5, 0, 0.5],   [ 0,-1, 0]),
            ([0.5, 0.5, 1],   [ 0, 0, 1]),
            ([0.5, 0.5, 0],   [ 0, 0,-1]),
        ]
        tractions = [
            np.array([ s_norm[0],  s_norm[5],  s_norm[4]]),   # +X
            np.array([-s_norm[0], -s_norm[5], -s_norm[4]]),   # -X
            np.array([ s_norm[5],  s_norm[1],  s_norm[3]]),   # +Y
            np.array([-s_norm[5], -s_norm[1], -s_norm[3]]),   # -Y
            np.array([ s_norm[4],  s_norm[3],  s_norm[2]]),   # +Z
            np.array([-s_norm[4], -s_norm[3], -s_norm[2]]),   # -Z
        ]

        for (center, normal), traction in zip(face_defs, tractions):
            mag = np.linalg.norm(traction)
            if mag < 0.02:
                continue
            direction = traction / mag          # unit direction for orientation
            center = np.array(center, dtype=float)
            normal = np.array(normal, dtype=float)
            color = '#ff4444' if np.dot(traction, normal) > 0 else '#44aaff'
            arrow = pv.Arrow(start=center, direction=direction,
                             tip_length=0.25, tip_radius=0.08,
                             shaft_radius=0.03, scale=mag * scale)
            plotter.add_mesh(arrow, color=color, opacity=0.9)

    # ---- public view methods (same signatures as PolycrystalCanvas) ----

    def show_loading_diagram(self, stress_voigt):
        """Show 3D RVE with boundary stress arrows."""
        self._stop_animation()
        self._anim_bar.setVisible(False)
        self._anim_mesh = None
        self._anim_actor = None
        self.plotter.clear()

        self._add_rve_wireframe(self.plotter)
        self._add_stress_arrows(self.plotter, stress_voigt)

        s = stress_voigt / 1e6
        info_text = (f'Applied Stress (MPa)\n'
                     f'sig11={s[0]:.0f}  sig22={s[1]:.0f}  sig33={s[2]:.0f}\n'
                     f'sig23={s[3]:.0f}  sig13={s[4]:.0f}  sig12={s[5]:.0f}')
        self.plotter.add_text(info_text, position='upper_left',
                              color='white', font_size=10)
        self.plotter.camera_position = 'iso'
        self.plotter.reset_camera()

    def show_grain_map(self, grain_ids, slice_idx=None):
        """Show 3D grain boundary surfaces — GPU-rendered by VTK."""
        self._stop_animation()
        self._anim_bar.setVisible(False)
        self._anim_mesh = None
        self._anim_actor = None
        self.plotter.clear()

        N = grain_ids.shape[0]
        n_grains = len(np.unique(grain_ids))

        quads, ga, gb, ci, cj, ck = self._extract_boundary_quads(grain_ids)

        if len(quads) > 0:
            mesh = self._quads_to_mesh(quads, scalars=ga.astype(float),
                                       scalar_name='Grain ID')
            self.plotter.add_mesh(
                mesh, scalars='Grain ID', cmap='tab20',
                opacity=0.5, show_edges=False,
                scalar_bar_args={'title': 'Grain ID', 'color': 'white',
                                 'title_font_size': 12, 'label_font_size': 10})

        self._add_rve_wireframe(self.plotter)
        self.plotter.add_text(
            f'{n_grains} grains, {N} cubed grid',
            position='upper_left', color='white', font_size=12)
        self.plotter.camera_position = 'iso'
        self.plotter.reset_camera()

    def show_results(self, eps_field, sig_field, grain_ids,
                     stress_applied=None, slice_idx=None):
        """Show VM stress on grain boundary surfaces — single viewport."""
        self._stop_animation()
        self._anim_mesh = None
        self._anim_actor = None
        self.plotter.clear()

        quads, ga, gb, ci, cj, ck = self._extract_boundary_quads(grain_ids)
        if len(quads) == 0:
            return

        # Von Mises stress on boundary faces
        vm = fft_solver.von_mises_stress(sig_field)
        vm_vals = vm[ci, cj, ck] / 1e6
        mesh_vm = self._quads_to_mesh(quads, scalars=vm_vals,
                                      scalar_name='VM Stress (MPa)')

        self.plotter.add_mesh(
            mesh_vm, scalars='VM Stress (MPa)', cmap='hot',
            opacity=0.6, show_edges=False,
            scalar_bar_args={'title': 'VM Stress (MPa)', 'color': 'white',
                             'title_font_size': 12, 'label_font_size': 10})
        if stress_applied is not None:
            self._add_stress_arrows(self.plotter, stress_applied, scale=0.25)
        self._add_rve_wireframe(self.plotter)

        # Compute displacement stats for the title
        _, u_mag = fft_solver.compute_displacement_field(eps_field)
        u_max = np.max(u_mag) * 1e3  # mm
        vm_max = np.max(vm) / 1e6

        self.plotter.add_text(
            f'Von Mises Stress  |  VM_max={vm_max:.1f} MPa  |  u_max={u_max:.4f} mm',
            position='upper_left', color='white', font_size=11)
        self.plotter.camera_position = 'iso'
        self.plotter.reset_camera()

        # Prepare animation data (displacement at boundary vertices)
        self._prepare_animation_single(eps_field, sig_field, grain_ids,
                                       quads, ci, cj, ck,
                                       vm_vals, stress_applied)
        self._anim_bar.setVisible(True)

    def show_results_incremental(self, step_results, grain_ids,
                                 stress_applied=None, cp_state=None):
        """Show final-step VM stress and prepare multi-step animation data.

        Parameters:
            step_results: list of (eps_field, sig_field, frac) per load step
            grain_ids:    (N,N,N) grain ID array
            stress_applied: (6,) full target stress in Pa
            cp_state:     CrystalPlasticityState or None (for plastic fields)
        """
        self._stop_animation()
        self._anim_mesh = None
        self._anim_actor = None
        self.plotter.clear()

        quads, ga, gb, ci, cj, ck = self._extract_boundary_quads(grain_ids)
        if len(quads) == 0:
            return

        # Show final step
        eps_final, sig_final, _ = step_results[-1]
        vm = fft_solver.von_mises_stress(sig_final)
        vm_vals = vm[ci, cj, ck] / 1e6
        mesh_vm = self._quads_to_mesh(quads, scalars=vm_vals,
                                      scalar_name='VM Stress (MPa)')

        self.plotter.add_mesh(
            mesh_vm, scalars='VM Stress (MPa)', cmap='hot',
            opacity=0.6, show_edges=False,
            scalar_bar_args={'title': 'VM Stress (MPa)', 'color': 'white',
                             'title_font_size': 12, 'label_font_size': 10})
        if stress_applied is not None:
            self._add_stress_arrows(self.plotter, stress_applied, scale=0.25)
        self._add_rve_wireframe(self.plotter)

        _, u_mag = fft_solver.compute_displacement_field(eps_final)
        u_max = np.max(u_mag) * 1e3
        vm_max = np.max(vm) / 1e6
        n_steps = len(step_results)
        self.plotter.add_text(
            f'VM Stress ({n_steps} load steps)  |  VM_max={vm_max:.1f} MPa  '
            f'|  u_max={u_max:.4f} mm',
            position='upper_left', color='white', font_size=11)
        self.plotter.camera_position = 'iso'
        self.plotter.reset_camera()

        # Prepare per-step animation data
        self._prepare_animation_steps(step_results, grain_ids, quads, ci, cj, ck,
                                      stress_applied, cp_state=cp_state)
        self._anim_bar.setVisible(True)

    # ---- deformation animation ----

    def _prepare_animation_single(self, eps_field, sig_field, grain_ids,
                                  quads, ci, cj, ck,
                                  vm_vals, stress_applied):
        """Prepare animation from a single solve (linear interpolation 0→1)."""
        u_field, u_mag_field = fft_solver.compute_displacement_field(eps_field)
        N = grain_ids.shape[0]
        h = 1.0 / N
        n_quads = len(vm_vals)
        verts = quads.reshape(-1, 3)
        vi = np.clip((verts[:, 0] / h - 0.5).astype(int), 0, N - 1)
        vj = np.clip((verts[:, 1] / h - 0.5).astype(int), 0, N - 1)
        vk = np.clip((verts[:, 2] / h - 0.5).astype(int), 0, N - 1)
        u_verts_phys = u_field[vi, vj, vk]

        # Full-field values at face voxels
        sig_faces = sig_field[ci, cj, ck] / 1e6  # (n_quads, 6) in MPa
        eps_faces = eps_field[ci, cj, ck]          # (n_quads, 6)
        u_mag_faces = u_mag_field[ci, cj, ck] * 1e3  # mm

        # Normalise so mag=100 → max shift = 30% of RVE side
        u_max_abs = np.max(np.abs(u_verts_phys)) if n_quads > 0 else 1.0
        target_max_shift = 0.3
        norm_factor = target_max_shift / max(u_max_abs, 1e-30)

        # Build frame list: 61 frames from 0 to 1
        n_frames = 60
        step_disps = []
        step_vm = []
        step_sig = []
        step_eps = []
        step_u_mag = []
        for i in range(n_frames + 1):
            t = i / n_frames
            step_disps.append(u_verts_phys * norm_factor * t)
            step_vm.append(vm_vals * t)  # scale VM proportionally
            step_sig.append(sig_faces * t)
            step_eps.append(eps_faces * t)
            step_u_mag.append(u_mag_faces * t)

        self._anim_data = {
            'base_points': verts.copy(),
            'step_disps': step_disps,     # list of (n_verts, 3)
            'step_vm': step_vm,           # list of (n_quads,)
            'step_sig': step_sig,         # list of (n_quads, 6) MPa
            'step_eps': step_eps,         # list of (n_quads, 6)
            'step_u_mag': step_u_mag,     # list of (n_quads,) mm
            'n_steps': n_frames,
            'n_quads': n_quads,
            'stress_applied': stress_applied,
        }
        self._anim_n_frames = n_frames
        self._frame_slider.setRange(0, n_frames)
        self._frame_slider.setValue(0)
        self._frame_num_label.setText(f"0 / {n_frames}")

    def _prepare_animation_steps(self, step_results, grain_ids, quads,
                                 ci, cj, ck, stress_applied, cp_state=None):
        """Prepare animation from multiple load steps — each step is a keyframe."""
        N = grain_ids.shape[0]
        h = 1.0 / N
        n_quads = quads.shape[0]
        verts = quads.reshape(-1, 3)
        vi = np.clip((verts[:, 0] / h - 0.5).astype(int), 0, N - 1)
        vj = np.clip((verts[:, 1] / h - 0.5).astype(int), 0, N - 1)
        vk = np.clip((verts[:, 2] / h - 0.5).astype(int), 0, N - 1)

        # Compute displacement + VM + full tensor at each load step
        raw_disps = []  # physical displacement per step
        raw_vm = []     # VM stress per step
        raw_sig = []    # stress components (n_quads, 6) in MPa
        raw_eps = []    # strain components (n_quads, 6)
        raw_u_mag = []  # displacement magnitude (n_quads,) in mm
        for eps_field, sig_field, frac in step_results:
            u_field, u_mag_field = fft_solver.compute_displacement_field(eps_field)
            u_v = u_field[vi, vj, vk]
            raw_disps.append(u_v)
            vm = fft_solver.von_mises_stress(sig_field)
            raw_vm.append(vm[ci, cj, ck] / 1e6)
            raw_sig.append(sig_field[ci, cj, ck] / 1e6)
            raw_eps.append(eps_field[ci, cj, ck])
            raw_u_mag.append(u_mag_field[ci, cj, ck] * 1e3)

        # Normalise displacements so mag=100 → max shift = 30% of RVE side
        all_u = np.array(raw_disps)  # (n_steps, n_verts, 3)
        u_max_abs = np.max(np.abs(all_u)) if n_quads > 0 else 1.0
        target_max_shift = 0.3
        norm_factor = target_max_shift / max(u_max_abs, 1e-30)

        # Build frames: step 0 = undeformed, then each load step
        step_disps = [np.zeros_like(verts)]  # frame 0 = no displacement
        step_vm = [np.zeros(n_quads)]        # frame 0 = zero stress
        step_sig = [np.zeros((n_quads, 6))]  # frame 0 = zero stress tensor
        step_eps = [np.zeros((n_quads, 6))]  # frame 0 = zero strain tensor
        step_u_mag = [np.zeros(n_quads)]     # frame 0 = zero displacement
        for i in range(len(step_results)):
            step_disps.append(raw_disps[i] * norm_factor)
            step_vm.append(raw_vm[i])
            step_sig.append(raw_sig[i])
            step_eps.append(raw_eps[i])
            step_u_mag.append(raw_u_mag[i])

        n_frames = len(step_results)  # not counting the zero frame separately
        self._anim_data = {
            'base_points': verts.copy(),
            'step_disps': step_disps,          # n_steps+1 entries (incl. zero)
            'step_vm': step_vm,                # n_steps+1 entries (incl. zero)
            'step_sig': step_sig,              # n_steps+1 (n_quads, 6) MPa
            'step_eps': step_eps,              # n_steps+1 (n_quads, 6)
            'step_u_mag': step_u_mag,          # n_steps+1 (n_quads,) mm
            'n_steps': n_frames,
            'n_quads': n_quads,
            'stress_applied': stress_applied,
        }

        # --- Crystal-plasticity fields (if available) ---
        if cp_state is not None and hasattr(cp_state, '_history_snapshots'):
            snaps = cp_state._history_snapshots
            step_ep_eq = [np.zeros(n_quads)]
            step_acc_slip = [np.zeros(n_quads)]
            step_slip_res = [np.zeros(n_quads)]
            for snap in snaps:
                step_ep_eq.append(snap['ep_eq'][ci, cj, ck])
                step_acc_slip.append(snap['acc_slip'][ci, cj, ck])
                step_slip_res.append(snap['slip_res'][ci, cj, ck])
            self._anim_data['step_ep_eq'] = step_ep_eq
            self._anim_data['step_acc_slip'] = step_acc_slip
            self._anim_data['step_slip_res'] = step_slip_res

            # Finite-strain extra fields (if available)
            if snaps and 'ssd_density' in snaps[0]:
                step_ssd = [np.zeros(n_quads)]
                step_gnd = [np.zeros(n_quads)]
                step_bs = [np.zeros(n_quads)]
                step_misor = [np.zeros(n_quads)]
                for snap in snaps:
                    step_ssd.append(snap['ssd_density'][ci, cj, ck])
                    step_gnd.append(snap['gnd_density'][ci, cj, ck])
                    step_bs.append(snap['backstress'][ci, cj, ck])
                    step_misor.append(snap['misorientation'][ci, cj, ck])
                self._anim_data['step_ssd_density'] = step_ssd
                self._anim_data['step_gnd_density'] = step_gnd
                self._anim_data['step_backstress'] = step_bs
                self._anim_data['step_misorientation'] = step_misor

        self._anim_n_frames = n_frames
        self._frame_slider.setRange(0, n_frames)
        self._frame_slider.setValue(0)
        self._frame_num_label.setText(f"0 / {n_frames}")

    def _toggle_animation(self):
        """Start / stop the deformation animation."""
        if self._anim_running:
            self._stop_animation()
        else:
            self._start_animation()

    def _start_animation(self):
        if self._anim_data is None:
            return
        self._anim_running = True
        self._anim_frame = 0
        self._anim_forward = True
        self._play_btn.setText("⏹  Stop")

        # Build the mesh ONCE; we'll update .points in-place each frame
        self._setup_anim_scene()
        # Slower interval for fewer load steps, faster for many frames
        interval = max(33, 500 // max(self._anim_n_frames, 1))
        self._anim_timer.start(interval)

    def _stop_animation(self):
        self._anim_timer.stop()
        self._anim_running = False
        self._play_btn.setText("▶  Play")

    def _on_scale_changed(self, val):
        self._scale_val.setText(f"{val}×")
        # If paused, update the current frame with the new magnification
        if not self._anim_running and self._anim_mesh is not None:
            self._apply_warp(self._anim_frame)

    def _on_frame_preview(self, val):
        """Lightweight preview while dragging — update label only."""
        self._frame_num_label.setText(f"{val} / {self._anim_n_frames}")

    def _on_frame_scrub(self, val):
        """Full frame update when slider is released."""
        if self._scrubbing:
            return
        self._anim_frame = val
        self._frame_num_label.setText(f"{val} / {self._anim_n_frames}")
        if self._anim_mesh is not None:
            self._apply_warp(val)

    def _on_field_changed(self, idx):
        """User changed the display field — rebuild scene with new colormap."""
        if self._anim_data is not None and self._anim_mesh is not None:
            was_running = self._anim_running
            if was_running:
                self._anim_timer.stop()
            self._setup_anim_scene()
            self._apply_warp(self._anim_frame)
            if was_running:
                interval = max(33, 500 // max(self._anim_n_frames, 1))
                self._anim_timer.start(interval)

    def _get_field_scalars(self, frame_idx):
        """Return (values, title, cmap) for the currently selected field."""
        d = self._anim_data
        if d is None:
            return np.zeros(1), 'Field', 'hot'
        idx = self._field_combo.currentIndex()
        if idx == 0:  # VM Stress
            return d['step_vm'][frame_idx], 'VM Stress (MPa)', 'hot'
        elif 1 <= idx <= 6:  # σ components
            comp = idx - 1
            labels = ['\u03c3\u2081\u2081 (MPa)', '\u03c3\u2082\u2082 (MPa)',
                      '\u03c3\u2083\u2083 (MPa)', '\u03c3\u2082\u2083 (MPa)',
                      '\u03c3\u2081\u2083 (MPa)', '\u03c3\u2081\u2082 (MPa)']
            return d['step_sig'][frame_idx][:, comp], labels[comp], 'coolwarm'
        elif 7 <= idx <= 12:  # ε components
            comp = idx - 7
            labels = ['\u03b5\u2081\u2081', '\u03b5\u2082\u2082', '\u03b5\u2083\u2083',
                      '\u03b5\u2082\u2083', '\u03b5\u2081\u2083', '\u03b5\u2081\u2082']
            return d['step_eps'][frame_idx][:, comp], labels[comp], 'coolwarm'
        elif idx == 13:  # |u|
            return d['step_u_mag'][frame_idx], '|u| (mm)', 'viridis'
        elif idx == 14:  # Equivalent plastic strain
            key = 'step_ep_eq'
            if key in d:
                return d[key][frame_idx], '\u03b5\u1d56 eq', 'magma'
            return d['step_vm'][frame_idx] * 0, '\u03b5\u1d56 eq (N/A)', 'magma'
        elif idx == 15:  # Accumulated slip
            key = 'step_acc_slip'
            if key in d:
                return d[key][frame_idx], 'Accum. Slip', 'inferno'
            return d['step_vm'][frame_idx] * 0, 'Accum. Slip (N/A)', 'inferno'
        elif idx == 16:  # Slip resistance
            key = 'step_slip_res'
            if key in d:
                return d[key][frame_idx], 'Slip Resist. (MPa)', 'plasma'
            return d['step_vm'][frame_idx] * 0, 'Slip Resist. (N/A)', 'plasma'
        elif idx == 17:  # SSD Density
            key = 'step_ssd_density'
            if key in d:
                return d[key][frame_idx], 'SSD Density (m⁻²)', 'viridis'
            return d['step_vm'][frame_idx] * 0, 'SSD (N/A)', 'viridis'
        elif idx == 18:  # GND Density
            key = 'step_gnd_density'
            if key in d:
                return d[key][frame_idx], 'GND Density (m⁻²)', 'viridis'
            return d['step_vm'][frame_idx] * 0, 'GND (N/A)', 'viridis'
        elif idx == 19:  # Back-stress
            key = 'step_backstress'
            if key in d:
                return d[key][frame_idx], 'Back-stress (MPa)', 'RdBu_r'
            return d['step_vm'][frame_idx] * 0, 'χ (N/A)', 'RdBu_r'
        elif idx == 20:  # Misorientation
            key = 'step_misorientation'
            if key in d:
                return d[key][frame_idx], 'Misorientation (°)', 'jet'
            return d['step_vm'][frame_idx] * 0, 'Misor. (N/A)', 'jet'
        else:
            return d['step_vm'][frame_idx], 'VM Stress (MPa)', 'hot'

    def _setup_anim_scene(self):
        """Clear plotter once and add mesh + wireframe + text for animation."""
        d = self._anim_data
        if d is None:
            return

        self.plotter.clear()

        # Build initial (undeformed) mesh — use frame 0 data
        n_quads = d['n_quads']
        base = d['base_points']  # (n_verts, 3)
        scalars_0, title_0, cmap_0 = self._get_field_scalars(0)

        # Create VTK face array
        faces = np.empty(n_quads * 5, dtype=np.int64)
        faces[0::5] = 4
        for k in range(4):
            faces[(k + 1)::5] = np.arange(n_quads) * 4 + k
        mesh = pv.PolyData(base.astype(np.float64), faces)
        mesh.cell_data['Field'] = scalars_0

        self._anim_mesh = mesh

        # Use last step's field range for initial colorbar
        scalars_last, _, _ = self._get_field_scalars(d['n_steps'])
        vmin = float(np.min(scalars_last)) if len(scalars_last) > 0 else 0.0
        vmax = float(np.max(scalars_last)) if len(scalars_last) > 0 else 1.0
        if abs(vmax - vmin) < 1e-10:
            vmin, vmax = vmin - 0.5, vmax + 0.5

        self._anim_actor = self.plotter.add_mesh(
            mesh, scalars='Field', cmap=cmap_0,
            clim=[vmin, vmax],
            opacity=0.6, show_edges=False,
            scalar_bar_args={'title': title_0, 'color': 'white',
                             'title_font_size': 12, 'label_font_size': 10})

        # Reference wireframe
        cube = pv.Box(bounds=[0, 1, 0, 1, 0, 1])
        self.plotter.add_mesh(cube, style='wireframe', color='#334466',
                              line_width=1, opacity=0.4)

        self._anim_text_actor = self.plotter.add_text(
            'Deformation  t=0%  ×1',
            position='upper_left', color='white', font_size=11,
            name='anim_title')

        self.plotter.camera_position = 'iso'
        self.plotter.reset_camera()

    def _apply_warp(self, frame_idx):
        """Update mesh points + VM scalars for a given step frame."""
        d = self._anim_data
        if d is None or self._anim_mesh is None:
            return

        mag = self._scale_slider.value() / 100.0  # slider 1-500 → 0.01-5.0
        n_steps = d['n_steps']

        # Clamp frame index
        frame_idx = max(0, min(frame_idx, n_steps))

        base = d['base_points']
        disp = d['step_disps'][frame_idx]

        warped = base + mag * disp
        self._anim_mesh.points = warped.astype(np.float64)

        # Get selected field scalars and update colorbar
        scalars, title, _ = self._get_field_scalars(frame_idx)
        self._anim_mesh.cell_data['Field'] = scalars
        vmin_f = float(np.min(scalars)) if len(scalars) > 0 else 0.0
        vmax_f = float(np.max(scalars)) if len(scalars) > 0 else 1e-6
        if abs(vmax_f - vmin_f) < 1e-10:
            vmin_f, vmax_f = vmin_f - 0.5, vmax_f + 0.5

        # Update both the VTK mapper range and the scalar bar
        mapper = self._anim_actor.GetMapper()
        mapper.SetScalarRange(vmin_f, vmax_f)
        self.plotter.update_scalar_bar_range([vmin_f, vmax_f])

        # Update title text
        mag_display = self._scale_slider.value()
        frac = frame_idx / n_steps if n_steps > 0 else 0
        self.plotter.add_text(
            f'Load step {frame_idx}/{n_steps}  ({frac:.0%})   \u00d7{mag_display}',
            position='upper_left', color='white', font_size=11,
            name='anim_title')

        self.plotter.render()

    def _anim_step(self):
        """Advance one animation frame (ping-pong through load steps)."""
        if self._anim_data is None:
            self._stop_animation()
            return

        n = self._anim_n_frames
        if self._anim_forward:
            self._anim_frame += 1
            if self._anim_frame >= n:
                self._anim_forward = False
        else:
            self._anim_frame -= 1
            if self._anim_frame <= 0:
                self._anim_forward = True

        self._apply_warp(self._anim_frame)

        # Sync the frame slider without triggering its callback
        self._scrubbing = True
        self._frame_slider.setValue(self._anim_frame)
        self._frame_num_label.setText(f"{self._anim_frame} / {n}")
        self._scrubbing = False

    def show_convergence(self, info):
        """Show convergence summary text."""
        self._stop_animation()
        self._anim_bar.setVisible(False)
        self._anim_mesh = None
        self._anim_actor = None
        self.plotter.clear()
        errors = info.get('errors', [])
        final = info.get('final_stress_error', info.get('final_error', 0))
        iters = info.get('iterations', 0)
        text = (f'Solver Convergence\n\n'
                f'Iterations: {iters}\n'
                f'Final error: {final:.2e}\n'
                f'Converged: {info.get("converged", False)}')
        self.plotter.add_text(text, position='upper_left',
                              color='white', font_size=14)


# ============================================================================
# Abaqus-Style Model Tree Panel
# ============================================================================

class ModelTreePanel(QtWidgets.QWidget):
    """Collapsible model tree inspired by Abaqus CAE.

    Top-level items are category headers (Microstructure, Material, etc.).
    Expanding a header reveals its editable properties as child rows with
    embedded widgets (spinboxes, combos, buttons).
    """
    run_requested = QtCore.pyqtSignal()
    generate_requested = QtCore.pyqtSignal()
    preview_loading = QtCore.pyqtSignal()

    TREE_STYLE = """
    QTreeWidget {
        background: #0d0d20;
        border: none;
        font-size: 13px;
        outline: none;
    }
    QTreeWidget::item {
        color: #ccccdd;
        padding: 6px 2px;
        min-height: 26px;
        border: none;
    }
    QTreeWidget::item:selected {
        background: #1a2a55;
    }
    QTreeWidget::branch:has-children:!has-siblings:closed,
    QTreeWidget::branch:closed:has-children:has-siblings {
        image: none;
        border-image: none;
    }
    QTreeWidget::branch:open:has-children:!has-siblings,
    QTreeWidget::branch:open:has-children:has-siblings {
        image: none;
        border-image: none;
    }
    QHeaderView::section {
        background: #0d0d20;
        color: #88aadd;
        border: none;
        border-bottom: 1px solid #334466;
        padding: 6px;
        font-weight: bold;
        font-size: 13px;
    }
    QDoubleSpinBox, QSpinBox {
        background: #1a1a3e; border: 1px solid #334466;
        border-radius: 3px; padding: 3px 4px; color: white; font-size: 12px;
    }
    QComboBox {
        background: #1a1a3e; border: 1px solid #334466;
        border-radius: 3px; padding: 3px 4px; color: white; font-size: 12px;
    }
    QComboBox QAbstractItemView {
        background: #1a1a3e; color: white; selection-background-color: #2255aa;
    }
    QPushButton {
        background: #2255aa; color: white; border: none; border-radius: 4px;
        padding: 6px 10px; font-size: 12px; font-weight: bold;
    }
    QPushButton:hover { background: #3366cc; }
    QPushButton:pressed { background: #1144aa; }
    QProgressBar {
        background: #1a1a3e; border: 1px solid #334466; border-radius: 3px;
        text-align: center; color: white; font-size: 11px;
    }
    QProgressBar::chunk { background: #2255aa; border-radius: 2px; }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(370)
        self.setStyleSheet("background-color: #0d0d20;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Title banner ---
        banner = QtWidgets.QLabel("  Crystal Plasticity FFT")
        banner.setStyleSheet(
            "background: #0a0a18; color: #5599ff; font-size: 16px;"
            " font-weight: bold; padding: 10px 8px;"
            " border-bottom: 1px solid #334466;")
        banner.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        layout.addWidget(banner)

        # --- Tree ---
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Property", "Value"])
        self.tree.header().setStretchLastSection(True)
        self.tree.header().resizeSection(0, 140)
        self.tree.setRootIsDecorated(True)
        self.tree.setAnimated(True)
        self.tree.setIndentation(18)
        self.tree.setStyleSheet(self.TREE_STYLE)
        self.tree.setFocusPolicy(QtCore.Qt.NoFocus)
        layout.addWidget(self.tree, stretch=1)

        # Build tree sections
        self._build_microstructure()
        self._build_material()
        self._build_plasticity()
        self._build_boundary_conditions()
        self._build_solver()
        self._build_job()

        # Expand all by default
        self.tree.expandAll()

        # Apply defaults
        self._apply_material(0)
        self._apply_load_preset(0)

    # ---- helpers ----

    def _make_header(self, icon, text):
        """Create a bold top-level tree item."""
        item = QtWidgets.QTreeWidgetItem(self.tree, [f" {icon}  {text}"])
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsSelectable)
        font = item.font(0)
        font.setBold(True)
        font.setPointSize(10)
        item.setFont(0, font)
        item.setForeground(0, QtGui.QBrush(QtGui.QColor('#88aadd')))
        return item

    def _add_widget_row(self, parent, label, widget):
        """Add a child row with a label in col 0 and a widget in col 1."""
        child = QtWidgets.QTreeWidgetItem(parent, [label])
        child.setFlags(child.flags() & ~QtCore.Qt.ItemIsSelectable)
        child.setForeground(0, QtGui.QBrush(QtGui.QColor('#aabbcc')))
        self.tree.setItemWidget(child, 1, widget)
        return child

    def _add_button_row(self, parent, button):
        """Add a full-width button spanning both columns."""
        child = QtWidgets.QTreeWidgetItem(parent, [""])
        child.setFlags(child.flags() & ~QtCore.Qt.ItemIsSelectable)
        self.tree.setItemWidget(child, 0, button)
        return child

    # ---- Microstructure section ----

    def _build_microstructure(self):
        hdr = self._make_header("\u2B22", "Microstructure")

        self.grid_spin = QtWidgets.QSpinBox()
        self.grid_spin.setRange(8, 128)
        self.grid_spin.setValue(32)
        self.grid_spin.setSingleStep(8)
        self._add_widget_row(hdr, "Grid N", self.grid_spin)

        self.grains_spin = QtWidgets.QSpinBox()
        self.grains_spin.setRange(2, 500)
        self.grains_spin.setValue(20)
        self._add_widget_row(hdr, "Grains", self.grains_spin)

        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        self._add_widget_row(hdr, "Seed", self.seed_spin)

        gen_btn = QtWidgets.QPushButton("Generate")
        gen_btn.setStyleSheet(
            "background: #1a6b3f; padding: 5px 8px; font-size: 11px;")
        gen_btn.setCursor(QtCore.Qt.PointingHandCursor)
        gen_btn.clicked.connect(self.generate_requested.emit)
        self._add_button_row(hdr, gen_btn)

    # ---- Material section ----

    def _build_material(self):
        hdr = self._make_header("\u2B23", "Material")

        self.mat_combo = QtWidgets.QComboBox()
        self.mat_combo.addItems([
            'Copper', 'Aluminium', 'Nickel', 'Iron (BCC approx)', 'Custom'])
        self.mat_combo.currentIndexChanged.connect(self._apply_material)
        self._add_widget_row(hdr, "Preset", self.mat_combo)

        self.c11_spin = QtWidgets.QDoubleSpinBox()
        self.c11_spin.setRange(0, 2000); self.c11_spin.setDecimals(1)
        self.c11_spin.setSuffix(' GPa'); self.c11_spin.setValue(168.4)
        self._add_widget_row(hdr, "C\u2081\u2081", self.c11_spin)

        self.c12_spin = QtWidgets.QDoubleSpinBox()
        self.c12_spin.setRange(0, 2000); self.c12_spin.setDecimals(1)
        self.c12_spin.setSuffix(' GPa'); self.c12_spin.setValue(121.4)
        self._add_widget_row(hdr, "C\u2081\u2082", self.c12_spin)

        self.c44_spin = QtWidgets.QDoubleSpinBox()
        self.c44_spin.setRange(0, 2000); self.c44_spin.setDecimals(1)
        self.c44_spin.setSuffix(' GPa'); self.c44_spin.setValue(75.4)
        self._add_widget_row(hdr, "C\u2084\u2084", self.c44_spin)

    # ---- Plasticity section ----

    def _build_plasticity(self):
        hdr = self._make_header("\u2694", "Plasticity")

        self.plasticity_toggle = QtWidgets.QComboBox()
        self.plasticity_toggle.addItems(['Elastic Only', 'Crystal Plasticity (EVPFFT)'])
        self.plasticity_toggle.setCurrentIndex(1)  # default to EVPFFT
        self.plasticity_toggle.currentIndexChanged.connect(self._on_plasticity_toggle)
        self._add_widget_row(hdr, "Model", self.plasticity_toggle)

        # Formulation: Small Strain / Finite Strain
        self.formulation_combo = QtWidgets.QComboBox()
        self.formulation_combo.addItems(['Small Strain', 'Finite Strain (F=FeFp)'])
        self.formulation_combo.setCurrentIndex(0)
        self.formulation_combo.setToolTip(
            "Small strain: ε, σ (Cauchy)\n"
            "Finite strain: F=Fe·Fp, P (1st Piola-Kirchhoff)")
        self._formulation_row = self._add_widget_row(hdr, "Formulation",
                                                      self.formulation_combo)

        # Hardening model
        self.hardening_combo = QtWidgets.QComboBox()
        self.hardening_combo.addItems(['Voce (phenomenological)',
                                        'Kocks-Mecking (dislocation density)'])
        self.hardening_combo.setCurrentIndex(0)
        self.hardening_combo.currentIndexChanged.connect(self._on_hardening_toggle)
        self.hardening_combo.setToolTip(
            "Voce: τ₀, τ_s, h₀ (phenomenological)\n"
            "Kocks-Mecking: k₁, k₂, α_T (dislocation-density based)")
        self._hardening_row = self._add_widget_row(hdr, "Hardening",
                                                    self.hardening_combo)

        # Flow rule parameters (Voce — existing)
        self.tau0_spin = QtWidgets.QDoubleSpinBox()
        self.tau0_spin.setRange(1, 1000); self.tau0_spin.setDecimals(1)
        self.tau0_spin.setSuffix(' MPa'); self.tau0_spin.setValue(50.0)
        self.tau0_spin.setToolTip("Initial slip resistance \u03c4\u2080")
        self._tau0_row = self._add_widget_row(hdr, "\u03c4\u2080 (CRSS)", self.tau0_spin)

        self.tau_s_spin = QtWidgets.QDoubleSpinBox()
        self.tau_s_spin.setRange(10, 2000); self.tau_s_spin.setDecimals(1)
        self.tau_s_spin.setSuffix(' MPa'); self.tau_s_spin.setValue(200.0)
        self.tau_s_spin.setToolTip("Saturation slip resistance \u03c4_s")
        self._taus_row = self._add_widget_row(hdr, "\u03c4_s (sat.)", self.tau_s_spin)

        self.h0_spin = QtWidgets.QDoubleSpinBox()
        self.h0_spin.setRange(10, 5000); self.h0_spin.setDecimals(0)
        self.h0_spin.setSuffix(' MPa'); self.h0_spin.setValue(500.0)
        self.h0_spin.setToolTip("Initial hardening modulus h\u2080")
        self._h0_row = self._add_widget_row(hdr, "h\u2080", self.h0_spin)

        self.q_lat_spin = QtWidgets.QDoubleSpinBox()
        self.q_lat_spin.setRange(0.5, 3.0); self.q_lat_spin.setDecimals(2)
        self.q_lat_spin.setValue(1.40)
        self.q_lat_spin.setToolTip("Latent hardening ratio q")
        self._qlat_row = self._add_widget_row(hdr, "q (latent)", self.q_lat_spin)

        # --- Kocks-Mecking parameters ---
        self.km_k1_spin = QtWidgets.QDoubleSpinBox()
        self.km_k1_spin.setRange(1e6, 1e11); self.km_k1_spin.setDecimals(0)
        self.km_k1_spin.setValue(7e8); self.km_k1_spin.setSuffix(' m⁻¹')
        self.km_k1_spin.setToolTip("Storage coefficient k₁ (m⁻¹)")
        self._km_k1_row = self._add_widget_row(hdr, "k₁ (storage)", self.km_k1_spin)

        self.km_k2_spin = QtWidgets.QDoubleSpinBox()
        self.km_k2_spin.setRange(0.1, 100); self.km_k2_spin.setDecimals(1)
        self.km_k2_spin.setValue(10.0)
        self.km_k2_spin.setToolTip("Dynamic recovery coefficient k₂")
        self._km_k2_row = self._add_widget_row(hdr, "k₂ (recovery)", self.km_k2_spin)

        self.km_alpha_spin = QtWidgets.QDoubleSpinBox()
        self.km_alpha_spin.setRange(0.05, 1.0); self.km_alpha_spin.setDecimals(2)
        self.km_alpha_spin.setValue(0.30)
        self.km_alpha_spin.setToolTip("Taylor coefficient α_T")
        self._km_alpha_row = self._add_widget_row(hdr, "α_T (Taylor)",
                                                    self.km_alpha_spin)

        self.km_rho0_spin = QtWidgets.QDoubleSpinBox()
        self.km_rho0_spin.setRange(1e8, 1e16); self.km_rho0_spin.setDecimals(0)
        self.km_rho0_spin.setValue(1e12); self.km_rho0_spin.setSuffix(' m⁻²')
        self.km_rho0_spin.setToolTip("Initial SSD density ρ₀ (m⁻²)")
        self._km_rho0_row = self._add_widget_row(hdr, "ρ₀ (initial)", self.km_rho0_spin)

        # --- Back-stress (Armstrong-Frederick) ---
        self.backstress_check = QtWidgets.QCheckBox("Armstrong-Frederick back-stress")
        self.backstress_check.setChecked(False)
        self.backstress_check.setToolTip("Enable kinematic hardening (Bauschinger effect)")
        child_bs = QtWidgets.QTreeWidgetItem(hdr, [""])
        child_bs.setFlags(child_bs.flags() & ~QtCore.Qt.ItemIsSelectable)
        self.tree.setItemWidget(child_bs, 0, self.backstress_check)
        self._bs_check_row = child_bs

        self.af_c1_spin = QtWidgets.QDoubleSpinBox()
        self.af_c1_spin.setRange(1e6, 1e12); self.af_c1_spin.setDecimals(0)
        self.af_c1_spin.setValue(1e9); self.af_c1_spin.setSuffix(' Pa')
        self.af_c1_spin.setToolTip("Direct hardening modulus c₁ (Pa)")
        self._af_c1_row = self._add_widget_row(hdr, "c₁ (direct)", self.af_c1_spin)

        self.af_c2_spin = QtWidgets.QDoubleSpinBox()
        self.af_c2_spin.setRange(0.1, 200); self.af_c2_spin.setDecimals(1)
        self.af_c2_spin.setValue(10.0)
        self.af_c2_spin.setToolTip("Dynamic recovery c₂  (χ_sat=c₁/c₂)")
        self._af_c2_row = self._add_widget_row(hdr, "c₂ (recovery)", self.af_c2_spin)

        # --- GND hardening ---
        self.gnd_check = QtWidgets.QCheckBox("GND hardening (non-local)")
        self.gnd_check.setChecked(False)
        self.gnd_check.setToolTip("Compute Nye tensor from curl(Fp) and add GND resistance")
        child_gnd = QtWidgets.QTreeWidgetItem(hdr, [""])
        child_gnd.setFlags(child_gnd.flags() & ~QtCore.Qt.ItemIsSelectable)
        self.tree.setItemWidget(child_gnd, 0, self.gnd_check)
        self._gnd_check_row = child_gnd

        self.temp_spin = QtWidgets.QDoubleSpinBox()
        self.temp_spin.setRange(77, 2500); self.temp_spin.setDecimals(0)
        self.temp_spin.setSuffix(' K'); self.temp_spin.setValue(1123)
        self.temp_spin.setToolTip("Temperature for OTIS thermal activation")
        self._temp_row = self._add_widget_row(hdr, "T (temp)", self.temp_spin)

        self.rate_ref_combo = QtWidgets.QComboBox()
        self.rate_ref_combo.addItems(['1e5', '1e6', '1e7', '1e8', '1e9'])
        self.rate_ref_combo.setCurrentIndex(2)  # default 1e7
        self.rate_ref_combo.setToolTip("Reference slip rate \u03b3\u0307\u2080 (s\u207b\u00b9)")
        self._rate_ref_row = self._add_widget_row(hdr, "\u03b3\u0307\u2080 ref", self.rate_ref_combo)

        # Preset
        self.cp_preset = QtWidgets.QComboBox()
        self.cp_preset.addItems([
            'Copper (FCC)', 'Aluminium (FCC)', 'Nickel (FCC)', 'Custom'])
        self.cp_preset.currentIndexChanged.connect(self._apply_cp_preset)
        self._add_widget_row(hdr, "CP Preset", self.cp_preset)

        # Show plasticity params (EVPFFT is default)
        self._set_plasticity_visible(True)
        self._on_hardening_toggle(0)  # initialize visibility

    def _on_plasticity_toggle(self, idx):
        self._set_plasticity_visible(idx == 1)

    def _set_plasticity_visible(self, visible):
        for row in [self._tau0_row, self._taus_row, self._h0_row,
                     self._qlat_row, self._temp_row, self._rate_ref_row,
                     self._formulation_row, self._hardening_row,
                     self._bs_check_row, self._gnd_check_row,
                     self._af_c1_row, self._af_c2_row]:
            row.setHidden(not visible)
        # KM rows depend on hardening combo
        self._on_hardening_toggle(self.hardening_combo.currentIndex()
                                  if visible else -1)

    def _on_hardening_toggle(self, idx):
        """Show Voce or Kocks-Mecking parameters based on selection."""
        is_cp = not self._tau0_row.isHidden() or idx >= 0
        voce = is_cp and idx == 0
        km = is_cp and idx == 1
        for row in [self._tau0_row, self._taus_row, self._h0_row]:
            row.setHidden(not voce)
        for row in [self._km_k1_row, self._km_k2_row,
                     self._km_alpha_row, self._km_rho0_row]:
            row.setHidden(not km)

    def _apply_cp_preset(self, idx):
        # (τ₀ MPa, τ_s MPa, h₀ MPa, q, T K, rate_ref_idx)
        presets = [
            # Copper: Kalidindi (1992)
            (16.0, 148.0, 180.0, 1.40, 1123, 2),
            # Aluminium: Bassani & Wu (1991) approx
            (8.0,  55.0,  200.0, 1.40, 1123, 2),
            # Nickel: OTIS driver defaults
            (26.0, 260.0, 400.0, 1.40, 1123, 2),
            None,  # Custom
        ]
        if idx < len(presets) and presets[idx] is not None:
            tau0, tau_s, h0, qlat, temp, rr_idx = presets[idx]
            self.tau0_spin.setValue(tau0)
            self.tau_s_spin.setValue(tau_s)
            self.h0_spin.setValue(h0)
            self.q_lat_spin.setValue(qlat)
            self.temp_spin.setValue(temp)
            self.rate_ref_combo.setCurrentIndex(rr_idx)

    # ---- Boundary Conditions section ----

    def _build_boundary_conditions(self):
        hdr = self._make_header("\u2192", "Boundary Conditions")

        self.load_preset = QtWidgets.QComboBox()
        self.load_preset.addItems([
            'Uniaxial X  (5 GPa)',
            'Uniaxial Y  (5 GPa)',
            'Uniaxial Z  (5 GPa)',
            'Biaxial XY  (5 GPa)',
            'Hydrostatic  (5 GPa)',
            'Shear XY  (2.5 GPa)',
            'Shear XZ  (2.5 GPa)',
            'Custom',
        ])
        self.load_preset.currentIndexChanged.connect(self._apply_load_preset)
        self._add_widget_row(hdr, "Preset", self.load_preset)

        # Stress tensor components
        voigt_labels = [
            ('\u03c3\u2081\u2081', 'sig11'), ('\u03c3\u2082\u2082', 'sig22'),
            ('\u03c3\u2083\u2083', 'sig33'), ('\u03c3\u2082\u2083', 'sig23'),
            ('\u03c3\u2081\u2083', 'sig13'), ('\u03c3\u2081\u2082', 'sig12'),
        ]
        self.stress_spins = {}
        for display, key in voigt_labels:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-10000, 10000)
            spin.setDecimals(0)
            spin.setSingleStep(100)
            spin.setSuffix(' MPa')
            self.stress_spins[display] = spin
            self._add_widget_row(hdr, display, spin)

        # Description label
        self.load_desc = QtWidgets.QLabel("")
        self.load_desc.setStyleSheet(
            "color: #88aadd; font-size: 10px; padding: 2px 4px;")
        self.load_desc.setWordWrap(True)
        self._add_widget_row(hdr, "", self.load_desc)

        preview_btn = QtWidgets.QPushButton("Preview Loading")
        preview_btn.setStyleSheet(
            "background: #664400; padding: 5px 8px; font-size: 11px;")
        preview_btn.setCursor(QtCore.Qt.PointingHandCursor)
        preview_btn.clicked.connect(self.preview_loading.emit)
        self._add_button_row(hdr, preview_btn)

    # ---- Solver section ----

    def _build_solver(self):
        hdr = self._make_header("\u2699", "Solver")

        self.solver_combo = QtWidgets.QComboBox()
        self.solver_combo.addItems([
            'CG (Zeman 2010)', 'Basic (Moulinec-Suquet)'])
        self._add_widget_row(hdr, "Method", self.solver_combo)

        self.tol_combo = QtWidgets.QComboBox()
        self.tol_combo.addItems(['1e-3', '1e-4', '1e-5', '1e-6'])
        self.tol_combo.setCurrentIndex(0)
        self._add_widget_row(hdr, "FFT Tol", self.tol_combo)

        self.stress_tol_combo = QtWidgets.QComboBox()
        self.stress_tol_combo.addItems(['1e-2', '1e-3', '1e-4'])
        self.stress_tol_combo.setCurrentIndex(0)
        self._add_widget_row(hdr, "Stress Tol", self.stress_tol_combo)

        self.maxiter_spin = QtWidgets.QSpinBox()
        self.maxiter_spin.setRange(10, 5000)
        self.maxiter_spin.setValue(200)
        self._add_widget_row(hdr, "Max Iter", self.maxiter_spin)

        self.steps_spin = QtWidgets.QSpinBox()
        self.steps_spin.setRange(1, 50)
        self.steps_spin.setValue(5)
        self.steps_spin.setToolTip(
            "Number of incremental load steps (stress ramped 0 \u2192 target)")
        self._add_widget_row(hdr, "Load Steps", self.steps_spin)

        # --- v2 advanced options ---
        self.deriv_combo = QtWidgets.QComboBox()
        self.deriv_combo.addItems(['continuous', 'finite_difference', 'rotated'])
        self.deriv_combo.setToolTip(
            "Discrete derivative scheme for Green's operator\n"
            "  continuous: standard k=2\u03c0\u03be (classical)\n"
            "  finite_difference: Willot sin(\u03c0\u03be)/h\n"
            "  rotated: Willot-Pellegrini (best for composites)")
        self._add_widget_row(hdr, "Derivatives", self.deriv_combo)

        self.refmed_combo = QtWidgets.QComboBox()
        self.refmed_combo.addItems(['mean', 'contrast_aware'])
        self.refmed_combo.setToolTip(
            "Reference medium selection\n"
            "  mean: Voigt (arithmetic) average\n"
            "  contrast_aware: geometric mean of per-voxel K,\u03bc")
        self._add_widget_row(hdr, "Ref. Medium", self.refmed_combo)

        self.anderson_spin = QtWidgets.QSpinBox()
        self.anderson_spin.setRange(0, 20)
        self.anderson_spin.setValue(0)
        self.anderson_spin.setToolTip(
            "Anderson acceleration window (Basic scheme only).\n"
            "0 = off, 3-5 = typical. Reduces iteration count ~30%.")
        self._add_widget_row(hdr, "Anderson m", self.anderson_spin)

        self.debug_check = QtWidgets.QCheckBox("Debug checks")
        self.debug_check.setToolTip(
            "Run physics invariant checks: symmetry, macro strain/stress, energy")
        self.debug_check.setChecked(False)
        child_dbg = QtWidgets.QTreeWidgetItem(hdr, [""])
        child_dbg.setFlags(child_dbg.flags() & ~QtCore.Qt.ItemIsSelectable)
        self.tree.setItemWidget(child_dbg, 0, self.debug_check)

    # ---- Job section ----

    def _build_job(self):
        hdr = self._make_header("\u25B6", "Job")

        self.run_btn = QtWidgets.QPushButton("Submit Job")
        self.run_btn.setStyleSheet(
            "background: #2255aa; padding: 7px 10px; font-size: 13px;")
        self.run_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.run_requested.emit)
        self._add_button_row(hdr, self.run_btn)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        child = QtWidgets.QTreeWidgetItem(hdr, [""])
        child.setFlags(child.flags() & ~QtCore.Qt.ItemIsSelectable)
        self.tree.setItemWidget(child, 0, self.progress)
        # Make progress span both columns via a container
        self._progress_item = child

        self.result_text = QtWidgets.QLabel("")
        self.result_text.setStyleSheet(
            "font-size: 11px; color: #88cc88; padding: 2px 4px;")
        self.result_text.setWordWrap(True)
        child2 = QtWidgets.QTreeWidgetItem(hdr, [""])
        child2.setFlags(child2.flags() & ~QtCore.Qt.ItemIsSelectable)
        self.tree.setItemWidget(child2, 0, self.result_text)

    # ---- presets ----

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
        # Remind user that changing material requires re-running
        names = ['Copper', 'Aluminium', 'Nickel', 'Iron (BCC)', 'Custom']
        name = names[idx] if idx < len(names) else 'Custom'
        self.result_text.setText(f"Material → {name}  (re-run to update results)")
        self.result_text.setStyleSheet(
            "font-size: 11px; color: #ffcc00; padding: 2px 4px;")

    def _apply_load_preset(self, idx):
        presets = [
            (5000,    0,    0,    0,    0,    0,
             "Uniaxial tension along X:\n\u03c3\u2081\u2081 = 5 GPa on \u00b1X faces"),
            (   0, 5000,    0,    0,    0,    0,
             "Uniaxial tension along Y:\n\u03c3\u2082\u2082 = 5 GPa on \u00b1Y faces"),
            (   0,    0, 5000,    0,    0,    0,
             "Uniaxial tension along Z:\n\u03c3\u2083\u2083 = 5 GPa on \u00b1Z faces"),
            (5000, 5000,    0,    0,    0,    0,
             "Biaxial tension in XY plane (5 GPa)"),
            (5000, 5000, 5000,    0,    0,    0,
             "Hydrostatic stress:\nEqual 5 GPa on all faces"),
            (   0,    0,    0,    0,    0, 2500,
             "Pure shear in XY plane (2.5 GPa)"),
            (   0,    0,    0,    0, 2500,    0,
             "Pure shear in XZ plane (2.5 GPa)"),
            None,
        ]
        if idx < len(presets) and presets[idx] is not None:
            s11, s22, s33, s23, s13, s12, desc = presets[idx]
            self.stress_spins['\u03c3\u2081\u2081'].setValue(s11)
            self.stress_spins['\u03c3\u2082\u2082'].setValue(s22)
            self.stress_spins['\u03c3\u2083\u2083'].setValue(s33)
            self.stress_spins['\u03c3\u2082\u2083'].setValue(s23)
            self.stress_spins['\u03c3\u2081\u2083'].setValue(s13)
            self.stress_spins['\u03c3\u2081\u2082'].setValue(s12)
            self.load_desc.setText(desc)
        else:
            self.load_desc.setText("Custom \u2014 set components manually")

    def get_stress_voigt_pa(self):
        """Return macroscopic stress as Voigt vector in Pa."""
        keys = ['\u03c3\u2081\u2081', '\u03c3\u2082\u2082', '\u03c3\u2083\u2083',
                '\u03c3\u2082\u2083', '\u03c3\u2081\u2083', '\u03c3\u2081\u2082']
        return np.array([self.stress_spins[k].value() * 1e6 for k in keys])


# ============================================================================
# Main Window (single-panel layout — no tabs)
# ============================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Crystal Plasticity FFT Simulator")
        self.setMinimumSize(1300, 800)
        self.setStyleSheet("background-color: #0f0f23;")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h_layout = QtWidgets.QHBoxLayout(central)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)

        # Left: model tree
        self.controls = ModelTreePanel()
        self.controls.generate_requested.connect(self._generate_microstructure)
        self.controls.run_requested.connect(self._run_polycrystal)
        self.controls.preview_loading.connect(self._preview_loading)
        h_layout.addWidget(self.controls)

        # Right: 3D viewer
        self.viewer = PolycrystalViewer()
        h_layout.addWidget(self.viewer, stretch=1)

        # State
        self.grain_ids = None
        self.euler_angles = None

    # ---- Preview loading diagram ----
    def _preview_loading(self):
        stress_pa = self.controls.get_stress_voigt_pa()
        self.viewer.show_loading_diagram(stress_pa)

    # ---- Generate microstructure ----
    def _generate_microstructure(self):
        N = self.controls.grid_spin.value()
        n_grains = self.controls.grains_spin.value()
        seed = self.controls.seed_spin.value()

        self.controls.result_text.setText("Generating microstructure...")
        self.controls.result_text.setStyleSheet(
            "font-size: 11px; color: #ffcc00;")
        QtWidgets.QApplication.processEvents()

        self.grain_ids, self.centers, self.euler_angles = \
            micro.generate_voronoi_microstructure(N, n_grains, seed=seed)

        stats = micro.grain_statistics(self.grain_ids)
        self.viewer.show_grain_map(self.grain_ids)

        self.controls.result_text.setText(
            f"Microstructure OK  |  {N}\u00b3 = {stats['total_voxels']} voxels"
            f"  |  {stats['n_grains']} grains")
        self.controls.result_text.setStyleSheet(
            "font-size: 11px; color: #88cc88;")

    # ---- Run FFT simulation (incremental load steps) ----
    def _run_polycrystal(self):
        if self.grain_ids is None:
            self.controls.result_text.setText(
                "Generate microstructure first!")
            self.controls.result_text.setStyleSheet(
                "font-size: 11px; color: #ff6666;")
            return

        use_cp = self.controls.plasticity_toggle.currentIndex() == 1
        if use_cp:
            self._run_evpfft()
        else:
            self._run_elastic()

    def _run_elastic(self):
        """Run elastic FFT solver (no plasticity)."""
        self.controls.run_btn.setEnabled(False)
        self.controls.progress.setVisible(True)
        self.controls.progress.setValue(5)
        self.controls.result_text.setText("Building stiffness field...")
        self.controls.result_text.setStyleSheet(
            "font-size: 11px; color: #ffcc00;")
        QtWidgets.QApplication.processEvents()

        C11 = self.controls.c11_spin.value() * 1e9
        C12 = self.controls.c12_spin.value() * 1e9
        C44 = self.controls.c44_spin.value() * 1e9

        C_field = micro.build_local_stiffness_field_fast(
            self.grain_ids, self.euler_angles, C11, C12, C44)

        S_macro_full = self.controls.get_stress_voigt_pa()
        tol_fft = float(self.controls.tol_combo.currentText())
        tol_stress = float(self.controls.stress_tol_combo.currentText())
        max_iter = self.controls.maxiter_spin.value()
        use_cg = self.controls.solver_combo.currentIndex() == 0
        solver_name = 'CG' if use_cg else 'Basic'
        n_steps = self.controls.steps_spin.value()

        # Incremental load stepping
        step_results = []
        total_fft_iters = 0
        total_newton_iters = 0

        for step in range(1, n_steps + 1):
            frac = step / n_steps
            S_step = S_macro_full * frac

            pct = int(10 + 80 * step / n_steps)
            self.controls.progress.setValue(pct)
            self.controls.result_text.setText(
                f"Step {step}/{n_steps}  ({frac:.0%})  —  {solver_name}")
            QtWidgets.QApplication.processEvents()

            try:
                eps_field, sig_field, info = fft_solver.solve_stress_controlled(
                    C_field, S_step,
                    tol_fft=tol_fft, tol_stress=tol_stress,
                    max_iter_fft=max_iter, max_iter_stress=30,
                    solver='cg' if use_cg else 'basic',
                    verbose=True)
            except Exception as e:
                self.controls.result_text.setText(
                    f"Error at step {step}/{n_steps}: {e}")
                self.controls.result_text.setStyleSheet(
                    "font-size: 11px; color: #ff6666;")
                self.controls.run_btn.setEnabled(True)
                self.controls.progress.setVisible(False)
                return

            total_fft_iters += info['iterations']
            total_newton_iters += info.get('stress_iterations', 0)
            step_results.append(
                (eps_field.copy(), sig_field.copy(), frac))

        eps_field, sig_field, _ = step_results[-1]

        self.controls.progress.setValue(95)
        QtWidgets.QApplication.processEvents()

        self.viewer.show_results_incremental(
            step_results, self.grain_ids,
            stress_applied=S_macro_full)

        if hasattr(self.viewer, '_start_animation'):
            self.viewer._start_animation()

        self.controls.progress.setValue(100)

        vm = fft_solver.von_mises_stress(sig_field)
        N = self.grain_ids.shape[0]

        lines = [
            f"Elastic  ({solver_name}, {n_steps} steps)",
            f"  {N}\u00b3 grid  |  {total_fft_iters} FFT iters  |  "
            f"{total_newton_iters} Newton iters",
            f"  VM: mean={np.mean(vm)/1e6:.1f}  max={np.max(vm)/1e6:.1f} MPa",
        ]
        self.controls.result_text.setText("\n".join(lines))
        self.controls.result_text.setStyleSheet(
            "font-size: 11px; color: #88cc88;")

        self.controls.run_btn.setEnabled(True)
        self.controls.progress.setVisible(False)

    def _run_evpfft(self):
        """Run crystal plasticity EVPFFT solver (small-strain or finite-strain)."""
        self.controls.run_btn.setEnabled(False)
        self.controls.progress.setVisible(True)
        self.controls.progress.setValue(5)
        self.controls.result_text.setText("Initializing crystal plasticity...")
        self.controls.result_text.setStyleSheet(
            "font-size: 11px; color: #ffcc00;")
        QtWidgets.QApplication.processEvents()

        C11 = self.controls.c11_spin.value() * 1e9
        C12 = self.controls.c12_spin.value() * 1e9
        C44 = self.controls.c44_spin.value() * 1e9
        tau0 = self.controls.tau0_spin.value() * 1e6
        tau_s = self.controls.tau_s_spin.value() * 1e6
        h0 = self.controls.h0_spin.value() * 1e6
        q_lat = self.controls.q_lat_spin.value()
        temperature = self.controls.temp_spin.value()
        rate_ref = float(self.controls.rate_ref_combo.currentText())

        # Formulation and hardening model
        use_finite_strain = (self.controls.formulation_combo.currentIndex() == 1)
        use_km = (self.controls.hardening_combo.currentIndex() == 1)
        enable_bs = self.controls.backstress_check.isChecked()
        enable_gnd = self.controls.gnd_check.isChecked()

        S_macro_full = self.controls.get_stress_voigt_pa()
        tol_fft = float(self.controls.tol_combo.currentText())
        tol_stress = float(self.controls.stress_tol_combo.currentText())
        max_iter = self.controls.maxiter_spin.value()
        n_steps = self.controls.steps_spin.value()
        deriv_scheme = self.controls.deriv_combo.currentText()
        ref_mode = self.controls.refmed_combo.currentText()

        self.controls.progress.setValue(10)
        label = "Finite-Strain EVPFFT" if use_finite_strain else "EVPFFT"
        self.controls.result_text.setText(f"{label}: 0/{n_steps} steps...")
        QtWidgets.QApplication.processEvents()

        def progress_callback(inc, eps_f, sig_f, st):
            pct = int(10 + 80 * inc / n_steps)
            self.controls.progress.setValue(pct)
            sig_mean = np.mean(sig_f.reshape(-1, 6), axis=0)
            self.controls.result_text.setText(
                f"{label} step {inc}/{n_steps}  "
                f"\u03c3\u2081\u2081={sig_mean[0]/1e6:.1f} MPa")
            QtWidgets.QApplication.processEvents()

        try:
            if use_finite_strain:
                import finite_strain as fs

                km_params = {
                    'k1': self.controls.km_k1_spin.value(),
                    'k2': self.controls.km_k2_spin.value(),
                    'alpha_taylor': self.controls.km_alpha_spin.value(),
                    'rho0': self.controls.km_rho0_spin.value(),
                    'q_latent': q_lat,
                }
                af_params = {
                    'c1': self.controls.af_c1_spin.value(),
                    'c2': self.controls.af_c2_spin.value(),
                }
                voce_params = {
                    'tau0': tau0, 'tau_s': tau_s, 'h0': h0, 'q_latent': q_lat,
                }

                state = fs.FiniteStrainState(
                    self.grain_ids, self.euler_angles,
                    C11=C11, C12=C12, C44=C44,
                    temperature=temperature,
                    rate_slip_ref=rate_ref,
                    hardening_model='kocks_mecking' if use_km else 'voce',
                    km_params=km_params if use_km else None,
                    voce_params=voce_params if not use_km else None,
                    af_params=af_params if enable_bs else None,
                    enable_backstress=enable_bs,
                    enable_gnd=enable_gnd)

                history = fs.solve_evpfft_finite_strain(
                    state, S_macro_full,
                    n_increments=n_steps, dt=0.1,
                    tol_fft=tol_fft, tol_stress=tol_stress,
                    max_iter_fft=max_iter, max_iter_stress=30,
                    verbose=True, callback=progress_callback,
                    derivative_scheme=deriv_scheme,
                    ref_medium_mode=ref_mode,
                    implicit_constitutive=False,
                    compute_gnd_every=1 if enable_gnd else 0)

            else:
                # Small-strain EVPFFT (original path)
                hardening = constitutive.VoceHardening(
                    tau0=tau0, tau_s=tau_s, h0=h0, q_latent=q_lat)
                state = constitutive.CrystalPlasticityState(
                    self.grain_ids, self.euler_angles,
                    hardening=hardening,
                    C11=C11, C12=C12, C44=C44,
                    temperature=temperature,
                    rate_slip_ref=rate_ref)

                history = constitutive.solve_evpfft(
                    state, S_macro_full,
                    n_increments=n_steps, dt=0.1,
                    tol_fft=tol_fft, tol_stress=tol_stress,
                    max_iter_fft=max_iter, max_iter_stress=30,
                    verbose=True, callback=progress_callback,
                    derivative_scheme=deriv_scheme,
                    ref_medium_mode=ref_mode)

        except Exception as e:
            self.controls.result_text.setText(f"{label} Error: {e}")
            self.controls.result_text.setStyleSheet(
                "font-size: 11px; color: #ff6666;")
            self.controls.run_btn.setEnabled(True)
            self.controls.progress.setVisible(False)
            import traceback; traceback.print_exc()
            return

        # Convert to step_results format for visualization
        step_results = history

        n_snaps = len(getattr(state, '_history_snapshots', []))
        ep_max = np.max(state.get_von_mises_plastic_strain_field())
        print(f"[{label}] {n_snaps} snapshots stored, "
              f"max εᵖ_eq = {ep_max:.6f}")

        self.controls.progress.setValue(95)
        QtWidgets.QApplication.processEvents()

        self.viewer.show_results_incremental(
            step_results, self.grain_ids,
            stress_applied=S_macro_full,
            cp_state=state)

        # Auto-switch field selector to plastic strain for CP results
        if hasattr(self.viewer, '_field_combo'):
            self.viewer._field_combo.setCurrentIndex(14)  # εᵖ eq

        if hasattr(self.viewer, '_start_animation'):
            self.viewer._start_animation()

        self.controls.progress.setValue(100)

        eps_final, sig_final, _ = step_results[-1]
        vm = fft_solver.von_mises_stress(sig_final)
        ep_eq = state.get_von_mises_plastic_strain_field()
        N = self.grain_ids.shape[0]

        lines = [
            f"{label} complete  ({n_steps} steps)",
            f"  {N}\u00b3 grid  |  T={temperature:.0f} K",
        ]
        if use_finite_strain:
            lines.append(
                f"  Hardening: {'Kocks-Mecking' if use_km else 'Voce'}"
                f"  |  Back-stress: {'ON' if enable_bs else 'OFF'}"
                f"  |  GND: {'ON' if enable_gnd else 'OFF'}")
            if hasattr(state, 'rho_ssd') and state.rho_ssd is not None:
                lines.append(
                    f"  Mean ρ_SSD: {np.mean(state.rho_ssd):.2e} m⁻²"
                    f"  |  Max GND: {np.max(state.rho_gnd):.2e} m⁻²")
            if hasattr(state, 'chi') and state.chi is not None:
                lines.append(
                    f"  Max |χ|: {np.max(np.abs(state.chi))/1e6:.1f} MPa"
                    f"  |  Max misor.: "
                    f"{np.max(state.get_misorientation_field()):.2f}°")
        else:
            lines.append(
                f"  \u03c4\u2080={tau0/1e6:.0f}  "
                f"\u03c4_s={tau_s/1e6:.0f}  h\u2080={h0/1e6:.0f} MPa")
        lines.extend([
            f"  VM: max={np.max(vm)/1e6:.1f} MPa  |  "
            f"\u03b5\u1d56_eq max={np.max(ep_eq):.5f}",
            f"  Mean slip res.: {np.mean(state.slip_resistance)/1e6:.1f} MPa",
        ])
        self.controls.result_text.setText("\n".join(lines))
        self.controls.result_text.setStyleSheet(
            "font-size: 11px; color: #88cc88;")

        self.controls.run_btn.setEnabled(True)
        self.controls.progress.setVisible(False)


# ============================================================================
# Entry point
# ============================================================================
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')

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
