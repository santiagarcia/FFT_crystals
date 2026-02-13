"""
GIF: Flow Rule — Thermal Activation Kinetics
=============================================
Visualizes the constitutive flow rule:

    gamma_dot^alpha = gamma_dot_0 * exp[-DeltaG(tau) / (k_B T)] * sgn(tau)

Two-panel animation:
  Left:  Energy barrier diagram — shows how applied stress lowers DeltaG
  Right: gamma_dot vs tau curve — the S-shaped exponential activation
Sweeps through increasing tau, then shows temperature effect.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')

# ─── Physical parameters (matching computeFlowRule.f90) ───────────────────────
GAMMA_DOT_0 = 1.0e7          # reference slip rate (1/s)
DELTA_G0 = 9.5e-19           # activation energy (J)
K_B = 1.3806e-23             # Boltzmann constant (J/K)
P = 0.78                     # barrier shape
Q = 1.15                     # barrier shape
R_THERMAL = 539e6            # thermal resistance (Pa)
R_ATHERMAL = 0.0             # athermal resistance (Pa)

TEXT_FX = [pe.withStroke(linewidth=2, foreground='white')]


def flow_rule(tau, T, r_thermal=R_THERMAL):
    """Compute gamma_dot for given tau and T."""
    tau_eff = np.abs(tau) - R_ATHERMAL
    tau_eff = np.maximum(tau_eff, 0.0)
    ratio = np.minimum(tau_eff / r_thermal, 0.9999)
    delta_g = DELTA_G0 * (1.0 - ratio**P)**Q
    gamma_dot = GAMMA_DOT_0 * np.exp(-delta_g / (K_B * T)) * np.sign(tau)
    # Zero out where tau_eff <= 0
    gamma_dot = np.where(tau_eff > 0, gamma_dot, 0.0)
    return gamma_dot


def delta_g_func(tau, r_thermal=R_THERMAL):
    """Compute DeltaG as function of tau."""
    tau_eff = np.abs(tau) - R_ATHERMAL
    tau_eff = np.maximum(tau_eff, 0.0)
    ratio = np.minimum(tau_eff / r_thermal, 0.9999)
    return DELTA_G0 * (1.0 - ratio**P)**Q


# ─── Figure ───────────────────────────────────────────────────────────────────
fig, (ax_barrier, ax_curve) = plt.subplots(1, 2, figsize=(13, 6), facecolor='white')

# ── Left panel: Energy barrier ────────────────────────────────────────────────
ax_barrier.set_facecolor('white')
ax_barrier.set_xlim(-0.5, 3.5)
ax_barrier.set_ylim(-0.3, 1.35)
ax_barrier.set_xlabel('Reaction Coordinate', fontsize=11)
ax_barrier.set_ylabel('Energy', fontsize=11)
ax_barrier.set_title('Activation Energy Barrier', fontsize=13, fontweight='bold')
ax_barrier.spines['top'].set_visible(False)
ax_barrier.spines['right'].set_visible(False)
ax_barrier.set_xticks([])
ax_barrier.set_yticks([])

# Static energy landscape (double-well)
x_land = np.linspace(-0.3, 3.3, 300)
# Simple double-well: minimum at x=0.5, barrier at x=1.5, minimum at x=2.5
E_base = 0.5 * (1.0 - np.cos(2.0 * np.pi * (x_land - 0.5) / 2.0))
# Clamp edges
E_base = np.clip(E_base, 0, None)
# Shape it nicely
E_landscape = 0.3 * np.exp(-((x_land - 0.5)**2) / 0.15) + \
              0.9 * np.exp(-((x_land - 1.5)**2) / 0.25) - \
              0.1 * np.exp(-((x_land - 2.5)**2) / 0.15)
E_landscape = E_landscape - E_landscape.min() + 0.05

ax_barrier.plot(x_land, E_landscape, 'k-', lw=2, alpha=0.3, label='No stress')

# Pre-compute reference barrier height (valley → peak on unstressed curve)
# Valley: local min near x≈0.5;  Peak: local max near x≈1.5
# Use derivative sign change to find true local extrema
def find_local_min(x, E, x_lo, x_hi):
    """Find the local minimum of E between x_lo and x_hi."""
    mask = (x >= x_lo) & (x <= x_hi)
    idx = np.where(mask)[0]
    local_idx = idx[np.argmin(E[idx])]
    return local_idx

def find_local_max(x, E, x_lo, x_hi):
    """Find the local maximum of E between x_lo and x_hi."""
    mask = (x >= x_lo) & (x <= x_hi)
    idx = np.where(mask)[0]
    local_idx = idx[np.argmax(E[idx])]
    return local_idx

# Reference extrema
idx_ref_valley = find_local_min(x_land, E_landscape, 0.2, 1.0)
idx_ref_peak   = find_local_max(x_land, E_landscape, 0.8, 2.2)
DG0 = E_landscape[idx_ref_peak] - E_landscape[idx_ref_valley]

def draw_dg_arrow(ax_b, E_curve, color='#D32F2F', label_suffix='', dynamic_list=None):
    """Draw ΔG(τ) arrow at the valley's x-position.
    Bottom tip sits right on the valley minimum.
    Top tip at the peak's energy level, with a dotted line over to the peak."""
    i_valley = find_local_min(x_land, E_curve, 0.2, 1.0)
    i_peak   = find_local_max(x_land, E_curve, 0.8, 2.2)
    
    x_valley = x_land[i_valley]
    x_peak   = x_land[i_peak]
    E_valley = E_curve[i_valley]
    E_peak   = E_curve[i_peak]
    dg = max(E_peak - E_valley, 0.0)
    pct = dg / DG0 * 100 if DG0 > 0.01 else 100.0
    
    if dg < 0.005:
        return
    
    # Arrow at the valley's x, from valley up to peak energy level
    arr = ax_b.annotate('', xy=(x_valley, E_valley), xytext=(x_valley, E_peak),
                        arrowprops=dict(arrowstyle='<->', color=color, lw=2))
    if dynamic_list is not None:
        dynamic_list.append(arr)
    
    # Horizontal dotted line from arrow top to the actual peak point
    tick_top, = ax_b.plot([x_valley, x_peak], [E_peak, E_peak],
                          color=color, lw=1, ls=':', alpha=0.5)
    if dynamic_list is not None:
        dynamic_list.append(tick_top)
    
    # Horizontal dotted line at valley level to signal the valley
    tick_bot, = ax_b.plot([x_valley - 0.15, x_valley], [E_valley, E_valley],
                          color=color, lw=1, ls=':', alpha=0.5)
    if dynamic_list is not None:
        dynamic_list.append(tick_bot)
    
    # Label to the right of the arrow
    txt = ax_b.text(x_valley + 0.12, (E_peak + E_valley) / 2,
                    f'$\\Delta G(\\tau)$\n({pct:.0f}%){label_suffix}',
                    color=color, fontsize=10, va='center', fontweight='bold')
    if dynamic_list is not None:
        dynamic_list.append(txt)

# ── Right panel: gamma_dot vs tau ─────────────────────────────────────────────
ax_curve.set_facecolor('white')
T_ref = 1123.0  # K
tau_range = np.linspace(-R_THERMAL * 0.95, R_THERMAL * 0.95, 500)
gdot_full = flow_rule(tau_range, T_ref)

# Static: faint full curve
ax_curve.plot(tau_range / 1e6, gdot_full / 1e6, color='#CCCCCC', lw=1.5, alpha=0.5)
ax_curve.axhline(0, color='gray', lw=0.5)
ax_curve.axvline(0, color='gray', lw=0.5)
ax_curve.set_xlabel('$\\tau_i$ (MPa)', fontsize=11)
ax_curve.set_ylabel('$\\dot{\\gamma}_i$ ($\\times 10^6$ s$^{-1}$)', fontsize=11)
ax_curve.set_title('Plastic Shear Rate', fontsize=13, fontweight='bold')
ax_curve.spines['top'].set_visible(False)
ax_curve.spines['right'].set_visible(False)

# Equation annotation
eq_box = ax_curve.text(0.5, 0.95,
    '$\\dot{\\gamma}^\\alpha = \\dot{\\gamma}_0 \\,'
    '\\exp\\!\\left[-\\dfrac{\\Delta G(\\tau)}{k_B T}\\right]'
    '\\mathrm{sgn}(\\tau)$',
    transform=ax_curve.transAxes, ha='center', va='top',
    fontsize=13, color='#333333',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1', edgecolor='#FFB300',
              alpha=0.9))

# ─── Mutable artists ─────────────────────────────────────────────────────────
dynamic = []


def clear_dynamic():
    for a in dynamic:
        try:
            a.remove()
        except Exception:
            pass
    dynamic.clear()


# ─── Animation Phases ─────────────────────────────────────────────────────────
# Phase 1 (0-39):   Sweep tau from 0 to max, showing barrier lowering + curve tracing
# Phase 2 (40-59):  Sweep back (negative tau) — show sgn(tau)
# Phase 3 (60-89):  Show temperature effect — overlay curves at different T
# Phase 4 (90-109): Hold final with all info

PHASE1_END = 50
PHASE2_END = 80
PHASE3_END = 120
PHASE4_END = 145
TOTAL_FRAMES = PHASE4_END


def update(frame):
    clear_dynamic()

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Sweep positive tau, trace curve, lower barrier
    # ═══════════════════════════════════════════════════════════════════════════
    if frame < PHASE1_END:
        t = frame / (PHASE1_END - 1)
        tau_current = t * R_THERMAL * 0.90

        # ── Left: barrier diagram ──
        # Tilt landscape by tau (stress lowers forward barrier)
        tilt = t * 0.55
        E_tilted = E_landscape - tilt * (x_land - 0.5) / 2.0
        E_tilted = E_tilted - E_tilted[0] + 0.05

        l, = ax_barrier.plot(x_land, E_tilted, color='#1565C0', lw=2.5)
        dynamic.append(l)

        # Delta G arrow — remaining barrier on the stressed curve
        # (from valley to peak, both on the blue curve)
        draw_dg_arrow(ax_barrier, E_tilted, dynamic_list=dynamic)

        # Stress label
        txt2 = ax_barrier.text(0.05, 0.92,
                               f'$\\tau = {tau_current/1e6:.0f}$ MPa\n$T = {1123}$ K',
                               transform=ax_barrier.transAxes, fontsize=11,
                               va='top', color='#1565C0', fontweight='bold')
        dynamic.append(txt2)

        # Caption
        if t < 0.3:
            cap = ax_barrier.text(0.5, 0.02, 'Stress lowers the energy barrier …',
                                  transform=ax_barrier.transAxes, ha='center',
                                  fontsize=10, color='#666666', style='italic')
            dynamic.append(cap)

        # ── Right: trace curve up to current tau ──
        mask = (tau_range >= 0) & (tau_range <= tau_current)
        if np.any(mask):
            gdot_partial = flow_rule(tau_range[mask], T_ref)
            l2, = ax_curve.plot(tau_range[mask] / 1e6, gdot_partial / 1e6,
                                color='#1565C0', lw=3)
            dynamic.append(l2)

        # Current point dot
        gdot_now = flow_rule(np.array([tau_current]), T_ref)[0]
        dot = ax_curve.plot(tau_current / 1e6, gdot_now / 1e6, 'o',
                            color='#FF6F00', ms=10, zorder=5)
        dynamic.extend(dot)

        # Value label
        if gdot_now > 0:
            txt3 = ax_curve.text(tau_current / 1e6 + 10, gdot_now / 1e6 + 0.15,
                                 f'$\\dot{{\\gamma}} = {gdot_now:.1e}$',
                                 fontsize=9, color='#E65100')
            dynamic.append(txt3)

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Sweep negative tau — show sgn(tau) symmetry
    # ═══════════════════════════════════════════════════════════════════════════
    elif frame < PHASE2_END:
        t = (frame - PHASE1_END) / (PHASE2_END - PHASE1_END - 1)
        tau_neg = -t * R_THERMAL * 0.90

        # Left: barrier still lowers (DeltaG depends on |tau|, not tau)
        E_tilted = E_landscape - t * 0.55 * (x_land - 0.5) / 2.0
        E_tilted = E_tilted - E_tilted[0] + 0.05
        l, = ax_barrier.plot(x_land, E_tilted, color='#C62828', lw=2.5)
        dynamic.append(l)

        # Delta G arrow — same logic as Phase 1 (remaining barrier)
        draw_dg_arrow(ax_barrier, E_tilted, dynamic_list=dynamic)

        txt2 = ax_barrier.text(0.05, 0.92,
                               f'$\\tau = {tau_neg/1e6:.0f}$ MPa\n$T = {1123}$ K\n'
                               f'$\\mathrm{{sgn}}(\\tau) = -1$',
                               transform=ax_barrier.transAxes, fontsize=11,
                               va='top', color='#C62828', fontweight='bold')
        dynamic.append(txt2)

        cap = ax_barrier.text(0.5, 0.02,
                              'Negative $\\tau$ → slip in opposite direction',
                              transform=ax_barrier.transAxes, ha='center',
                              fontsize=10, color='#666666', style='italic')
        dynamic.append(cap)

        # Right: full positive curve + growing negative curve
        gdot_pos = flow_rule(tau_range[tau_range >= 0], T_ref)
        l_pos, = ax_curve.plot(tau_range[tau_range >= 0] / 1e6, gdot_pos / 1e6,
                               color='#1565C0', lw=3)
        dynamic.append(l_pos)

        mask_neg = (tau_range <= 0) & (tau_range >= tau_neg)
        if np.any(mask_neg):
            gdot_neg = flow_rule(tau_range[mask_neg], T_ref)
            l_neg, = ax_curve.plot(tau_range[mask_neg] / 1e6, gdot_neg / 1e6,
                                   color='#C62828', lw=3)
            dynamic.append(l_neg)

        # Current point
        gdot_now = flow_rule(np.array([tau_neg]), T_ref)[0]
        dot = ax_curve.plot(tau_neg / 1e6, gdot_now / 1e6, 'o',
                            color='#FF6F00', ms=10, zorder=5)
        dynamic.extend(dot)

        # sgn label
        sgn_txt = ax_curve.text(0.25, 0.15,
                                '$\\mathrm{sgn}(\\tau) = -1$\n→ negative $\\dot{\\gamma}$',
                                transform=ax_curve.transAxes, fontsize=10,
                                color='#C62828', fontweight='bold',
                                bbox=dict(facecolor='#FFEBEE', edgecolor='#C62828',
                                          alpha=0.8, boxstyle='round,pad=0.3'))
        dynamic.append(sgn_txt)

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3: Temperature effect — overlay curves at different T
    # ═══════════════════════════════════════════════════════════════════════════
    elif frame < PHASE3_END:
        t = (frame - PHASE2_END) / (PHASE3_END - PHASE2_END - 1)

        # Left: barrier is FIXED (T does NOT change the barrier)
        # Show the same stressed curve, and illustrate kBT as a scale
        tau_fixed = 0.6 * R_THERMAL
        tilt_fixed = 0.6 * 0.55
        E_tilted = E_landscape - tilt_fixed * (x_land - 0.5) / 2.0
        E_tilted = E_tilted - E_tilted[0] + 0.05
        l, = ax_barrier.plot(x_land, E_tilted, color='#1565C0', lw=2.5)
        dynamic.append(l)

        T_temps = [600, 900, 1123, 1400]
        T_colors = ['#1A237E', '#1565C0', '#2E7D32', '#E65100']
        T_names = ['600 K', '900 K', '1123 K', '1400 K']

        # How many T curves to show on right panel
        n_show = min(len(T_temps), int(t * len(T_temps)) + 1)

        txt2 = ax_barrier.text(0.05, 0.92,
                               f'$\\tau = {tau_fixed/1e6:.0f}$ MPa (fixed)\n'
                               f'$\\Delta G$ is the same',
                               transform=ax_barrier.transAxes, fontsize=11,
                               va='top', color='#333333', fontweight='bold')
        dynamic.append(txt2)

        # Show ΔG arrow (remaining barrier, unchanged)
        draw_dg_arrow(ax_barrier, E_tilted, label_suffix='\n(fixed)', dynamic_list=dynamic)

        # Show kBT scale bars stacked vertically below the curves
        for j in range(n_show):
            kbt_scaled = K_B * T_temps[j] / DELTA_G0 * 0.9
            y_bar = -0.25 + j * 0.08
            rect = ax_barrier.barh(y_bar, kbt_scaled, height=0.04,
                                   left=0.3, color=T_colors[j], alpha=0.6)
            for r in rect:
                dynamic.append(r)
            txt_t = ax_barrier.text(0.3 + kbt_scaled + 0.05, y_bar + 0.02,
                                    T_names[j],
                                    color=T_colors[j], fontsize=9, va='center',
                                    fontweight='bold')
            dynamic.append(txt_t)

        # Label the bars
        dynamic.append(ax_barrier.text(0.3, -0.28, '$k_B T$',
                       fontsize=9, color='#555555', va='top'))

        cap = ax_barrier.text(0.5, 0.02,
                              '$T$ does not change the barrier — it changes how easily it is overcome',
                              transform=ax_barrier.transAxes, ha='center',
                              fontsize=9, color='#666666', style='italic')
        dynamic.append(cap)

        # Right: overlay curves for each T
        for j in range(n_show):
            gdot_T = flow_rule(tau_range, T_temps[j])
            lw = 3 if j == n_show - 1 else 1.8
            alpha = 1.0 if j == n_show - 1 else 0.5
            l_t, = ax_curve.plot(tau_range / 1e6, gdot_T / 1e6,
                                 color=T_colors[j], lw=lw, alpha=alpha,
                                 label=T_names[j])
            dynamic.append(l_t)

        # Legend
        leg = ax_curve.legend(loc='lower right', fontsize=9, framealpha=0.8,
                              title='Temperature', title_fontsize=10)
        dynamic.append(leg)

        temp_note = ax_curve.text(0.03, 0.95,
                                  'Higher $T$ → faster slip\nat same $\\tau$',
                                  transform=ax_curve.transAxes, fontsize=10,
                                  va='top', color='#333333',
                                  bbox=dict(facecolor='#FFF8E1', edgecolor='#FFB300',
                                            alpha=0.8, boxstyle='round,pad=0.3'))
        dynamic.append(temp_note)

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 4: Hold final — summary
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        # Left: clean barrier with summary
        tau_fixed = 0.6 * R_THERMAL
        tilt_fixed = 0.6 * 0.55
        E_tilted = E_landscape - tilt_fixed * (x_land - 0.5) / 2.0
        E_tilted = E_tilted - E_tilted[0] + 0.05
        l, = ax_barrier.plot(x_land, E_tilted, color='#1565C0', lw=2.5)
        dynamic.append(l)

        draw_dg_arrow(ax_barrier, E_tilted, dynamic_list=dynamic)

        # Summary box
        summary = ax_barrier.text(0.5, 0.92,
            'Geometry → where slip occurs\n'
            'Kinetics → how fast',
            transform=ax_barrier.transAxes, ha='center', va='top',
            fontsize=12, color='#333333', fontweight='bold',
            bbox=dict(facecolor='#E3F2FD', edgecolor='#1565C0',
                      alpha=0.9, boxstyle='round,pad=0.4'))
        dynamic.append(summary)

        # Right: full curve at T=1123 K with annotations
        gdot_full_ref = flow_rule(tau_range, T_ref)
        l_final, = ax_curve.plot(tau_range / 1e6, gdot_full_ref / 1e6,
                                 color='#1565C0', lw=3)
        dynamic.append(l_final)

        # Annotate key features
        props = [
            (0.75, 0.55, '• Scalar quantity\n• Signed ($\\pm$)\n• Output of flow rule',
             '#333333', '#E8F5E9', '#4CAF50'),
        ]
        for px, py, text, tc, fc, ec in props:
            box = ax_curve.text(px, py, text,
                                transform=ax_curve.transAxes, fontsize=10,
                                color=tc, va='top',
                                bbox=dict(facecolor=fc, edgecolor=ec,
                                          alpha=0.9, boxstyle='round,pad=0.4'))
            dynamic.append(box)

    return []


anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=140, blit=False)

plt.tight_layout()
out_path = 'gif_flow_rule.gif'
print(f"Rendering {TOTAL_FRAMES} frames …")
anim.save(out_path, writer='pillow', fps=10, dpi=120,
          savefig_kwargs={'facecolor': 'white'})
print(f"Saved  →  {out_path}")
plt.close()
