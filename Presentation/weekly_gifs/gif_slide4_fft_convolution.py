"""
Slide 4 GIF — Why FFT Appears: Lippmann-Schwinger in Fourier Space
==================================================================
Visualizes:
  1) Real-space polarization τ(x)
  2) FFT → τ_hat(k)
  3) Multiply by Γ0(k) → ε_hat(k)
  4) IFFT → ε(x)
  5) Enforce ε_hat(k=0) = E

Step-by-step pipeline animation.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'mathtext.fontset': 'cm',
})

N = 32
np.random.seed(42)

# Generate fake fields
n_grains = 8
seeds = np.random.rand(n_grains, 2)
xx, yy = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
grain_id = np.zeros((N, N), int)
for iy in range(N):
    for ix in range(N):
        dists = np.sqrt((seeds[:, 0] - xx[iy, ix])**2 + (seeds[:, 1] - yy[iy, ix])**2)
        grain_id[iy, ix] = np.argmin(dists)

C_grain = 80 + 100 * np.random.rand(n_grains)
C0 = np.mean(C_grain)
tau_field = (C_grain[grain_id] - C0) * gaussian_filter(np.random.randn(N, N), sigma=2) * 0.01

# FFT of tau
tau_hat = np.fft.fft2(tau_field)
tau_hat_mag = np.log1p(np.abs(np.fft.fftshift(tau_hat)))

# Fake Gamma0 operator (just a filter for visualization)
kx = np.fft.fftfreq(N, d=1.0/N)
ky = np.fft.fftfreq(N, d=1.0/N)
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2
K2[0, 0] = 1  # avoid div by zero
Gamma0 = 1.0 / K2  # simplified
Gamma0[0, 0] = 0  # zero mode
Gamma0_vis = np.log1p(np.abs(np.fft.fftshift(Gamma0)))

eps_hat = -Gamma0 * tau_hat
eps_hat[0, 0] = 0.01 * N * N  # macroscopic strain E
eps_field = np.real(np.fft.ifft2(eps_hat))
eps_hat_mag = np.log1p(np.abs(np.fft.fftshift(eps_hat)))

# ─── Figure: Pipeline ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5), facecolor='white')

# 5 panels
positions = [
    [0.02, 0.18, 0.17, 0.55],   # τ(x)
    [0.22, 0.18, 0.17, 0.55],   # τ_hat(k)
    [0.42, 0.18, 0.17, 0.55],   # Γ0(k)
    [0.62, 0.18, 0.17, 0.55],   # ε_hat(k)
    [0.82, 0.18, 0.17, 0.55],   # ε(x)
]

titles = [
    r'$\tau(\mathbf{x})$',
    r'$\hat{\tau}(\mathbf{k})$',
    r'$\hat{\Gamma}^0(\mathbf{k})$',
    r'$\hat{\varepsilon}(\mathbf{k})$',
    r'$\varepsilon(\mathbf{x})$',
]

subtitles = [
    'Polarization\n(real space)',
    'FFT',
    'Green operator\n(Fourier space)',
    'Multiply',
    'Strain\n(IFFT back)',
]

data = [tau_field, tau_hat_mag, Gamma0_vis, eps_hat_mag, eps_field]
cmaps = ['PuOr_r', 'inferno', 'viridis', 'inferno', 'coolwarm']

axs = []
ims = []
title_txts = []
sub_txts = []

for i, (pos, title, sub, d, cm) in enumerate(zip(positions, titles, subtitles, data, cmaps)):
    ax = fig.add_axes(pos)
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(np.zeros((N, N)), cmap=cm, origin='lower', aspect='equal')
    im.set_clim(d.min(), d.max())
    im.set_alpha(0)
    axs.append(ax)
    ims.append(im)

    tt = ax.set_title(title, fontsize=14, fontweight='bold', color='#333', pad=4)
    tt.set_alpha(0)
    title_txts.append(tt)

    st = fig.text(pos[0] + pos[2]/2, 0.08, sub, ha='center', fontsize=9,
                  color='#777', va='top')
    st.set_alpha(0)
    sub_txts.append(st)

# Arrows between panels
arrow_positions = [(0.195, 0.46), (0.395, 0.46), (0.595, 0.46), (0.795, 0.46)]
arrow_labels = ['FFT', r'$\times$', r'$=$', 'IFFT']
arrow_colors = ['#1565C0', '#D32F2F', '#333', '#2E7D32']
arrow_txts = []
for (ax_pos, lbl, col) in zip(arrow_positions, arrow_labels, arrow_colors):
    t = fig.text(ax_pos[0], ax_pos[1], r'$\longrightarrow$' + '\n' + lbl,
                 ha='center', va='center', fontsize=12, fontweight='bold',
                 color=col)
    t.set_alpha(0)
    arrow_txts.append(t)

# Top equation
eq_main = fig.text(0.50, 0.92,
    r'$\hat{\varepsilon}(\mathbf{k}) = -\hat{\Gamma}^0(\mathbf{k}) : \hat{\tau}(\mathbf{k})$'
    r'$,\quad \hat{\varepsilon}(\mathbf{0}) = \mathbf{E}$',
    ha='center', fontsize=16, fontweight='bold', color='#1565C0',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#1565C0', alpha=0))

# ─── Animation ────────────────────────────────────────────────────────────────
# Each panel lights up in sequence, with arrows
TOTAL = 90

def update(frame):
    # 5 stages, ~15 frames each
    stage = frame / 15.0

    for i in range(5):
        if stage > i:
            t = min((stage - i) * 1.5, 1.0)
            ims[i].set_data(data[i])
            ims[i].set_alpha(t)
            title_txts[i].set_alpha(t)
            sub_txts[i].set_alpha(t)

    for i in range(4):
        if stage > i + 0.5:
            t = min((stage - i - 0.5) * 2, 1.0)
            arrow_txts[i].set_alpha(t)

    # Show main equation after all panels visible
    if stage > 4.5:
        t = min((stage - 4.5) * 2, 1.0)
        eq_main.set_alpha(t)
        eq_main.get_bbox_patch().set_alpha(t)

    # Zero-mode emphasis
    if stage > 5.2:
        # Highlight the center of ε_hat
        if not hasattr(update, '_zero_marked'):
            axs[3].plot(N//2, N//2, 'o', color='#D32F2F', ms=10, mew=2, mfc='none')
            axs[3].text(N//2 + 2, N//2 + 2, r'$\mathbf{k}=0$', color='#D32F2F',
                       fontsize=9, fontweight='bold')
            update._zero_marked = True

    return []

anim = FuncAnimation(fig, update, frames=TOTAL, interval=130, blit=False)
outpath = r'c:\Users\vfn333\Documents\GitHub\FFT_crystals\Presentation\weekly_gifs\slide4_fft_convolution.gif'
anim.save(outpath, writer='pillow', fps=8, dpi=130)
plt.close()
print(f'Saved {outpath}')
