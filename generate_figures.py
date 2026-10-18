import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import matplotlib.patches as patches

# Austere, classic academic style (clean, sober, publication-standard)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 10,
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ==============================================================================
# FIGURE 1: beta_temperature.pdf / beta_temperature.png
# ==============================================================================
def generate_beta_temperature():
    x = np.linspace(-1.75, 1.75, 600)
    # Double-well Energy: E(x) = x^4 - 2x^2 + 0.3x
    E = x**4 - 2 * x**2 + 0.3 * x
    
    betas = [0.0, 0.5, 3.0]
    titles = [
        r"$\beta = 0.0$ (High Temp.)",
        r"$\beta = 0.5$ (Moderate Temp.)",
        r"$\beta = 3.0$ (Low Temp.)"
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
    
    for ax, beta, title in zip(axes, betas, titles):
        # Calculate Boltzmann density p(x) = exp(-beta * E(x)) / Z
        if beta == 0.0:
            p = np.ones_like(x) / (x[-1] - x[0])
        else:
            unnorm = np.exp(-beta * (E - np.min(E)))
            Z = simpson(unnorm, x)
            p = unnorm / Z
        
        # Simple, sober curves: dashed gray/black for Energy, solid black for Density
        ax.plot(x, E, linestyle='--', color='#555555', linewidth=1.4, label=r'Energy $E(x)$')
        ax.plot(x, p, linestyle='-', color='#000000', linewidth=1.8, label=r'Density $p(x)$')
        
        # NO fill_between (completely clean/austere)
        ax.set_title(title, pad=8, color='#000000')
        ax.set_xlabel(r'$x$', labelpad=4)
        ax.set_xlim(-1.75, 1.75)
        ax.set_ylim(-1.55, 3.15)
        
        # Minimalist classic spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#000000')
        ax.spines['bottom'].set_color('#000000')
        ax.tick_params(direction='out', length=3.5, width=0.8)
        
    axes[0].set_ylabel(r'Value', labelpad=4)
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 0.97), frameon=True,
                   facecolor='white', framealpha=0.95, edgecolor='#E2E8F0', fontsize=9.5)
    
    plt.tight_layout()
    plt.savefig('beta_temperature.pdf')
    plt.savefig('beta_temperature.png')
    plt.close()
    print("Generated austere beta_temperature.pdf and beta_temperature.png")

# ==============================================================================
# FIGURE 2: heatmaps_IZ.pdf
# ==============================================================================
def generate_heatmaps_IZ():
    np.random.seed(42)
    d = 16
    
    # 1. Redundant representation
    A = np.random.randn(d, d) * 0.35
    C_redundant = np.dot(A, A.T)
    diag_inv = 1.0 / np.sqrt(np.diag(C_redundant))
    C_redundant = C_redundant * np.outer(diag_inv, diag_inv)
    np.fill_diagonal(C_redundant, 1.0)
    C_redundant[0:5, 0:5] += 0.32
    C_redundant[8:13, 8:13] += 0.36
    C_redundant = np.clip(C_redundant, -0.05, 1.0)
    np.fill_diagonal(C_redundant, 1.0)
    
    # 2. Barlow's Ideal
    noise = np.random.randn(d, d) * 0.018
    noise = (noise + noise.T) / 2.0
    C_ideal = np.eye(d) + noise
    np.fill_diagonal(C_ideal, 1.0)
    C_ideal = np.clip(C_ideal, 0.0, 1.0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    plt.subplots_adjust(left=0.08, right=0.88, wspace=0.45)
    
    cmap = 'magma'
    
    # Left: Redundant
    im1 = ax1.imshow(C_redundant, cmap=cmap, vmin=0, vmax=1, aspect='equal', interpolation='nearest')
    ax1.set_title("Redundant Representation\n(Correlated Features)", pad=8, fontsize=11, color='#000000')
    ax1.set_xlabel("Feature Dimension $j$", labelpad=5)
    ax1.set_ylabel("Feature Dimension $i$", labelpad=5)
    ax1.set_xticks([0, 5, 10, 15])
    ax1.set_yticks([0, 5, 10, 15])
    
    # Right: Barlow's Ideal
    im2 = ax2.imshow(C_ideal, cmap=cmap, vmin=0, vmax=1, aspect='equal', interpolation='nearest')
    ax2.set_title("Barlow's Ideal\n($\\mathcal{I}(Z) \\approx I_d$ Independence)", pad=8, fontsize=11, color='#000000')
    ax2.set_xlabel("Feature Dimension $j$", labelpad=5)
    ax2.set_xticks([0, 5, 10, 15])
    ax2.tick_params(left=False, labelleft=False)
    
    # Central arrow between the two plots
    arrow = patches.FancyArrowPatch((0.445, 0.48), (0.535, 0.48),
                                    transform=fig.transFigure,
                                    arrowstyle='-|>',
                                    mutation_scale=20,
                                    linewidth=1.8,
                                    color='#000000')
    fig.patches.append(arrow)
    fig.text(0.49, 0.55, "Barlow\nOptimization", ha='center', va='bottom',
             fontsize=9.5, color='#000000')
    
    # Colorbar on right
    cbar_ax = fig.add_axes([0.91, 0.16, 0.02, 0.68])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label("Correlation / Mutual Information", labelpad=6, fontsize=10)
    cbar.outline.set_linewidth(0.8)
    cbar.outline.set_edgecolor('#000000')
    
    plt.savefig('heatmaps_IZ.pdf')
    plt.close()
    print("Generated heatmaps_IZ.pdf")

# ==============================================================================
# FIGURE 3: cramer_wold.pdf
# ==============================================================================
def generate_cramer_wold():
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    plt.subplots_adjust(wspace=0.32, left=0.06, right=0.97)
    
    # Panel 1: 2D Gaussian density contour lines + projection vectors (NO contourf fill)
    ax1 = axes[0]
    
    delta = 0.02
    x = np.arange(-3.2, 3.2, delta)
    y = np.arange(-3.2, 3.2, delta)
    X, Y = np.meshgrid(x, y)
    Z = (1.0 / (2 * np.pi)) * np.exp(-0.5 * (X**2 + Y**2))
    
    # Pure contour lines in clean monochrome/dark slate (classic sober math illustration)
    levels = np.array([0.02, 0.05, 0.09, 0.13])
    cs = ax1.contour(X, Y, Z, levels=levels, colors='#444444', linewidths=0.9)
    
    # Direction 1: u1 = (1, 0)
    theta1 = 0.0
    u1 = np.array([np.cos(theta1), np.sin(theta1)])
    # Direction 2: u2 = (cos 50 deg, sin 50 deg)
    theta2 = np.radians(50)
    u2 = np.array([np.cos(theta2), np.sin(theta2)])
    
    # Dashed projection lines through origin
    line_len = 2.8
    ax1.plot([-line_len * u1[0], line_len * u1[0]], [-line_len * u1[1], line_len * u1[1]],
             linestyle=':', color='#777777', linewidth=1.0)
    ax1.plot([-line_len * u2[0], line_len * u2[0]], [-line_len * u2[1], line_len * u2[1]],
             linestyle=':', color='#777777', linewidth=1.0)
    
    # Vectors (simple, clean arrows)
    vec_len = 2.1
    ax1.annotate('', xy=(vec_len * u1[0], vec_len * u1[1]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.5", lw=1.8, color='#000000'))
    ax1.annotate('', xy=(vec_len * u2[0], vec_len * u2[1]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.5", lw=1.8, color='#000000'))
    
    ax1.text(vec_len * u1[0] + 0.15, vec_len * u1[1] + 0.08, r'$\mathbf{u}_1$',
             color='#000000', fontsize=12)
    ax1.text(vec_len * u2[0] + 0.1, vec_len * u2[1] + 0.1, r'$\mathbf{u}_2$',
             color='#000000', fontsize=12)
    
    ax1.set_title(r'Latent Density Contours $p(z_1, z_2)$', pad=8, color='#000000')
    ax1.set_xlabel(r'$z_1$', labelpad=4)
    ax1.set_ylabel(r'$z_2$', labelpad=4)
    ax1.set_xlim(-3.1, 3.1)
    ax1.set_ylim(-3.1, 3.1)
    ax1.axhline(0, color='#999999', linewidth=0.5, linestyle='-')
    ax1.axvline(0, color='#999999', linewidth=0.5, linestyle='-')
    ax1.set_aspect('equal')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#000000')
    ax1.spines['bottom'].set_color('#000000')
    ax1.tick_params(direction='out', length=3.5, width=0.8)
    
    # Panel 2: 1D projection along u1 (pure line, NO fill_between)
    ax2 = axes[1]
    h = np.linspace(-3.5, 3.5, 400)
    p_h = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * h**2)
    
    ax2.plot(h, p_h, color='#000000', linewidth=1.6)
    ax2.set_title(r'Marginal Density along $\mathbf{u}_1$', pad=8, color='#000000')
    ax2.set_xlabel(r'Projected Variable $h_1 = \mathbf{u}_1^\top \mathbf{z} \sim \mathcal{N}(0, 1)$', labelpad=4)
    ax2.set_ylabel(r'Density $p(h_1)$', labelpad=4)
    ax2.set_xlim(-3.4, 3.4)
    ax2.set_ylim(0, 0.44)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#000000')
    ax2.spines['bottom'].set_color('#000000')
    ax2.tick_params(direction='out', length=3.5, width=0.8)
    
    # Panel 3: 1D projection along u2 (pure line, NO fill_between)
    ax3 = axes[2]
    ax3.plot(h, p_h, color='#000000', linewidth=1.6)
    ax3.set_title(r'Marginal Density along $\mathbf{u}_2$', pad=8, color='#000000')
    ax3.set_xlabel(r'Projected Variable $h_2 = \mathbf{u}_2^\top \mathbf{z} \sim \mathcal{N}(0, 1)$', labelpad=4)
    ax3.set_ylabel(r'Density $p(h_2)$', labelpad=4)
    ax3.set_xlim(-3.4, 3.4)
    ax3.set_ylim(0, 0.44)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color('#000000')
    ax3.spines['bottom'].set_color('#000000')
    ax3.tick_params(direction='out', length=3.5, width=0.8)
    
    plt.savefig('cramer_wold.pdf')
    plt.close()
    print("Generated austere cramer_wold.pdf")

if __name__ == '__main__':
    generate_beta_temperature()
    generate_heatmaps_IZ()
    generate_cramer_wold()
