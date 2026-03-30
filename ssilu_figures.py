"""
Generate SSiLU figures for backup slides in the presentation.

Author:  Nikolaos M. Matzakos
         nikmatz@aspete.gr  |  ORCID 0000-0001-8647-6082
License: MIT
Figure 1: SSiLU activation and derivative for several (alpha, beta) pairs
Figure 2: Derivative comparison — tanh' vs SiLU' vs SSiLU'
Figure 3: Trajectory-level delta illustration (why SSiLU keeps delta away from 0)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

x = np.linspace(-8, 8, 1000)

# ── Activation functions ──
def tanh_act(x):
    return np.tanh(x)

def silu(x):
    sig = 1.0 / (1.0 + np.exp(-x))
    return x * sig

def ssilu(x, alpha, beta):
    sig = 1.0 / (1.0 + np.exp(-beta * x))
    return alpha * beta * x * sig

# ── Derivatives (analytical) ──
def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

def silu_deriv(x):
    sig = 1.0 / (1.0 + np.exp(-x))
    return sig + x * sig * (1.0 - sig)

def ssilu_deriv(x, alpha, beta):
    sig = 1.0 / (1.0 + np.exp(-beta * x))
    return alpha * beta * (sig + beta * x * sig * (1.0 - sig))

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures") + os.sep
os.makedirs(outdir, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Figure 1: Derivative comparison — tanh' vs SiLU' vs SSiLU'
# This is the KEY figure: shows how tanh' → 0 fast, SiLU' → 1,
# and SSiLU' can be tuned to stay bounded away from 0.
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(7, 4))

ax.plot(x, tanh_deriv(x), color="#2166ac", linewidth=2.2, label=r"$\tanh^\prime(x)$")
ax.plot(x, silu_deriv(x), color="#d6604d", linewidth=2.2, label=r"$\mathrm{SiLU}^\prime(x)$")
ax.plot(x, ssilu_deriv(x, 1.0, 0.5), color="#1b7837", linewidth=2.2,
        linestyle="--", label=r"$\mathrm{SSiLU}^\prime(x;\,\alpha\!=\!1,\,\beta\!=\!0.5)$")
ax.plot(x, ssilu_deriv(x, 1.2, 0.7), color="#762a83", linewidth=2.2,
        linestyle="-.", label=r"$\mathrm{SSiLU}^\prime(x;\,\alpha\!=\!1.2,\,\beta\!=\!0.7)$")

ax.axhline(0, color="gray", linewidth=0.5, zorder=0)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$|\sigma^\prime(x)|$")
ax.set_title("Derivative comparison: activation sensitivity")
ax.legend(loc="upper left", framealpha=0.9)
ax.set_ylim(-0.05, 1.35)
ax.grid(True, alpha=0.3)

fig.savefig(outdir + "ssilu_deriv_comparison.pdf")
fig.savefig(outdir + "ssilu_deriv_comparison.png")
plt.close(fig)
print("✓ ssilu_deriv_comparison.pdf")

# ═══════════════════════════════════════════════════════════════
# Figure 2: Activation curves — tanh vs SiLU vs SSiLU
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(7, 4))

ax.plot(x, tanh_act(x), color="#2166ac", linewidth=2.2, label=r"$\tanh(x)$")
ax.plot(x, silu(x), color="#d6604d", linewidth=2.2, label=r"$\mathrm{SiLU}(x)$")
ax.plot(x, ssilu(x, 1.0, 0.5), color="#1b7837", linewidth=2.2,
        linestyle="--", label=r"$\mathrm{SSiLU}(x;\,\alpha\!=\!1,\,\beta\!=\!0.5)$")
ax.plot(x, ssilu(x, 1.2, 0.7), color="#762a83", linewidth=2.2,
        linestyle="-.", label=r"$\mathrm{SSiLU}(x;\,\alpha\!=\!1.2,\,\beta\!=\!0.7)$")

ax.axhline(0, color="gray", linewidth=0.5, zorder=0)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\sigma(x)$")
ax.set_title("Activation functions")
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.3)

fig.savefig(outdir + "ssilu_activation_comparison.pdf")
fig.savefig(outdir + "ssilu_activation_comparison.png")
plt.close(fig)
print("✓ ssilu_activation_comparison.pdf")

# ═══════════════════════════════════════════════════════════════
# Figure 3: "Saturation zone" — heatmap-style showing where
# |σ'| < threshold for each activation. Illustrates how tanh
# has a large dead zone vs SSiLU.
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

threshold = 0.1  # below this we consider "saturated"

for ax, (name, deriv_fn, color) in zip(axes, [
    (r"$\tanh$", tanh_deriv, "#2166ac"),
    (r"$\mathrm{SiLU}$", silu_deriv, "#d6604d"),
    (r"$\mathrm{SSiLU}$" + "\n" + r"$(\alpha\!=\!1.2,\,\beta\!=\!0.7)$",
     lambda x: ssilu_deriv(x, 1.2, 0.7), "#762a83"),
]):
    y = deriv_fn(x)
    ax.fill_between(x, 0, 1.4, where=(np.abs(y) < threshold),
                    color="red", alpha=0.15, label=r"$|\sigma^\prime| < 0.1$ (saturation)")
    ax.fill_between(x, 0, 1.4, where=(np.abs(y) >= threshold),
                    color="green", alpha=0.08, label=r"$|\sigma^\prime| \geq 0.1$ (active)")
    ax.plot(x, np.abs(y), color=color, linewidth=2)
    ax.axhline(threshold, color="red", linewidth=1, linestyle=":", alpha=0.7)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel(r"$x$")
    ax.set_ylim(0, 1.35)
    ax.grid(True, alpha=0.2)

axes[0].set_ylabel(r"$|\sigma^\prime(x)|$")
axes[0].legend(loc="upper right", fontsize=7, framealpha=0.9)

fig.suptitle(r"Saturation zones: where $|\sigma^\prime(x)| < 0.1$ (red) impairs Jacobian sensitivity",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(outdir + "ssilu_saturation_zones.pdf", bbox_inches="tight")
fig.savefig(outdir + "ssilu_saturation_zones.png", bbox_inches="tight")
plt.close(fig)
print("✓ ssilu_saturation_zones.pdf")

# ═══════════════════════════════════════════════════════════════
# Figure 4: Effect of α and β on SSiLU derivative
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Varying α (fixed β=1)
for alpha, ls in [(0.8, ":"), (1.0, "-"), (1.3, "--")]:
    ax1.plot(x, ssilu_deriv(x, alpha, 1.0), linewidth=2, linestyle=ls,
             label=rf"$\alpha={alpha}$")
ax1.set_title(r"Varying $\alpha$ (fixed $\beta=1$)")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$\sigma_M^\prime(x)$")
ax1.legend(framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 1.6)

# Varying β (fixed α=1)
for beta, ls in [(0.5, ":"), (1.0, "-"), (2.0, "--")]:
    ax2.plot(x, ssilu_deriv(x, 1.0, beta), linewidth=2, linestyle=ls,
             label=rf"$\beta={beta}$")
ax2.set_title(r"Varying $\beta$ (fixed $\alpha=1$)")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$\sigma_M^\prime(x)$")
ax2.legend(framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.1, 2.3)

fig.tight_layout()
fig.savefig(outdir + "ssilu_param_effects.pdf")
fig.savefig(outdir + "ssilu_param_effects.png")
plt.close(fig)
print("✓ ssilu_param_effects.pdf")

print("\nAll SSiLU figures generated.")
