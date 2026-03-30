"""
numerical_experiment.py
─────────────────────────────────────────────────────────────────────────────
Author:  Nikolaos M. Matzakos
         nikmatz@aspete.gr  |  ORCID 0000-0001-8647-6082
License: MIT

Numerical verification of the main results in:

  N. M. Matzakos, "Activation Saturation and Floquet Spectrum
  Collapse in Neural ODEs", arXiv preprint, 2026.

Experiments
───────────
A  Jacobian attenuation (Theorem thm:main, §3).
   Architecture:  f(h; s) = W₂ · tanh(s·(W₁·h + b₁)) + b₂
   Verification:  ‖Df_θ(h₀; s)‖₂  ≤  C_W(s) · δ(h₀, s)

B  Geometry of Floquet–Liouville obstruction (Theorem thm:liouville, §4).
   The bound |ln det M_γ| ≤ d·C(U)·T holds only when the orbit
   is in a region without zero pre-activations (zero crossings).

C  Phase portraits of ḣ = f_θ(h; s) for mild / moderate / strong saturation.
   Separate 256-unit MLP trained WITH bias (frozen b₁, row-norm
   projection) so that δ → 0 as s → ∞.

D  Reference: exact Stuart–Landau monodromy vs bound.

Dependencies:  numpy, scipy, matplotlib
Execution:    python numerical_experiment.py
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SEED   = 42
HIDDEN = 32
d      = 2
T_SL   = 2.0 * np.pi
SL_DET = np.exp(-4.0 * np.pi)   # ≈ 3.487e-6
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  tanh-MLP
# ─────────────────────────────────────────────────────────────────────────────
class TanhMLP:
    """
    Single-layer tanh-MLP  f: ℝ² → ℝ²

        f(h; s) = W₂ · tanh(s·(W₁·h + b₁)) + b₂

    Weights freeze after .fit(). The parameter s controls saturation.
    """

    def __init__(self, n_hidden=HIDDEN):
        self.n_hidden = n_hidden
        self.W1 = np.random.randn(n_hidden, d) * 0.5
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(d, n_hidden) * 0.5
        self.b2 = np.zeros(d)

    def __call__(self, h, s=1.0):
        a = s * (self.W1 @ h + self.b1)
        return self.W2 @ np.tanh(a) + self.b2

    def jacobian(self, h, s=1.0):
        """Jacobian: Df(h;s) = W₂ · diag(sech²(s·(W₁h+b₁))) · s·W₁."""
        a = s * (self.W1 @ h + self.b1)
        D = np.diag(1.0 - np.tanh(a)**2)
        return self.W2 @ D @ (s * self.W1)

    def delta(self, h, s=1.0):
        """Saturation level: δ = max_i sech²(s·(W₁h+b₁)_i)."""
        a = s * (self.W1 @ h + self.b1)
        return float(np.max(1.0 - np.tanh(a)**2))

    def CW(self, s=1.0):
        """Weight constant: C_W(s) = s·‖W₁‖₂·‖W₂‖₂."""
        return s * np.linalg.norm(self.W1, ord=2) * np.linalg.norm(self.W2, ord=2)

    def fit(self, n_iters=7000, lr=0.012, verbose=True, bias_const=0.0,
            N_data=1500, use_adam=False):
        """
        Fit MSE to f_SL(x,y) = (x−y−xr², x+y−yr²), 0.1 ≤ r ≤ 2.

        bias_const : if > 0, NON-TRAINABLE constant is added to b₁
                     during forward pass. b₁ FREEZES (not updated)
                     so the optimizer doesn't undo the shift.
                     After training, the constant is integrated into b₁.
        use_adam   : if True, use Adam optimizer.
        """
        N  = N_data
        r  = np.random.uniform(0.1, 2.0, N)
        th = np.random.uniform(0, 2*np.pi, N)
        H  = np.column_stack([r*np.cos(th), r*np.sin(th)])
        r2 = r**2
        F  = np.column_stack([H[:,0]-H[:,1]-H[:,0]*r2,
                               H[:,0]+H[:,1]-H[:,1]*r2])

        # Which parameters are trained
        freeze_b1 = (bias_const > 0)
        if freeze_b1:
            params = [self.W1, self.W2, self.b2]  # b₁ frozen
        else:
            params = [self.W1, self.b1, self.W2, self.b2]

        # Adam state
        if use_adam:
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            m = [np.zeros_like(p) for p in params]
            v = [np.zeros_like(p) for p in params]

        best, bloss = None, np.inf
        for it in range(n_iters):
            A   = H @ self.W1.T + self.b1 + bias_const
            Z   = np.tanh(A)
            Fp  = Z @ self.W2.T + self.b2
            err = Fp - F
            loss = 0.5 * np.mean(err**2)

            # Gradients (partial derivatives)
            g   = err / N
            gW2 = g.T @ Z
            gb2 = g.sum(0)
            dZ  = g @ self.W2 * (1.0 - Z**2)
            gW1 = dZ.T @ H

            if freeze_b1:
                grads = [gW1, gW2, gb2]
            else:
                gb1 = dZ.sum(0)
                grads = [gW1, gb1, gW2, gb2]

            if use_adam:
                for i, (grad, p) in enumerate(zip(grads, params)):
                    m[i] = beta1 * m[i] + (1 - beta1) * grad
                    v[i] = beta2 * v[i] + (1 - beta2) * grad**2
                    mh = m[i] / (1 - beta1**(it+1))
                    vh = v[i] / (1 - beta2**(it+1))
                    p -= lr * mh / (np.sqrt(vh) + eps)
            else:
                for grad, p in zip(grads, params):
                    p -= lr * grad

            # Projection: keep ‖W₁_j‖ ≤ bias_const - margin
            # so that on the unit circle (‖h‖=1):
            #   a_j = W₁_j·h + bias_const ≥ bias_const - ‖W₁_j‖ ≥ margin > 0
            # margin ≥ 0.5 ensures sech²(s·margin) → 0 as s increases
            if bias_const > 0:
                margin = 0.5
                max_norm = bias_const - margin
                rnorms = np.linalg.norm(self.W1, axis=1, keepdims=True)
                mask   = rnorms > max_norm
                if mask.any():
                    scale = np.where(mask, max_norm / rnorms, 1.0)
                    self.W1 *= scale

            if loss < bloss:
                bloss = loss
                best  = (self.W1.copy(), self.b1.copy(),
                         self.W2.copy(), self.b2.copy())
            if verbose and it % 2000 == 0:
                print(f"      iter {it:5d}: MSE = {loss:.6f}")
        self.W1, self.b1, self.W2, self.b2 = best
        # Incorporate the non-trainable constant into b₁
        self.b1 += bias_const
        if verbose:
            print(f"      Final MSE = {bloss:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def circle_orbit(R, n=400):
    """Points on circle of radius R: array (n, 2)."""
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([R*np.cos(t), R*np.sin(t)])


def C_orbit(mlp, orbit, s):
    """C(U) = max_{h ∈ orbit} ‖Df_θ(h; s)‖₂."""
    return max(np.linalg.norm(mlp.jacobian(h, s), ord=2) for h in orbit)


def delta_orbit(mlp, orbit, s):
    """δ(U,s) = max_{h ∈ orbit} max_i sech²(s·a_i(h))."""
    return max(mlp.delta(h, s) for h in orbit)


def lndetM_laj(mlp, orbit, s, T):
    """
    Liouville–Abel–Jacobi integral along a closed curve:
        ln det M = ∫₀ᵀ Tr(Df_θ(γ(t); s)) dt    (midpoint rule).
    orbit: (n, 2) — n evenly spaced points on the curve.
    """
    traces = np.array([np.trace(mlp.jacobian(h, s)) for h in orbit])
    return traces.mean() * T


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Experiment A — Jacobian attenuation (Theorem thm:main, §3)
# ─────────────────────────────────────────────────────────────────────────────
def experiment_A(mlp, h0=None, s_values=None):
    if h0 is None:
        h0 = np.array([0.8, 0.4])
    if s_values is None:
        s_values = np.logspace(-0.5, 1.8, 60)

    norms, bounds, deltas, CWs = [], [], [], []
    orb_max, orb_bnd = [], []
    gamma1 = circle_orbit(1.0)

    for s in s_values:
        J = mlp.jacobian(h0, s)
        norms.append(np.linalg.norm(J, ord=2))
        deltas.append(mlp.delta(h0, s))
        CWs.append(mlp.CW(s))
        bounds.append(CWs[-1] * deltas[-1])
        orb_max.append(C_orbit(mlp, gamma1, s))
        orb_bnd.append(max(mlp.CW(s) * mlp.delta(h, s) for h in gamma1))

    return dict(s=s_values,
                actual_norm=np.array(norms),
                bound=np.array(bounds),
                delta=np.array(deltas),
                CW=np.array(CWs),
                orbit_norm_max=np.array(orb_max),
                orbit_bound_max=np.array(orb_bnd))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Experiment B — Floquet–Liouville obstruction (Theorem thm:liouville, §4)
# ─────────────────────────────────────────────────────────────────────────────
def experiment_B(mlp, s_values=None, T=T_SL, n_pts=500, bias_offsets=None):
    """
    Compares C(U,s) on the unit-circle orbit for TWO versions of the network:

    · UNBIASED  (b₁ ≈ 0, as trained): pre-activation a_j(t) = W₁_j·γ(t)
      is a pure sinusoid that crosses zero ⟹ sech²(s·0)=1 always
      ⟹  C(U,s) does NOT decay; the Floquet obstruction is NOT active.

    · BIASED  (b₁ shifted by +c): a_j(t) = W₁_j·γ(t) + (b₁_j + c)
      If c > max_j ‖(W₁_{j1}, W₁_{j2})‖₂  (radius of sinusoid),
      then a_j(t) > 0 for all t ⟹ sech²(s·a_j) → 0 exponentially
      ⟹  C(U,s) → 0  ⟹  Floquet obstruction IS active.

    This models a Neural ODE whose operating point is well inside the
    saturation zone (large pre-activations on the limit cycle).

    Physical interpretation: the obstruction is NOT generic — it applies to
    networks whose periodic orbit passes through the genuinely saturated
    region of the state space.
    """
    if s_values is None:
        s_values = np.logspace(-0.3, 1.4, 40)
    if bias_offsets is None:
        # Row norms of W1: the bias needs to exceed these to prevent zero crossings
        row_norms = np.linalg.norm(mlp.W1, axis=1)    # (HIDDEN,)
        c_min     = row_norms.max()                    # minimum effective bias
        bias_offsets = [0.0, 1.5 * c_min, 3.0 * c_min, 6.0 * c_min]

    orbit1 = circle_orbit(1.0, n_pts)

    results = {}
    for c in bias_offsets:
        # Temporarily shift b1
        b1_save = mlp.b1.copy()
        mlp.b1  = b1_save + c

        C_vals, bound_vals = [], []
        for s in s_values:
            C_vals.append(C_orbit(mlp, orbit1, s))
            bound_vals.append(d * C_vals[-1] * T)

        mlp.b1 = b1_save      # restore

        # Check if orbit avoids zero crossings (all pre-activations > 0)
        # c must exceed c_min = max row-norm of W1 to guarantee a_j > 0
        row_norms_W1 = np.linalg.norm(mlp.W1, axis=1)
        c_min_val = row_norms_W1.max()
        label = (f'$c={c:.1f}$' if c > 0 else '$c=0$ (unbiased)')

        results[c] = dict(s=s_values,
                          C=np.array(C_vals),
                          bound=np.array(bound_vals),
                          label=label,
                          bias=c,
                          avoids_zero=(c >= c_min_val))

    return results, bias_offsets


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Experiment D — Reference verification Stuart–Landau (Theorem thm:liouville)
# ─────────────────────────────────────────────────────────────────────────────
def experiment_D():
    """
    For the exact Stuart–Landau system, verify numerically:
      (a) LAJ identity: ∫ Tr(Df_SL(γ(t))) dt = −4π  (exact: det M = e^{−4π})
      (b) Floquet–Liouville bound:  d · C_SL(U) · T  ≥  |ln det M_SL| = 4π
    """
    def f_SL(h):
        x, y = h
        r2 = x**2 + y**2
        return np.array([x - y - x*r2, x + y - y*r2])

    def Df_SL(h):
        x, y = h
        r2 = x**2 + y**2
        return np.array([[1 - 3*x**2 - y**2, -1 - 2*x*y],
                         [1 - 2*x*y,           1 - x**2 - 3*y**2]])

    gamma1 = circle_orbit(1.0, 1000)
    T = T_SL

    # LAJ integral (numerical)
    traces = np.array([np.trace(Df_SL(h)) for h in gamma1])
    lndetM_num = traces.mean() * T

    # Analytical: Tr(Df_SL(cos t, sin t)) = (1-3cos²t-sin²t) + (1-cos²t-3sin²t)
    #                                      = 2 - 4 = −2  (constant!)
    lndetM_exact = -4 * np.pi  # = T * (-2)

    # C_SL(U): max spectral norm of Df_SL on unit circle
    J_norms = [np.linalg.norm(Df_SL(h), ord=2) for h in gamma1]
    C_SL    = max(J_norms)
    bound_SL = d * C_SL * T

    return dict(
        lndetM_numerical = lndetM_num,
        lndetM_exact     = lndetM_exact,
        detM_numerical   = np.exp(lndetM_num),
        detM_exact       = SL_DET,
        C_SL             = C_SL,
        bound            = bound_SL,
        Tr_constant      = traces.mean(),  # should be ≈ -2
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Helper function for phase portrait
# ─────────────────────────────────────────────────────────────────────────────
def phase_portrait(rhs_fn, ax, title, T_int=30.0, xlim=2.5):
    """
    Generic phase portrait.  rhs_fn(t, h) → dh/dt.
    Integration from 8 angles × 4 radii.
    Dots with black outline: initial conditions.
    """
    n_angles = 8
    angles   = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    radii    = [0.3, 0.7, 1.3, 1.8]
    palette  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    esc_r = xlim + 0.5

    for j, th in enumerate(angles):
        for ri, r0 in enumerate(radii):
            h0 = np.array([r0*np.cos(th), r0*np.sin(th)])

            def escape(t, h):
                return esc_r - np.linalg.norm(h)
            escape.terminal  = True
            escape.direction = -1

            sol = solve_ivp(rhs_fn, [0, T_int], h0,
                            method='RK45', rtol=1e-9, atol=1e-11,
                            max_step=0.02, events=escape)
            xs, ys = sol.y[0], sol.y[1]
            # Orbits near limit cycle: thicker
            lw  = 1.2 if r0 in (0.7, 1.3) else 0.7
            alp = 0.85 if r0 in (0.7, 1.3) else 0.5
            ax.plot(xs, ys, lw=lw, color=palette[j], alpha=alp)

            # Initial condition indicator
            ax.plot(*h0, 'o', ms=4.0, color=palette[j], zorder=7,
                    markeredgecolor='k', markeredgewidth=0.6)

    # Reference: Stuart–Landau limit cycle (unit circle)
    phi = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(phi), np.sin(phi), 'k--', lw=1.6,
            label='SL limit cycle', zorder=3)

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-xlim, xlim)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('$h_1$', fontsize=10)
    ax.set_ylabel('$h_2$', fontsize=10)
    ax.legend(fontsize=6.5, loc='lower right',
              framealpha=0.85, edgecolor='gray')
    ax.grid(True, alpha=0.2, linewidth=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Plot generation
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {1: '#1f77b4', 2: '#ff7f0e', 5: '#2ca02c', 10: '#d62728'}
LSMAP  = {1: '-o',      2: '-s',      5: '-^',      10: '-D'}


def make_plots(res_A, res_B, res_D, mlp, outdir):
    """res_B: dict keyed by bias offset value, each value is a data dict."""
    plt.rcParams.update({
        'font.family':    'serif',
        'font.size':      11,
        'axes.labelsize': 12,
        'axes.titlesize': 11,
        'figure.dpi':     150,
        'lines.linewidth': 2.0,
        'legend.fontsize': 9,
        'grid.alpha':      0.3,
    })
    os.makedirs(outdir, exist_ok=True)

    # ═════════════════════════════════════════════════════════════════════
    # Figure 1: Experiment A — Jacobian attenuation
    # ═════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2    = axes1
    s = res_A['s']

    # Left panel: norm vs bound
    ax1.semilogy(s, res_A['actual_norm'], 'b-',  lw=2.2,
                 label=r'$\|Df_\theta(h_0;\,s)\|_2$')
    ax1.semilogy(s, res_A['bound'],       'r--', lw=2.2,
                 label=r'$C_W(s)\cdot\delta(s)$  (upper bound)')
    ax1.fill_between(s, res_A['actual_norm'], res_A['bound'],
                     alpha=0.10, color='red', label='gap')
    ax1.set_xlabel(r'Pre-activation scale $s$')
    ax1.set_ylabel(r'$\|Df_\theta\|_2$  (log scale)')
    ax1.set_title(r'(a)  $\|Df_\theta\| \leq C_W \cdot \delta$  (Thm.\ main)',
                  fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, which='both', alpha=0.25)

    # Right panel: bound decomposition
    ax2.semilogy(s, res_A['delta'], 'g-',  lw=2.2,
                 label=r'$\delta(s)$  (saturation)')
    ax2.semilogy(s, res_A['CW'],    'm--', lw=2.2,
                 label=r'$C_W(s) = s\,\|W_1\|\,\|W_2\|$')
    ax2.semilogy(s, res_A['bound'], 'r-',  lw=1.5, alpha=0.6,
                 label=r'$C_W \cdot \delta$')
    ax2.set_xlabel(r'Pre-activation scale $s$')
    ax2.set_ylabel('Value  (log scale)')
    ax2.set_title(r'(b)  Bound decomposition: $\delta$ decays, $C_W$ grows',
                  fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, which='both', alpha=0.25)

    fig1.suptitle(
        'Illustration A \u2014 Jacobian Attenuation',
        fontsize=13, fontweight='bold',
    )
    fig1.tight_layout()
    for ext in ('pdf', 'png'):
        p = os.path.join(outdir, f'exp_A_jacobian.{ext}')
        fig1.savefig(p, bbox_inches='tight', dpi=200)
        print(f'  Saved: {p}')
    plt.close(fig1)

    # ═════════════════════════════════════════════════════════════════════
    # Figure 2: Experiment B — Floquet obstruction: biased vs unbiased
    # Single panel: bound d·C(U)·T as function of s
    # ═════════════════════════════════════════════════════════════════════
    fig2, ax_b = plt.subplots(figsize=(8, 5))
    bias_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    lss         = ['-o', '-s', '-^', '-D']

    for idx, (c_val, data) in enumerate(res_B.items()):
        col = bias_colors[idx % len(bias_colors)]
        ls  = lss[idx % len(lss)]
        ax_b.semilogy(data['s'], data['bound'], ls, ms=3, color=col,
                      lw=1.8, label=data['label'])

    ax_b.axhline(4*np.pi, color='k', ls=':', lw=1.5,
                 label=r'$|\ln\det M_{SL}| = 4\pi$')
    ax_b.set_xlabel(r'Pre-activation scale $s$', fontsize=12)
    ax_b.set_ylabel(r'Floquet bound  $d \cdot C(U,s) \cdot T$  (log scale)',
                    fontsize=12)
    ax_b.set_title(
        'Illustration B \u2014 Floquet\u2013Liouville Obstruction (Floquet\u2013Liouville)',
        fontsize=13, fontweight='bold')
    ax_b.legend(title='Bias offset $c$', fontsize=9, title_fontsize=10)
    ax_b.grid(True, which='both', alpha=0.25)
    ax_b.text(0.97, 0.97,
              '$c = 0$: zero-crossings on orbit\n'
              r'$\Rightarrow\;\delta = 1\;\Rightarrow$ bound grows with $s$'
              '\n\n'
              '$c > 0$: all pre-activations $> 0$\n'
              r'$\Rightarrow\;\delta \to 0\;\Rightarrow$ bound $\to 0$',
              transform=ax_b.transAxes, ha='right', va='top', fontsize=9,
              bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow',
                        ec='orange', alpha=0.95))

    fig2.tight_layout()
    for ext in ('pdf', 'png'):
        p = os.path.join(outdir, f'exp_B_floquet.{ext}')
        fig2.savefig(p, bbox_inches='tight', dpi=200)
        print(f'  Saved: {p}')
    plt.close(fig2)

    # ═════════════════════════════════════════════════════════════════════
    # Figure 3: Experiment D — LAJ verification Stuart–Landau
    # Two panels: (a) bar chart comparison, (b) table of values
    # ═════════════════════════════════════════════════════════════════════
    c = res_D
    fig3, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                         gridspec_kw={'width_ratios': [3, 2]})

    # Left panel: bar chart — numerical vs exact
    labels = [r'$\mathrm{Tr}(Df_{SL})$',
              r'$\ln\det M_\gamma$',
              r'$\det M_\gamma$  ($\times 10^{6}$)']
    numerical = [c['Tr_constant'],
                 c['lndetM_numerical'],
                 c['detM_numerical'] * 1e6]
    exact     = [-2.0,
                 c['lndetM_exact'],
                 c['detM_exact'] * 1e6]

    x_pos = np.arange(len(labels))
    w = 0.35
    bars1 = ax_d1.bar(x_pos - w/2, numerical, w, label='Numerical',
                      color='steelblue', edgecolor='k', linewidth=0.5)
    bars2 = ax_d1.bar(x_pos + w/2, exact, w, label='Exact',
                      color='lightsalmon', edgecolor='k', linewidth=0.5)
    ax_d1.set_xticks(x_pos)
    ax_d1.set_xticklabels(labels, fontsize=10)
    ax_d1.set_ylabel('Value')
    ax_d1.set_title('(a)  LAJ identity verification', fontsize=11)
    ax_d1.legend(fontsize=9)
    ax_d1.grid(axis='y', alpha=0.25)
    ax_d1.axhline(0, color='k', lw=0.5)

    # Right panel: table of values
    ax_d2.axis('off')
    table_data = [
        [r'Tr$(Df_{SL})$', f'{c["Tr_constant"]:.5f}', '$-2$'],
        [r'$\ln\det M$', f'{c["lndetM_numerical"]:.5f}',
         f'$-4\\pi = {c["lndetM_exact"]:.5f}$'],
        [r'$\det M$', f'{c["detM_numerical"]:.3e}',
         f'$e^{{-4\\pi}} = {c["detM_exact"]:.3e}$'],
        [r'Floquet bound', f'{c["bound"]:.2f}',
         f'$\\geq 4\\pi = {4*np.pi:.2f}$  '],
    ]
    tbl = ax_d2.table(cellText=table_data,
                      colLabels=['Quantity', 'Numerical', 'Exact / Bound'],
                      loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)
    # Color header
    for j in range(3):
        tbl[0, j].set_facecolor('#4682B4')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    ax_d2.set_title('(b)  Numerical vs exact values', fontsize=11)

    fig3.suptitle(
        'Illustration D \u2014 Stuart\u2013Landau reference: '
        'LAJ identity and Floquet\u2013Liouville bound',
        fontsize=13, fontweight='bold')
    fig3.tight_layout()
    for ext in ('pdf', 'png'):
        p = os.path.join(outdir, f'exp_D_stuart_landau.{ext}')
        fig3.savefig(p, bbox_inches='tight', dpi=200)
        print(f'  Saved: {p}')
    plt.close(fig3)

    # ═════════════════════════════════════════════════════════════════════
    # Figure 4: Phase portraits — 256-unit MLP trained WITH bias
    # ═════════════════════════════════════════════════════════════════════
    #
    # Separate wider network (256 neurons) trained WITH
    # non-trainable constant bias, frozen b₁, and row-norm
    # projection. This ensures:
    #   (a) good approximation of SL at s=1
    #   (b) all pre-activations > 0 on orbit → δ → 0 as s
    #
    # Panel 0: exact Stuart–Landau (reference).
    # Panels 1–3: biased MLP at s = 1, 4, 15 with δ and ln det M.

    print('\n  [Phase portraits] Training 256-unit MLP with bias …')
    bias_c = 2.5
    np.random.seed(SEED + 1)
    mlp_wide = TanhMLP(n_hidden=256)
    mlp_wide.fit(n_iters=40000, lr=0.001, verbose=True,
                 bias_const=bias_c, N_data=3000, use_adam=True)

    # Check that pre-activations are positive on unit circle
    orb_check = circle_orbit(1.0, 500)
    min_pa = min(
        (mlp_wide.W1 @ h + mlp_wide.b1).min() for h in orb_check
    )
    print(f'  bias={bias_c:.3f}, min pre-act on orbit={min_pa:.4f} '
          f'{"✓>0" if min_pa > 0 else "✗ PROBLEM"}')

    def f_SL(t, h):
        x, y = h
        r2 = x**2 + y**2
        return np.array([x - y - x*r2, x + y - y*r2])

    # Exact Floquet integral of SL
    lndetM_exact = -4.0 * np.pi

    # s = 1 (mild saturation), 4 (moderate), 15 (strong)
    # Larger range of s → more visible gradual loss of stability
    s_vals  = [1.0, 4.0, 15.0]
    # Larger T_int for high s so slow dynamics are visible
    T_ints  = [25.0, 30.0, 40.0]
    fig4, axes4 = plt.subplots(1, 4, figsize=(18.0, 4.8))

    # Panel 0: exact Stuart–Landau
    phase_portrait(f_SL, axes4[0],
                   'Stuart\u2013Landau (exact)',
                   T_int=25.0)
    ann_sl = (f'$\\ln\\det M = {lndetM_exact:.2f}$\n'
              f'$\\det M = {np.exp(lndetM_exact):.1e}$')
    axes4[0].text(0.03, 0.97, ann_sl, transform=axes4[0].transAxes,
                  fontsize=8, ha='left', va='top',
                  bbox=dict(boxstyle='round,pad=0.3',
                            fc='white', ec='gray', alpha=0.92))

    # Panels 1–3: biased MLP at increasing s
    sat_labels = ['mild', 'moderate', 'strong']
    for ax, sv, Ti, slbl in zip(axes4[1:], s_vals, T_ints, sat_labels):
        phase_portrait(lambda t, h, _s=sv: mlp_wide(h, _s),
                       ax, '', T_int=Ti)

        # Compute δ and actual Floquet integral
        dv     = delta_orbit(mlp_wide, orb_check, sv)
        lndetM = lndetM_laj(mlp_wide, orb_check, sv, T_SL)
        detM   = np.exp(lndetM)

        ax.set_title(f'Neural ODE,  $s = {sv:.0f}$  ({slbl})',
                     fontsize=10.5, fontweight='bold')
        ann = (f'$\\delta = {dv:.4f}$\n'
               f'$\\ln\\det M = {lndetM:.2f}$\n'
               f'$\\det M = {detM:.2e}$')
        ax.text(0.03, 0.97, ann, transform=ax.transAxes,
                fontsize=8, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.3',
                          fc='white', ec='darkorange', alpha=0.92))

    fig4.suptitle(
        u'Illustration C \u2014 Phase portraits:  '
        r'increasing $s$ drives $\delta\to 0$, '
        r'$\det M_\gamma \to 1$  '
        u'(Floquet\u2013Liouville obstruction)',
        fontsize=12, fontweight='bold', y=1.01,
    )
    fig4.tight_layout()
    for ext in ('pdf', 'png'):
        p = os.path.join(outdir, f'exp_C_phase.{ext}')
        fig4.savefig(p, bbox_inches='tight', dpi=200)
        print(f'  Saved: {p}')
    plt.close(fig4)

    # ═════════════════════════════════════════════════════════════════════
    # Figure 5: Summary panel (A + B + D in one figure)
    # ═════════════════════════════════════════════════════════════════════
    fig5 = plt.figure(figsize=(14, 10))
    gs   = GridSpec(2, 3, fig5, hspace=0.45, wspace=0.40)

    # ── A: Jacobian attenuation ───────────────────────────────────────────
    ax_a1 = fig5.add_subplot(gs[0, :2])
    ax_a1.semilogy(res_A['s'], res_A['actual_norm'], 'b-', lw=2,
                   label=r'$\|Df_\theta\|_2$')
    ax_a1.semilogy(res_A['s'], res_A['bound'],       'r--',lw=2,
                   label=r'$C_W \cdot \delta$  (upper bound)')
    ax_a1.fill_between(res_A['s'], res_A['actual_norm'], res_A['bound'],
                       alpha=0.08, color='red')
    ax_a1.set_xlabel(r'Pre-activation scale $s$')
    ax_a1.set_ylabel(r'$\|Df_\theta\|_2$  (log)')
    ax_a1.set_title('(A)  Jacobian Attenuation  (Thm.\ main)', fontsize=11)
    ax_a1.legend(fontsize=9); ax_a1.grid(True, which='both', alpha=0.25)

    ax_a2 = fig5.add_subplot(gs[0, 2])
    ax_a2.semilogy(res_A['s'], res_A['delta'], 'g-', lw=2,
                   label=r'$\delta(s)$')
    ax_a2.semilogy(res_A['s'], res_A['CW'],    'm--',lw=2,
                   label=r'$C_W(s)$')
    ax_a2.semilogy(res_A['s'], res_A['bound'], 'r-', lw=1.5, alpha=0.6,
                   label=r'$C_W \cdot \delta$')
    ax_a2.set_xlabel(r'$s$')
    ax_a2.set_title('Bound decomposition', fontsize=10)
    ax_a2.legend(fontsize=8); ax_a2.grid(True, which='both', alpha=0.25)

    # ── B: Floquet obstruction ────────────────────────────────────────────
    bias_colors_sum = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ax_b = fig5.add_subplot(gs[1, :2])
    for idx, (c_val, data) in enumerate(res_B.items()):
        col = bias_colors_sum[idx % 4]
        ax_b.semilogy(data['s'], data['bound'], '-', ms=3, color=col,
                      lw=1.8, label=data['label'])
    ax_b.axhline(4*np.pi, color='k', ls=':', lw=1.2,
                 label=r'$4\pi$')
    ax_b.set_xlabel(r'Pre-activation scale $s$')
    ax_b.set_ylabel(r'Floquet bound  $d \cdot C(U) \cdot T$')
    ax_b.set_title(u'(B)  Floquet\u2013Liouville Obstruction',
                   fontsize=11)
    ax_b.legend(title='Bias $c$', fontsize=8); ax_b.grid(True, which='both',
                                                          alpha=0.25)

    # ── D: Stuart–Landau summary ──────────────────────────────────────────
    ax_d = fig5.add_subplot(gs[1, 2])
    ax_d.axis('off')
    c = res_D
    # Table of values instead of large text block
    rows = [
        [r'Tr$(Df_{SL})$', f'{c["Tr_constant"]:.4f}', '$-2$'],
        [r'$\ln\det M$', f'{c["lndetM_numerical"]:.4f}',
         f'${c["lndetM_exact"]:.4f}$'],
        [r'$\det M$', f'{c["detM_numerical"]:.2e}',
         f'${c["detM_exact"]:.2e}$'],
        ['Bound', f'{c["bound"]:.2f}',
         f'$\\geq {4*np.pi:.2f}$'],
    ]
    tbl = ax_d.table(cellText=rows,
                     colLabels=['', 'Num.', 'Exact'],
                     loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)
    for j in range(3):
        tbl[0, j].set_facecolor('#4682B4')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    ax_d.set_title('(D)  Stuart\u2013Landau reference', fontsize=10)

    fig5.suptitle(
        u'Summary \u2014 Numerical verification of Theorems 1 and 2\n'
        r'tanh-MLP $\mathbb{R}^2\!\to\![32]\!\to\!\mathbb{R}^2$, '
        u'trained to Stuart\u2013Landau oscillator',
        fontsize=12, fontweight='bold', y=1.01,
    )
    for ext in ('pdf', 'png'):
        p = os.path.join(outdir, f'summary_panel.{ext}')
        fig5.savefig(p, bbox_inches='tight', dpi=200)
        print(f'  Saved: {p}')
    plt.close(fig5)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main program
# ─────────────────────────────────────────────────────────────────────────────
def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    sep    = '═' * 62
    print(sep)
    print('  Saturation Attenuation & Floquet–Liouville Obstruction')
    print('  Numerical Verification')
    print(sep)

    # ── 1a. Train unbiased MLP (for Experiments A, B, D) ─────────
    print('\n[1a] Training UNBIASED tanh-MLP on Stuart–Landau …')
    np.random.seed(SEED)
    mlp = TanhMLP()
    mlp.fit(n_iters=7000, verbose=True)

    row_norms = np.linalg.norm(mlp.W1, axis=1)
    c_min     = row_norms.max()
    print(f'     c_min = max row-norm W₁ = {c_min:.4f}')

    # ── 2. Experiment A ──────────────────────────────────────────────────
    print('\n[2]  Experiment A — Jacobian attenuation …')
    res_A = experiment_A(mlp, h0=np.array([0.8, 0.4]),
                          s_values=np.logspace(-0.5, 1.8, 60))
    print(f'     Bound always satisfied: '
          f'{np.all(res_A["actual_norm"] <= res_A["bound"] * 1.001)}')
    print(f'     ‖Df‖ range: '
          f'{res_A["actual_norm"].min():.3e} – {res_A["actual_norm"].max():.3e}')

    # ── 3. Experiment B ──────────────────────────────────────────────────
    print('\n[3]  Experiment B — Floquet obstruction (biased vs unbiased) …')
    s_B         = np.logspace(-0.3, 1.4, 40)
    res_B, bias_offsets = experiment_B(mlp, s_values=s_B)
    for c_val, data in res_B.items():
        avoids = 'WITHOUT zero crossings' if data['avoids_zero'] else 'WITH zero crossings'
        print(f'     c={c_val:5.2f}: bound range '
              f'{data["bound"].min():.3e} – {data["bound"].max():.3e}  '
              f'[{avoids}]')

    # ── 4. Experiment D ──────────────────────────────────────────────────
    print('\n[4]  Experiment D — LAJ verification Stuart–Landau …')
    res_D = experiment_D()
    print(f'     Tr(Df_SL) = {res_D["Tr_constant"]:.5f}  (exact: -2.0)')
    print(f'     LAJ integral: {res_D["lndetM_numerical"]:.5f}  '
          f'(exact: -4π = {res_D["lndetM_exact"]:.5f})')
    print(f'     det M numerical: {res_D["detM_numerical"]:.4e}  '
          f'(exact: {res_D["detM_exact"]:.4e})')
    print(f'     Thm. 2 bound: {res_D["bound"]:.4f} ≥ 4π = {4*np.pi:.4f}  '
          f'✓ {res_D["bound"] >= 4*np.pi}')

    # ── 5. Plot generation ─────────────────────────────────────
    print('\n[5]  Generating figures …')
    make_plots(res_A, res_B, res_D, mlp, outdir)

    print(f'\n{sep}')
    print(f'  Done.  Figures: {outdir}/')
    print(sep)


if __name__ == '__main__':
    main()
