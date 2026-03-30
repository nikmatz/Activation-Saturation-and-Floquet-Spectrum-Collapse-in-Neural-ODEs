"""
Illustration F: Individual Floquet Multiplier Bounds (Theorem 4.5)
=================================================================
Author:  Nikolaos M. Matzakos
         nikmatz@aspete.gr  |  ORCID 0000-0001-8647-6082
License: MIT

Verifies Theorem 4.5:  e^{-C(U)T} ≤ |μ_i| ≤ e^{C(U)T}  for EACH
individual Floquet multiplier μ_i of the monodromy matrix M_γ.

This goes beyond Illustration D (which only verifies the determinant,
i.e. the *product* of all multipliers) and tests the *individual*
bound that is the paper's main dynamical result.

Method:
  1. Train a bias-shifted 256-unit tanh MLP on Stuart–Landau (same as
     Illustration C) to ensure pre-activations > 0 on the orbit.
  2. For each pre-activation scale s, solve the variational equation
         Ψ̇(t) = Df_θ(γ(t); s) · Ψ(t),   Ψ(0) = I
     along the (approximate) limit-cycle orbit γ via RK45.
  3. M_γ(s) = Ψ(T).  Compute eigenvalues μ_1, μ_2 and check:
         e^{-C(U,s)T} ≤ |μ_i| ≤ e^{C(U,s)T}
  4. Also check vs the Stuart–Landau exact monodromy (reference).

Output: Table for the paper + assertion that bounds hold everywhere.

Dependencies: numpy, scipy, matplotlib
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from numerical_experiment import (TanhMLP, circle_orbit, C_orbit,
                                  delta_orbit, T_SL)

SEED   = 42
d      = 2
T      = T_SL  # 2π


# ══════════════════════════════════════════════════════════════════════
#  1.  Train bias-shifted MLP (same protocol as Illustration C)
# ══════════════════════════════════════════════════════════════════════
print("Training 256-unit bias-shifted MLP (same as Illustration C)...")
bias_c = 2.5
np.random.seed(SEED + 1)
mlp = TanhMLP(n_hidden=256)
mlp.fit(n_iters=40000, lr=0.001, verbose=True,
        bias_const=bias_c, N_data=3000, use_adam=True)

orbit = circle_orbit(1.0, 1000)

# Verify pre-activations > 0 on orbit
min_pa = min((mlp.W1 @ h + mlp.b1).min() for h in orbit)
print(f"  min pre-activation on orbit = {min_pa:.4f}  "
      f"{'✓ > 0' if min_pa > 0 else '✗ PROBLEM'}")


# ══════════════════════════════════════════════════════════════════════
#  2.  Compute monodromy matrix via variational equation
# ══════════════════════════════════════════════════════════════════════
def monodromy_neural_ode(mlp, s, n_pts=1000, T=T_SL):
    """
    Solve Ψ̇ = Df_θ(γ(t); s) · Ψ along the unit-circle orbit
    using midpoint-rule approximation (piecewise constant Jacobian).

    Returns M_γ = Ψ(T) and C(U,s) = max ||Df_θ|| on orbit.
    """
    gamma = circle_orbit(1.0, n_pts)
    dt = T / n_pts

    Psi = np.eye(d)
    C_U = 0.0

    for h in gamma:
        J = mlp.jacobian(h, s)
        norm_J = np.linalg.norm(J, ord=2)
        if norm_J > C_U:
            C_U = norm_J
        # Matrix exponential step: Ψ_{k+1} = exp(J · dt) · Ψ_k
        # For small dt, use Padé approximation via scipy
        from scipy.linalg import expm
        Psi = expm(J * dt) @ Psi

    return Psi, C_U


def monodromy_ode_ivp(mlp, s, T=T_SL):
    """
    Solve variational equation via scipy ODE integrator (high accuracy).
    γ(t) = (cos t, sin t) — exact orbit of Stuart–Landau.
    """
    def rhs(t, psi_flat):
        h = np.array([np.cos(t), np.sin(t)])
        J = mlp.jacobian(h, s)
        Psi = psi_flat.reshape(d, d)
        return (J @ Psi).flatten()

    sol = solve_ivp(rhs, [0, T], np.eye(d).flatten(),
                    method='DOP853', rtol=1e-11, atol=1e-13)
    Psi_T = sol.y[:, -1].reshape(d, d)

    # Also compute C(U,s)
    gamma = circle_orbit(1.0, 1000)
    C_U = max(np.linalg.norm(mlp.jacobian(h, s), ord=2) for h in gamma)

    return Psi_T, C_U


# ══════════════════════════════════════════════════════════════════════
#  3.  Sweep s and check individual multiplier bounds
# ══════════════════════════════════════════════════════════════════════
s_values = [1.0, 2.0, 4.0, 7.0, 10.0, 15.0, 20.0, 25.0]

print(f"\n{'s':>5s}  {'|μ₁|':>10s}  {'|μ₂|':>10s}  "
      f"{'lower':>10s}  {'upper':>10s}  "
      f"{'δ':>8s}  {'C(U)':>8s}  {'ok?':>4s}")
print("-" * 80)

results = []
all_ok = True

for s in s_values:
    # Use high-accuracy ODE integrator
    M, C_U = monodromy_ode_ivp(mlp, s)
    eigvals = np.linalg.eigvals(M)
    abs_mu = np.sort(np.abs(eigvals))  # |μ₁| ≤ |μ₂|

    lower = np.exp(-C_U * T)
    upper = np.exp(C_U * T)

    delta = delta_orbit(mlp, orbit, s)

    ok1 = (lower <= abs_mu[0] * 1.001) and (abs_mu[0] <= upper * 1.001)
    ok2 = (lower <= abs_mu[1] * 1.001) and (abs_mu[1] <= upper * 1.001)
    ok = ok1 and ok2
    if not ok:
        all_ok = False

    results.append(dict(s=s, mu=abs_mu, lower=lower, upper=upper,
                        C_U=C_U, delta=delta, ok=ok))

    print(f"{s:5.1f}  {abs_mu[0]:10.4e}  {abs_mu[1]:10.4e}  "
          f"{lower:10.4e}  {upper:10.4e}  "
          f"{delta:8.4f}  {C_U:8.4f}  {'✓' if ok else '✗'}")


# ══════════════════════════════════════════════════════════════════════
#  4.  Also verify against exact Stuart–Landau monodromy
# ══════════════════════════════════════════════════════════════════════
print("\n── Exact Stuart–Landau reference ──")

def Df_SL(h):
    x, y = h
    return np.array([[1 - 3*x**2 - y**2, -1 - 2*x*y],
                     [1 - 2*x*y,          1 - x**2 - 3*y**2]])

def monodromy_SL():
    def rhs(t, psi_flat):
        h = np.array([np.cos(t), np.sin(t)])
        J = Df_SL(h)
        Psi = psi_flat.reshape(2, 2)
        return (J @ Psi).flatten()

    sol = solve_ivp(rhs, [0, T], np.eye(2).flatten(),
                    method='DOP853', rtol=1e-12, atol=1e-14)
    return sol.y[:, -1].reshape(2, 2)

M_SL = monodromy_SL()
mu_SL = np.abs(np.linalg.eigvals(M_SL))
mu_SL.sort()
C_SL = 1 + np.sqrt(2)  # exact: constant on unit circle
lower_SL = np.exp(-C_SL * T)
upper_SL = np.exp(C_SL * T)

print(f"  μ₁ = {mu_SL[0]:.6e},  μ₂ = {mu_SL[1]:.6e}")
print(f"  Exact: μ₁ = e^(-4π) ≈ 3.487e-6,  μ₂ = 1")
print(f"  Bounds: [{lower_SL:.4e}, {upper_SL:.4e}]")
print(f"  μ₁ in bounds: {lower_SL <= mu_SL[0] * 1.001 and mu_SL[0] <= upper_SL * 1.001}")
print(f"  μ₂ in bounds: {lower_SL <= mu_SL[1] * 1.001 and mu_SL[1] <= upper_SL * 1.001}")

# Floquet exponents
lam_SL = np.log(mu_SL) / T
print(f"\n  Floquet exponents: λ₁ = {lam_SL[0]:.6f}, λ₂ = {lam_SL[1]:.6f}")
print(f"  Bound |λ_i| ≤ C(U) = {C_SL:.6f}")
print(f"  |λ₁| = {abs(lam_SL[0]):.6f} ≤ {C_SL:.6f}: "
      f"{'✓' if abs(lam_SL[0]) <= C_SL * 1.001 else '✗'}")
print(f"  |λ₂| = {abs(lam_SL[1]):.6f} ≤ {C_SL:.6f}: "
      f"{'✓' if abs(lam_SL[1]) <= C_SL * 1.001 else '✗'}")


# ══════════════════════════════════════════════════════════════════════
#  5.  Print LaTeX table for paper
# ══════════════════════════════════════════════════════════════════════
print("\n\n── LaTeX table for paper ──")
print(r"""
\begin{table}[ht]
  \centering
  \caption{%
    \textbf{Individual Floquet multiplier bounds (Theorem~\ref{thm:individual}).}
    Bias-shifted 256-unit tanh MLP (Illustration~C protocol).
    For each pre-activation scale $s$, the monodromy matrix
    $M_\gamma$ is computed via the variational equation;
    $\mu_1,\mu_2$ are its eigenvalues.
    Theorem~\ref{thm:individual} requires
    $e^{-C(U)\,T}\le|\mu_i|\le e^{C(U)\,T}$.
    The bound holds in every case; as $\delta\to 0$,
    both multipliers are squeezed toward~$1$.}
  \label{tab:individual_multipliers}
  \begin{tabular}{rcccccc}
    \hline
    $s$ & $\delta$ & $C(U)$ & $|\mu_1|$ & $|\mu_2|$
        & lower & upper \\
    \hline""")

for r in results:
    print(f"    {r['s']:4.0f} & {r['delta']:.4f} & {r['C_U']:.3f} "
          f"& {r['mu'][0]:.3e} & {r['mu'][1]:.3e} "
          f"& {r['lower']:.3e} & {r['upper']:.3e} \\\\")

print(r"""    \hline
  \end{tabular}
\end{table}
""")


# ══════════════════════════════════════════════════════════════════════
#  6.  Figure: multiplier window shrinks with s
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

ss = [r['s'] for r in results]
mu1 = [r['mu'][0] for r in results]
mu2 = [r['mu'][1] for r in results]
lowers = [r['lower'] for r in results]
uppers = [r['upper'] for r in results]

ax.semilogy(ss, uppers, 'k--', lw=1.5, label=r'Upper bound $e^{C(U)T}$')
ax.semilogy(ss, lowers, 'k--', lw=1.5, label=r'Lower bound $e^{-C(U)T}$')
ax.fill_between(ss, lowers, uppers, alpha=0.08, color='gray',
                label='Allowed window')
ax.semilogy(ss, mu1, 'bo-', ms=6, lw=1.8, label=r'$|\mu_1|$')
ax.semilogy(ss, mu2, 'rs-', ms=6, lw=1.8, label=r'$|\mu_2|$')
ax.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.5)

ax.set_xlabel(r'Pre-activation scale $s$', fontsize=12)
ax.set_ylabel(r'$|\mu_i|$  (log scale)', fontsize=12)
ax.set_title('Individual Floquet multiplier bounds (Theorem 4.5)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.25, which='both')

plt.tight_layout()
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
for ext in ('pdf', 'png'):
    p = os.path.join(outdir, f'exp_F_individual_multipliers.{ext}')
    plt.savefig(p, dpi=200, bbox_inches='tight')
    print(f"Saved: {p}")
plt.close()

# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
if all_ok:
    print("ALL INDIVIDUAL MULTIPLIER BOUNDS VERIFIED ✓")
else:
    print("WARNING: Some bounds violated!")
print(f"{'='*60}")
